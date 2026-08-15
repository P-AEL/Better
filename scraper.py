import re
import hashlib
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = "http://ufcstats.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 30
MAX_RETRIES = 3
MAX_CHALLENGE_ATTEMPTS = 5_000_000

session = requests.Session()
session.headers.update(HEADERS)


def normalize_ufcstats_url(url):
    """Use UFCStats' reachable canonical host for relative and absolute links."""
    if not url:
        return url

    url = urljoin(BASE_URL, url)
    return (
        url.replace("https://www.ufcstats.com", BASE_URL, 1)
        .replace("https://ufcstats.com", BASE_URL, 1)
        .replace("http://www.ufcstats.com", BASE_URL, 1)
    )


def page_summary(soup):
    title = soup.title.get_text(" ", strip=True) if soup.title else "no title"
    preview = soup.get_text(" ", strip=True)[:300]
    return f"Page title: {title!r}. Preview: {preview!r}"


def solve_browser_challenge(html, url):
    """Solve UFCStats' small JavaScript proof-of-work challenge without executing JS."""
    match = re.search(
        r'var\s+nonce\s*=\s*"([^"]+)".*?new Array\((\d+)\s*\+\s*1\)',
        html,
        flags=re.DOTALL,
    )
    if not match:
        return False

    nonce, difficulty_text = match.groups()
    difficulty = int(difficulty_text)
    if not 1 <= difficulty <= 8:
        raise RuntimeError(f"Unsupported UFCStats challenge difficulty: {difficulty}")

    prefix = "0" * difficulty
    for answer in range(MAX_CHALLENGE_ATTEMPTS):
        digest = hashlib.sha256(f"{nonce}:{answer}".encode("utf-8")).hexdigest()
        if digest.startswith(prefix):
            challenge_url = urljoin(url, "/__c")
            response = session.post(
                challenge_url,
                data={"nonce": nonce, "n": answer},
                headers={"Referer": url},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            return True

    raise RuntimeError(
        f"UFCStats challenge exceeded {MAX_CHALLENGE_ATTEMPTS} attempts"
    )


def get_soup(url, required_selector=None):
    """Holt den HTML-Content und gibt einen BeautifulSoup-Parser zurück."""
    url = normalize_ufcstats_url(url)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            resp.raise_for_status()

            if solve_browser_challenge(resp.text, url):
                resp = session.get(url, timeout=TIMEOUT)
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            if required_selector:
                table = soup.select_one(required_selector)
                if not table or not table.tbody:
                    last_error = RuntimeError(
                        f"Expected table '{required_selector}' not found at {url}. "
                        f"{page_summary(soup)}"
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(attempt * 2)
                        continue
                    break

            return soup
        except requests.RequestException as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(attempt * 2)

    raise RuntimeError(f"Could not fetch valid UFCStats data from {url}: {last_error}") from last_error


def require_table(soup, selector, url):
    table = soup.select_one(selector)
    if table and table.tbody:
        return table

    raise RuntimeError(
        f"Expected table '{selector}' not found at {url}. "
        f"{page_summary(soup)}"
    )

def scrape_events_list():
    """
    Scrapt alle abgeschlossenen UFC-Events.
    Liefert einen DataFrame mit event_name, event_date, location und link.
    """
    events = []
    url = f"{BASE_URL}/statistics/events/completed?page=all"
    soup = get_soup(url, required_selector="table.b-statistics__table-events")
    table = require_table(soup, "table.b-statistics__table-events", url)

    for row in table.tbody.find_all("tr"):
        cols = row.find_all("td")
        a_tag = row.select_one('a[href*="/event-details/"]')
        date_span = row.select_one(".b-statistics__date")

        if len(cols) < 2 or not a_tag or not date_span:
            continue

        events.append({
            "event_name": a_tag.get_text(strip=True),
            "event_date": date_span.get_text(strip=True),
            "location":   cols[1].get_text(strip=True),
            "link":       normalize_ufcstats_url(a_tag["href"])
        })

    if not events:
        raise RuntimeError(f"No completed UFC events parsed from {url}")

    return pd.DataFrame(events)

def scrape_fights_for_event(event_name, event_url):
    """Scrapt alle Fights eines Events."""
    event_url = normalize_ufcstats_url(event_url)
    soup  = get_soup(event_url, required_selector="table.b-fight-details__table")
    table = require_table(soup, "table.b-fight-details__table", event_url)
    fights = []

    for row in table.tbody.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 10:
            continue

        # fight-details-Link aus onclick oder aus <a>
        onclick = row.get("onclick", "")
        m = re.search(r"href='([^']+)'", onclick)
        fight_link = m.group(1) if m else None
        if not fight_link:
            link_tag = cols[0].find("a")
            fight_link = link_tag["href"] if link_tag else None

        # Fighter
        fighters = cols[1].find_all("a")
        if len(fighters) < 2:
            continue
        fighter_red, fighter_blue = fighters[0].get_text(strip=True), fighters[1].get_text(strip=True)

        result_tokens = [text.lower() for text in cols[0].stripped_strings]
        if "win" in result_tokens:
            result = "win"
            winner = fighter_red
        elif result_tokens and all(text == "draw" for text in result_tokens):
            result = "draw"
            winner = None
        elif result_tokens and all(text == "nc" for text in result_tokens):
            result = "nc"
            winner = None
        else:
            result = "scheduled"
            winner = None

        # KD, STR, TD, SUB (je zwei Werte pro Zelle)
        def parse_two(cell):
            parts = cell.get_text(separator="|", strip=True).split("|")
            return parts[0], parts[1] if len(parts) > 1 else None

        kd_red,  kd_blue  = parse_two(cols[2])
        str_red, str_blue = parse_two(cols[3])
        td_red,  td_blue  = parse_two(cols[4])
        sub_red, sub_blue = parse_two(cols[5])

        weight_class = cols[6].get_text(strip=True)
        method       = cols[7].get_text(separator=" ", strip=True)
        rnd          = cols[8].get_text(strip=True)
        time_text    = cols[9].get_text(strip=True)

        fights.append({
            "event_name":   event_name,
            "fighter_red":  fighter_red,
            "fighter_blue": fighter_blue,
            "result":       result,
            "winner":       winner,
            "kd_red":       kd_red,
            "kd_blue":      kd_blue,
            "str_red":      str_red,
            "str_blue":     str_blue,
            "td_red":       td_red,
            "td_blue":      td_blue,
            "sub_red":      sub_red,
            "sub_blue":     sub_blue,
            "weight_class": weight_class,
            "method":       method,
            "round":        rnd,
            "time":         time_text,
            "fight_link":   fight_link
        })

    return fights

def main():
    # 1) Alle Events (mit Pagination) scrapen
    df_events = scrape_events_list()
    print(f"Gefundene Events: {len(df_events)}")

    # 2) Fights pro Event scrapen
    all_fights = []
    for _, ev in tqdm(df_events.iterrows(), total=len(df_events), desc="Events"):
        all_fights.extend(scrape_fights_for_event(ev.event_name, ev.link))
        time.sleep(1)

    df_fights = pd.DataFrame(all_fights)
    print(f"Gefundene Fights: {len(df_fights)}")

    # 3) CSVs speichern
    out_dir = Path("data/scraped_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_events.to_csv(out_dir / "ufc_events.csv", index=False)
    df_fights.to_csv(out_dir / "ufc_fights.csv", index=False)

if __name__ == "__main__":
    main()
