import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm

BASE_URL = "http://ufcstats.com"

def get_soup(url):
    """Holt den HTML-Content und gibt einen BeautifulSoup-Parser zurück."""
    resp = requests.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def scrape_events_list():
    """
    Scrapt alle abgeschlossenen UFC-Events über alle Seiten hinweg.
    Liefert einen DataFrame mit event_name, event_date, location und link.
    """
    events = []
    page = 1

    while True:
        if page == 1:
            url = f"{BASE_URL}/statistics/events/completed"
        else:
            url = f"{BASE_URL}/statistics/events/completed?page={page}"

        soup = get_soup(url)
        table = soup.find("table", class_="b-statistics__table-events")
        rows = table.tbody.find_all("tr")
        found = 0

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            content   = cols[0].find("i", class_="b-statistics__table-content")
            a_tag     = content.find("a")
            date_span = content.find("span", class_="b-statistics__date")
            events.append({
                "event_name": a_tag.get_text(strip=True),
                "event_date": date_span.get_text(strip=True),
                "location":   cols[1].get_text(strip=True),
                "link":       a_tag["href"]
            })
            found += 1

        if found == 0:
            break  # keine neuen Events auf dieser Seite → Ende
        page += 1
        time.sleep(1)

    return pd.DataFrame(events)

def scrape_fights_for_event(event_name, event_url):
    """Scrapt alle Fights eines Events."""
    soup  = get_soup(event_url)
    table = soup.find("table", class_="b-fight-details__table")
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
        fighter_red, fighter_blue = fighters[0].get_text(strip=True), fighters[1].get_text(strip=True)

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

        # Gewinner: immer der erstgenannte Fighter
        winner = fighter_red

        fights.append({
            "event_name":   event_name,
            "fighter_red":  fighter_red,
            "fighter_blue": fighter_blue,
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
    df_events.to_csv("ufc_events.csv", index=False)
    df_fights.to_csv("ufc_fights.csv", index=False)

if __name__ == "__main__":
    main()
