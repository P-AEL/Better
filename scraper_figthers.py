import string
import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "http://ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; your-bot/0.1)"}

def get_soup(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def get_dob(profile_url):
    """Ziehe DOB von der Fighter-Profilseite. Rückgabe: z.B. 'Jul 03, 1983' oder None."""
    try:
        soup = get_soup(profile_url)
    except Exception:
        return None

    # Struktur auf ufcstats: <li class="b-list__box-list-item"> mit
    # <span class="b-list__box-list-item__title">DOB:</span>
    # <span class="b-list__box-list-item__text">Jul 03, 1983</span>
    for li in soup.select("li.b-list__box-list-item"):
        title = li.find("span", class_="b-list__box-list-item__title")
        text  = li.find("span", class_="b-list__box-list-item__text")
        if title and text and title.get_text(strip=True).startswith("DOB"):
            return text.get_text(strip=True)

    # Fallback: Regex über den ganzen Block
    m = re.search(r"DOB:\s*([A-Za-z]{3}\s+\d{2},\s+\d{4})",
                  soup.get_text(" ", strip=True))
    if m:
        return m.group(1)
    return None

def scrape_all_fighters():
    fighters = []

    for char in string.ascii_uppercase:
        print(f"Scraping fighters starting with '{char}'")
        page = 1
        while True:
            url = f"{BASE_URL}/statistics/fighters?char={char}&page=all"
            if page > 1:
                url += f"&page={page}"
            soup = get_soup(url)

            # finde die richtige Tabelle (erste TH = "First")
            table = None
            for tbl in soup.find_all("table"):
                thead = tbl.find("thead")
                if not thead:
                    continue
                ths = thead.find_all("th")
                if ths and ths[0].get_text(strip=True).lower() == "first":
                    table = tbl
                    break

            if table is None:
                break

            rows = table.tbody.find_all("tr")
            if not rows:
                break

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 11:
                    continue
                link = cols[0].find("a")
                if not link:
                    continue

                first = link.get_text(strip=True)
                last  = cols[1].get_text(strip=True)
                profile_url = link["href"]

                print(f"Scrape Fighter: {first} {last}")
                dob = get_dob(profile_url)  # <-- HIER wird DOB geholt

                fighters.append({
                    "first_name":  first,
                    "last_name":   last,
                    "nickname":    cols[2].get_text(strip=True) or None,
                    "height":      cols[3].get_text(strip=True),
                    "weight":      cols[4].get_text(strip=True),
                    "reach":       cols[5].get_text(strip=True),
                    "stance":      cols[6].get_text(strip=True),
                    "wins":        cols[7].get_text(strip=True),
                    "losses":      cols[8].get_text(strip=True),
                    "draws":       cols[9].get_text(strip=True),
                    "belt":        cols[10].get_text(strip=True) or None,
                    "profile_url": profile_url,
                    "dob":         dob
                })

                time.sleep(0.3)  # kleiner Delay pro Profil-Call

            page += 1
            time.sleep(0.5)  # polite delay zwischen fighter-list Seiten
            break  # deine ursprüngliche Logik: nur page=all einmal laufen
    return pd.DataFrame(fighters)

def main():
    df = scrape_all_fighters()
    print(f"Gefundene Fighter: {len(df)}")
    df.to_csv("ufc_fighters_basic_with_dob.csv", index=False)
    print("Gespeichert als ufc_fighters_basic_with_dob.csv")

if __name__ == "__main__":
    main()
