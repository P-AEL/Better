# -*- coding: utf-8 -*-
import csv
import io
import re
import time
from pathlib import Path
from urllib.parse import quote_plus, urljoin
from datetime import datetime, timezone
import pandas as pd
from bs4 import BeautifulSoup
import cloudscraper

try:
    from rapidfuzz import process, fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

BASE = "https://www.bestfightodds.com"
SEARCH_URL = BASE + "/search?query={query}"
REQUEST_DELAY = 0.20
TIMEOUT = 8
MAX_RETRIES = 3
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "scraped_data"
ODDS_COLUMNS = [
    "event_name", "event_date", "fighter_red", "fighter_blue",
    "red_open", "blue_open",
    "red_close_low", "red_close_high", "blue_close_low", "blue_close_high",
    "red_open_decimal", "blue_open_decimal",
    "red_bfo_url", "blue_bfo_url",
    "bfo_event_date", "bfo_event_url", "match_status",
]

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
)

def log(msg): print(msg, flush=True)
def sleep(): time.sleep(REQUEST_DELAY)

def get_soup(url):
    for attempt in range(MAX_RETRIES):
        try:
            sleep()
            r = scraper.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)

def clean_name(s):
    if s is None: return None
    return re.sub(r"\s+", " ", str(s)).strip()

def strip_ordinals(s):  # "Jul 26th 2025" -> "Jul 26 2025"
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s or "")

def parse_bfo_date(text):
    if not text: return None
    text = strip_ordinals(text.strip())
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except Exception:
            pass
    return None

def american_to_decimal(odd_str):
    try:
        o = int(str(odd_str).strip())
    except Exception:
        return None
    return 1 + (o/100.0) if o > 0 else 1 + (100.0/abs(o))

# ---------------------------
# 1) Fighter auf BFO finden
# ---------------------------
def search_fighter_url(query_name):
    url = SEARCH_URL.format(query=quote_plus(query_name))
    soup = get_soup(url)
    candidates = []
    for a in soup.select('a[href*="/fighters/"]'):
        href = a.get("href") or ""
        text = a.get_text(" ", strip=True)
        if "/fighters/" in href and text:
            candidates.append((clean_name(text), urljoin(BASE, href)))
    if not candidates:
        return (None, None)

    qn = clean_name(query_name).lower()
    for name, link in candidates:
        if name and name.lower() == qn:
            return (name, link)
    if HAVE_RAPIDFUZZ:
        names = [c[0] for c in candidates]
        best = process.extractOne(query_name, names, scorer=fuzz.WRatio)
        if best and best[1] >= 90:
            return candidates[names.index(best[0])]
    return candidates[0]

# --------------------------------------------
# 2) Fighter-Seite parsen (Zeilenpaare)
#    Wir lesen: Open, Current, "Closing-guess"
# --------------------------------------------
def parse_moneyline_cells(tr):
    """Gibt (open, current, close_guess) als Strings zurück (z.B. '+152')."""
    mls = tr.select("td.moneyline span")
    vals = [ml.get_text(strip=True) for ml in mls if ml.get_text(strip=True)]
    # Heuristik: 1. = Open, 2. = Current, letzte = "Closing/Range"-Wert (falls nur 1 Zahl, nutzen wir sie)
    open_v   = vals[0] if len(vals) >= 1 and re.match(r'^[+-]?\d{2,4}$', vals[0]) else None
    current  = vals[1] if len(vals) >= 2 and re.match(r'^[+-]?\d{2,4}$', vals[1]) else None
    close_v  = None
    # suche irgendeinen moneyline-Wert weiter rechts als "closing guess"
    for v in reversed(vals):
        if re.match(r'^[+-]?\d{2,4}$', v):
            close_v = v
            break
    # Fallbacks
    if close_v is None: close_v = current or open_v
    if current is None: current = close_v or open_v
    return open_v, current, close_v

def parse_event_info_from_row(tr):
    """Versucht Datum + Event-Link aus der (Gegner-)Zeile zu lesen."""
    # Datum sitzt oft in td.item-non-mobile
    date = None
    event_url = None
    # Datum
    for td in tr.select("td.item-non-mobile"):
        dt = parse_bfo_date(td.get_text(" ", strip=True))
        if dt:
            date = dt
            break
    # Event-Link (kann fehlen!)
    a = tr.select_one('a[href*="/events/"]')
    if a and a.get("href"):
        event_url = urljoin(BASE, a["href"])
    return date, event_url

def parse_fighter_page_pairs(fighter_url):
    soup = get_soup(fighter_url)
    pairs = []

    # Page Fighter Name (zur Kontrolle)
    page_name_el = soup.select_one("h1")
    page_name = clean_name(page_name_el.get_text(strip=True)) if page_name_el else None

    tbody = soup.select_one("tbody")
    if not tbody:
        # Fallback: nimm alle tr im Dokument
        trs = soup.select("tr")
    else:
        trs = tbody.find_all("tr", recursive=False) or tbody.select("tr")

    # Iterate: eine "main-row" + nächste Zeile (= Gegner)
    for i, tr in enumerate(trs):
        if "main-row" not in (tr.get("class") or []):
            continue

        # Zeile 1: page fighter (A)
        A_open, A_curr, A_close = parse_moneyline_cells(tr)

        # nächste Zeile suchen (B)
        if i + 1 >= len(trs):
            continue
        tr_b = trs[i+1]

        # Gegner-Name in th.oppcell a
        opp_a = tr_b.select_one("th.oppcell a")
        opp_name = clean_name(opp_a.get_text(strip=True)) if opp_a else None
        if not opp_name:
            continue

        # Zeile 2: Gegner (B)
        B_open, B_curr, B_close = parse_moneyline_cells(tr_b)
        ev_date, ev_url = parse_event_info_from_row(tr_b)

        pairs.append({
            "event_date": ev_date,
            "event_url": ev_url,
            "fighter_A": page_name,   # Seiten-Fighter (z. B. Robert Whittaker)
            "fighter_B": opp_name,    # Gegner
            "fighter_A_url": fighter_url,
            "fighter_B_url": urljoin(BASE, opp_a["href"]) if opp_a.get("href") else None,
            "A_open": A_open, "A_curr": A_curr, "A_close": A_close,
            "B_open": B_open, "B_curr": B_curr, "B_close": B_close
        })

    return pairs

# --------------------------------------------
# 3) Matching: df-Zeile -> passendes Paar
# --------------------------------------------
def normalize(s): return clean_name(s).lower() if s else None

def pick_pair_for_row(row, red_pairs, blue_pairs):
    red  = normalize(row.get("fighter_red"))
    blue = normalize(row.get("fighter_blue"))
    evd  = row.get("event_date")
    if pd.notnull(evd):
        try: evd = pd.to_datetime(evd).date()
        except Exception: evd = None

    # Kandidaten: alle Paare, in denen {red,blue} vorkommen (Reihenfolge egal)
    def both_in(p):
        names = {normalize(p["fighter_A"]), normalize(p["fighter_B"])}
        return red in names and blue in names

    cands = [p for p in (red_pairs + blue_pairs) if both_in(p)]
    if not cands:
        return None

    # scorings: exakt gleiches datum bevorzugen, sonst jüngstes mit irgendeinem datum
    def score(p):
        s = 0
        if evd and p["event_date"] == evd: s += 2
        if p["event_date"]: s += 1
        return s

    cands.sort(key=lambda p: (score(p), p["event_date"] or datetime(1900,1,1).date()), reverse=True)
    return cands[0]

def load_next_event_fights(data_dir=DATA_DIR, today=None):
    data_dir = Path(data_dir)
    fights = pd.read_csv(data_dir / "ufc_fights.csv")
    events = pd.read_csv(data_dir / "ufc_events.csv")
    scheduled = fights[fights["result"].eq("scheduled")].copy()
    if scheduled.empty:
        raise RuntimeError("No scheduled fights are available")

    events = events[["event_name", "event_date"]].copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    scheduled = scheduled.merge(events, on="event_name", how="left")
    event_dates = scheduled[["event_name", "event_date"]].drop_duplicates()
    today = today or datetime.now(timezone.utc).date()
    upcoming = event_dates[event_dates["event_date"].dt.date >= today]
    candidates = upcoming if not upcoming.empty else event_dates.dropna(subset=["event_date"])
    if candidates.empty:
        event_name = scheduled.iloc[0]["event_name"]
    else:
        event_name = candidates.sort_values("event_date").iloc[0]["event_name"]

    selected = scheduled[scheduled["event_name"].eq(event_name)].copy()
    selected["event_date"] = selected["event_date"].dt.date
    return event_name, selected.reset_index(drop=True)


def empty_odds_row(row):
    return {
        "event_name": row.get("event_name"),
        "event_date": row.get("event_date"),
        "fighter_red": clean_name(row.get("fighter_red")),
        "fighter_blue": clean_name(row.get("fighter_blue")),
        "red_open": None, "blue_open": None,
        "red_close_low": None, "red_close_high": None,
        "blue_close_low": None, "blue_close_high": None,
        "red_open_decimal": None, "blue_open_decimal": None,
        "red_bfo_url": None, "blue_bfo_url": None,
        "bfo_event_date": None, "bfo_event_url": None,
        "match_status": "no_match",
    }


def fetch_fighter_pairs(name, cache):
    if name in cache:
        return cache[name]
    try:
        found_name, url = search_fighter_url(name)
        pairs = parse_fighter_page_pairs(url) if url else []
        for pair in pairs:
            pair["fighter_A"] = found_name or name
        cache[name] = (url, pairs)
        log(f"[Parsed] {name}: {len(pairs)} fights from {url}")
    except Exception as exc:
        log(f"[WARN] Could not fetch {name}: {exc}")
        cache[name] = (None, [])
    return cache[name]


def build_event_odds(event_fights):
    cache = {}
    rows = []
    for _, fight in event_fights.iterrows():
        red = clean_name(fight["fighter_red"])
        blue = clean_name(fight["fighter_blue"])
        red_url, red_pairs = fetch_fighter_pairs(red, cache)
        match = pick_pair_for_row(fight, red_pairs, [])

        blue_url = None
        if match is None:
            blue_url, blue_pairs = fetch_fighter_pairs(blue, cache)
            match = pick_pair_for_row(fight, red_pairs, blue_pairs)

        out = empty_odds_row(fight)
        out["red_bfo_url"] = red_url
        out["blue_bfo_url"] = blue_url
        if match:
            same_orientation = (
                normalize(match["fighter_A"]) == normalize(red)
                and normalize(match["fighter_B"]) == normalize(blue)
            )
            if same_orientation:
                ro, rcg = match["A_open"], match["A_close"]
                bo, bcg = match["B_open"], match["B_close"]
                out["red_bfo_url"] = match.get("fighter_A_url") or red_url
                out["blue_bfo_url"] = match.get("fighter_B_url") or blue_url
            else:
                ro, rcg = match["B_open"], match["B_close"]
                bo, bcg = match["A_open"], match["A_close"]
                out["red_bfo_url"] = match.get("fighter_B_url") or red_url
                out["blue_bfo_url"] = match.get("fighter_A_url") or blue_url

            ro, bo = ro or rcg, bo or bcg
            rcg, bcg = rcg or ro, bcg or bo
            complete = all((ro, bo, rcg, bcg))
            out.update({
                "red_open": ro,
                "blue_open": bo,
                "red_close_low": rcg, "red_close_high": rcg,
                "blue_close_low": bcg, "blue_close_high": bcg,
                "red_open_decimal": american_to_decimal(ro),
                "blue_open_decimal": american_to_decimal(bo),
                "bfo_event_date": match.get("event_date"),
                "bfo_event_url": match.get("event_url"),
                "match_status": "matched" if complete else "partial",
            })
        rows.append(out)
    return pd.DataFrame(rows, columns=ODDS_COLUMNS)


def write_event_odds(path, current, event_name):
    path = Path(path)
    if not path.exists():
        current.to_csv(path, index=False)
        return

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        current.to_csv(path, index=False)
        return

    header = lines[0]
    columns = next(csv.reader([header]))
    event_index = columns.index("event_name")
    normalized_event = normalize(event_name)
    kept = []
    for line in lines[1:]:
        values = next(csv.reader([line]))
        if len(values) > event_index and normalize(values[event_index]) != normalized_event:
            kept.append(line)

    newline = "\r\n" if header.endswith("\r\n") else "\n"
    buffer = io.StringIO(newline="")
    current.to_csv(buffer, index=False, header=False, lineterminator=newline)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(header)
        handle.write(buffer.getvalue())
        handle.writelines(kept)


def main(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    out_csv = data_dir / "ufc_fight_odds.csv"
    event_name, event_fights = load_next_event_fights(data_dir)
    log(f"[Next event] {event_name}: {len(event_fights)} fights")
    current = build_event_odds(event_fights)
    write_event_odds(out_csv, current, event_name)
    out = pd.read_csv(out_csv)
    log(f"[DONE] Updated {out_csv}: {len(out)} total rows")
    log(current["match_status"].value_counts(dropna=False).to_string())
    return out

if __name__ == "__main__":
    main()
