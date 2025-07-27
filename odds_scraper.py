# -*- coding: utf-8 -*-
import re
import time
from pathlib import Path
from urllib.parse import quote_plus, urljoin
from datetime import datetime
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
REQUEST_DELAY = 0.25
TIMEOUT = 10

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
)

def log(msg): print(msg, flush=True)
def sleep(): time.sleep(REQUEST_DELAY)

def get_soup(url):
    sleep()
    r = scraper.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")

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

# --------------------------------------------
# 4) Main
# --------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    in_csv  = here / "ufc_fights.csv"
    out_csv = here / "ufc_fight_odds.csv"

    if not in_csv.exists():
        log(f"[ERR] Datei nicht gefunden: {in_csv}")
        return

    df = pd.read_csv(in_csv)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    else:
        df["event_date"] = pd.NaT

    # Platzhalter „View Matchup“ raus
    mask = (~df["fighter_red"].astype(str).str.contains("View", case=False, na=False)) & \
           (~df["fighter_blue"].astype(str).str.contains("View", case=False, na=False))
    df = df[mask].reset_index(drop=True)

    # Alle Namen -> BFO-URLs
    names = sorted(set(clean_name(x) for x in pd.concat([df["fighter_red"], df["fighter_blue"]]).dropna()))
    name_to_url = {}
    for n in names:
        _, url = search_fighter_url(n)
        name_to_url[n] = url
        log(f"[Fighter URL] {n} -> {url}")

    # Seiten parsen
    name_to_pairs = {}
    for n, url in name_to_url.items():
        if not url:
            name_to_pairs[n] = []
            continue
        try:
            pr = parse_fighter_page_pairs(url)
            name_to_pairs[n] = pr
            log(f"[Parsed] {n}: {len(pr)} fights (pairs)")
        except Exception as e:
            log(f"[ERROR] parsing {n}: {e}")
            name_to_pairs[n] = []

    # Matchen & Output bauen
    out_rows = []
    for _, row in df.iterrows():
        red  = clean_name(row["fighter_red"])
        blue = clean_name(row["fighter_blue"])

        red_url  = name_to_url.get(red)
        blue_url = name_to_url.get(blue)

        red_pairs  = name_to_pairs.get(red, [])
        blue_pairs = name_to_pairs.get(blue, [])

        match = pick_pair_for_row(row, red_pairs, blue_pairs)

        out = {
            "event_name": row.get("event_name"),
            "event_date": row.get("event_date"),
            "fighter_red": red,
            "fighter_blue": blue,
            "red_open": None, "blue_open": None,
            "red_close_low": None, "red_close_high": None,
            "blue_close_low": None, "blue_close_high": None,
            "red_open_decimal": None, "blue_open_decimal": None,
            "red_bfo_url": red_url, "blue_bfo_url": blue_url,
            "bfo_event_url": None,
            "match_status": "no_match"
        }

        if match:
            # Mappe A/B -> rot/blau
            def eq(a,b): return normalize(a) == normalize(b)
            if eq(match["fighter_A"], red) and eq(match["fighter_B"], blue):
                ro, rcg = match["A_open"], match["A_close"]
                bo, bcg = match["B_open"], match["B_close"]
            else:
                ro, rcg = match["B_open"], match["B_close"]
                bo, bcg = match["A_open"], match["A_close"]

            # Wir haben i. d. R. keine Range, sondern einen Close-Wert -> setze low==high==close_guess
            out.update({
                "red_open": ro,
                "blue_open": bo,
                "red_close_low": rcg, "red_close_high": rcg,
                "blue_close_low": bcg, "blue_close_high": bcg,
                "red_open_decimal": american_to_decimal(ro) if ro else None,
                "blue_open_decimal": american_to_decimal(bo) if bo else None,
                "bfo_event_url": match.get("event_url"),
                "match_status": "matched" if (ro or bo or rcg or bcg) else "partial"
            })

        out_rows.append(out)

    out = pd.DataFrame(out_rows)
    cols = [
        "event_name","event_date","fighter_red","fighter_blue",
        "red_open","blue_open",
        "red_close_low","red_close_high","blue_close_low","blue_close_high",
        "red_open_decimal","blue_open_decimal",
        "red_bfo_url","blue_bfo_url","bfo_event_url","match_status"
    ]
    out[cols].to_csv(out_csv, index=False)
    log(f"[DONE] {out_csv} geschrieben: {len(out)} Zeilen")
    log(out["match_status"].value_counts(dropna=False).to_string())
    log("Beispiel:")
    log(out.head(12).to_string())
    return out

if __name__ == "__main__":
    main()
