import unittest
from unittest.mock import Mock, patch
import hashlib

from bs4 import BeautifulSoup
import requests

import scraper


EVENTS_HTML = """
<html><head><title>Stats | UFC</title></head><body>
  <table class="b-statistics__table-events"><tbody>
    <tr><th>Name/date</th><th>Location</th></tr>
    <tr>
      <td><i class="b-statistics__table-content">
        <a href="http://ufcstats.com/event-details/event-1">UFC Test</a>
        <span class="b-statistics__date">August 01, 2026</span>
      </i></td>
      <td>Las Vegas, Nevada, USA</td>
    </tr>
  </tbody></table>
</body></html>
"""

FIGHTS_HTML = """
<html><head><title>Stats | UFC</title></head><body>
  <table class="b-fight-details__table"><tbody>
    <tr onclick="location.href='http://ufcstats.com/fight-details/fight-1'">
      <td><p>W</p><p>L</p></td>
      <td><p><a>Red Fighter</a></p><p><a>Blue Fighter</a></p></td>
      <td><p>1</p><p>0</p></td>
      <td><p>25</p><p>18</p></td>
      <td><p>2</p><p>1</p></td>
      <td><p>0</p><p>1</p></td>
      <td>Lightweight</td>
      <td>Decision Unanimous</td>
      <td>3</td>
      <td>5:00</td>
    </tr>
  </tbody></table>
</body></html>
"""

CHALLENGE_HTML = """
<html><head><title>Loading...</title></head><body>
<p>Checking your browser...</p>
<script>
var nonce="test-nonce", target=new Array(1+1).join('0');
</script>
</body></html>
"""


class ScraperTests(unittest.TestCase):
    def test_normalizes_ufcstats_urls(self):
        expected = "http://ufcstats.com/event-details/abc"
        self.assertEqual(scraper.normalize_ufcstats_url("/event-details/abc"), expected)
        self.assertEqual(
            scraper.normalize_ufcstats_url("http://ufcstats.com/event-details/abc"),
            expected,
        )

    @patch("scraper.get_soup")
    def test_scrapes_event_list(self, get_soup):
        get_soup.return_value = BeautifulSoup(EVENTS_HTML, "html.parser")

        events = scraper.scrape_events_list()

        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].event_name, "UFC Test")
        self.assertEqual(events.iloc[0].event_date, "August 01, 2026")
        self.assertEqual(events.iloc[0].location, "Las Vegas, Nevada, USA")
        self.assertEqual(
            events.iloc[0].link,
            "http://ufcstats.com/event-details/event-1",
        )

    @patch("scraper.get_soup")
    def test_scrapes_fight(self, get_soup):
        get_soup.return_value = BeautifulSoup(FIGHTS_HTML, "html.parser")

        fights = scraper.scrape_fights_for_event(
            "UFC Test", "http://ufcstats.com/event-details/event-1"
        )

        self.assertEqual(len(fights), 1)
        self.assertEqual(fights[0]["fighter_red"], "Red Fighter")
        self.assertEqual(fights[0]["fighter_blue"], "Blue Fighter")
        self.assertEqual(fights[0]["str_red"], "25")
        self.assertEqual(fights[0]["str_blue"], "18")
        self.assertEqual(
            fights[0]["fight_link"],
            "http://ufcstats.com/fight-details/fight-1",
        )

    @patch("scraper.time.sleep")
    @patch("scraper.session.post")
    @patch("scraper.session.get")
    def test_solves_browser_challenge(self, session_get, session_post, _sleep):
        challenge = Mock()
        challenge.text = CHALLENGE_HTML
        challenge.raise_for_status.return_value = None
        valid = Mock()
        valid.text = EVENTS_HTML
        valid.raise_for_status.return_value = None
        session_get.side_effect = [challenge, valid]
        session_post.return_value.raise_for_status.return_value = None

        soup = scraper.get_soup(
            "http://ufcstats.com/statistics/events/completed?page=all",
            required_selector="table.b-statistics__table-events",
        )

        self.assertIsNotNone(soup.select_one("table.b-statistics__table-events"))
        challenge_data = session_post.call_args.kwargs["data"]
        digest = hashlib.sha256(
            f"test-nonce:{challenge_data['n']}".encode("utf-8")
        ).hexdigest()
        self.assertTrue(digest.startswith("0"))
        self.assertEqual(session_get.call_count, 2)

    @patch("scraper.time.sleep")
    @patch("scraper.session.get")
    def test_retries_page_without_expected_table(self, session_get, _sleep):
        invalid = Mock()
        invalid.text = "<html><title>Temporary error</title><body>Try again</body></html>"
        invalid.raise_for_status.return_value = None
        valid = Mock()
        valid.text = EVENTS_HTML
        valid.raise_for_status.return_value = None
        session_get.side_effect = [invalid, valid]

        soup = scraper.get_soup(
            "http://ufcstats.com/statistics/events/completed?page=all",
            required_selector="table.b-statistics__table-events",
        )

        self.assertIsNotNone(soup.select_one("table.b-statistics__table-events"))
        self.assertEqual(session_get.call_count, 2)

    @patch("scraper.time.sleep")
    @patch("scraper.session.get")
    def test_reports_invalid_page_after_retries(self, session_get, _sleep):
        response = Mock()
        response.text = "<html><title>Access denied</title><body>Blocked</body></html>"
        response.raise_for_status.return_value = None
        session_get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "Access denied"):
            scraper.get_soup(
                "https://www.ufcstats.com/statistics/events/completed?page=all",
                required_selector="table.b-statistics__table-events",
            )

        self.assertEqual(session_get.call_count, scraper.MAX_RETRIES)

    @patch("scraper.time.sleep")
    @patch("scraper.session.get")
    def test_retries_request_error(self, session_get, _sleep):
        valid = Mock()
        valid.text = EVENTS_HTML
        valid.raise_for_status.return_value = None
        session_get.side_effect = [requests.ConnectionError("offline"), valid]

        soup = scraper.get_soup(
            "https://www.ufcstats.com/statistics/events/completed?page=all",
            required_selector="table.b-statistics__table-events",
        )

        self.assertEqual(soup.title.get_text(strip=True), "Stats | UFC")
        self.assertEqual(session_get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
