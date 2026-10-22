"""Download the latest 10-K for a ticker from SEC EDGAR."""
import json
import time

import requests

from .config import RAW_DIR, SEC_USER_AGENT

HEADERS = {"User-Agent": SEC_USER_AGENT}
_ticker_map: dict | None = None


def _get(url: str) -> requests.Response:
    time.sleep(0.2)  # stay well under SEC's 10 req/s limit
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def cik_for(ticker: str) -> str:
    global _ticker_map
    if _ticker_map is None:
        data = _get("https://www.sec.gov/files/company_tickers.json").json()
        _ticker_map = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}
    return _ticker_map[ticker.upper().replace(".", "-")]


def latest_10k(ticker: str) -> dict:
    """Return {url, period, filing_date} for the most recent 10-K."""
    cik = cik_for(ticker)
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    recent = sub["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            acc = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]
            report_date = recent["reportDate"][i]
            return {
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}",
                "period": f"FY{report_date[:4]}",
                "filing_date": recent["filingDate"][i],
            }
    raise LookupError(f"no 10-K found for {ticker}")


def download_10k(ticker: str) -> tuple[str, dict]:
    """Download (with on-disk cache) and return (html, meta)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RAW_DIR / f"{ticker}_10k.json"
    html_path = RAW_DIR / f"{ticker}_10k.html"
    if meta_path.exists() and html_path.exists():
        return html_path.read_text(), json.loads(meta_path.read_text())
    meta = latest_10k(ticker)
    html = _get(meta["url"]).text
    html_path.write_text(html)
    meta_path.write_text(json.dumps(meta))
    return html, meta
