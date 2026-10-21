"""FastAPI backend for the efficient-frontier app.

Thin JSON layer over the existing Python brains (engine.py / data.py / universe.py) —
all the MPT math is reused unchanged. Serves the built React app (web/dist) at /.

Run (dev):   uvicorn api:app --reload --port 8000      (from the frontier/ dir)
"""
from __future__ import annotations

import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
import data
import engine
import universe

app = FastAPI(title="Efficient Frontier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=4)
def _riskfree(_bucket: int):
    try:
        import yfinance as yf
        s = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(s.iloc[-1]) / 100, True
    except Exception:
        return 0.041, False


# Full ~610-asset solves take minutes on the 0.1-CPU free tier — longer than one HTTP request
# can hang on — so /api/solve runs them as background jobs the page polls. _solve_lock serializes
# the compute so only one runs at a time (peak RAM stays ~one solve, well under 512 MB); _jobs
# doubles as the result cache, capped so memory stays bounded.
# ponytail: in-memory job store — fine for a single instance; it just re-solves after a restart.
_solve_lock = threading.Lock()
_jobs_lock = threading.Lock()
_jobs: "OrderedDict[str, dict]" = OrderedDict()   # param-key -> {status, result, error}
_JOBS_MAX = 32


def _run_solve(key: str, args: tuple) -> None:
    try:
        with _solve_lock:
            res = _solve(*args)
        with _jobs_lock:
            if key in _jobs:
                _jobs[key].update(status="done", result=res)
    except Exception as e:                            # a bad param set marks the job, never wedges it
        with _jobs_lock:
            if key in _jobs:
                _jobs[key].update(status="error", error=str(e))


def _solve(cov_model: str, return_model: str, rf: float, erp: float, bl_conf: float,
           include_stocks: bool, max_assets: int, n_points: int, max_holdings: int) -> dict:
    u = universe.build_universe(include_stocks=include_stocks)
    tickers = u["ticker"].tolist()[:max_assets]
    px = data.download_history(tickers, period="2y")
    mu, cov, tk = data.estimate_inputs(px, cov_model=cov_model, return_model=return_model,
                                       rf=rf, erp=erp, bl_conf=bl_conf)
    meta = u.set_index("ticker").loc[tk]
    sigma = np.sqrt(np.diag(cov))
    # full universe for the scatter
    assets = [{"ticker": tk[i], "name": str(meta.iloc[i]["name"]),
               "category": str(meta.iloc[i]["category"]),
               "vol": round(float(sigma[i]) * 100, 3), "ret": round(float(mu[i]) * 100, 3)}
              for i in range(len(tk))]

    # holdings cap: build the frontier on just the top-N names (ranked by the full
    # tangency weight), so everything stays a coherent <=N-asset portfolio.
    if 0 < max_holdings < len(tk):
        rank = engine.tangency(mu, cov, rf)["w"]
        hold = sorted(int(i) for i in np.argsort(rank)[::-1][:max_holdings])
    else:
        hold = list(range(len(tk)))
    muH, covH = mu[hold], cov[np.ix_(hold, hold)]
    tkH = [tk[i] for i in hold]
    nmH = [str(meta.iloc[i]["name"]) for i in hold]

    fr = engine.efficient_frontier(muH, covH, n_points=n_points)
    tan = engine.tangency(muH, covH, rf)
    w_mv = engine.min_variance(muH, covH)
    mv_ret, mv_vol = engine.port_stats(w_mv, muH, covH)

    return {
        "assets": assets,
        "frontier": [{"vol": round(p["vol"] * 100, 4), "ret": round(p["ret"] * 100, 4)} for p in fr],
        "frontier_w": [[round(float(x), 5) for x in p["w"]] for p in fr],
        "tangency": {"vol": tan["vol"] * 100, "ret": tan["ret"] * 100, "sharpe": tan["sharpe"],
                     "w": [round(float(x), 5) for x in tan["w"]]},
        "minvar": {"vol": mv_vol * 100, "ret": mv_ret * 100},
        "rf": rf, "n_assets": len(tk), "tickers": tkH, "names": nmH,
    }


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/riskfree")
def riskfree():
    rate, live = _riskfree(int(time.time() // 900))
    return {"rate": rate, "is_live": live}


@app.get("/api/solve")
def solve(cov_model: str = "ledoit_wolf", return_model: str = "capm", rf: float = 0.041,
          erp: float = 0.05, bl_conf: float = 1.0, include_stocks: bool = True,
          max_assets: int = 400, n_points: int = 30, max_holdings: int = 0):
    """Start (or reuse) a solve job. Returns the result immediately if it's already cached,
    else a job id to poll at /api/solve_result — keeps the long full-universe solve off the
    HTTP timeout."""
    args = (cov_model, return_model, round(rf, 4), round(erp, 4), round(bl_conf, 2),
            include_stocks, max_assets, n_points, max_holdings)
    key = repr(args)
    with _jobs_lock:
        j = _jobs.get(key)
        if j and j["status"] == "done":
            _jobs.move_to_end(key)
            return {"status": "done", "result": j["result"]}
        if j and j["status"] == "running":
            return {"status": "running", "job": key}
        _jobs[key] = {"status": "running", "result": None, "error": None}
        _jobs.move_to_end(key)
        while len(_jobs) > _JOBS_MAX:
            _jobs.popitem(last=False)                 # evict the oldest cached solve
    threading.Thread(target=_run_solve, args=(key, args), daemon=True).start()
    return {"status": "running", "job": key}


@app.get("/api/solve_result")
def solve_result(job: str):
    """Poll a solve job: status is running | done (+result) | error (+error) | unknown (re-POST)."""
    with _jobs_lock:
        j = _jobs.get(job)
        if j is None:
            return {"status": "unknown"}
        if j["status"] == "done":
            _jobs.move_to_end(job)
            return {"status": "done", "result": j["result"]}
        if j["status"] == "error":
            return {"status": "error", "error": j["error"]}
        return {"status": "running"}


class PricesReq(BaseModel):
    tickers: list[str]


@app.post("/api/prices")
def prices(req: PricesReq):
    pv = data.live_prices(req.tickers).reindex(req.tickers).ffill()
    return {t: (float(pv[t]) if pd.notna(pv[t]) else None) for t in req.tickers}


# range token -> (yfinance period, interval). Intraday for short windows, daily/weekly for long.
_HIST_RANGE = {
    "1d": ("1d", "5m"), "5d": ("5d", "30m"), "1mo": ("1mo", "1d"),
    "1y": ("1y", "1d"), "ytd": ("ytd", "1d"), "5y": ("5y", "1wk"),
}


@lru_cache(maxsize=512)
def _history(ticker: str, rng: str, _bucket: int) -> dict:
    import yfinance as yf
    period, interval = _HIST_RANGE.get(rng, ("1y", "1d"))
    h = yf.Ticker(ticker).history(period=period, interval=interval)["Close"].dropna()
    return {"t": [t.strftime("%Y-%m-%d %H:%M:%S") for t in h.index],
            "c": [round(float(x), 4) for x in h.values]}


@app.get("/api/history")
def history(ticker: str, range: str = "1mo"):
    """Free close-price series for one ticker over the requested range (yfinance).
    Cached in 2-minute buckets so repeated clicks are instant but stay ~live."""
    try:
        return _history(ticker.upper(), range, int(time.time() // 120))
    except Exception:
        return {"t": [], "c": []}


def _dedup_titles(rows: list) -> list:
    """Drop syndicated reprints — the same story carried by several outlets. Key on the
    title stripped to lowercase alphanumerics, keep the first (Google ranks it highest)."""
    out, seen = [], set()
    for r in rows:
        key = re.sub(r"[^a-z0-9]", "", r["title"].lower())
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


@lru_cache(maxsize=256)
def _news(ticker: str, name: str, _bucket: int) -> list:
    # Google News RSS — free, no API key. Quote the ticker + name + a finance hint so a
    # cryptic ETF label (e.g. "20y+ UST") still returns ticker-specific headlines.
    q = f'"{ticker}" {name} stock OR ETF'.strip()
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=en-US&gl=US&ceid=US:en"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=8).read()
    rows = []
    for it in ET.fromstring(raw).findall(".//item")[:20]:   # pull extra so dedup still leaves a full list
        title = it.findtext("title") or ""
        src = it.find("source")                          # Google News RSS: <source url=...>Publisher</source>
        pub = src.text if src is not None and src.text else ""
        if pub and title.endswith(f" - {pub}"):          # strip the redundant "- Publisher" suffix
            title = title[: -len(f" - {pub}")]
        if "news, quote" in title.lower():               # Yahoo quote-page boilerplate ("… Stock Price, News, Quote & History"), not a headline
            continue
        rows.append({"title": title, "publisher": pub,
                     "link": it.findtext("link") or "", "published": (it.findtext("pubDate") or "")[:16]})
    return _dedup_titles(rows)[:6]


@app.get("/api/news")
def news(ticker: str, name: str = ""):
    """Recent headlines for one asset via Google News RSS (free). Cached in 10-minute buckets."""
    try:
        return {"items": _news(ticker.upper(), name.strip(), int(time.time() // 600))}
    except Exception:
        return {"items": []}


# Serve the built React app (added last so /api/* routes win). Present only after a build.
DIST = Path(__file__).parent / "web" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="spa")


if __name__ == "__main__":
    # self-check: syndicated reprints (same story, different outlet / punctuation) collapse to one
    rows = [{"title": "TLT Makes a Big Bet on Long Bonds", "publisher": "24/7 Wall St."},
            {"title": "TLT makes a BIG bet on long bonds!", "publisher": "Yahoo Finance"},
            {"title": "Apple stock slips", "publisher": "Benzinga"}]
    assert [r["publisher"] for r in _dedup_titles(rows)] == ["24/7 Wall St.", "Benzinga"]
    print("dedup OK")
