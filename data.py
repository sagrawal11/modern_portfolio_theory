"""Live data pipeline: download history, estimate mu + a well-conditioned covariance.

Expected returns from trailing means are the fragile input (we show that in the UI);
the covariance is where scale bites. With ~500 assets and ~2y of daily data (~500
obs), the raw sample covariance is nearly singular — so we shrink it (Ledoit-Wolf)
toward a structured target, which keeps it invertible and the optimizer sane.

Prices are cached to data_cache/ so re-runs and the per-minute live poll are cheap.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

CACHE = Path(__file__).parent / "data_cache"
TRADING_DAYS = 252
RET_CLIP = 0.25   # winsorize daily returns to ±25% — kills bad-tick / unadjusted-split artifacts
                  # (e.g. a free-data glitch printing a +177% day) that would otherwise dominate variance


def download_history(tickers: list[str], period: str = "2y", max_age_h: float = 12.0) -> pd.DataFrame:
    """Adjusted daily closes for `tickers`. Cached to parquet; refetched if stale."""
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"prices_{period}.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_h * 3600:
        px = pd.read_parquet(cache)
        if set(tickers).issubset(px.columns):
            return px[tickers]
    raw = yf.download(tickers, period=period, interval="1d",
                      auto_adjust=True, progress=False, threads=True)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    px = px.dropna(how="all")
    px.to_parquet(cache)
    return px


def bayes_stein(mu_daily: np.ndarray, cov_daily: np.ndarray, T: int):
    """Jorion (1986) Bayes-Stein shrinkage of the mean toward the min-variance return.

    Raw trailing means are wildly noisy and make the optimizer chase hindsight winners
    (the Sharpe-10 nonsense). This pulls each asset's mean toward mu0 — the return of the
    global minimum-variance portfolio — by an amount phi that grows when the estimates
    look noisy (small T) and the cross-sectional spread looks like luck. Daily units in,
    daily units out; annualize afterwards. Returns (mu_shrunk, phi).
    """
    N = len(mu_daily)
    Sinv = np.linalg.inv(cov_daily + 1e-12 * np.eye(N))
    ones = np.ones(N)
    mu0 = (ones @ Sinv @ mu_daily) / (ones @ Sinv @ ones)
    d = mu_daily - mu0 * ones
    Q = float(d @ Sinv @ d)
    phi = float(np.clip((N + 2) / ((N + 2) + T * Q), 0.0, 1.0))   # shrinkage weight
    return (1 - phi) * mu_daily + phi * mu0 * ones, phi


def capm_returns(R: np.ndarray, rf: float, erp: float) -> np.ndarray:
    """Equilibrium-style expected returns: mu_i = rf + beta_i * ERP.

    Market proxy = the equal-weight mean of the universe. Uses each asset's *systematic
    risk* (beta), not its noisy trailing mean, so it never chases hindsight winners — the
    tangency stays sane (Sharpe near the market's). ERP is the one macro assumption.
    """
    rm = R.mean(axis=1)
    mc = rm - rm.mean()
    betas = ((R - R.mean(axis=0)).T @ mc) / (mc @ mc)
    return rf + betas * erp


def factor_cov_daily(R: np.ndarray, k: int = 15) -> np.ndarray:
    """Principal-component (statistical) factor model for the covariance.

    Keeps the top-k return factors as the systematic covariance and everything else as an
    idiosyncratic diagonal: Sigma = (low-rank systematic) + diag(residual variance). That
    structure is always well-conditioned and invertible — the right tool for the stock
    block, where hundreds of names share a handful of real drivers (market, sectors).
    Diagonal (total variances) is preserved exactly; only the noisy off-diagonals are
    rank-reduced.
    """
    T = len(R)
    Rc = R - R.mean(axis=0)
    _, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    eig = (S ** 2) / (T - 1)                      # eigenvalues of the sample covariance
    V = Vt.T
    k = int(min(k, len(eig)))
    syst = (V[:, :k] * eig[:k]) @ V[:, :k].T      # top-k systematic part
    total_var = (Rc ** 2).sum(axis=0) / (T - 1)
    resid = np.clip(total_var - np.diag(syst), 1e-10, None)
    return syst + np.diag(resid)


def ledoit_wolf_cov(R: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage applied to the CORRELATION matrix, preserving each asset's
    real sample variance.

    sklearn's plain LedoitWolf shrinks the whole covariance toward a *spherical* target
    (one average variance on every diagonal). In a universe spanning T-bills (~0.2% vol)
    and tech stocks (~30% vol) that drags the near-cash bonds up to ~13% vol — so the
    optimizer can't tell a Treasury from an equity and a low-risk portfolio has nothing
    safe to hold. Standardizing to unit variance first shrinks only the correlations
    (target = identity), then rescaling by the true per-asset vols keeps SHV at 0.2%
    while the matrix stays well-conditioned.
    """
    sd = R.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    corr = LedoitWolf().fit(R / sd).covariance_      # shrink the correlation matrix only
    cov = corr * np.outer(sd, sd)
    np.fill_diagonal(cov, sd ** 2)                   # pin true per-asset variances exactly
    return cov


def black_litterman(mu_prior: np.ndarray, mu_view: np.ndarray, cov: np.ndarray,
                    tau: float = 0.05, conf: float = 8.0) -> np.ndarray:
    """Black-Litterman posterior blending a prior with a view (annualized units).

    Prior = CAPM equilibrium (robust); view = the historical mean (data, but noisy).
    Posterior is the precision-weighted combination with pick matrix P = I (a view on
    every asset). `conf` scales trust in the historical view: conf -> 0 gives the pure
    CAPM prior, large conf leans historical. This is the principled dial between the
    robust and the fragile return estimates.
    """
    tauS_inv = np.linalg.inv(tau * cov)
    omega_inv = np.diag(conf / np.clip(np.diag(cov), 1e-12, None))
    return np.linalg.solve(tauS_inv + omega_inv, tauS_inv @ mu_prior + omega_inv @ mu_view)


def momentum_returns(R: np.ndarray, rf: float, tilt: float = 0.10,
                     lookback: int = 252, skip: int = 21) -> np.ndarray:
    """Cross-sectional 12-1 momentum tilt → expected returns.

    Rank assets by trailing ~11-month return (excluding the last month, the classic
    12-1 window that skips short-term reversal), z-score across the universe, and set
    mu = rf + tilt*z. High-momentum names get high expected returns, so the optimizer
    *concentrates* in the current leaders — and it *rotates* as momentum shifts over time.
    This is an active bet that momentum persists (a well-documented factor), not pure MPT.
    """
    T = R.shape[0]
    window = R[-min(lookback, T):-skip] if T > skip else R
    mom = np.prod(1 + window, axis=0) - 1
    z = (mom - mom.mean()) / (mom.std() + 1e-9)
    return rf + tilt * z


def estimate_inputs(prices: pd.DataFrame, cov_model: str = "ledoit_wolf", return_model: str = "capm",
                    rf: float = 0.04, erp: float = 0.05, bl_conf: float = 8.0,
                    k_factors: int = 15, min_obs: int = 60):
    """-> (mu, cov, tickers). Annualized.

    cov_model: 'ledoit_wolf' (shrinkage, default) or 'factor' (PCA factor model).
    return_model: 'capm' (robust, default); 'black_litterman' (CAPM prior + historical
    view, blended by bl_conf); 'bayes_stein'; 'raw' (fragile trailing mean).
    Complete-case aligned returns so the covariance is positive-definite.
    """
    rets = prices.pct_change(fill_method=None)
    keep = rets.columns[rets.notna().sum() >= min_obs]
    rets = rets[keep].dropna().clip(-RET_CLIP, RET_CLIP)   # complete cases + winsorize bad ticks
    tickers = list(rets.columns)
    R, T = rets.values, len(rets)
    cov_daily = factor_cov_daily(R, k_factors) if cov_model == "factor" else ledoit_wolf_cov(R)
    cov = cov_daily * TRADING_DAYS
    if return_model == "capm":
        mu = capm_returns(R, rf, erp)
    elif return_model == "black_litterman":
        mu = black_litterman(capm_returns(R, rf, erp), R.mean(axis=0) * TRADING_DAYS, cov, conf=bl_conf)
    elif return_model == "bayes_stein":
        mu = bayes_stein(R.mean(axis=0), cov_daily, T)[0] * TRADING_DAYS
    elif return_model == "momentum":
        mu = momentum_returns(R, rf)
    else:
        mu = R.mean(axis=0) * TRADING_DAYS
    return mu, cov, tickers


def condition_number(cov: np.ndarray) -> float:
    """Ratio of largest/smallest eigenvalue — how close the matrix is to singular."""
    ev = np.linalg.eigvalsh(cov)
    return float(ev.max() / max(ev.min(), 1e-18))


def live_prices(tickers: list[str]) -> pd.Series:
    """Latest available price per ticker — for marking the portfolio live.

    Tries 1-minute intraday first; falls back to daily closes when the market is
    closed or intraday comes back empty.
    """
    for period, interval in (("1d", "1m"), ("5d", "1d")):
        raw = yf.download(tickers, period=period, interval=interval,
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            continue
        px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        px = px.ffill().dropna(how="all")
        if len(px):
            return px.iloc[-1]
    raise RuntimeError("no live prices available")


if __name__ == "__main__":
    import universe
    u = universe.build_universe(include_stocks=False)      # ETFs only, quick
    sample = u["ticker"].tolist()[:45]
    print(f"downloading {len(sample)} ETFs...")
    px = download_history(sample, period="2y")
    print(f"prices: {px.shape[0]} days x {px.shape[1]} tickers")
    for cm in ("ledoit_wolf", "factor"):
        mu, cov, tk = estimate_inputs(px, cov_model=cm)
        print(f"  {cm:12s}:  {len(tk)} assets  cond(cov)={condition_number(cov):,.0f}  "
              f"mu range [{mu.min()*100:+.1f}%, {mu.max()*100:+.1f}%]")
