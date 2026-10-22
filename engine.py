"""Modern Portfolio Theory engine — efficient frontier, tangency portfolio, and CAL.

Given expected returns (mu), a covariance matrix (cov), and a risk-free rate, this
computes everything the teacher's spreadsheet does *plus* the piece it leaves out:
the risk-free asset, the Capital Allocation Line, the tangency (max-Sharpe) portfolio,
and the risk-slider mapping that turns "how much risk can you stomach" into an actual
allocation.

Everything is in DECIMAL units internally (0.08 = 8%), annualized.

Optimization is a quadratic program solved with `quadprog`. Long-only by default
(w >= 0 for every asset) — no shorting, ever, so no asset can take a negative weight.
That scales cleanly to hundreds of assets, where scipy's general solver would crawl.

Two honesty notes:
  * The long-only frontier tops out at the single highest-return asset — you cannot
    out-return your best asset without leverage or shorting. That cap is real.
  * The tangency portfolio trusts `mu`; expected returns are the fragile input, which
    is why we also expose the minimum-variance portfolio, which ignores `mu` entirely.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import quadprog

DATA = Path(__file__).parent / "teacher_inputs.json"


# --------------------------------------------------------------------------- data

def load_inputs(path: str | Path = DATA) -> dict:
    """Load assets + build the covariance matrix from correlation and sigmas."""
    d = json.loads(Path(path).read_text())
    mu = np.array(d["mu"], float)
    sigma = np.array(d["sigma"], float)
    corr = np.array(d["corr"], float)
    cov = corr * np.outer(sigma, sigma)          # Sigma = D corr D, D = diag(sigma)
    return {
        "assets": d["assets"], "groups": d["groups"],
        "mu": mu, "sigma": sigma, "corr": corr, "cov": cov,
        "raw": d,
    }


# ---------------------------------------------------------------------- portfolio

def port_stats(w, mu, cov) -> tuple[float, float]:
    """Return (expected_return, volatility) for weight vector w."""
    w = np.asarray(w, float)
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return ret, vol


def _qp(cov, A_eq, b_eq, long_only: bool = True) -> np.ndarray | None:
    """min wᵀ cov w  s.t.  A_eq w = b_eq  (and w >= 0 if long_only).

    Returns None if the constraints are infeasible (e.g. a target return the
    long-only universe can't reach).
    """
    n = cov.shape[0]
    G = np.ascontiguousarray((cov + cov.T) / 2 + 1e-10 * np.eye(n))   # symmetric, PD
    a = np.zeros(n)
    A_eq = np.atleast_2d(np.asarray(A_eq, float))
    b_eq = np.asarray(b_eq, float)
    if long_only:
        C = np.vstack([A_eq, np.eye(n)]).T
        b = np.concatenate([b_eq, np.zeros(n)])
    else:
        C = A_eq.T
        b = b_eq
    try:
        return quadprog.solve_qp(G, a, C, b, meq=len(b_eq))[0]
    except ValueError:
        return None


def min_variance(mu, cov, long_only: bool = True) -> np.ndarray:
    """Global minimum-variance portfolio. Ignores mu — robust to return-forecast error."""
    n = len(mu)
    return _qp(cov, np.ones((1, n)), [1.0], long_only)


def efficient_frontier(mu, cov, n_points: int = 50, long_only: bool = True,
                       on_progress=None) -> list[dict]:
    """Trace the efficient half of the frontier as [{ret, vol, w}, ...].

    Targets run from the global-min-variance return up to the max achievable return
    (the top-return asset when long-only). on_progress(done, total), if given, is called
    after each point so a caller can report real progress (this loop is the slow part).
    """
    mu = np.asarray(mu, float)
    w_mv = min_variance(mu, cov, long_only)
    r_mv = float(w_mv @ mu)
    r_max = float(mu.max()) if long_only else float(mu.max()) * 1.5
    pts = []
    targets = np.linspace(r_mv, r_max, n_points)
    for i, t in enumerate(targets):
        w = _qp(cov, [np.ones(len(mu)), mu], [1.0, t], long_only)
        if w is not None:
            ret, vol = port_stats(w, mu, cov)
            pts.append({"ret": ret, "vol": vol, "w": w})
        if on_progress:
            on_progress(i + 1, n_points)
    return pts


def tangency(mu, cov, rf: float, long_only: bool = True) -> dict:
    """Max-Sharpe portfolio — the risky portfolio the CAL is tangent to.

    Long-only solve uses the standard reformulation: min yᵀΣy s.t. (mu-rf)ᵀy = 1,
    y >= 0, then rescale w = y / sum(y). Trusts the expected returns.
    """
    mu = np.asarray(mu, float)
    n = len(mu)
    excess = mu - rf
    if long_only:
        y = _qp(cov, excess.reshape(1, n), [1.0], long_only=True)
        if y is None or y.sum() <= 0:          # nothing beats rf — fall back to min-var
            w = min_variance(mu, cov, True)
        else:
            w = y / y.sum()
    else:
        w = np.linalg.solve(cov, excess)
        w = w / w.sum()
    ret, vol = port_stats(w, mu, cov)
    return {"w": w, "ret": ret, "vol": vol, "sharpe": (ret - rf) / vol if vol else 0.0}


def cal_point(target_vol: float, tan: dict, rf: float, allow_leverage: bool = True) -> dict:
    """A point on the Capital Allocation Line at the requested volatility.

    Mix of risk-free asset and the tangency portfolio. y = fraction in the risky
    tangency portfolio. y > 1 means borrowing at rf to lever up (a *negative*
    risk-free weight); with allow_leverage=False this is capped at y = 1 (fully
    invested, no borrowing) so no weight is ever negative.
    """
    y = target_vol / tan["vol"] if tan["vol"] > 0 else 0.0
    if not allow_leverage:
        y = min(y, 1.0)
    ret = rf + y * (tan["ret"] - rf)
    vol = y * tan["vol"]
    return {
        "y": y, "ret": ret, "vol": vol,
        "w_risky": tan["w"] * y,          # per-asset risky weights (always >= 0)
        "w_riskfree": 1.0 - y,            # < 0 only when levering (borrowing)
    }


# --------------------------------------------------------------------------- check

def demo() -> None:
    """Self-check: reproduce the teacher's spreadsheet numbers from scratch."""
    d = load_inputs()
    mu, cov = d["mu"], d["cov"]
    val = d["raw"]["validation"]

    # 1) demo portfolio (40% Europe / 40% Sweden / 20% Commodities) matches the sheet
    w = np.array(d["raw"]["demo_weights_pct"]) / 100.0
    ret, vol = port_stats(w, mu, cov)
    assert abs(vol * 100 - val["demo_sd"]) < 1e-3, (vol * 100, val["demo_sd"])
    assert abs(ret * 100 - val["demo_return"]) < 1e-3, (ret * 100, val["demo_return"])
    print(f"demo portfolio   vol={vol*100:.4f}%  ret={ret*100:.4f}%   (sheet "
          f"{val['demo_sd']:.4f}% / {val['demo_return']:.4f}%)  OK")

    # 2) my efficient frontier matches his Solver points
    fr = efficient_frontier(mu, cov, n_points=120)
    vols = np.array([p["vol"] for p in fr]) * 100
    rets = np.array([p["ret"] for p in fr]) * 100
    order = np.argsort(vols)
    vols, rets = vols[order], rets[order]
    assert (np.array([p["w"] for p in fr]) >= -1e-8).all(), "a frontier weight went negative!"
    print("\n  target risk %   my return %   sheet return %   diff")
    worst = 0.0
    for risk_pct, sheet_ret in val["frontier_risk_return_pct"]:
        if risk_pct < vols.min() or risk_pct > vols.max():
            continue
        mine = float(np.interp(risk_pct, vols, rets))
        worst = max(worst, abs(mine - sheet_ret))
        if int(risk_pct) % 4 == 0:
            print(f"    {risk_pct:9.0f}   {mine:10.2f}   {sheet_ret:12.2f}   {mine - sheet_ret:+.3f}")
    print(f"  worst frontier diff vs sheet: {worst:.3f} pp")
    assert worst < 0.5, f"frontier drift too large: {worst} pp"

    # 3) long-only frontier caps at the best single asset, all weights >= 0
    assert abs(rets.max() - mu.max() * 100) < 0.05, (rets.max(), mu.max() * 100)
    print(f"\n  frontier caps at {rets.max():.2f}% = best asset "
          f"({d['assets'][int(mu.argmax())]} @ {mu.max()*100:.2f}%)  OK")

    # 4) tangency is long-only (no negative weights) + CAL sanity at rf = 4%
    rf = 0.04
    tan = tangency(mu, cov, rf)
    assert (tan["w"] >= -1e-8).all(), "tangency has a negative weight!"
    top = sorted(zip(d["assets"], tan["w"]), key=lambda x: -x[1])[:5]
    print(f"\n  tangency @ rf={rf*100:.0f}%:  ret={tan['ret']*100:.2f}%  "
          f"vol={tan['vol']*100:.2f}%  Sharpe={tan['sharpe']:.3f}  min weight={tan['w'].min():+.4f}")
    print("  top holdings: " + ", ".join(f"{a} {w*100:.0f}%" for a, w in top if w > 1e-3))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    demo()
