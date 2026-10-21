# Modern Portfolio Theory - a live efficient-frontier allocator

This is an interactive tool that builds the efficient frontier - basically the set of portfolios that squeeze the most expected return out of every unit of risk you're willing to take on - across a universe of ~130 diverse ETFs plus the entire S&P 500, and then lets you dial in your own risk tolerance and watch the optimal allocation fall out in real time. It takes the classic Markowitz mean-variance optimization that you'd normally meet as a static chart in a finance textbook and makes it genuinely live: real prices, a real risk-free rate pulled from the 3-month T-bill, and a real quadratic-program solver re-running in the background every time you move a slider.

I built it because the textbook version always stops right where it gets interesting. The spreadsheet gives you a frontier and a couple of portfolios on it, but it never lets you actually turn the knobs, and it quietly leaves out the risk-free asset and the Capital Allocation Line, which are the whole point of how you'd really blend a risky portfolio with cash. So this does all of that, and it's honest about the parts that are fragile - expected returns are the shakiest input in the entire pipeline, so you can swap between a robust CAPM-implied estimate, momentum, Black-Litterman, Bayes-Stein shrinkage, and the raw historical mean, and literally see the optimizer go from sane to chasing hindsight winners in front of you.

## What it does

- **Live efficient frontier** over ~610 assets, with the tangency (max-Sharpe) portfolio, the global minimum-variance portfolio, and the Capital Allocation Line all drawn in.
- **A risk slider** that maps "how much volatility can you stomach" straight to a concrete allocation, including the option to lever up by borrowing at the risk-free rate.
- **Five expected-return models and two covariance models** so you can feel how much the answer depends on what you assume, which is the real lesson of mean-variance optimization.
- **Click any dot or holding** for its live price chart over multiple ranges plus recent news headlines, so the universe isn't just abstract points on a scatter.
- **A max-holdings cap** that concentrates the portfolio into its top N names, because a long-only CAPM optimizer will happily spread a tiny sliver across almost everything, and usually you don't actually want to hold 380 positions.

## Running it locally

Backend (FastAPI) - from the repo root:

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Frontend (React + Vite) - in a second terminal:

```bash
cd web
npm install
npm run dev
```

Then open http://localhost:5173. The dev server proxies `/api` to the backend on port 8000, so the two talk to each other automatically.

## How it's deployed

It's split across two hosts because the backend is too heavy for a serverless platform - numpy, pandas, scikit-learn and a QP solver blow past the size and timeout limits - so:

- **Frontend -> Vercel**, as a static Vite build.
- **Backend -> Render**, as a single FastAPI web service (see `render.yaml`).
- **Vercel rewrites `/api/*` to the Render service**, so from the browser's point of view it's all one same-origin app and there's no CORS to worry about.

Prices are cached to `data_cache/` as parquet and refreshed every 12 hours, and that cache is committed so the free-tier backend serves a fast first solve instead of cold-downloading two years of history for 600 tickers on the first request.

## The honesty notes

Two things this is deliberately upfront about, because they're where MPT quietly misleads people:

- **The long-only frontier caps at your single best asset.** You can't out-return your highest-return holding without leverage or shorting - that ceiling is real, not a bug.
- **The tangency portfolio trusts the expected returns completely**, and those are the fragile input. That's exactly why the minimum-variance portfolio (which ignores expected returns entirely) is always on the chart, and why a Sharpe above ~2 trips a warning telling you the returns are overfit to hindsight.
