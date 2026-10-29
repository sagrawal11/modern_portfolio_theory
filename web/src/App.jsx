import { useEffect, useRef, useState } from 'react'
import Plotly from 'plotly.js-dist-min'

const PALETTE = ['#8aadf4', '#ed8796', '#f5a97f', '#8bd5ca', '#a6da95', '#eed49f', '#c6a0f6',
  '#f0c6c6', '#ee99a0', '#91d7e3', '#7dc4e4', '#b7bdf8', '#f4dbd6', '#cad3f5']

const RET_MODELS = [
  ['capm', 'CAPM-implied (robust, diversifies)'],
  ['momentum', 'Momentum (active, concentrates + rotates)'],
  ['black_litterman', 'Black-Litterman'],
  ['bayes_stein', 'Bayes-Stein shrinkage'],
  ['raw', 'Historical mean (fragile)'],
]

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// A full-universe solve takes minutes on the free-tier backend, so /api/solve starts a job and
// returns a job id we poll at /api/solve_result instead of blocking one long request. While the
// solve pins the single CPU the server may briefly return 502s, and hit() swallows those so we just
// keep polling. Resolves with the result, or null if the caller cancelled (deps changed) mid-poll.
async function solveApi(p, cancelled, onProgress) {
  const qs = new URLSearchParams(p).toString()
  const hit = async (url) => {                           // null on any transient failure -> keep waiting
    try { const x = await fetch(url); return x.ok ? await x.json() : null } catch { return null }
  }
  const tick = (r) => { if (onProgress && r && r.progress != null) onProgress(r.progress, r.stage || '') }
  const start = await hit(`/api/solve?${qs}`)
  if (start && start.status === 'done') return start.result
  tick(start)
  let job = start && start.job
  for (let i = 0; i < 600; i++) {                        // ~600 * 1.2s ≈ 12 min ceiling
    if (cancelled()) return null
    await sleep(1200)
    if (cancelled()) return null
    const r = job ? await hit(`/api/solve_result?job=${encodeURIComponent(job)}`) : await hit(`/api/solve?${qs}`)
    if (!r) continue                                     // 502/network blip during a busy solve, retry
    if (r.status === 'done') return r.result
    if (r.status === 'error') throw new Error(r.error || 'solve failed')
    if (r.status === 'unknown') { job = null; continue } // instance restarted, job gone, re-POST next loop
    if (r.job) job = r.job
    tick(r)
  }
  throw new Error('solve timed out')
}

// Map the risk slider to an allocation (client-side). Risky weights are always >= 0.
// Classic mode scales the max-Sharpe portfolio you'd actually hold: the capped one (d.port) when
// Max holdings is set, else the full-universe tangency. sel carries its own tickers/names so the
// allocation table lists the right holdings.
function selPortfolio(d, mode, target, rf, lev) {
  if (!d) return null
  if (mode === 'classic') {
    const t = d.port || { ret: d.tangency.ret, vol: d.tangency.vol, w: d.tangency.w, tickers: d.tickers, names: d.names }
    let y = target / t.vol
    if (!lev) y = Math.min(y, 1)
    return { ret: rf * 100 + y * (t.ret - rf * 100), vol: y * t.vol, w: t.w.map((x) => x * y), wRf: 1 - y, lever: y, tickers: t.tickers, names: t.names }
  }
  let bi = 0, bd = 1e9
  d.frontier.forEach((p, i) => { const dd = Math.abs(p.vol - target); if (dd < bd) { bd = dd; bi = i } })
  return { ret: d.frontier[bi].ret, vol: d.frontier[bi].vol, w: d.frontier_w[bi], wRf: 0, lever: 1, tickers: d.tickers, names: d.names }
}

const HIST_RANGES = [['1d', '1D'], ['5d', '1W'], ['1mo', '1M'], ['1y', '1Y'], ['ytd', 'YTD'], ['5y', '5Y']]

// Shared animated loader. overlay = full-cover blurred takeover (main chart); small = compact inline (news).
function Spinner({ label, overlay, small }) {
  const inner = (
    <div className={overlay ? 'load-box' : 'load-inline'}>
      <div className={small ? 'spinner spinner-sm' : 'spinner'} />
      {label && <span className="load-label">{label}</span>}
    </div>
  )
  return overlay ? <div className="load-overlay">{inner}</div> : inner
}

// Determinate progress bar for the main solve. The backend reports real progress (data ->
// covariance -> each frontier point), and the bar resets to 0 whenever a new solve starts.
function ProgressOverlay({ progress, stage }) {
  const pct = Math.round((progress || 0) * 100)
  return (
    <div className="load-overlay">
      <div className="load-box">
        <div className="pbar"><div className="pbar-fill" style={{ width: `${Math.max(pct, 2)}%` }} /></div>
        <div className="pbar-meta"><span>{stage || 'Solving…'}</span><span className="pbar-pct">{pct}%</span></div>
      </div>
    </div>
  )
}

// Click-through live price chart for a single asset (dot on the scatter, or allocation row).
function AssetModal({ ticker, name, onClose }) {
  const [range, setRange] = useState('1mo')
  const [hist, setHist] = useState(null)
  const [loading, setLoading] = useState(true)
  const [news, setNews] = useState(null)
  const [about, setAbout] = useState(null)
  const ref = useRef()
  const sub = name.includes(' — ') ? name.split(' — ').slice(1).join(' — ') : name

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    let cancel = false
    setNews(null)
    fetch(`/api/news?ticker=${encodeURIComponent(ticker)}&name=${encodeURIComponent(sub)}`)
      .then((r) => r.json())
      .then((d) => { if (!cancel) setNews(d.items || []) })
      .catch(() => { if (!cancel) setNews([]) })
    return () => { cancel = true }
  }, [ticker, sub])

  useEffect(() => {
    let cancel = false
    setAbout(null)
    fetch(`/api/about?ticker=${encodeURIComponent(ticker)}`)
      .then((r) => r.json())
      .then((d) => { if (!cancel) setAbout(d) })
      .catch(() => { if (!cancel) setAbout({}) })
    return () => { cancel = true }
  }, [ticker])

  useEffect(() => {
    let cancel = false
    setLoading(true)
    fetch(`/api/history?ticker=${encodeURIComponent(ticker)}&range=${range}`)
      .then((r) => r.json())
      .then((d) => { if (!cancel) { setHist(d); setLoading(false) } })
      .catch(() => { if (!cancel) { setHist({ t: [], c: [] }); setLoading(false) } })
    return () => { cancel = true }
  }, [ticker, range])

  useEffect(() => {
    if (!hist?.c?.length || !ref.current) return
    const up = hist.c[hist.c.length - 1] >= hist.c[0]
    const col = up ? '#8bd5ca' : '#ed8796'
    const intraday = hist.t.some((s) => !s.endsWith(' 00:00:00'))   // intraday points carry a clock time
    Plotly.react(ref.current, [{
      x: hist.t, y: hist.c, mode: 'lines', type: 'scatter', line: { color: col, width: 2 },   // SVG line, y autoscales to the price range
      hovertemplate: '%{x|%b %d %Y %H:%M}<br>$%{y:.2f}<extra></extra>',
    }], {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#9aa5b1', family: 'Inter, sans-serif', size: 11 },
      // drop the market-closed gaps so overnight/weekend stretches don't draw as diagonal lines
      xaxis: { gridcolor: '#232b3d', zeroline: false, automargin: true,
        rangebreaks: (intraday ? [{ pattern: 'hour', bounds: [16, 9.5] }] : []).concat([{ pattern: 'day of week', bounds: [6, 1] }]) },
      yaxis: { gridcolor: '#232b3d', zeroline: false, tickprefix: '$', automargin: true },
      margin: { l: 10, r: 14, t: 8, b: 28 }, showlegend: false,
    }, { displayModeBar: false, responsive: true })
  }, [hist])

  const c = hist?.c, last = c?.length ? c[c.length - 1] : null
  const chg = c?.length ? last - c[0] : 0
  const pct = c?.length && c[0] ? (chg / c[0]) * 100 : 0
  const up = chg >= 0

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div><div className="modal-tk">{ticker}</div><div className="muted">{sub}</div></div>
          <button className="modal-x" onClick={onClose}>✕</button>
        </div>
        {about?.summary && <div className="modal-about">{about.summary}
          {about.website && <> <a href={about.website} target="_blank" rel="noreferrer">{about.website.replace(/^https?:\/\/(www\.)?/, '').replace(/\/$/, '')}</a></>}</div>}
        {last != null && <div className="modal-price">${last.toFixed(2)}
          <span className={up ? 'up' : 'down'}> {up ? '▲' : '▼'} {chg >= 0 ? '+' : ''}{chg.toFixed(2)} ({pct >= 0 ? '+' : ''}{pct.toFixed(2)}%)</span>
          <span className="muted" style={{ fontSize: 11, marginLeft: 8, fontWeight: 400 }}>over {HIST_RANGES.find(([k]) => k === range)[1]}</span></div>}
        <div className="seg modal-ranges">
          {HIST_RANGES.map(([k, lbl]) => <button key={k} className={range === k ? 'on' : ''} onClick={() => setRange(k)}>{lbl}</button>)}
        </div>
        <div className="modal-chart">
          <div ref={ref} style={{ width: '100%', height: '100%' }}></div>
          {loading && <div className="modal-overlay"><Spinner label="loading price history…" /></div>}
          {!loading && !c?.length && <div className="modal-overlay muted">no data for this range</div>}
        </div>
        <div className="modal-news">
          <div className="news-head">Recent headlines</div>
          {news === null && <Spinner small label="loading headlines…" />}
          {news?.length === 0 && <div className="muted">No recent headlines found.</div>}
          {news?.map((a, i) => (
            <a key={i} className="news-item" href={a.link} target="_blank" rel="noreferrer">
              <div className="news-title">{a.title}</div>
              <div className="news-meta">{a.publisher}{a.published ? ' · ' + a.published : ''}</div>
            </a>
          ))}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Plain-language help. Every bit of jargon on the page gets a "?" that opens
// one of these. Written in plain language, aimed at non-experts.
// ---------------------------------------------------------------------------
const GLOSSARY = {
  efficient_frontier: { t: 'Efficient frontier', b: `Imagine every possible mix of investments you could own. For each amount of risk you're willing to take, exactly one mix gives you the most expected reward. String all of those best-possible mixes together and you get a curve, and that curve is the efficient frontier. Basically it's the "you can't do any better than this" line: anything below it is a worse deal, and anything above it just isn't reachable. Finding this line is the whole point of the tool.` },
  expected_return: { t: 'Expected return', b: `This is our best guess for how much an investment grows in a year, written as a percent. An expected return of 8% is basically us saying "we think $100 turns into about $108 in a year." It's a guess and not a promise, the real number will bounce around, but it's the number we aim with.` },
  volatility: { t: 'Risk (volatility)', b: `Volatility is how much an investment's price bounces up and down. A calm, steady investment has low volatility, and a wild one that swings hard has high volatility. We treat risk as volatility because big swings are exactly what can hurt you if you need your money at a bad moment. "Annualized" just means we scaled the bounciness to a one-year view so everything's comparable.` },
  sharpe: { t: 'Sharpe ratio', b: `The Sharpe ratio is basically reward-per-unit-of-risk, it tells you whether a bumpy ride was actually worth it. You take the extra return you earned above a totally safe investment, then divide by how bumpy the ride was. Higher is better: a Sharpe of 1 is solid, 2 is excellent, and anything way above that is usually too good to be true.` },
  capm: { t: 'CAPM', b: `CAPM is a classic recipe for guessing expected returns. Instead of trusting an investment's past (which can fool you), it says the more something moves with the whole market, the more return people should demand to hold it. So a jumpy stock that amplifies the market's moves "should" pay more than a sleepy one. It's the robust, play-it-safe option here, and it tends to spread your money across lots of names.` },
  momentum: { t: 'Momentum', b: `Momentum bets on a simple, almost stubborn idea: stuff that's been going up lately tends to keep going up for a while. So this model leans into the recent winners. It's the active, aggressive choice, it concentrates your money into fewer names and rotates into whatever's hot, which can pay off big but is a bumpier, riskier ride.` },
  black_litterman: { t: 'Black-Litterman', b: `Black-Litterman is a clever way to blend two things: the market's own built-in wisdom (from CAPM) and any opinions you want to add on top. You start from the sensible market baseline, then nudge it with your views and how strongly you hold them. Basically it's "trust the crowd, but let me lean it a little," and it keeps portfolios from going haywire the way raw guesses can.` },
  bayes_stein: { t: 'Bayes-Stein shrinkage', b: `Raw historical averages are noisy, a couple of lucky years can make a so-so stock look amazing. Bayes-Stein "shrinks" every estimate toward the group average, basically pulling the wild numbers back toward something more believable. It's like saying "I don't fully trust this stock's hot streak, so let me tone it down," and the result is steadier guesses that get fooled by luck less often.` },
  raw: { t: 'Historical mean', b: `This just takes each investment's average past return and assumes the future looks the same. It's the simplest guess you can make, but it's fragile: the past is full of flukes, and betting the exact same winners keep winning usually backfires. It's here mostly to show you why the smarter models exist.` },
  erp: { t: 'Equity risk premium', b: `This is the extra reward investors expect for putting money in the stock market instead of a totally safe spot like a government bond. If safe pays 4% and you expect stocks to pay 9%, the premium is that 5% gap. It's the "why bother taking the risk" payment, and sliding it changes how juicy CAPM thinks stocks are.` },
  bl_conf: { t: 'Trust in history', b: `This slider controls how much the Black-Litterman model listens to the actual past returns versus the safe CAPM baseline. At 0 it ignores history completely and just trusts the market's wisdom. Turn it up and it believes the historical numbers more and more. Basically it's a dial between "play it safe" and "trust what actually happened."` },
  covariance: { t: 'Covariance model', b: `Covariance is about how investments move together: do two stocks tend to rise and fall at the same time, or does one zig while the other zags? That matters a ton, because owning things that move apart is what makes a portfolio safer. A covariance model is just the method we use to estimate all those together-or-apart relationships across hundreds of investments at once. Ledoit-Wolf is the steady, cleaned-up default; Factor (PCA) explains everything through a few big hidden forces like "the overall market."` },
  max_holdings: { t: 'Max holdings', b: `This caps how many different investments your final portfolio is allowed to hold. Set it to 20 and the tool picks the best 20 names and puts everything there; set it to 0 and it can hold as many as it likes. It's handy because owning 300 tiny slivers is a pain in real life, sometimes you just want a tight, focused basket.` },
  philosophy: { t: 'Philosophy', b: `This picks the overall game plan. "Classic + CAL" builds the textbook best-reward-per-risk portfolio and then blends it with safe cash to hit your comfort level. "Robust min-var" instead hunts for the calmest, most stable portfolio it can, caring less about chasing return and more about not getting surprised. One chases the ideal, the other plays defense.` },
  leverage: { t: 'Leverage', b: `Leverage means borrowing money to invest more than you actually have. If you have $100 but want $130 working for you, you borrow $30. It can boost your gains, but it magnifies your losses the same way, so it adds real risk. It's off by default, and you have to opt in deliberately.` },
  target_vol: { t: 'Risk tolerance', b: `This is you telling the tool how bumpy a ride you can stomach, measured as volatility. A low number means "keep it calm, no big swings," and a high number means "I can handle the rollercoaster if it means more reward." The tool then builds the best portfolio it can for exactly that comfort level, and "feasible" just shows the range that's actually possible.` },
  universe: { t: 'Universe', b: `The universe is simply the full menu of investments the tool is allowed to choose from, things like ETFs, bonds, commodities, and optionally hundreds of individual S&P 500 stocks. A bigger universe means more options to build a great portfolio, but also more number-crunching. "Max assets" just limits how many items from that menu we throw into the math.` },
  sp500: { t: 'S&P 500', b: `The S&P 500 is a list of about 500 of the biggest, most important companies in the US, think Apple, Microsoft, and Coca-Cola. People use it as shorthand for "the US stock market" overall. Checking this box lets the tool pick from those individual companies, not just big bundled funds.` },
  assets: { t: 'Assets', b: `This is just the count of how many different investments made it into the math for this run. More assets means more raw material to build a good portfolio from, but also more for the computer to chew through.` },
  risk_free: { t: 'Risk-free rate', b: `The risk-free rate is what you'd earn in the safest possible parking spot for money, basically a short-term US government IOU (a 3-month Treasury bill) that's about as close to guaranteed as money gets. Everything else is measured against it: an investment is only "worth it" if it beats just sitting in that safe spot. "Live" means we pulled today's real rate.` },
  cal: { t: 'Capital Allocation Line', b: `This is the straight line on the chart that starts at the safe risk-free spot and runs through the best portfolio (the star). Every point on it is just a different mix of "safe cash" and "that best portfolio." Want less risk? Slide toward cash. Want more? Lean into the portfolio, or even borrow to buy extra. It's basically your risk dial drawn as a line.` },
  tangency: { t: 'Tangency (max Sharpe)', b: `The tangency portfolio has the best reward-per-risk on the whole chart (the highest Sharpe). It sits where the straight capital allocation line just touches the curve. In classic theory this is the portfolio everyone should hold, and you then dial your risk up or down by mixing it with safe cash. That's why it's marked with a star.` },
  min_var: { t: 'Minimum variance', b: `This is the single calmest portfolio you can build out of everything available, the mix that bounces around the least. It won't give you the highest return, but it's the steadiest ride, which is exactly what some people want. On the chart it's the leftmost point of the curve, because left means low risk.` },
  your_portfolio: { t: 'Your portfolio', b: `This is the white dot on the chart, the specific mix the tool is recommending for you right now based on everything you've set. Move the risk slider or change a setting and watch it slide to a new spot. The allocation table below is the exact breakdown of that dot.` },
  estimation_error: { t: 'Estimation error', b: `Everything here is built on estimates of the future, and estimates are never perfect; that gap is estimation error. It matters because the math will happily build a "perfect" portfolio around a number that was really just lucky noise. That's why an unusually high Sharpe is a red flag rather than a jackpot: the model has overfit a fluke. The robust models exist to counter this.` },
  allocation: { t: 'Allocation & weight', b: `Your allocation is just how you split your money across investments, and each "weight" is the percent in one thing. A weight of 10% means 10 cents of every dollar goes there. Add up all the weights and you get 100%, your whole portfolio. This table is literally the tool's recommended recipe.` },
  diversification: { t: 'Diversification', b: `Diversification is the classic "don't put all your eggs in one basket" idea. If you spread money across investments that don't all move together, one of them falling doesn't sink the whole portfolio. It lowers your risk without necessarily giving up return, which is rare in investing.` },
  long_only: { t: 'Long-only (no shorting)', b: `Long-only means the tool can only buy investments hoping they go up, it can't "short" them (a fancier bet that profits when something falls). So every weight here is zero or positive, never negative. It keeps things simpler and safer, which is how most normal people actually invest.` },
}

// Ordered walkthrough. Each step highlights a real element on the page (by its
// data-tour attribute) and explains it. Keep the copy short, it lives in a tooltip.
const TOUR = [
  { sel: '[data-tour="chart"]', t: 'The map', b: `Each dot is an investment, placed by its risk and its expected return. The highlighted curve is the efficient frontier: the best return available at each level of risk. Adjust the controls on the left and the chart updates live.` },
  { sel: '[data-tour="returns"]', t: '1. Estimating returns', b: `How much will each investment earn? No one knows for sure, so you choose a method. CAPM is the robust, diversifying default; Momentum leans into recent winners. Each choice produces a different portfolio.` },
  { sel: '[data-tour="cov"]', t: '2. How things move together', b: `How do investments move relative to each other? Holding assets that move apart is what reduces risk. These two models estimate those relationships in different ways; Ledoit-Wolf is the steady default.` },
  { sel: '[data-tour="maxhold"]', t: '3. Keep it focused', b: `Holding hundreds of tiny positions is impractical. Use this to cap the portfolio to its best few names, or leave it at 0 to hold everything.` },
  { sel: '[data-tour="philosophy"]', t: '4. Choose an approach', b: `"Classic + CAL" builds the textbook-optimal portfolio and blends it with cash. "Robust min-var" instead targets the calmest, most stable mix. One pursues the ideal; the other prioritizes stability.` },
  { sel: '[data-tour="risk"]', t: '5. Your risk tolerance', b: `How much volatility can you accept? Slide left for a calmer portfolio, right for a more aggressive one. The tool rebuilds the best portfolio for that level of risk.` },
  { sel: '[data-tour="universe"]', t: '6. The investment menu', b: `This is everything the tool can choose from. Enable the S&P 500 to include hundreds of individual companies, or keep it focused. A larger universe means more computation.` },
  { sel: '[data-tour="cards"]', t: 'Headline numbers', b: `These four update live as you adjust settings. Expected return and risk are the reward and the volatility; Sharpe is the reward-per-risk score, where higher is better.` },
  { sel: '[data-tour="chart"]', t: 'Reading the chart', b: `The orange star is the single best (max-Sharpe) portfolio. The white dot is your pick for your risk level. The dashed line blends that portfolio with cash. Click any dot for its live price chart.` },
  { sel: '[data-tour="alloc"]', t: 'The allocation', b: `The exact breakdown: which investments, and how much of each. Click any row to open that investment's live chart and recent news.` },
  { sel: '[data-tour="replay"]', t: "You're all set", b: `Every term on the page has a "?" next to it that opens a plain-language explanation. You can replay this tour anytime from here.` },
]

function GlossaryModal({ term, onClose }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const g = GLOSSARY[term]
  if (!g) return null
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal gloss" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="gloss-title">{g.t}</div>
          <button type="button" className="modal-x" onClick={onClose}>✕</button>
        </div>
        {g.b.split('\n\n').map((p, i) => <p key={i} className="gloss-body">{p}</p>)}
      </div>
    </div>
  )
}

function WelcomeModal({ onStart, onSkip }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onSkip() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onSkip])
  return (
    <div className="modal-backdrop">
      <div className="modal welcome" onClick={(e) => e.stopPropagation()}>
        <h2 className="welcome-h">Welcome</h2>
        <p className="welcome-p">An interactive tool for building an optimal investment portfolio with modern portfolio theory: the mean-variance framework used across the industry, made transparent enough to explore and understand.</p>
        <p className="welcome-p">Much of the page uses finance terminology. Every term has a "?" next to it that opens a plain-language explanation. You can take a short guided tour first, or explore on your own.</p>
        <div className="welcome-actions">
          <button type="button" className="btn-primary" onClick={onStart}>Take the tour (60s)</button>
          <button type="button" className="btn-ghost" onClick={onSkip}>Explore on my own</button>
        </div>
      </div>
    </div>
  )
}

const POP_W = 320
// Spotlight tour. One <div> carries a huge box-shadow (the dim) with a transparent
// hole over the target; a separate mask swallows clicks so the app can't be nudged
// mid-tour. No tour library; the shadow trick is the whole highlight.
function TourGuide({ step, onNext, onBack, onClose }) {
  const [rect, setRect] = useState(null)
  const s = TOUR[step]
  const last = step === TOUR.length - 1

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowRight') { if (last) onClose(); else onNext() }
      else if (e.key === 'ArrowLeft' && step > 0) onBack()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [step, last, onNext, onBack, onClose])

  useEffect(() => {
    const el = s.sel ? document.querySelector(s.sel) : null
    if (!el) { setRect(null); return }
    const measure = () => {
      const r = el.getBoundingClientRect()
      setRect({ top: r.top, left: r.left, width: r.width, height: r.height })
    }
    el.scrollIntoView({ block: 'center', inline: 'nearest' })
    const id = requestAnimationFrame(measure)            // measure after the scroll lands
    window.addEventListener('resize', measure)
    window.addEventListener('scroll', measure, true)     // capture: catch the sidebar's own scroll too
    return () => {
      cancelAnimationFrame(id)
      window.removeEventListener('resize', measure)
      window.removeEventListener('scroll', measure, true)
    }
  }, [s.sel])

  // Narrow sidebar targets -> float the card to their right. Wide main-area targets
  // -> dock the card top or bottom, whichever is away from the highlighted element.
  const vw = window.innerWidth, vh = window.innerHeight
  let pop
  if (rect && rect.left + rect.width + 16 + POP_W <= vw) {
    pop = { top: Math.min(Math.max(rect.top, 16), Math.max(vh - 300, 16)), left: rect.left + rect.width + 16, width: POP_W }
  } else {
    const dockTop = rect && rect.top + rect.height / 2 > vh / 2
    pop = dockTop ? { top: 24, left: '50%', transform: 'translateX(-50%)', width: POP_W }
      : { bottom: 24, left: '50%', transform: 'translateX(-50%)', width: POP_W }
  }

  return (
    <>
      <div className="tour-mask" style={{ background: rect ? 'transparent' : 'rgba(7,9,14,0.72)' }} />
      {rect && <div className="tour-ring" style={{ top: rect.top - 6, left: rect.left - 6, width: rect.width + 12, height: rect.height + 12 }} />}
      <div className="tour-pop" style={pop}>
        <div className="tour-step">Step {step + 1} of {TOUR.length}</div>
        <div className="tour-title">{s.t}</div>
        <p className="tour-body">{s.b}</p>
        <div className="tour-dots">{TOUR.map((_, i) => <span key={i} className={i === step ? 'on' : ''} />)}</div>
        <div className="tour-nav">
          <button type="button" className="btn-ghost sm" onClick={onClose}>Skip</button>
          <div style={{ flex: 1 }} />
          {step > 0 && <button type="button" className="btn-ghost sm" onClick={onBack}>Back</button>}
          <button type="button" className="btn-primary sm" onClick={last ? onClose : onNext}>{last ? 'Done' : 'Next'}</button>
        </div>
      </div>
    </>
  )
}

export default function App() {
  const [cov, setCov] = useState('ledoit_wolf')
  const [ret, setRet] = useState('capm')
  const [stocks, setStocks] = useState(true)
  const [maxA, setMaxA] = useState(400)
  const [erp, setErp] = useState(5)
  const [blConf, setBlConf] = useState(1)
  const [mode, setMode] = useState('classic')
  const [lev, setLev] = useState(false)
  const [maxHold, setMaxHold] = useState(25)
  const [rfInfo, setRfInfo] = useState(null)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [target, setTarget] = useState(10)
  const [detail, setDetail] = useState(null)   // {ticker, name} of the asset whose chart is open
  const [showSmall, setShowSmall] = useState(false)   // expand the "under 0.5%" sliver group
  const [progress, setProgress] = useState(0)         // 0..1 solve progress from the backend
  const [stage, setStage] = useState('')              // current solve stage label
  const [help, setHelp] = useState(null)              // glossary term key open in a modal, or null
  const [showWelcome, setShowWelcome] = useState(false)
  const [tourStep, setTourStep] = useState(null)      // null = tour off, else current step index
  const chartRef = useRef()

  const rf = rfInfo?.rate ?? 0.041

  useEffect(() => {                                    // first visit -> welcome + offer the tour
    try { if (!localStorage.getItem('ef_tour_v1')) setShowWelcome(true) } catch { /* private mode: skip */ }
  }, [])
  const seenTour = () => { try { localStorage.setItem('ef_tour_v1', '1') } catch { /* ignore */ } }
  const startTour = () => { setShowWelcome(false); seenTour(); setTourStep(0) }
  // The "?" next to any piece of jargon. preventDefault stops it toggling a checkbox when nested in a <label>.
  const H = (k) => (
    <button type="button" className="help" aria-label={`What does ${GLOSSARY[k].t} mean?`}
      onClick={(e) => { e.preventDefault(); e.stopPropagation(); setHelp(k) }}>?</button>
  )

  useEffect(() => {
    fetch('/api/riskfree').then((r) => r.json()).then(setRfInfo).catch(() => setRfInfo({ rate: 0.041, is_live: false }))
  }, [])

  useEffect(() => {
    let cancel = false
    const run = setTimeout(() => {                        // debounce rapid slider moves into one solve
      setLoading(true)
      setProgress(0); setStage('')                        // reset the bar on every recompute
      solveApi({ cov_model: cov, return_model: ret, rf: rf.toFixed(4), erp: (erp / 100).toFixed(4),
        bl_conf: blConf, include_stocks: stocks, max_assets: maxA, n_points: 20, max_holdings: maxHold },
        () => cancel,
        (frac, st) => { if (!cancel) { setProgress(frac); setStage(st) } })
        .then((d) => { if (!cancel && d) { setData(d); setLoading(false) } })
        .catch(() => { if (!cancel) setLoading(false) })
    }, 350)
    return () => { cancel = true; clearTimeout(run) }
  }, [cov, ret, stocks, maxA, erp, blConf, rf, maxHold])

  const maxV = data ? Math.max(...data.frontier.map((p) => p.vol)) : 30
  const tanVol = data ? (data.port || data.tangency).vol : 30   // your held portfolio's max-Sharpe vol
  const bounds = !data ? [0, 30]
    : mode === 'classic' ? [0, lev ? maxV : Math.min(tanVol, maxV)]
    : [data.minvar.vol, maxV]

  useEffect(() => { setTarget((t) => Math.min(Math.max(t, bounds[0]), bounds[1])) }, [data, mode, lev]) // eslint-disable-line react-hooks/exhaustive-deps

  const sel = selPortfolio(data, mode, target, rf, lev)
  const sharpe = sel && sel.vol ? (sel.ret / 100 - rf) / (sel.vol / 100) : 0

  useEffect(() => {
    if (!data || !chartRef.current) return
    const A = data.assets, cats = [...new Set(A.map((a) => a.category))]
    const traces = cats.map((c, i) => {
      const pts = A.filter((a) => a.category === c)
      return { x: pts.map((a) => a.vol), y: pts.map((a) => a.ret), mode: 'markers', type: 'scattergl', name: c,
        marker: { size: 6, color: PALETTE[i % PALETTE.length], line: { width: 0.4, color: '#0b0e14' } },
        text: pts.map((a) => `${a.ticker} — ${a.name}`), showlegend: cats.length <= 14,
        hovertemplate: '%{text}<br>risk %{x:.1f}%<br>return %{y:.1f}%<extra></extra>' }
    })
    const fr = data.frontier, t = data.tangency
    // cap x-axis near the 95th-pct asset vol so a few ultra-volatile names don't compress the view
    const avols = A.map((a) => a.vol).sort((a, b) => a - b)
    const xcap = Math.max(Math.max(...fr.map((p) => p.vol)) * 1.25, avols[Math.floor(avols.length * 0.95)] || 0)
    const sh = (t.ret / 100 - rf) / (t.vol / 100)
    traces.push({ x: fr.map((p) => p.vol), y: fr.map((p) => p.ret), mode: 'lines', name: 'Efficient frontier', line: { color: '#e6edf3', width: 2.5 } })
    traces.push({ x: [0, xcap], y: [rf * 100, (rf + sh * xcap / 100) * 100], mode: 'lines', name: 'Capital Allocation Line', line: { color: '#f5a97f', width: 1.5, dash: 'dash' } })
    traces.push({ x: [t.vol], y: [t.ret], mode: 'markers', name: 'Tangency (max Sharpe)', marker: { symbol: 'star', size: 16, color: '#f5a97f' } })
    traces.push({ x: [data.minvar.vol], y: [data.minvar.ret], mode: 'markers', name: 'Minimum variance', marker: { symbol: 'diamond', size: 11, color: '#8aadf4' } })
    if (sel) traces.push({ x: [sel.vol], y: [sel.ret], mode: 'markers', name: 'Your portfolio', marker: { symbol: 'circle', size: 14, color: '#fff', line: { width: 2, color: '#f5a97f' } } })
    traces.push({ x: [0], y: [rf * 100], mode: 'markers', name: 'Risk-free', marker: { symbol: 'square', size: 8, color: '#8bd5ca' } })
    Plotly.react(chartRef.current, traces, {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#9aa5b1', family: 'Inter, sans-serif', size: 12 },
      xaxis: { title: 'Risk — annualized volatility (%)', gridcolor: '#232b3d', zeroline: false, range: [0, xcap] },
      yaxis: { title: 'Expected return (%)', gridcolor: '#232b3d', zeroline: false },
      margin: { l: 55, r: 16, t: 16, b: 46 }, legend: { orientation: 'h', y: -0.16, font: { size: 10 } },
      hoverlabel: { bgcolor: '#161b26' },
    }, { displayModeBar: false, responsive: true })
    const gd = chartRef.current                      // open the live chart on a scatter-dot click
    gd.removeAllListeners?.('plotly_click')
    gd.on('plotly_click', (e) => {
      const txt = e.points?.[0]?.text                // only asset dots carry "TICKER — name"; overlays don't
      if (txt) setDetail({ ticker: txt.split(' — ')[0], name: txt })
    })
  }, [data, sel, rf])

  let bigRows = [], smallRows = [], nPos = 0, totalW = 0, smallW = 0
  if (data && sel) {
    const assetRows = []
    sel.tickers.forEach((tk, i) => { if (sel.w[i] > 1e-5) assetRows.push([`${tk} — ${sel.names[i]}`, sel.w[i] * 100, tk]) })
    assetRows.sort((a, b) => b[1] - a[1])
    nPos = assetRows.length
    const cashW = sel.wRf * 100
    const cash = Math.abs(cashW) > 0.01 ? [['Cash (risk-free)', cashW, null]] : []
    bigRows = cash.concat(assetRows.filter((r) => r[1] >= 0.5))   // >= 0.5% shown individually
    smallRows = assetRows.filter((r) => r[1] < 0.5)               // slivers, folded into one expandable row
    smallW = smallRows.reduce((s, r) => s + r[1], 0)
    totalW = cashW + assetRows.reduce((s, r) => s + r[1], 0)
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Efficient Frontier {H('efficient_frontier')}</h1>
        <div className="sub">Live strategic allocator · long-only, no shorting {H('long_only')}</div>
        <button type="button" className="tour-link" data-tour="replay" onClick={() => setTourStep(0)}>▶ New here? Take the 60-second tour</button>

        <div className="grp" data-tour="returns"><label>Expected returns {H('expected_return')}</label>
          <div className="chip">{RET_MODELS.map(([k, v]) =>
            <div key={k} className="chip-row">
              <button className={ret === k ? 'on' : ''} onClick={() => setRet(k)}>{v}</button>{H(k)}
            </div>)}
          </div>
        </div>

        {(ret === 'capm' || ret === 'black_litterman') &&
          <div className="grp"><label>Equity risk premium {H('erp')} · <span className="val">{erp}%</span></label>
            <input type="range" min="3" max="8" step="0.5" value={erp} onChange={(e) => setErp(+e.target.value)} /></div>}
        {ret === 'black_litterman' &&
          <div className="grp"><label>Trust in history {H('bl_conf')} · <span className="val">{blConf}</span> <span className="muted">(0 = pure CAPM)</span></label>
            <input type="range" min="0" max="20" step="0.5" value={blConf} onChange={(e) => setBlConf(+e.target.value)} /></div>}

        <div className="grp" data-tour="cov"><label>Covariance model {H('covariance')}</label><div className="seg">
          <button className={cov === 'ledoit_wolf' ? 'on' : ''} onClick={() => setCov('ledoit_wolf')}>Ledoit-Wolf</button>
          <button className={cov === 'factor' ? 'on' : ''} onClick={() => setCov('factor')}>Factor (PCA)</button>
        </div></div>

        <div className="grp" data-tour="maxhold"><label>Max holdings {H('max_holdings')} · <span className="val">{maxHold === 0 ? 'off (hold all)' : maxHold}</span></label>
          <input type="range" min="0" max="50" step="1" value={maxHold} onChange={(e) => setMaxHold(+e.target.value)} />
          <div className="muted">cap the portfolio to its top N names (0 = uncapped)</div></div>

        <div className="grp" data-tour="philosophy"><label>Philosophy {H('philosophy')}</label><div className="seg">
          <button className={mode === 'classic' ? 'on' : ''} onClick={() => setMode('classic')}>Classic + CAL</button>
          <button className={mode === 'robust' ? 'on' : ''} onClick={() => setMode('robust')}>Robust min-var</button>
        </div></div>

        {mode === 'classic' &&
          <div className="grp"><label><input type="checkbox" checked={lev} onChange={(e) => setLev(e.target.checked)} />Allow leverage (borrow at rf) {H('leverage')}</label></div>}

        <div className="grp" data-tour="risk"><label>Risk tolerance {H('target_vol')} · target vol <span className="val">{target.toFixed(1)}%</span></label>
          <input type="range" min={bounds[0]} max={bounds[1]} step="0.25" value={target} onChange={(e) => setTarget(+e.target.value)} />
          <div className="muted">feasible {bounds[0].toFixed(1)}%–{bounds[1].toFixed(1)}%</div></div>

        <div className="grp" data-tour="universe"><label>Universe {H('universe')}</label>
          <label className="muted"><input type="checkbox" checked={stocks} onChange={(e) => setStocks(e.target.checked)} />include S&P 500 single stocks {H('sp500')}</label>
          <div style={{ margin: '8px 0 2px' }} className="muted">max assets {H('assets')} <span className="val">{maxA}</span></div>
          <input type="range" min="50" max={stocks ? 612 : 130} step="10" value={maxA} onChange={(e) => setMaxA(+e.target.value)} /></div>

        <div className="rf">rf {rfInfo?.is_live ? 'live' : '—'} <span className="badge">{(rf * 100).toFixed(2)}%</span> US 3M T-bill {H('risk_free')}</div>
      </aside>

      <main className="main">
        {data && data.tangency.sharpe > 2 && ret !== 'momentum' &&
          <div className="warn">Tangency Sharpe <b>{data.tangency.sharpe.toFixed(1)}</b>, implausibly high. That's estimation
            error {H('estimation_error')}: these trailing-mean returns overfit hindsight winners. Switch Expected returns to <b>CAPM</b>, or use the
            <b> Robust min-var</b> philosophy.</div>}
        <div className="cards" data-tour="cards">
          <div className="card"><div className="k">Expected return {H('expected_return')}</div><div className="v">{sel ? sel.ret.toFixed(2) : '—'}%</div></div>
          <div className="card"><div className="k">Risk (vol) {H('volatility')}</div><div className="v">{sel ? sel.vol.toFixed(2) : '—'}%</div></div>
          <div className="card"><div className="k">Sharpe {H('sharpe')}</div><div className="v">{sharpe.toFixed(2)}</div></div>
          <div className="card"><div className="k">Assets {H('assets')}</div><div className="v">{data ? data.n_assets : '—'}</div></div>
        </div>
        <div className="chart-wrap" data-tour="chart">
          <div className="chart" ref={chartRef}></div>
          {loading && <ProgressOverlay progress={progress} stage={stage} />}
        </div>
        <div className="chartkey">
          <span>On the chart:</span>
          <span className="kk"><span className="sw" style={{ background: '#e6edf3' }} />Efficient frontier {H('efficient_frontier')}</span>
          <span className="kk"><span className="sw" style={{ background: '#f5a97f' }} />Capital Allocation Line {H('cal')}</span>
          <span className="kk"><span className="sw" style={{ background: '#f5a97f' }} />Tangency ★ {H('tangency')}</span>
          <span className="kk"><span className="sw" style={{ background: '#8aadf4' }} />Minimum variance {H('min_var')}</span>
          <span className="kk"><span className="sw" style={{ background: '#fff', border: '2px solid #f5a97f' }} />Your portfolio {H('your_portfolio')}</span>
          <span className="kk"><span className="sw" style={{ background: '#8bd5ca' }} />Risk-free {H('risk_free')}</span>
        </div>
        {sel && sel.lever > 1.001 &&
          <div className="muted" style={{ marginTop: 10 }}>Levered {sel.lever.toFixed(2)}×, borrowing {((sel.lever - 1) * 100).toFixed(0)}% at the risk-free rate (you opted in).</div>}
        <div className="alloc" data-tour="alloc">
          <div className="muted" style={{ marginBottom: 8 }}>
            Allocation {H('allocation')} · <b style={{ color: 'var(--text)' }}>{nPos}</b> position{nPos === 1 ? '' : 's'} · sums to <b style={{ color: 'var(--text)' }}>{totalW.toFixed(0)}%</b> · click any row or dot for its live chart
            {nPos > 25 && <> · broadly diversified {H('diversification')}: CAPM spreads thin; use Momentum + Max holdings to get a concentrated, rotating ~20-name portfolio</>}
          </div>
          <table><thead><tr><th>Asset</th><th>Weight</th></tr></thead>
            <tbody>
              {bigRows.map(([a, w, tk]) => <tr key={a} className={tk ? 'clickable' : ''} onClick={tk ? () => setDetail({ ticker: tk, name: a }) : undefined}><td>{a}</td><td>{w.toFixed(2)}%</td></tr>)}
              {smallRows.length > 0 &&
                <tr className="clickable small-head" onClick={() => setShowSmall((s) => !s)}>
                  <td>{showSmall ? '▾' : '▸'} Under 0.5% · {smallRows.length} position{smallRows.length === 1 ? '' : 's'}</td><td>{smallW.toFixed(2)}%</td>
                </tr>}
              {showSmall && smallRows.map(([a, w, tk]) => <tr key={a} className="clickable small-row" onClick={() => setDetail({ ticker: tk, name: a })}><td>{a}</td><td>{w.toFixed(2)}%</td></tr>)}
            </tbody></table>
        </div>
      </main>
      {detail && <AssetModal ticker={detail.ticker} name={detail.name} onClose={() => setDetail(null)} />}
      {help && <GlossaryModal term={help} onClose={() => setHelp(null)} />}
      {showWelcome && <WelcomeModal onStart={startTour} onSkip={() => { setShowWelcome(false); seenTour() }} />}
      {tourStep !== null && <TourGuide step={tourStep}
        onNext={() => setTourStep((p) => Math.min(p + 1, TOUR.length - 1))}
        onBack={() => setTourStep((p) => Math.max(p - 1, 0))}
        onClose={() => setTourStep(null)} />}
    </div>
  )
}
