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
// solve pins the single CPU the server may briefly return 502s — hit() swallows those so we just
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
    if (!r) continue                                     // 502/network blip during a busy solve — retry
    if (r.status === 'done') return r.result
    if (r.status === 'error') throw new Error(r.error || 'solve failed')
    if (r.status === 'unknown') { job = null; continue } // instance restarted, job gone — re-POST next loop
    if (r.job) job = r.job
    tick(r)
  }
  throw new Error('solve timed out')
}

// Map the risk slider to an allocation (client-side). Risky weights are always >= 0.
function selPortfolio(d, mode, target, rf, lev) {
  if (!d) return null
  const t = d.tangency
  if (mode === 'classic') {
    let y = target / t.vol
    if (!lev) y = Math.min(y, 1)
    return { ret: rf * 100 + y * (t.ret - rf * 100), vol: y * t.vol, w: t.w.map((x) => x * y), wRf: 1 - y, lever: y }
  }
  let bi = 0, bd = 1e9
  d.frontier.forEach((p, i) => { const dd = Math.abs(p.vol - target); if (dd < bd) { bd = dd; bi = i } })
  return { ret: d.frontier[bi].ret, vol: d.frontier[bi].vol, w: d.frontier_w[bi], wRf: 0, lever: 1 }
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
  const chartRef = useRef()

  const rf = rfInfo?.rate ?? 0.041

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
  const bounds = !data ? [0, 30]
    : mode === 'classic' ? [0, lev ? maxV : Math.min(data.tangency.vol, maxV)]
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
    data.tickers.forEach((tk, i) => { if (sel.w[i] > 1e-5) assetRows.push([`${tk} — ${data.names[i]}`, sel.w[i] * 100, tk]) })
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
        <h1>Efficient Frontier</h1>
        <div className="sub">Live strategic allocator · long-only, no shorting</div>

        <div className="grp"><label>Expected returns</label>
          <div className="chip">{RET_MODELS.map(([k, v]) =>
            <button key={k} className={ret === k ? 'on' : ''} onClick={() => setRet(k)}>{v}</button>)}
          </div>
        </div>

        {(ret === 'capm' || ret === 'black_litterman') &&
          <div className="grp"><label>Equity risk premium · <span className="val">{erp}%</span></label>
            <input type="range" min="3" max="8" step="0.5" value={erp} onChange={(e) => setErp(+e.target.value)} /></div>}
        {ret === 'black_litterman' &&
          <div className="grp"><label>Trust in history · <span className="val">{blConf}</span> <span className="muted">(0 = pure CAPM)</span></label>
            <input type="range" min="0" max="20" step="0.5" value={blConf} onChange={(e) => setBlConf(+e.target.value)} /></div>}

        <div className="grp"><label>Covariance model</label><div className="seg">
          <button className={cov === 'ledoit_wolf' ? 'on' : ''} onClick={() => setCov('ledoit_wolf')}>Ledoit-Wolf</button>
          <button className={cov === 'factor' ? 'on' : ''} onClick={() => setCov('factor')}>Factor (PCA)</button>
        </div></div>

        <div className="grp"><label>Max holdings · <span className="val">{maxHold === 0 ? 'off (hold all)' : maxHold}</span></label>
          <input type="range" min="0" max="50" step="1" value={maxHold} onChange={(e) => setMaxHold(+e.target.value)} />
          <div className="muted">cap the portfolio to its top N names (0 = uncapped)</div></div>

        <div className="grp"><label>Philosophy</label><div className="seg">
          <button className={mode === 'classic' ? 'on' : ''} onClick={() => setMode('classic')}>Classic + CAL</button>
          <button className={mode === 'robust' ? 'on' : ''} onClick={() => setMode('robust')}>Robust min-var</button>
        </div></div>

        {mode === 'classic' &&
          <div className="grp"><label><input type="checkbox" checked={lev} onChange={(e) => setLev(e.target.checked)} />Allow leverage (borrow at rf)</label></div>}

        <div className="grp"><label>Risk tolerance · target vol <span className="val">{target.toFixed(1)}%</span></label>
          <input type="range" min={bounds[0]} max={bounds[1]} step="0.25" value={target} onChange={(e) => setTarget(+e.target.value)} />
          <div className="muted">feasible {bounds[0].toFixed(1)}%–{bounds[1].toFixed(1)}%</div></div>

        <div className="grp"><label>Universe</label>
          <label className="muted"><input type="checkbox" checked={stocks} onChange={(e) => setStocks(e.target.checked)} />include S&P 500 single stocks</label>
          <div style={{ margin: '8px 0 2px' }} className="muted">max assets <span className="val">{maxA}</span></div>
          <input type="range" min="50" max={stocks ? 612 : 130} step="10" value={maxA} onChange={(e) => setMaxA(+e.target.value)} /></div>

        <div className="rf">rf {rfInfo?.is_live ? 'live' : '—'} <span className="badge">{(rf * 100).toFixed(2)}%</span> US 3M T-bill</div>
      </aside>

      <main className="main">
        {data && data.tangency.sharpe > 2 && ret !== 'momentum' &&
          <div className="warn">⚠️ Tangency Sharpe <b>{data.tangency.sharpe.toFixed(1)}</b> — implausibly high. That's estimation
            error: these trailing-mean returns overfit hindsight winners. Switch Expected returns to <b>CAPM</b>, or use the
            <b> Robust min-var</b> philosophy.</div>}
        <div className="cards">
          <div className="card"><div className="k">Expected return</div><div className="v">{sel ? sel.ret.toFixed(2) : '—'}%</div></div>
          <div className="card"><div className="k">Risk (vol)</div><div className="v">{sel ? sel.vol.toFixed(2) : '—'}%</div></div>
          <div className="card"><div className="k">Sharpe</div><div className="v">{sharpe.toFixed(2)}</div></div>
          <div className="card"><div className="k">Assets</div><div className="v">{data ? data.n_assets : '—'}</div></div>
        </div>
        <div className="chart-wrap">
          <div className="chart" ref={chartRef}></div>
          {loading && <ProgressOverlay progress={progress} stage={stage} />}
        </div>
        {sel && sel.lever > 1.001 &&
          <div className="muted" style={{ marginTop: 10 }}>Levered {sel.lever.toFixed(2)}× — borrowing {((sel.lever - 1) * 100).toFixed(0)}% at the risk-free rate (you opted in).</div>}
        <div className="alloc">
          <div className="muted" style={{ marginBottom: 8 }}>
            Allocation · <b style={{ color: 'var(--text)' }}>{nPos}</b> position{nPos === 1 ? '' : 's'} · sums to <b style={{ color: 'var(--text)' }}>{totalW.toFixed(0)}%</b> · click any row or dot for its live chart
            {nPos > 25 && ' · broadly diversified — CAPM spreads thin; use Momentum + Max holdings to get a concentrated, rotating ~20-name portfolio'}
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
    </div>
  )
}
