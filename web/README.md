# Frontend — Efficient Frontier

The React + Vite single-page app for the efficient-frontier allocator. The whole
UI lives in `src/App.jsx`; styles in `src/index.css`.

## Develop

```bash
npm install
npm run dev     # http://localhost:5173 — proxies /api to the backend on :8000
```

## Build

```bash
npm run build   # static output in dist/, served by the FastAPI backend in production
npm run lint    # oxlint
```

See the repository root `README.md` for the full project: the backend, the MPT
math, and how it's deployed.
