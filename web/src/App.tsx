import { useEffect, useState } from 'react'

type ClassName = 'not_out' | 'full' | 'partial'

interface Weather {
  station: string
  visibility_sm: number | null
  ceiling_ft: number | null
  raw: string
}

interface State {
  timestamp_utc: string | null
  class_index: 0 | 1 | 2 | null
  class_name: ClassName | null
  is_out: boolean | null
  confidence: Record<ClassName, number> | null
  weather: Weather | null
  webcam_url: string
  model_version: string | null
}

interface Presentation {
  headline: string
  sub: string
  bg: string
  fg: string
  accent: string
}

const PRESENTATION: Record<ClassName | 'unknown', Presentation> = {
  full: {
    headline: 'YES.',
    sub: "She's out.",
    bg: 'bg-gradient-to-b from-sky-300 via-sky-200 to-amber-100',
    fg: 'text-slate-900',
    accent: 'text-rose-600',
  },
  partial: {
    headline: 'SORT OF.',
    sub: 'Peeking through.',
    bg: 'bg-gradient-to-b from-slate-400 via-slate-300 to-amber-200',
    fg: 'text-slate-900',
    accent: 'text-indigo-700',
  },
  not_out: {
    headline: 'NOPE.',
    sub: 'Not today.',
    bg: 'bg-gradient-to-b from-slate-800 via-slate-700 to-slate-900',
    fg: 'text-slate-100',
    accent: 'text-sky-300',
  },
  unknown: {
    headline: 'CHECKING…',
    sub: 'Asking the webcam.',
    bg: 'bg-slate-900',
    fg: 'text-slate-200',
    accent: 'text-slate-400',
  },
}

const STALE_MS = 60 * 60 * 1000
// Same-origin: the Worker in ../worker/index.ts serves it from R2, and
// `vite dev` proxies it (vite.config.ts). No cross-origin fetch, no CORS.
const STATE_URL = '/state.json'

function formatRelative(iso: string | null, now: number): string {
  if (!iso) return 'never'
  const then = Date.parse(iso)
  if (Number.isNaN(then)) return 'unknown'
  const diffSec = Math.round((now - then) / 1000)
  if (diffSec < 30) return 'just now'
  if (diffSec < 90) return '1 minute ago'
  if (diffSec < 3600) return `${Math.round(diffSec / 60)} minutes ago`
  if (diffSec < 5400) return '1 hour ago'
  return `${Math.round(diffSec / 3600)} hours ago`
}

function App() {
  const [state, setState] = useState<State | null>(null)
  const [error, setError] = useState<string | null>(null)
  // Wall clock as of the last poll. Read once per poll rather than during
  // render so staleness is a stable input, not a fresh Date.now() every paint.
  const [now, setNow] = useState(() => Date.now())
  const debug = new URLSearchParams(window.location.search).has('debug')

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const polledAt = Date.now()
      if (!cancelled) setNow(polledAt)
      try {
        const resp = await fetch(`${STATE_URL}?t=${polledAt}`)
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const json = (await resp.json()) as State
        if (!cancelled) {
          setState(json)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    }
    load()
    const id = window.setInterval(load, 60_000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  const className = state?.class_name ?? null
  const timestamp = state?.timestamp_utc ?? null
  const stale = timestamp ? now - Date.parse(timestamp) > STALE_MS : false
  const key: ClassName | 'unknown' = className && !stale ? className : 'unknown'
  const look = PRESENTATION[key]

  return (
    <div className={`min-h-full w-full ${look.bg} ${look.fg} flex flex-col`}>
      <main className="flex-1 flex flex-col items-center justify-center px-6 text-center">
        <h1
          className="font-black tracking-tight leading-none"
          style={{ fontSize: 'clamp(5rem, 22vw, 20rem)' }}
        >
          {look.headline}
        </h1>
        <p className="mt-4 text-2xl sm:text-3xl font-medium opacity-90">{look.sub}</p>
        <p className={`mt-10 text-sm uppercase tracking-widest ${look.accent}`}>
          {stale && timestamp
            ? `stale — last checked ${formatRelative(timestamp, now)}`
            : timestamp
            ? `checked ${formatRelative(timestamp, now)}`
            : error
            ? 'state unavailable'
            : 'loading…'}
        </p>
      </main>

      {debug && state && <DebugPanel state={state} />}

      <footer className="px-6 py-4 text-xs opacity-60 text-center">
        Mount Rainier · UW ATG webcam · METAR {state?.weather?.station ?? '—'} ·{' '}
        <a
          className="underline hover:opacity-100"
          href="https://github.com/robogeosociety/is-the-mountain-out"
        >
          source
        </a>
      </footer>
    </div>
  )
}

function DebugPanel({ state }: { state: State }) {
  const conf = state.confidence
  const weather = state.weather
  return (
    <section className="w-full max-w-xl mx-auto mb-8 px-6">
      <div className="rounded-2xl bg-black/30 backdrop-blur p-5 text-sm font-mono">
        <div className="mb-4">
          <div className="uppercase text-xs opacity-60 mb-2">confidence</div>
          {conf ? (
            (['not_out', 'full', 'partial'] as ClassName[]).map((k) => (
              <ConfidenceBar key={k} label={k} value={conf[k]} />
            ))
          ) : (
            <div className="opacity-60">unavailable</div>
          )}
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <div className="opacity-60">station</div>
          <div>{weather?.station ?? '—'}</div>
          <div className="opacity-60">visibility</div>
          <div>{weather?.visibility_sm != null ? `${weather.visibility_sm} SM` : '—'}</div>
          <div className="opacity-60">ceiling</div>
          <div>{weather?.ceiling_ft != null ? `${weather.ceiling_ft} ft` : 'none'}</div>
          <div className="opacity-60">model</div>
          <div>{state.model_version ?? '—'}</div>
          <div className="opacity-60">timestamp</div>
          <div>{state.timestamp_utc ?? '—'}</div>
        </div>
        {weather?.raw && (
          <div className="mt-3 pt-3 border-t border-white/10 break-all text-xs opacity-80">
            {weather.raw}
          </div>
        )}
      </div>
    </section>
  )
}

function ConfidenceBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 1000) / 10
  return (
    <div className="mb-1">
      <div className="flex justify-between text-xs">
        <span>{label}</span>
        <span>{pct.toFixed(1)}%</span>
      </div>
      <div className="h-1.5 rounded bg-white/10 overflow-hidden">
        <div className="h-full bg-white/70" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default App
