import { useState } from 'react'
import { downloadForecastUrl, predictForecast } from '../services/api'

function pathFor(rows, key, width, height, pad, min, max) {
  const range = max - min || 1
  return rows.map((row, index) => {
    const x = pad + index * (width - 2 * pad) / Math.max(1, rows.length - 1)
    const y = height - pad - (Number(row[key]) - min) * (height - 2 * pad) / range
    return `${x},${y}`
  }).join(' ')
}

export default function ForecastResultPanel({ result }) {
  const [historyFile, setHistoryFile] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState('')
  if (!result) return null
  const rows = result.predictions || []
  const values = rows.flatMap((row) => [row.actual == null ? null : Number(row.actual), Number(row.predicted), Number(row.naive)]).filter(Number.isFinite)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const width = 760, height = 280, pad = 32
  const modelLabel = {
    ridge: 'Ridge',
    lstm: 'LSTM',
    bilstm: 'Bidirectional LSTM',
    bilstm_stack: 'Stacked Bidirectional LSTM',
  }[result.config?.model_family] || 'Forecast model'
  const resolvedFamilies = Object.values(result.config?.resolved_model_family_by_horizon || {})
  const resolvedLabel = resolvedFamilies.length && resolvedFamilies.every((family) => family === 'ridge_fallback')
    ? `${modelLabel} (Ridge fallback)`
    : modelLabel
  return <section className="chart-card mt-5" aria-label="Forecast results">
    <div className="section-heading"><div><p className="card-kicker">Fixed-origin forecast / {result.config?.horizon} steps</p><h2>Forecast against actuals and naive baseline</h2></div></div>
    <div className="overflow-x-auto"><svg viewBox={`0 0 ${width} ${height}`} className="min-w-[600px] w-full" role="img" aria-label="Forecast, actual, and naive values">
      {[['actual', '#2259ce'], ['predicted', '#f26430'], ['naive', '#78877d']].map(([key, color]) => <polyline key={key} points={pathFor(key === 'actual' ? rows.filter((row) => row.actual != null) : rows, key, width, height, pad, min, max)} fill="none" stroke={color} strokeWidth="3" strokeDasharray={key === 'naive' ? '7 5' : undefined} />)}
    </svg></div>
    <div className="trajectory-legend"><span className="legend-item actual">Actual</span><span className="legend-item predicted">{resolvedLabel}</span><span>Naive last-value</span></div>
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{Object.entries(result.metrics || {}).map(([key, value]) => <div className="gen-metric-box" key={key}><span>{key.replaceAll('_', ' ')}</span><strong>{value == null ? 'n/a' : Number(value).toFixed(4)}</strong></div>)}</div>
    <p className="metric-key">Forecasting mode: {result.config?.forecast_mode}. Features use observations available at the fixed forecast origin. The naive baseline repeats the final training value.</p>
    {result.forecast_id && <div className="mt-4 flex flex-wrap items-center gap-3"><a className="primary-action compact" href={downloadForecastUrl(result.forecast_id)}>Download forecast model</a><label className="upload-predictions">Append observed history CSV (optional)<input type="file" accept=".csv,text/csv" onChange={(event) => setHistoryFile(event.target.files?.[0] || null)} /></label><button className="primary-action compact" onClick={async () => { try { setError(''); const response = await predictForecast(result.forecast_id, historyFile, result.config?.horizon); setPrediction(response.data.predictions) } catch (requestError) { setError(requestError.response?.data?.detail || requestError.message) } }}>Predict future</button></div>}
    {error && <p className="notice">{error}</p>}
    {prediction && <div className="comparison-table-wrap mt-3"><table className="comparison-table"><thead><tr><th>Horizon</th><th>Timestamp</th><th>Forecast</th></tr></thead><tbody>{prediction.map((row) => <tr key={row.horizon}><td>{row.horizon}</td><td>{row.index}</td><td>{Number(row.predicted).toFixed(4)}</td></tr>)}</tbody></table></div>}
  </section>
}
