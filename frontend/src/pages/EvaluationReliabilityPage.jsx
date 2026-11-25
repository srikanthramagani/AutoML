import { useState } from 'react'
import { AlertTriangle, ArrowLeft, Download, ShieldCheck } from 'lucide-react'
import { downloadEvaluationAudit, runCorrectedEvaluation } from '../services/api'

const severityStyles = {
  Critical: 'border-red-300 bg-red-50 text-red-800',
  High: 'border-orange-300 bg-orange-50 text-orange-800',
  Medium: 'border-amber-300 bg-amber-50 text-amber-800',
  Low: 'border-slate-300 bg-slate-50 text-slate-700',
  Informational: 'border-blue-200 bg-blue-50 text-blue-800',
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export default function EvaluationReliabilityPage({ job, onBack }) {
  const [excludedFeatures, setExcludedFeatures] = useState([])
  const [corrected, setCorrected] = useState(null)
  const [error, setError] = useState('')
  const [running, setRunning] = useState(false)
  const report = job?.evaluation_audit
  if (!job) return null

  async function exportReport(format) {
    try {
      setError('')
      const { data } = await downloadEvaluationAudit(job.job_id, format)
      saveBlob(data, `evaluation-audit-${job.job_id}.${format}`)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }

  async function evaluateCorrected() {
    try {
      setRunning(true)
      setError('')
      const response = await runCorrectedEvaluation(job.job_id, excludedFeatures)
      setCorrected(response.data)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally {
      setRunning(false)
    }
  }

  if (!report) {
    return (
      <main className="product-page max-w-7xl mx-auto px-4 py-8">
        <button className="text-action" onClick={onBack}><ArrowLeft size={15} /> Return to analysis</button>
        <section className="chart-card">
          <h1 className="text-xl font-bold">Evaluation Reliability</h1>
          <p className="text-sm text-slate-600">{job.audit_error || 'No audit report is available for this experiment.'}</p>
        </section>
      </main>
    )
  }

  const reliability = report.reliability || {}
  const findings = report.findings || []
  const integrity = report.summary || {}
  const suspectedFeatures = (integrity.potential_leakage_features || []).map((item) => item.feature)
  const metricRows = corrected?.comparison || []

  return (
    <main className="analysis-workbench product-page max-w-7xl mx-auto px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200/80 pb-4 mb-6">
        <button className="text-action" onClick={onBack}><ArrowLeft size={15} /> Return to analysis</button>
        <div className="flex gap-2">
          <button className="banner-action" onClick={() => exportReport('json')}><Download size={14} /> Export JSON</button>
          <button className="banner-action" onClick={() => exportReport('csv')}><Download size={14} /> Export CSV</button>
        </div>
      </div>

      <header className="mb-6">
        <p className="font-mono text-xs font-bold uppercase tracking-wider text-indigo-600">Evidence-based evaluation audit</p>
        <h1 className="mt-1 text-3xl font-extrabold tracking-tight text-slate-900">Evaluation Reliability</h1>
        <p className="mt-2 text-sm text-slate-600">Checks describe detected evidence; a high model score alone is not treated as leakage.</p>
      </header>

      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4 mb-6">
        <div className="chart-card mt-0">
          <span className="text-xs uppercase text-slate-500">Reliability score</span>
          <strong className="mt-2 block text-3xl text-slate-900">{reliability.score == null ? 'Not enough evidence' : `${reliability.score}/100`}</strong>
        </div>
        <div className="chart-card mt-0">
          <span className="text-xs uppercase text-slate-500">Audit status</span>
          <strong className="mt-2 block text-sm text-slate-900">{reliability.status || 'INCONCLUSIVE'}</strong>
        </div>
        <div className="chart-card mt-0">
          <span className="text-xs uppercase text-slate-500">Findings</span>
          <strong className="mt-2 block text-3xl text-slate-900">{findings.length}</strong>
        </div>
        <div className="chart-card mt-0">
          <span className="text-xs uppercase text-slate-500">Critical / High</span>
          <strong className="mt-2 block text-3xl text-slate-900">{integrity.critical_count || 0} / {integrity.high_count || 0}</strong>
        </div>
      </section>

      <section className="chart-card">
        <div className="section-heading">
          <div><h2>Dataset integrity</h2><p className="text-xs text-slate-500">{report.dataset?.name} · {report.dataset?.samples} samples · {report.dataset?.features} features · target: {report.dataset?.target_column || 'none'}</p></div>
          <ShieldCheck className="text-indigo-600" size={18} />
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-lg border p-3"><span className="text-xs text-slate-500">Duplicate rows</span><strong className="block text-lg">{report.dataset?.duplicate_record_count} ({report.dataset?.duplicate_record_pct}%)</strong></div>
          <div className="rounded-lg border p-3"><span className="text-xs text-slate-500">Missing values</span><strong className="block text-lg">{report.dataset?.missing_value_pct}%</strong></div>
          <div className="rounded-lg border p-3"><span className="text-xs text-slate-500">Exact train/test overlap</span><strong className="block text-lg">{integrity.duplicate_train_test_overlap || 0}</strong></div>
          <div className="rounded-lg border p-3"><span className="text-xs text-slate-500">Potential leakage features</span><strong className="block text-lg">{suspectedFeatures.length}</strong></div>
        </div>
        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-semibold">Reliability score evidence and weights</summary>
          <p className="mt-2 text-xs text-slate-600">{reliability.formula}</p>
          <div className="mt-2 grid gap-1 sm:grid-cols-2">
            {Object.entries(reliability.components || {}).map(([name, component]) => (
              <p key={name} className="text-xs text-slate-600"><strong>{name.replaceAll('_', ' ')} · {component.status}:</strong> {component.evidence}</p>
            ))}
          </div>
        </details>
      </section>

      <section className="chart-card">
        <h2 className="text-lg font-bold">Audit findings <span className="text-sm font-normal text-slate-500">({findings.length})</span></h2>
        {findings.length === 0 && <p className="mt-3 text-sm text-slate-600">No findings from the checks that ran. This is not a guarantee of absolute trustworthiness.</p>}
        <div className="mt-3 grid gap-2">
          {findings.map((finding) => (
            <details key={finding.id} className={`rounded-lg border p-3 ${severityStyles[finding.severity] || severityStyles.Low}`}>
              <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-semibold">
                <AlertTriangle size={15} /> {finding.title}
                <span className="ml-auto text-[10px] uppercase">{finding.severity}{finding.verified ? ' · verified' : ' · suspected'}</span>
              </summary>
              <div className="mt-3 space-y-2 text-xs">
                <p><strong>Explanation:</strong> {finding.explanation}</p>
                <p><strong>Evidence:</strong> {finding.evidence}</p>
                <p><strong>Potential impact:</strong> {finding.potential_impact}</p>
                <p><strong>Recommended action:</strong> {finding.recommended_correction}</p>
              </div>
            </details>
          ))}
        </div>
      </section>

      <section className="chart-card">
        <h2 className="text-lg font-bold">Original vs corrected evaluation</h2>
        <p className="mt-1 text-xs text-slate-500">Corrected evaluation clones and retrains the selected pipeline on a new split. Select any suspected feature for exclusion; no feature is removed without your choice.</p>
        {suspectedFeatures.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-3">
            {suspectedFeatures.map((feature) => (
              <label key={feature} className="flex items-center gap-2 text-xs">
                <input type="checkbox" checked={excludedFeatures.includes(feature)} onChange={(event) =>
                  setExcludedFeatures((items) => event.target.checked ? [...items, feature] : items.filter((item) => item !== feature))} />
                Exclude {feature}
              </label>
            ))}
          </div>
        )}
        <button className="mt-3 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-semibold text-white disabled:opacity-50" onClick={evaluateCorrected} disabled={running || !job.target_column}>
          {running ? 'Retraining and evaluating…' : 'Run corrected evaluation'}
        </button>
        {!job.target_column && <p className="mt-2 text-xs text-amber-700">A supervised target is required for corrected evaluation.</p>}
        {corrected?.status === 'not_possible' && <p className="mt-3 text-sm text-amber-700">{corrected.reason}</p>}
        {corrected?.status === 'complete' && (
          <div className="comparison-table-wrap">
            <table className="comparison-table">
              <thead><tr><th>Metric</th><th>Original evaluation</th><th>Corrected evaluation</th><th>Difference</th></tr></thead>
              <tbody>{metricRows.map((row) => <tr key={row.metric}><td>{row.metric}</td><td>{row.original ?? '—'}</td><td>{row.corrected}</td><td>{row.difference ?? '—'}</td></tr>)}</tbody>
            </table>
            <p className="px-3 py-2 text-xs text-slate-500">{corrected.note} · test set: {corrected.test_size} rows · excluded features: {corrected.excluded_features.join(', ') || 'none'}</p>
          </div>
        )}
        {error && <p role="alert" className="mt-3 text-sm text-red-700">{error}</p>}
      </section>

      <section className="chart-card">
        <h2 className="text-lg font-bold">Evaluation metadata</h2>
        <div className="mt-3 grid gap-2 text-xs text-slate-600 sm:grid-cols-2">
          <p><strong>Model:</strong> {report.evaluation?.model_name}</p>
          <p><strong>Validation:</strong> {report.evaluation?.validation_strategy} ({report.evaluation?.cv_folds || '—'} folds)</p>
          <p><strong>Test-set size:</strong> {report.evaluation?.test_set_size}</p>
          <p><strong>Random seed:</strong> {report.evaluation?.random_seed}</p>
          <p className="sm:col-span-2"><strong>Dataset fingerprint:</strong> <code>{report.dataset?.fingerprint}</code></p>
        </div>
      </section>
    </main>
  )
}
