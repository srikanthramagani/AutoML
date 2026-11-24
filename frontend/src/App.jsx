import { useEffect, useState } from 'react'
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  BarChart3,
  Boxes,
  ChevronDown,
  Database,
  Download,
  FlaskConical,
  Gauge,
  Layers3,
  Play,
  Settings2,
  Sparkles,
  UploadCloud,
} from 'lucide-react'
import { runForecast, startMultimodalTraining, uploadPredictionCsv } from './services/api'
import { useAutoML } from './hooks/useAutoML'
import DatasetProfile from './components/DatasetProfile'
import AnalysisPage from './pages/AnalysisPage'
import ExportPage from './pages/ExportPage'
import ForecastResultPanel from './components/ForecastResultPanel'

const navigation = [
  { id: 'run', label: 'Workspace', icon: Boxes },
  { id: 'analysis', label: 'Analysis', icon: BarChart3, requiresJob: true },
  { id: 'export', label: 'Export', icon: Download, requiresJob: true },
]

function App() {
  const { dataset, job, error, upload, train } = useAutoML()
  const [file, setFile] = useState(null)
  const [target, setTarget] = useState('')
  const [trials, setTrials] = useState(15)
  const [folds, setFolds] = useState(5)
  const [metric, setMetric] = useState('auto')
  const [task, setTask] = useState('auto')
  const [activeStep, setActiveStep] = useState(0)
  const [view, setView] = useState('run')
  const [keptIdentifiers, setKeptIdentifiers] = useState([])
  const [multimodal, setMultimodal] = useState(false)
  const [representation, setRepresentation] = useState('auto')
  const [fusion, setFusion] = useState('auto')
  const [multimodalResult, setMultimodalResult] = useState(null)
  const [columnOverrides, setColumnOverrides] = useState({})
  const [predictionOutput, setPredictionOutput] = useState(null)
  const [predictionError, setPredictionError] = useState('')
  const [forecastTimeColumn, setForecastTimeColumn] = useState('')
  const [forecastHorizon, setForecastHorizon] = useState(7)
  const [forecastModelFamily, setForecastModelFamily] = useState('ridge')
  const [forecastResult, setForecastResult] = useState(null)
  const [forecastError, setForecastError] = useState('')

  async function handleUpload(event) {
    const selected = event.target.files?.[0]
    if (!selected) return
    setFile(selected)
    setActiveStep(1)
    try {
      const result = await upload(selected)
      setTarget(result.target || '')
      setKeptIdentifiers((result.column_details || [])
        .filter((column) => column.identifier_candidate && !column.identifier_excluded_by_default)
        .map((column) => column.name))
    } catch (requestError) {
      setForecastError(requestError.response?.data?.detail || requestError.message)
    }
  }

  async function handleTrain(event) {
    event.preventDefault()
    try {
      setForecastError('')
      if (task === 'time_series_forecasting') {
        const response = await runForecast({
          dataset_id: dataset.id,
          target_column: target,
          datetime_column: forecastTimeColumn || null,
          horizon: Number(forecastHorizon),
          n_trials: Number(trials),
          cv_folds: Number(folds),
          model_family: forecastModelFamily,
        })
        setForecastResult(response.data)
        setActiveStep(3)
        return
      }
      if (multimodal) {
        const response = await startMultimodalTraining({
          dataset_id: dataset.id,
          target_column: target,
          representation,
          fusion,
          trials: Number(trials),
          cv_folds: Number(folds),
          overrides: columnOverrides,
        })
        setMultimodalResult(response.data)
        setActiveStep(3)
        return
      }
      await train({
        target_column: task === 'clustering' ? '' : target,
        metric,
        task,
        optimization_trials: Number(trials),
        cv_folds: Number(folds),
        keep_identifier_columns: keptIdentifiers,
      })
      setActiveStep(3)
    } catch (requestError) {
      setForecastError(requestError.response?.data?.detail || requestError.message)
    }
  }

  useEffect(() => {
    if (job?.status === 'complete') setActiveStep(3)
  }, [job?.status])

  const steps = [
    { label: 'Dataset', detail: 'CSV intake' },
    { label: 'Profile', detail: 'Schema scan' },
    { label: 'Configure', detail: 'Run settings' },
    { label: 'Review', detail: 'Results' },
  ]

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <button className="sidebar-brand" onClick={() => setView('run')} aria-label="Adaptive AutoML home">
          <span className="brand-glyph"><FlaskConical size={18} /></span>
          <span><strong>Adaptive</strong><small>AutoML Studio</small></span>
        </button>

        <nav className="sidebar-nav" aria-label="Primary navigation">
          <p className="sidebar-label">Platform</p>
          {navigation.map((item) => {
            const Icon = item.icon
            const disabled = item.requiresJob && job?.status !== 'complete'
            return (
              <button
                key={item.id}
                className={view === item.id ? 'sidebar-link active' : 'sidebar-link'}
                disabled={disabled}
                onClick={() => setView(item.id)}
              >
                <Icon size={17} />
                <span>{item.label}</span>
                {view === item.id && <span className="nav-indicator" />}
              </button>
            )
          })}
        </nav>

        <div className="sidebar-status">
          <span className="pulse-dot" />
          <div><strong>Compute ready</strong><small>Local engine connected</small></div>
        </div>
      </aside>

      <main className="app-main">
        {view === 'analysis' && job?.status === 'complete' && (
          <AnalysisPage job={job} dataset={dataset} onBack={() => setView('run')} onExport={() => setView('export')} />
        )}
        {view === 'export' && job?.status === 'complete' && (
          <ExportPage job={job} dataset={dataset} onBack={() => setView('analysis')} />
        )}

        {view === 'run' && (
          <>
            <header className="workspace-header">
              <div>
                <div className="breadcrumb"><span>Projects</span><span>/</span><strong>New experiment</strong></div>
                <h1>Model workspace</h1>
              </div>
              <div className="header-actions">
                {job?.status && <span className={`run-status ${job.status}`}><Activity size={14} /> {job.status}</span>}
                <button className="icon-button" aria-label="Workspace settings"><Settings2 size={18} /></button>
              </div>
            </header>

            <section className="progress-card" aria-label="Experiment progress">
              {steps.map((step, index) => (
                <button
                  type="button"
                  key={step.label}
                  className={`progress-step ${activeStep === index ? 'active' : ''} ${activeStep > index ? 'complete' : ''}`}
                  onClick={() => setActiveStep(index)}
                >
                  <span className="step-number">{activeStep > index ? '✓' : String(index + 1).padStart(2, '0')}</span>
                  <span><strong>{step.label}</strong><small>{step.detail}</small></span>
                </button>
              ))}
            </section>

            <div className="workspace-grid">
              <section className="workspace-content">
                <div className="section-heading-row">
                  <div>
                    <p className="kicker">01 · Data source</p>
                    <h2>Build a training run</h2>
                  </div>
                  {dataset && <span className="secure-badge"><Database size={14} /> Profiled successfully</span>}
                </div>

                <label className={`upload-zone ${file ? 'has-file' : ''}`}>
                  <input type="file" accept=".csv,text/csv" onChange={handleUpload} />
                  <span className="upload-icon"><UploadCloud size={22} /></span>
                  <span className="upload-copy">
                    <strong>{file ? file.name : 'Drop in a CSV dataset'}</strong>
                    <small>{file ? `${(file.size / 1024).toFixed(1)} KB · ready for profiling` : 'or click to browse · .csv up to 50 MB'}</small>
                  </span>
                  <span className="browse-button">{file ? 'Replace' : 'Browse'} <ArrowRight size={15} /></span>
                </label>
                <p className="notice">Use a real CSV with documented rows and columns. Repository datasets are available under <code>datasets/</code>; synthetic benchmark data is not used for this training workflow.</p>

                {dataset && (
                  <DatasetProfile
                    dataset={dataset}
                    keptIdentifiers={keptIdentifiers}
                    onKeepIdentifier={(columnName, keep) =>
                      setKeptIdentifiers((current) => keep
                        ? [...new Set([...current, columnName])]
                        : current.filter((name) => name !== columnName))}
                  />
                )}

                {dataset?.modality_report && (
                  <div className="modality-summary">
                    <div className="analysis-title"><Layers3 size={15} /><span>Modality detection</span></div>
                    <div className="modality-grid">
                      {Object.entries(dataset.modality_report)
                        .filter(([key, value]) => Array.isArray(value) && key !== 'high_cardinality')
                        .map(([kind, columns]) => (
                          <div key={kind}><span>{kind.replaceAll('_', ' ')}</span><strong>{columns.length ? columns.join(', ') : 'None detected'}</strong></div>
                        ))}
                    </div>
                  </div>
                )}

                {dataset?.column_details && (
                  <details className="column-overrides">
                    <summary>Review column assignments <ChevronDown size={15} /></summary>
                    <div className="override-list">
                      {dataset.column_details.filter((column) => column.name !== target).map((column) => (
                        <label className="override-row" key={column.name}>
                          <span><strong>{column.name}</strong><small>{column.inferred_type}</small></span>
                          <select
                            value={columnOverrides[column.name] || ''}
                            onChange={(event) => setColumnOverrides((state) => ({ ...state, [column.name]: event.target.value }))}
                          >
                            <option value="">Automatic</option>
                            {['numerical', 'categorical', 'boolean', 'datetime', 'text', 'identifiers', 'exclude'].map((item) => (
                              <option key={item} value={item}>{item}</option>
                            ))}
                          </select>
                        </label>
                      ))}
                    </div>
                  </details>
                )}
              </section>

              <aside className="run-config">
                <div className="config-header">
                  <div><p className="kicker">02 · Configuration</p><h3>Run settings</h3></div>
                  <Gauge size={18} />
                </div>

                <form onSubmit={handleTrain}>
                  <label className="config-field">
                    <span>Target column</span>
                    <input value={target} onChange={(event) => setTarget(event.target.value)} placeholder={dataset ? dataset.target || 'Select a column' : 'Upload dataset first'} />
                  </label>

                  <div className="config-field">
                    <span>Task type</span>
                    <select value={task} onChange={(event) => setTask(event.target.value)}>
                      <option value="auto">Auto-detect</option>
                      <option value="classification">Classification</option>
                      <option value="regression">Regression</option>
                      <option value="clustering">Clustering</option>
                      <option value="time_series_forecasting">Time-series forecasting</option>
                    </select>
                  </div>

                  <div className="config-two">
                    <label className="config-field"><span>Trials</span><input type="number" min="1" value={trials} onChange={(event) => setTrials(event.target.value)} /></label>
                    <label className="config-field"><span>CV folds</span><input type="number" min="2" value={folds} onChange={(event) => setFolds(event.target.value)} /></label>
                  </div>

                  <label className="config-field">
                    <span>Metric</span>
                    <select value={metric} onChange={(event) => setMetric(event.target.value)}>
                      <option value="auto">Auto detect</option>
                      <option value="roc_auc">ROC-AUC</option>
                      <option value="f1_weighted">F1 weighted</option>
                      <option value="r2">R²</option>
                      <option value="silhouette_score">Silhouette</option>
                    </select>
                  </label>

                  <label className="toggle-row">
                    <span><strong>Tabular + text classification</strong><small>Combine structured columns with written text</small></span>
                    <input type="checkbox" checked={multimodal} onChange={(event) => setMultimodal(event.target.checked)} />
                  </label>

                  {multimodal && (
                    <div className="multimodal-options">
                      <label className="config-field"><span>Text representation</span><select value={representation} onChange={(event) => setRepresentation(event.target.value)}><option value="auto">Let AutoML choose</option><option value="tfidf">TF-IDF — fast, interpretable word patterns</option><option value="sentence_transformer">Sentence transformer — semantic meaning</option></select></label>
                      <label className="config-field"><span>Fusion strategy</span><select value={fusion} onChange={(event) => setFusion(event.target.value)}><option value="auto">Let AutoML choose</option><option value="early">Early fusion — combine features before one model</option><option value="late">Late fusion — train separate branches, then blend probabilities</option></select></label>
                      <p className="notice">Early fusion is usually simplest and works well with TF-IDF. Late fusion is useful when tabular and text signals should remain independent before a calibrated probability blend.</p>
                    </div>
                  )}

                  {task === 'time_series_forecasting' && (
                    <>
                      <div className="config-two">
                        <label className="config-field"><span>Datetime column</span><input value={forecastTimeColumn} onChange={(event) => setForecastTimeColumn(event.target.value)} placeholder="Optional — leave blank for row order" /></label>
                        <label className="config-field"><span>Horizon</span><input type="number" min="1" max="90" value={forecastHorizon} onChange={(event) => setForecastHorizon(event.target.value)} /></label>
                      </div>
                      <label className="config-field">
                        <span>Forecast model</span>
                        <select value={forecastModelFamily} onChange={(event) => setForecastModelFamily(event.target.value)}>
                          <option value="ridge">Ridge — fast, stable baseline</option>
                          <option value="lstm">LSTM — sequence learning</option>
                          <option value="bilstm">Bidirectional LSTM — learns both directions of each sequence</option>
                          <option value="bilstm_stack">Stacked Bidirectional LSTM — deeper sequence representation</option>
                        </select>
                      </label>
                      <p className="notice">The forecast uses chronological holdout data and compares against a last-value baseline. Neural models require the optional TensorFlow forecasting dependency.</p>
                    </>
                  )}

                  <button className="launch-button" disabled={!file} type="submit">
                    <span><Play size={17} fill="currentColor" /> Launch experiment</span>
                    <ArrowRight size={17} />
                  </button>
                </form>

                {(error || forecastError) && <div className="notice error">{forecastError || error}</div>}
              </aside>
            </div>

            {job && (
              <section className={`job-banner ${job.status}`}>
                <div className="job-icon"><Sparkles size={18} /></div>
                <div><strong>{job.status === 'complete' ? 'Experiment complete' : `Experiment ${job.status}`}</strong><small>{job.best_model ? `Champion model · ${job.best_model}` : `Job ${job.job_id?.slice(0, 8) || 'pending'}`}</small></div>
                {job.status === 'complete' && <button onClick={() => setView('analysis')}>View analysis <ArrowRight size={15} /></button>}
              </section>
            )}

            {job?.error && <div className="notice error">{job.error}</div>}
            {forecastResult && <ForecastResultPanel result={forecastResult} />}

            {multimodalResult && (
              <section className="job-banner complete multimodal-banner">
                <div className="job-icon"><BarChart3 size={18} /></div>
                <div><strong>{multimodalResult.selected_model}</strong><small>Tabular + text · CV score {Number(multimodalResult.cv_score).toFixed(3)} · {multimodalResult.representation === 'tfidf' ? 'TF-IDF text features' : 'Sentence transformer'} · {multimodalResult.fusion === 'early' ? 'Early fusion' : 'Late probability fusion'} </small></div>
                <label className="text-upload">Predict CSV<input type="file" accept=".csv" onChange={async (event) => {
                  try {
                    setPredictionError('')
                    const result = await uploadPredictionCsv(multimodalResult.job_id, event.target.files[0])
                    setPredictionOutput(result.data)
                  } catch (requestError) {
                    setPredictionError(requestError.response?.data?.detail || requestError.message)
                  }
                }} /></label>
                {predictionOutput && <a className="banner-action" href={predictionOutput.prediction_download}>Download {predictionOutput.rows} predictions</a>}
                {predictionError && <span className="notice error">{predictionError}</span>}
              </section>
            )}
          </>
        )}
      </main>
    </div>
  )
}

export default App
