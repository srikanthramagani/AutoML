import { useEffect, useState } from 'react'
import { startMultimodalTraining, uploadPredictionCsv, runForecast } from './services/api'
import { Activity, ArrowUpRight, BarChart3, Database, Download, FlaskConical, UploadCloud } from 'lucide-react'
import { useAutoML } from './hooks/useAutoML'
import DatasetProfile from './components/DatasetProfile'
import AnalysisPage from './pages/AnalysisPage'
import ExportPage from './pages/ExportPage'
import ForecastResultPanel from './components/ForecastResultPanel'

const steps = ['Dataset', 'Profile', 'Configure', 'Results']

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
      if (task === 'time_series_forecasting') setForecastError(requestError.response?.data?.detail || requestError.message)
    }
  }

  async function handleTrain(event) {
    event.preventDefault()
    try {
      if (task === 'time_series_forecasting') {
        const response = await runForecast({ dataset_id: dataset.id, target_column: target, datetime_column: forecastTimeColumn || null, horizon: Number(forecastHorizon), n_trials: Number(trials), cv_folds: Number(folds) })
        setForecastResult(response.data)
        setForecastError('')
        setActiveStep(3)
        return
      }
      if (multimodal) {
        const response = await startMultimodalTraining({dataset_id: dataset.id, target_column: target, representation, fusion, trials: Number(trials), cv_folds: Number(folds), overrides: columnOverrides})
        // This API is synchronous today; present the actual completed result.
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
        keep_identifier_columns: keptIdentifiers
      })
      setActiveStep(3)
    } catch {
      // The API contract is intentionally scaffold-only in this phase.
    }
  }

  useEffect(() => {
    if (job?.status === 'complete') {
      setActiveStep(3)
      setView('analysis')
    }
  }, [job?.status])

  return (
    <main className="shell">
      <nav className="topbar">
        <button className="brand" onClick={() => setView('run')}><span className="brand-mark"><FlaskConical size={17} /></span><span>adaptive / automl</span></button>
        <div className="topnav"><button className={view === 'run' ? 'active' : ''} onClick={() => setView('run')}>Run workspace</button><button className={view === 'analysis' ? 'active' : ''} disabled={job?.status !== 'complete'} onClick={() => setView('analysis')}><BarChart3 size={14} /> Analysis</button><button className={view === 'export' ? 'active' : ''} disabled={job?.status !== 'complete'} onClick={() => setView('export')}><Download size={14} /> Export</button></div>
        <div className="status"><span className="status-dot" /> phase 1 research workspace</div>
      </nav>

      {view === 'analysis' && job?.status === 'complete' && <AnalysisPage job={job} dataset={dataset} onBack={() => setView('run')} onExport={() => setView('export')} />}
      {view === 'export' && job?.status === 'complete' && <ExportPage job={job} dataset={dataset} onBack={() => setView('analysis')} />}

      {view === 'run' && <><section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Tabular intelligence laboratory</p>
          <h1>Turn a raw table into a <em>reproducible</em> model.</h1>
          <p className="hero-text">Upload a CSV, inspect its shape, and prepare a complete preprocessing and model-search run.</p>
        </div>
        <div className="hero-orbit"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><span>01</span></div>
      </section>

      <section className="workspace">
        <aside className="rail">
          <p className="rail-label">Workflow</p>
          {steps.map((step, index) => <button className={`rail-step ${activeStep === index ? 'active' : ''}`} key={step} onClick={() => setActiveStep(index)}><span>0{index + 1}</span>{step}<ArrowUpRight size={14} /></button>)}
          <div className="rail-note"><Database size={16} /><p>Metadata only<br /><span>Files stay in storage.</span></p></div>
        </aside>

        <section className="panel">
          <header className="panel-header"><div><p className="eyebrow">01 / dataset intake</p><h2>Start with your dataset</h2></div><Activity size={20} className="muted-icon" /></header>
          <label className="dropzone"><input type="file" accept=".csv,text/csv" onChange={handleUpload} /><UploadCloud size={27} /><strong>{file ? file.name : 'Drop a CSV here or browse'}</strong><span>{file ? `${(file.size / 1024).toFixed(1)} KB ready for inspection` : 'CSV files only · maximum 50 MB'}</span></label>

          {dataset && <DatasetProfile dataset={dataset} keptIdentifiers={keptIdentifiers} onKeepIdentifier={(columnName, keep) => setKeptIdentifiers((current) => keep ? [...new Set([...current, columnName])] : current.filter((name) => name !== columnName))} />}
          {dataset?.modality_report && <div className="modality-summary"><strong>Detected modalities</strong><div>{Object.entries(dataset.modality_report).filter(([key, value]) => Array.isArray(value) && key !== 'high_cardinality').map(([kind, columns]) => <p key={kind}><span>{kind.replace('_', ' ')}</span> {columns.length ? columns.join(', ') : 'None detected'}</p>)}</div><small>Review the text and identifier assignments before multimodal training. Overrides are available through the API.</small></div>}
          {dataset?.column_details && <details className="modality-summary"><summary>Manually reassign columns</summary>{dataset.column_details.filter((column) => column.name !== target).map((column) => <label className="override-row" key={column.name}><span>{column.name} · {column.inferred_type}</span><select value={columnOverrides[column.name] || ''} onChange={(event) => setColumnOverrides((state) => ({...state,[column.name]:event.target.value}))}><option value="">Automatic</option>{['numerical','categorical','boolean','datetime','text','identifiers','exclude'].map((item) => <option key={item} value={item}>{item}</option>)}</select></label>)}</details>}

          <div className="section-rule" />
          <form onSubmit={handleTrain} className="config-grid">
            <div className="field wide"><label htmlFor="target">Target column</label><input id="target" value={target} onChange={(event) => setTarget(event.target.value)} placeholder="e.g. churn, price" /><small>Leave blank to let the loader use the final column.</small></div>
            <div className="field wide"><label className="inline-choice"><input type="checkbox" checked={multimodal} onChange={(e) => setMultimodal(e.target.checked)} /> Train tabular + text model</label><small>Uses detected text fields, validation folds, Optuna model selection, and exports the full pipeline.</small></div>
            {multimodal && <><div className="field"><label htmlFor="representation">Text representation</label><select id="representation" value={representation} onChange={(e) => setRepresentation(e.target.value)}><option value="auto">Automatic selection</option><option value="tfidf">TF-IDF</option><option value="sentence_transformer">Sentence Transformer (optional package)</option></select></div><div className="field"><label htmlFor="fusion">Fusion strategy</label><select id="fusion" value={fusion} onChange={(e) => setFusion(e.target.value)}><option value="auto">Automatic selection</option><option value="early">Early fusion</option><option value="late">Late fusion</option></select></div></>}
            <div className="field"><label htmlFor="folds">CV folds</label><input id="folds" type="number" min="2" value={folds} onChange={(event) => setFolds(event.target.value)} /></div>
            <div className="field"><label htmlFor="trials">Optimization trials</label><input id="trials" type="number" min="1" value={trials} onChange={(event) => setTrials(event.target.value)} /></div>
            <div className="field"><label htmlFor="metric">Evaluation metric</label><select id="metric" value={metric} onChange={(event) => setMetric(event.target.value)}><option value="auto">Auto detect</option><option value="roc_auc">ROC-AUC</option><option value="f1_weighted">F1 weighted</option><option value="r2">R²</option><option value="silhouette_score">Silhouette Score (Clustering)</option></select></div>
            <div className="field"><label>Task family</label><div className="segmented"><button type="button" className={task === 'auto' ? 'selected' : ''} onClick={() => setTask('auto')}>Auto</button><button type="button" className={task === 'classification' ? 'selected' : ''} onClick={() => setTask('classification')}>Classify</button><button type="button" className={task === 'regression' ? 'selected' : ''} onClick={() => setTask('regression')}>Regress</button><button type="button" className={task === 'clustering' ? 'selected' : ''} onClick={() => setTask('clustering')}>Cluster</button><button type="button" className={task === 'time_series_forecasting' ? 'selected' : ''} onClick={() => setTask('time_series_forecasting')}>Forecast</button></div></div>
            {task === 'time_series_forecasting' && <><div className="field"><label htmlFor="time-column">Datetime column (optional)</label><input id="time-column" value={forecastTimeColumn} onChange={(event) => setForecastTimeColumn(event.target.value)} placeholder="e.g. timestamp" /></div><div className="field"><label htmlFor="forecast-horizon">Forecast horizon</label><input id="forecast-horizon" type="number" min="1" max="90" value={forecastHorizon} onChange={(event) => setForecastHorizon(event.target.value)} /></div></>}
            <button className="primary-action" disabled={!file} type="submit">Prepare training run <ArrowUpRight size={16} /></button>
          </form>
          {error && <div className="notice">{error}</div>}
          {job && <div className={`notice ${job.status === 'complete' ? 'success' : ''}`}>Run status: {job.status}{job.best_model ? ` · Best model: ${job.best_model}` : ''}</div>}
          {multimodalResult && <div className="completion-card"><BarChart3 size={17} /><div><strong>Multimodal validation complete · {multimodalResult.selected_model}</strong><p>CV macro F1 {Number(multimodalResult.cv_score).toFixed(3)} · holdout accuracy {Number(multimodalResult.metrics?.accuracy).toFixed(3)} · {multimodalResult.representation} / {multimodalResult.fusion} fusion.</p><label className="upload-predictions">Predict unseen CSV<input type="file" accept=".csv" onChange={async (event) => {try {setPredictionError(''); const result=await uploadPredictionCsv(multimodalResult.job_id,event.target.files[0]); setPredictionOutput(result.data)} catch(err) {setPredictionError(err.response?.data?.detail || err.message)}}} /></label>{predictionOutput && <><span>{predictionOutput.rows} predictions ready.</span><a href={predictionOutput.prediction_download}>Download predictions CSV</a></>}{predictionError && <span className="notice">{predictionError}</span>}<a className="primary-action compact" href={`/api/models/${multimodalResult.job_id}/download`}>Download pipeline</a></div></div>}
          {job?.error && <div className="notice">{job.error}</div>}
          {forecastError && <div className="notice">{forecastError}</div>}
          {forecastResult && <ForecastResultPanel result={forecastResult} />}
          {job?.status === 'complete' && <div className="completion-card"><BarChart3 size={17} /><div><strong>Analysis is ready</strong><p>{job.best_model} was selected. Open the Analysis page for the full comparison and Export for the trained pipeline.</p></div><button className="primary-action compact" onClick={() => setView('analysis')}>Open analysis <ArrowUpRight size={15} /></button></div>}
        </section>
      </section>
      </>}

      <footer><span>Adaptive AutoML Platform</span><span>Preprocessing → candidates → validation → export</span><span>v0.1 / scaffold</span></footer>
    </main>
  )
}

export default App
