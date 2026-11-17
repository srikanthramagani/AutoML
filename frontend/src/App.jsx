import { useEffect, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, Database, Download, FlaskConical, UploadCloud } from 'lucide-react'
import { useAutoML } from './hooks/useAutoML'
import DatasetProfile from './components/DatasetProfile'
import AnalysisPage from './pages/AnalysisPage'
import ExportPage from './pages/ExportPage'

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
    } catch {
      // The API contract is intentionally scaffold-only in this phase.
    }
  }

  async function handleTrain(event) {
    event.preventDefault()
    try {
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

          <div className="section-rule" />
          <form onSubmit={handleTrain} className="config-grid">
            <div className="field wide"><label htmlFor="target">Target column</label><input id="target" value={target} onChange={(event) => setTarget(event.target.value)} placeholder="e.g. churn, price" /><small>Leave blank to let the loader use the final column.</small></div>
            <div className="field"><label htmlFor="folds">CV folds</label><input id="folds" type="number" min="2" value={folds} onChange={(event) => setFolds(event.target.value)} /></div>
            <div className="field"><label htmlFor="trials">Optimization trials</label><input id="trials" type="number" min="1" value={trials} onChange={(event) => setTrials(event.target.value)} /></div>
            <div className="field"><label htmlFor="metric">Evaluation metric</label><select id="metric" value={metric} onChange={(event) => setMetric(event.target.value)}><option value="auto">Auto detect</option><option value="roc_auc">ROC-AUC</option><option value="f1_weighted">F1 weighted</option><option value="r2">R²</option><option value="silhouette_score">Silhouette Score (Clustering)</option></select></div>
            <div className="field"><label>Task family</label><div className="segmented"><button type="button" className={task === 'auto' ? 'selected' : ''} onClick={() => setTask('auto')}>Auto</button><button type="button" className={task === 'classification' ? 'selected' : ''} onClick={() => setTask('classification')}>Classify</button><button type="button" className={task === 'regression' ? 'selected' : ''} onClick={() => setTask('regression')}>Regress</button><button type="button" className={task === 'clustering' ? 'selected' : ''} onClick={() => setTask('clustering')}>Cluster</button></div></div>
            <button className="primary-action" disabled={!file} type="submit">Prepare training run <ArrowUpRight size={16} /></button>
          </form>
          {error && <div className="notice">{error}</div>}
          {job && <div className={`notice ${job.status === 'complete' ? 'success' : ''}`}>Run status: {job.status}{job.best_model ? ` · Best model: ${job.best_model}` : ''}</div>}
          {job?.error && <div className="notice">{job.error}</div>}
          {job?.status === 'complete' && <div className="completion-card"><BarChart3 size={17} /><div><strong>Analysis is ready</strong><p>{job.best_model} was selected. Open the Analysis page for the full comparison and Export for the trained pipeline.</p></div><button className="primary-action compact" onClick={() => setView('analysis')}>Open analysis <ArrowUpRight size={15} /></button></div>}
        </section>
      </section>
      </>}

      <footer><span>Adaptive AutoML Platform</span><span>Preprocessing → candidates → validation → export</span><span>v0.1 / scaffold</span></footer>
    </main>
  )
}

export default App
