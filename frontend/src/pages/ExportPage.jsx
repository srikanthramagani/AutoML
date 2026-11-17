import { ArrowLeft, CheckCircle2, Download, FileKey2, ShieldCheck } from 'lucide-react'
import ModelDownload from '../components/ModelDownload'

export default function ExportPage({ job, dataset, onBack }) {
  if (!job) return null
  const params = Object.entries(job.best_params || {})
  return <main className="product-page export-page">
    <div className="page-toolbar"><button className="text-action" onClick={onBack}><ArrowLeft size={16} /> Back to analysis</button></div>
    <header className="page-title"><div><p className="eyebrow">Model intelligence / export</p><h1>Reproducibility package</h1><p>Download the complete trained pipeline and keep the decisions behind it.</p></div><div className="run-badge"><CheckCircle2 size={16} /> Ready to download</div></header>
    <section className="export-layout"><div><ModelDownload job={job} /><div className="parameter-card"><div className="section-heading"><div><p className="card-kicker">Tuned configuration</p><h2>Why this model is ready</h2></div><FileKey2 size={19} /></div><p className="muted-copy">Optuna searched the configured space for {job.optimization_trials || '-'} trials and retained the parameters below for {job.best_model}.</p>{params.length ? <dl className="parameter-list">{params.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{typeof value === 'object' ? JSON.stringify(value) : String(value)}</dd></div>)}</dl> : <p className="empty-state">The winning estimator used its registered defaults.</p>}</div></div><aside className="package-card"><p className="card-kicker">Artifact contents</p><h2>One portable pipeline</h2><ul><li><CheckCircle2 size={15} /> Imputation and feature handling</li><li><CheckCircle2 size={15} /> Encoding and scaling</li><li><CheckCircle2 size={15} /> Tuned {job.best_model} estimator</li><li><CheckCircle2 size={15} /> Holdout metrics and metadata</li></ul><div className="compatibility"><ShieldCheck size={16} /><span>Joblib / scikit-learn pipeline<br /><small>Use the same compatible Python environment to load `.pkl` files.</small></span></div><div className="package-dataset"><span>Trained from</span><strong>{dataset?.filename || 'uploaded dataset'}</strong></div></aside></section>
  </main>
}
