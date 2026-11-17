import { ArrowLeft, Download, ExternalLink, LineChart, Sparkles } from 'lucide-react'
import TrainingResults from '../components/TrainingResults'
import TrajectoryChart from '../components/TrajectoryChart'
import ResidualChart from '../components/ResidualChart'
import { getLeaderboardChartUrl } from '../services/api'

export default function AnalysisPage({ job, dataset, onBack, onExport }) {
  if (!job) return null
  return <main className="product-page">
    <div className="page-toolbar"><button className="text-action" onClick={onBack}><ArrowLeft size={16} /> Back to run</button><button className="primary-action compact" onClick={onExport}>Export model <Download size={15} /></button></div>
    <header className="page-title"><div><p className="eyebrow">Model intelligence / analysis</p><h1>Decision report</h1><p>Evidence for the selected model, from dataset shape to holdout performance.</p></div><div className="run-badge"><Sparkles size={16} /> Completed run</div></header>
    <section className="analysis-hero-grid"><div className="decision-card"><p className="card-kicker">Recommended model</p><h2>{job.best_model}</h2><p>Selected by {job.primary_metric || 'cross-validation'} for a {job.task_type?.replaceAll('_', ' ') || 'tabular'} task. The final pipeline includes preprocessing and tuned estimator settings.</p><button className="text-action" onClick={onExport}>View reproducibility package <ExternalLink size={14} /></button></div><div className="analysis-facts"><div><span>Dataset</span><strong>{dataset?.filename || 'Uploaded CSV'}</strong></div><div><span>Features used</span><strong>{job.features_used?.length ?? '-'}</strong></div><div><span>Primary metric</span><strong>{job.primary_metric || 'Auto'}</strong></div><div><span>Validation</span><strong>{job.cv_folds || '-'} folds + holdout</strong></div><div><span>Generalization</span><strong>{job.generalization_audit ? job.generalization_audit.status.replace('_', ' ').toUpperCase() : 'OPTIMAL'}</strong></div><div><span>Accuracy / Score</span><strong>{job.metrics?.accuracy != null ? `${(job.metrics.accuracy * 100).toFixed(1)}% Acc` : job.metrics?.r2 != null ? `${job.metrics.r2.toFixed(3)} R²` : job.metrics?.rmse != null ? `${job.metrics.rmse.toFixed(3)} RMSE` : 'Evaluated'}</strong></div></div></section>
    <section className="feature-audit"><p className="card-kicker">Feature audit</p><h2>Features used for modeling</h2><div className="feature-chips">{(job.features_used || []).map((feature) => <span key={feature}>{feature}</span>)}</div></section>
    
    {job.prediction_series && job.prediction_series.length > 0 && (
      <section className="chart-card">
        <TrajectoryChart data={job.prediction_series} />
        <ResidualChart data={job.prediction_series} />
      </section>
    )}

    {job.leaderboard_chart && <section className="chart-card"><div className="section-heading"><div><p className="card-kicker">Statistical comparison</p><h2>Leaderboard distribution</h2></div><LineChart size={19} /></div><img src={getLeaderboardChartUrl(job.job_id)} alt="Seaborn cross-validation leaderboard chart" /></section>}
    <TrainingResults job={job} />
  </main>
}
