import { Activity, BarChart3, CheckCircle2, Clock3, Gauge, Info, ShieldCheck, Sparkles } from 'lucide-react'

const metricDefinitions = {
  accuracy: 'Share of predictions that are strictly correct (0 to 1; higher is better).',
  balanced_accuracy: 'Average recall across classes; accounts for dataset imbalance.',
  f1_weighted: 'Precision and recall harmonic mean, weighted by class support.',
  f1_macro: 'F1 score averaged equally across all classes.',
  precision_weighted: 'When the model predicts a class, how often it is correct (weighted).',
  precision_macro: 'Unweighted macro average precision across all classes.',
  recall_weighted: 'Share of ground truth examples correctly identified, weighted by class.',
  recall_macro: 'Unweighted macro average recall across all classes.',
  mcc: 'Matthews Correlation Coefficient (-1 to +1; 1 is perfect, 0 is chance).',
  roc_auc: 'Area Under ROC Curve; discriminative ability across thresholds.',
  log_loss: 'Cross-entropy log loss measuring probabilistic certainty; lower is better.',
  r2: 'Variance explained by a regression model; 1.0 is a perfect fit.',
  adjusted_r2: 'R² adjusted for feature count to penalize redundant predictors.',
  explained_variance: 'Proportion of dataset variance captured by predictions.',
  mae: 'Average absolute error; lower is better.',
  mse: 'Average squared error; lower is better.',
  rmse: 'Root Mean Squared Error in original target units; lower is better.',
  mape: 'Mean Absolute Percentage Error (%); lower is better.',
  max_error: 'Maximum single residual error across test samples.',
  median_absolute_error: 'Median residual error; robust against extreme outliers.',
  silhouette_score: 'Cluster separation and cohesion (-1 to +1; higher is better).',
  calinski_harabasz_score: 'Variance ratio between clusters vs within clusters; higher is better.',
  davies_bouldin_score: 'Average similarity between clusters; lower is better.',
}

function formatScore(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : 'n/a'
}

function formatSeconds(value) {
  const seconds = Number(value)
  return Number.isFinite(seconds) ? `${seconds.toFixed(2)}s` : 'n/a'
}

function buildReason(winner, records, scoring) {
  const winnerRecord = records.find((record) => record.model_name === winner)
  const runnerUp = records.find((record) => record.model_name !== winner)
  if (!winnerRecord) return 'This model was selected as the top valid candidate by cross-validation.'
  const margin = runnerUp ? Number(winnerRecord.cv_mean_score) - Number(runnerUp.cv_mean_score) : 0
  const stability = Number(winnerRecord.cv_std_score)
  const details = [`highest ${scoring || 'validation'} score (${formatScore(winnerRecord.cv_mean_score)})`]
  if (Number.isFinite(stability)) details.push(`stable fold variation of ${formatScore(stability)}`)
  if (margin > 0) details.push(`a ${formatScore(margin)} lead over the next candidate`)
  return `Chosen because it delivered the ${details.join(', ')}. The final pipeline was retrained on the full training partition before holdout validation.`
}

export default function TrainingResults({ job }) {
  const leaderboard = job.leaderboard || []
  const records = job.model_records || []
  const candidates = records.length ? records : leaderboard.map((model) => ({ ...model, cv_mean_score: model.mean_score, cv_std_score: model.std_score, is_valid: true }))
  const maxScore = Math.max(...leaderboard.map((model) => Number(model.mean_score) || 0), 1)
  const maxTime = Math.max(...candidates.map((record) => Number(record.training_time_seconds) || 0), 1)
  const metrics = Object.entries(job.metrics || {})
  const maxMetric = Math.max(...metrics.map(([, value]) => Math.abs(Number(value)) || 0), 1)
  const winner = leaderboard.find((model) => model.model_name === job.best_model) || leaderboard[0]
  const scoring = winner?.scoring || records[0]?.scoring || 'primary metric'
  const genAudit = job.generalization_audit

  return <section className="results" aria-label="Training results">
    <div className="results-heading">
      <div>
        <p className="eyebrow">04 / model analysis</p>
        <h3>Every candidate, measured</h3>
        <p className="results-subtitle">{job.task_type?.replaceAll('_', ' ') || 'task'} · ranked by {job.primary_metric || scoring} · {job.cv_folds || winner?.n_folds || '-'} folds used</p>
      </div>
      <BarChart3 size={20} className="muted-icon" />
    </div>

    {winner && <div className="winner-note"><CheckCircle2 size={19} /><div><strong>{winner.model_name} is the recommended model</strong><p>{buildReason(winner.model_name, records, scoring)}</p></div></div>}

    {/* Generalization & Fit Diagnostics */}
    {genAudit && (
      <div className={`generalization-card ${genAudit.status}`}>
        <div className="gen-header">
          <div>
            <span className="card-kicker">Bias-Variance & Generalization Diagnosis</span>
            <div className="gen-title-row">
              <h4>{genAudit.model_name}: {
                genAudit.status === 'well_fitted' ? 'Well-Fitted (Optimal Bias-Variance Balance)' :
                genAudit.status === 'overfitting' ? 'Overfitting Risk (High Variance Detected)' :
                'Underfitting Risk (High Bias Detected)'
              }</h4>
              <span className={`status-pill ${genAudit.status}`}>
                {genAudit.status.replace('_', ' ')} · {genAudit.risk_level} risk
              </span>
            </div>
          </div>
        </div>

        <div className="gen-metrics-grid">
          <div className="gen-metric-box">
            <span>Train Score ({genAudit.metric_name})</span>
            <strong>{formatScore(genAudit.train_score)}</strong>
            <small>Fitted training partition</small>
          </div>
          <div className="gen-metric-box">
            <span>Validation CV Score</span>
            <strong>{formatScore(genAudit.validation_score)}</strong>
            <small>Cross-validation average</small>
          </div>
          {genAudit.holdout_score != null && (
            <div className="gen-metric-box">
              <span>Holdout Test Score</span>
              <strong>{formatScore(genAudit.holdout_score)}</strong>
              <small>Unseen test partition</small>
            </div>
          )}
          <div className="gen-metric-box highlight">
            <span>Generalization Gap</span>
            <strong>{formatScore(genAudit.generalization_gap)}</strong>
            <small>{genAudit.gap_percentage}% divergence</small>
          </div>
        </div>

        <p className="gen-explanation">{genAudit.explanation}</p>

        {genAudit.recommendations?.length > 0 && (
          <div className="gen-recommendations">
            <strong>Recommended Mitigations:</strong>
            <ul>
              {genAudit.recommendations.map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    )}

    <div className="analysis-grid">
      <div className="analysis-block">
        <div className="analysis-title"><Gauge size={15} /><span>Cross-validation score</span></div>
        {leaderboard.map((model) => (
          <div className="bar-row" key={model.model_name}>
            <div className="bar-label">
              <span>{model.rank}. {model.model_name}</span>
              <strong>{formatScore(model.mean_score)}</strong>
            </div>
            <div className="bar-track">
              <i style={{ width: `${Math.max(2, (Number(model.mean_score) / maxScore) * 100)}%` }} />
            </div>
            <small>{model.scoring} · +/- {formatScore(model.std_score)} std dev · {model.n_folds || records.find((record) => record.model_name === model.model_name)?.cv_folds || '-'} folds</small>
          </div>
        ))}
      </div>
      <div className="analysis-block">
        <div className="analysis-title"><Clock3 size={15} /><span>Training time by candidate</span></div>
        {candidates.map((record) => (
          <div className="bar-row" key={record.model_name}>
            <div className="bar-label">
              <span>{record.model_name}</span>
              <strong>{formatSeconds(record.training_time_seconds)}</strong>
            </div>
            <div className="bar-track time">
              <i style={{ width: `${Math.max(2, (Number(record.training_time_seconds) / maxTime) * 100)}%` }} />
            </div>
            <small>{record.is_valid === false ? `Failed: ${record.error || 'candidate could not be evaluated'}` : `HPO: ${formatSeconds(record.optimization_time_seconds)} · ${record.optimization_trials || 0} trials`}</small>
          </div>
        ))}
      </div>
    </div>

    <div className="metric-analysis rounded-lg border border-[#c5d0c6] bg-white/25 p-4 shadow-sm">
      <div className="analysis-title"><BarChart3 size={15} /><span>Holdout evaluation metrics</span></div>
      <p className="mb-4 text-xs leading-5 text-[#6a7e73]">
        These scores are computed on unseen data partition not used to fit the final pipeline. Hover-free definitions are shown beside every bar; error metrics are marked lower-is-better.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {metrics.map(([name, value]) => (
          <div className="metric-bar" key={name}>
            <div className="bar-label">
              <span>{name.replaceAll('_', ' ')}</span>
              <strong>{formatScore(value)}</strong>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-[#d8e0d8]">
              <i className="block h-full rounded-full bg-[#b45b35]" style={{ width: `${Math.max(3, (Math.abs(Number(value)) / maxMetric) * 100)}%` }} />
            </div>
            <small>{metricDefinitions[name] || 'Evaluation metric returned by the training engine.'}</small>
          </div>
        ))}
      </div>
    </div>

    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Algorithm</th>
            <th>CV Mean</th>
            <th>Train Score</th>
            <th>Generalization Gap</th>
            <th>Fit Status</th>
            <th>Variation</th>
            <th>HPO Score</th>
            <th>Fit Time</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((record) => {
            const model = leaderboard.find((item) => item.model_name === record.model_name)
            const fitStatus = record.fit_status || 'well_fitted'
            return (
              <tr className={record.model_name === job.best_model ? 'winner-row' : ''} key={record.model_name}>
                <td>{model ? `#${model.rank}` : '-'}</td>
                <td><strong>{record.model_name}</strong></td>
                <td>{formatScore(record.cv_mean_score ?? model?.mean_score)}</td>
                <td>{record.train_score != null ? formatScore(record.train_score) : '-'}</td>
                <td>{record.generalization_gap != null ? formatScore(record.generalization_gap) : '-'}</td>
                <td>
                  <span className={`fit-pill ${fitStatus}`}>
                    {fitStatus.replace('_', ' ')}
                  </span>
                </td>
                <td>{formatScore(record.cv_std_score ?? model?.std_score)}</td>
                <td>{formatScore(record.optimization_score)}</td>
                <td>{formatSeconds(record.training_time_seconds)}</td>
                <td>
                  {record.model_name === job.best_model ? (
                    <span className="selected-label">Selected</span>
                  ) : record.is_valid === false ? (
                    <span className="failed-label">Failed</span>
                  ) : (
                    'Evaluated'
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>

    <div className="metric-key"><Info size={14} /><span>CV mean ranks algorithms. Holdout metrics measure the selected pipeline on unseen rows. A large bar is not automatically better for MAE, MSE, RMSE, or MAPE.</span></div>
    {job.excluded_identifier_columns?.length > 0 && <div className="metric-key"><Info size={14} /><span>Excluded before train/test split: {job.excluded_identifier_columns.join(', ')}. {job.excluded_identifier_columns.map((name) => job.identifier_reasons?.[name]).filter(Boolean).join(' · ')}</span></div>}
    {job.fairness_audit && <div className="audit-line"><ShieldCheck size={15} /><span>Fairness audit: {job.fairness_audit.status === 'not_run' ? 'not run for this configuration' : job.fairness_audit.status}</span></div>}
  </section>
}
