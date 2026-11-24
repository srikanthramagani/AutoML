import { useState } from 'react'
import {
  Activity,
  Award,
  BarChart3,
  CheckCircle2,
  Clock3,
  Compass,
  Cpu,
  ExternalLink,
  Flame,
  Gauge,
  HelpCircle,
  Info,
  Layers,
  LineChart,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Sliders,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { getLeaderboardChartUrl, getOptimizerComparisonChartUrl, getAllModelsChartUrl } from '../services/api'

const metricDefinitions = {
  accuracy: 'Share of predictions that are strictly correct (0 to 1; higher is better).',
  balanced_accuracy: 'Average recall across classes; accounts for dataset imbalance.',
  f1_weighted: 'Precision and recall harmonic mean, weighted by class support.',
  f1_macro: 'F1 score averaged equally across all classes; robust to class imbalance.',
  f1: 'Standard F1 score; balance of precision and recall.',
  precision_weighted: 'Weighted precision across all classes.',
  precision_macro: 'Unweighted macro average precision across all classes.',
  precision: 'Positive predictive value: fraction of positive calls that were correct.',
  recall_weighted: 'Weighted recall across all classes.',
  recall_macro: 'Unweighted macro average recall across all classes.',
  recall: 'Sensitivity: fraction of actual positive instances successfully detected.',
  mcc: 'Matthews Correlation Coefficient (-1 to +1; 1 is perfect prediction).',
  roc_auc: 'Area Under ROC Curve; discrimination ability across all thresholds.',
  log_loss: 'Cross-entropy loss measuring probabilistic certainty; lower is better.',
  r2: 'Coefficient of determination (variance explained); 1.0 is a perfect fit.',
  adjusted_r2: 'R² adjusted for feature count to penalize redundant predictors.',
  explained_variance: 'Proportion of dataset variance captured by predictions.',
  mae: 'Mean Absolute Error in original target units; lower is better.',
  mse: 'Mean Squared Error; penalizes large errors heavily; lower is better.',
  rmse: 'Root Mean Squared Error in original target units; lower is better.',
  mape: 'Mean Absolute Percentage Error (%); lower is better.',
  smape: 'Symmetric Mean Absolute Percentage Error (0% to 200%); lower is better.',
  smae: 'Scaled Mean Absolute Error (MAE normalized by actual mean); lower is better.',
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

export default function TrainingResults({ job }) {
  const [activeTab, setActiveTab] = useState('leaderboard')
  const leaderboard = job.leaderboard || []
  const records = job.model_records || []
  const candidates = records.length
    ? records
    : leaderboard.map((model) => ({
        ...model,
        cv_mean_score: model.mean_score,
        cv_std_score: model.std_score,
        is_valid: true,
      }))

  const winnerName = job.best_model || leaderboard[0]?.model_name || ''
  const [selectedModel, setSelectedModel] = useState(winnerName)

  const winner = leaderboard.find((model) => model.model_name === winnerName) || leaderboard[0]
  const selectedRecord = records.find((r) => r.model_name === selectedModel) || records[0] || {}
  const selectedLeaderboardItem = leaderboard.find((item) => item.model_name === selectedModel)

  const isMetricMinimize = ['rmse', 'mse', 'mae', 'smape', 'smae', 'log_loss', 'davies_bouldin_score'].includes(
    (job.primary_metric || '').toLowerCase().replace('neg_', '')
  )

  const maxScore = Math.max(...leaderboard.map((model) => Math.abs(Number(model.mean_score)) || 0), 1)
  const maxTime = Math.max(...candidates.map((record) => Number(record.training_time_seconds) || 0), 1)
  const metrics = Object.entries(job.metrics || {})
  const maxMetric = Math.max(...metrics.map(([, value]) => Math.abs(Number(value)) || 0), 1)

  const genAudit = job.generalization_audit
  const explainability = job.explainability
  const importance = explainability?.global_feature_importance || []

  const confusionCells = Object.entries(job.metrics || {}).filter(([key]) => /^confusion_matrix_\d+_\d+$/.test(key))
  const confusionSize = confusionCells.length ? Math.sqrt(confusionCells.length) : 0
  const confusion = Array.from({ length: confusionSize }, (_, row) =>
    Array.from({ length: confusionSize }, (_, col) => Number(job.metrics[`confusion_matrix_${row}_${col}`] || 0))
  )

  const selectedParams = selectedRecord.best_params || job.best_params || {}
  const selectedHistory = selectedRecord.optimization_history || []
  const selectedImportances = Object.entries(selectedRecord.parameter_importance || {}).sort((a, b) => b[1] - a[1])
  const validTrials = selectedHistory.filter((t) => t.state === 'COMPLETE' && Number.isFinite(Number(t.value)))
  const trialValues = validTrials.map((t) => Number(t.value))
  const lowVal = Math.min(...trialValues, 0)
  const highVal = Math.max(...trialValues, 1)

  return (
    <section className="results-container mt-8" aria-label="Automated Model Evaluation and Comparison">
      {/* ── Top Header & Tab Navigation ── */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-md bg-teal-50 px-2.5 py-1 text-xs font-semibold text-teal-800">
                <Sparkles size={13} className="text-teal-600" />
                Validated Execution Report
              </span>
              <span className="text-xs text-slate-400 font-mono">ID: {job.job_id?.slice(0, 8)}</span>
            </div>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">
              Model Intelligence & Performance Leaderboard
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Task:{' '}
              <span className="font-semibold text-slate-700 capitalize">
                {job.task_type?.replaceAll('_', ' ') || 'Supervised'}
              </span>{' '}
              · Ranked by:{' '}
              <span className="font-semibold text-indigo-600 uppercase font-mono text-xs">
                {job.primary_metric || 'Score'}
              </span>{' '}
              ({isMetricMinimize ? 'Lower is better' : 'Higher is better'}) ·{' '}
              <span className="font-semibold text-slate-700">{job.cv_folds || 5}-Fold Cross Validation</span>
            </p>
          </div>

          {/* Model Selection Callout Pill */}
          <div className="flex items-center gap-3 rounded-xl border border-teal-200 bg-gradient-to-r from-teal-50/80 to-emerald-50/80 p-3.5 shadow-sm">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-teal-600 text-white shadow-sm">
              <Award size={20} />
            </div>
            <div>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-teal-700 font-mono">
                {selectedModel === winnerName ? 'Recommended Winner' : 'Selected Candidate'}
              </span>
              <p className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                {selectedModel}
                {selectedModel === winnerName && (
                  <span className="inline-block rounded bg-teal-600 px-1.5 py-0.5 text-[10px] font-bold text-white uppercase">
                    Rank #1
                  </span>
                )}
              </p>
              <span className="text-xs text-slate-600 font-mono">
                CV Score: <strong>{formatScore(selectedRecord.cv_mean_score ?? selectedLeaderboardItem?.mean_score)}</strong> ± {formatScore(selectedRecord.cv_std_score ?? selectedLeaderboardItem?.std_score)}
              </span>
            </div>
          </div>
        </div>

        {/* Tab Controls */}
        <div className="mt-4 flex flex-wrap gap-2 pt-2">
          <button
            onClick={() => setActiveTab('leaderboard')}
            className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition ${
              activeTab === 'leaderboard'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <BarChart3 size={15} /> Leaderboard & Comparisons
          </button>
          <button
            onClick={() => setActiveTab('diagnostics')}
            className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition ${
              activeTab === 'diagnostics'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <Gauge size={15} /> Selected Model Diagnostics & HPO
          </button>
          <button
            onClick={() => setActiveTab('optimizers')}
            className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition ${
              activeTab === 'optimizers'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <Flame size={15} /> Optuna vs. Grid & Random Search
          </button>
          <button
            onClick={() => setActiveTab('holdout')}
            className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition ${
              activeTab === 'holdout'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <CheckCircle2 size={15} /> Holdout Metrics & Explainability
          </button>
        </div>
      </div>

      {/* ── TAB 1: LEADERBOARD & COMPARISONS ── */}
      {activeTab === 'leaderboard' && (
        <div className="mt-6 space-y-6">
          {/* Winner Reasoning Banner */}
          {winner && (
            <div className="flex items-start gap-4 rounded-xl border border-teal-200 bg-teal-50/60 p-4 shadow-sm">
              <CheckCircle2 size={24} className="mt-0.5 text-teal-600 shrink-0" />
              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h4 className="text-sm font-bold text-slate-900">
                    Winner Selected: {winner.model_name}
                  </h4>
                  <span className="rounded bg-teal-100 px-2 py-0.5 text-[11px] font-semibold text-teal-800">
                    Recommended Production Candidate
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                  Delivered the highest cross-validation score of{' '}
                  <strong className="font-mono text-slate-800">{formatScore(winner.mean_score)}</strong>{' '}
                  with reliable fold stability (±{formatScore(winner.std_score)}). The model was refit on the entire training partition before holdout evaluation.
                </p>
              </div>
            </div>
          )}

          {/* Side-by-Side Horizontal Bar Comparison Charts */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Chart 1: Cross-Validation Score Bars */}
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div className="flex items-center gap-2">
                  <BarChart3 size={17} className="text-indigo-600" />
                  <h3 className="text-sm font-bold text-slate-900">
                    Cross-Validation Performance by Model
                  </h3>
                </div>
                <span className="text-[11px] font-mono text-slate-500">
                  {job.primary_metric || 'score'}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Click any bar to select the model for detailed inspection and hyperparameter analysis:
              </p>

              <div className="mt-4 space-y-3">
                {leaderboard.map((model) => {
                  const isWinner = model.model_name === winnerName
                  const isSelected = model.model_name === selectedModel
                  const barPercent = Math.max(
                    4,
                    Math.min(100, (Math.abs(Number(model.mean_score)) / (maxScore || 1)) * 100)
                  )
                  return (
                    <div
                      key={model.model_name}
                      onClick={() => setSelectedModel(model.model_name)}
                      className={`cursor-pointer rounded-lg p-2.5 transition border ${
                        isSelected
                          ? 'border-indigo-400 bg-indigo-50/50 shadow-sm'
                          : 'border-transparent hover:bg-slate-50'
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-800 flex items-center gap-1.5">
                          <span className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-slate-200 text-[10px] font-bold text-slate-700">
                            {model.rank}
                          </span>
                          {model.model_name}
                          {isWinner && (
                            <span className="rounded bg-teal-600 px-1.5 py-0.2 text-[9px] font-bold text-white uppercase">
                              Winner
                            </span>
                          )}
                        </span>
                        <span className="font-mono font-bold text-slate-900">
                          {formatScore(model.mean_score)}{' '}
                          <span className="text-[10px] text-slate-400">±{formatScore(model.std_score)}</span>
                        </span>
                      </div>
                      <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            isWinner
                              ? 'bg-gradient-to-r from-teal-500 to-emerald-500'
                              : isSelected
                              ? 'bg-gradient-to-r from-indigo-500 to-blue-500'
                              : 'bg-slate-400'
                          }`}
                          style={{ width: `${barPercent}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Chart 2: Training & HPO Duration Bars */}
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div className="flex items-center gap-2">
                  <Clock3 size={17} className="text-slate-600" />
                  <h3 className="text-sm font-bold text-slate-900">Training & Tuning Latency</h3>
                </div>
                <span className="text-[11px] font-mono text-slate-500">Seconds</span>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Computational budget breakdown including HPO and fold fitting:
              </p>

              <div className="mt-4 space-y-3">
                {candidates.map((rec) => {
                  const timeSec = Number(rec.training_time_seconds) || 0.01
                  const barPercent = Math.max(4, Math.min(100, (timeSec / (maxTime || 1)) * 100))
                  const isSelected = rec.model_name === selectedModel
                  return (
                    <div
                      key={rec.model_name}
                      onClick={() => setSelectedModel(rec.model_name)}
                      className={`cursor-pointer rounded-lg p-2.5 transition border ${
                        isSelected
                          ? 'border-indigo-400 bg-indigo-50/50 shadow-sm'
                          : 'border-transparent hover:bg-slate-50'
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-700">{rec.model_name}</span>
                        <span className="font-mono text-slate-800 font-bold">
                          {formatSeconds(timeSec)}
                        </span>
                      </div>
                      <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-slate-500 transition-all duration-500"
                          style={{ width: `${barPercent}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Interactive Leaderboard Table */}
          <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-100 bg-slate-50/70 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-bold text-slate-900">Complete Model Comparison Matrix</h3>
                <p className="text-xs text-slate-500">
                  Select any row to view its complete hyperparameter configuration and fit audit
                </p>
              </div>
              <span className="text-xs font-mono text-slate-500">
                {candidates.length} Candidate Models Evaluated
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 font-mono text-[11px] uppercase tracking-wider text-slate-600">
                    <th className="px-4 py-3">Rank</th>
                    <th className="px-4 py-3">Algorithm</th>
                    <th className="px-4 py-3">CV Mean</th>
                    <th className="px-4 py-3">Std Dev</th>
                    <th className="px-4 py-3">Train Score</th>
                    <th className="px-4 py-3">Generalization Gap</th>
                    <th className="px-4 py-3">Bias-Variance Status</th>
                    <th className="px-4 py-3">Fit Time</th>
                    <th className="px-4 py-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {candidates.map((record) => {
                    const model = leaderboard.find((item) => item.model_name === record.model_name)
                    const isWinner = record.model_name === winnerName
                    const isSelected = record.model_name === selectedModel
                    const fitStatus = record.fit_status || 'well_fitted'

                    return (
                      <tr
                        key={record.model_name}
                        onClick={() => setSelectedModel(record.model_name)}
                        className={`cursor-pointer transition ${
                          isSelected
                            ? 'bg-indigo-50/80 font-medium'
                            : isWinner
                            ? 'bg-teal-50/40 hover:bg-teal-50/70'
                            : 'hover:bg-slate-50'
                        }`}
                      >
                        <td className="px-4 py-3">
                          <span
                            className={`inline-flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold ${
                              isWinner
                                ? 'bg-teal-600 text-white'
                                : 'bg-slate-200 text-slate-700'
                            }`}
                          >
                            {model ? model.rank : '-'}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-1.5">
                            <strong className="text-slate-900">{record.model_name}</strong>
                            {isWinner && (
                              <span className="rounded bg-teal-100 px-1.5 py-0.5 text-[9px] font-bold text-teal-800 uppercase">
                                Winner
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3 font-mono font-bold text-slate-900">
                          {formatScore(record.cv_mean_score ?? model?.mean_score)}
                        </td>
                        <td className="px-4 py-3 font-mono text-slate-500">
                          ±{formatScore(record.cv_std_score ?? model?.std_score)}
                        </td>
                        <td className="px-4 py-3 font-mono text-slate-700">
                          {record.train_score != null ? formatScore(record.train_score) : '-'}
                        </td>
                        <td className="px-4 py-3 font-mono">
                          {record.generalization_gap != null ? (
                            <span
                              className={
                                record.generalization_gap > 0.12 ? 'text-amber-600 font-semibold' : 'text-slate-600'
                              }
                            >
                              {formatScore(record.generalization_gap)}
                            </span>
                          ) : (
                            '-'
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                              fitStatus === 'well_fitted'
                                ? 'bg-emerald-100 text-emerald-800'
                                : fitStatus === 'overfitting'
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-rose-100 text-rose-800'
                            }`}
                          >
                            {fitStatus.replace('_', ' ')}
                          </span>
                        </td>
                        <td className="px-4 py-3 font-mono text-slate-600">
                          {formatSeconds(record.training_time_seconds)}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              setSelectedModel(record.model_name)
                              setActiveTab('diagnostics')
                            }}
                            className={`rounded-md px-2.5 py-1 text-[11px] font-semibold transition ${
                              isSelected
                                ? 'bg-indigo-600 text-white'
                                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                            }`}
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: SELECTED MODEL DIAGNOSTICS & HPO ── */}
      {activeTab === 'diagnostics' && (
        <div className="mt-6 space-y-6">
          {/* Header of Selected Model */}
          <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-indigo-200 bg-indigo-50/40 p-4">
            <div>
              <span className="text-xs font-mono uppercase tracking-wider text-indigo-700 font-bold">
                Detailed Model Inspector
              </span>
              <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                {selectedModel}
                {selectedModel === winnerName ? (
                  <span className="rounded bg-teal-600 px-2 py-0.5 text-xs font-bold text-white uppercase">
                    Rank #1 Selected Winner
                  </span>
                ) : (
                  <span className="rounded bg-slate-200 px-2 py-0.5 text-xs font-semibold text-slate-700">
                    Rank #{selectedLeaderboardItem?.rank || '-'} Alternative Candidate
                  </span>
                )}
              </h3>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500">Switch Model:</span>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-800 shadow-sm"
              >
                {candidates.map((c) => (
                  <option key={c.model_name} value={c.model_name}>
                    {c.model_name} {c.model_name === winnerName ? '(Winner)' : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Hyperparameter Settings */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
              <Sliders size={17} className="text-indigo-600" />
              <h3 className="text-sm font-bold text-slate-900">
                Tuned Hyperparameters ({Object.keys(selectedParams).length} Parameters Tuned)
              </h3>
            </div>

            {Object.keys(selectedParams).length > 0 ? (
              <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {Object.entries(selectedParams).map(([paramName, paramVal]) => (
                  <div
                    key={paramName}
                    className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50/70 px-3.5 py-2.5 text-xs"
                  >
                    <span className="font-mono text-slate-600">{paramName}</span>
                    <strong className="font-mono font-bold text-slate-900">
                      {String(paramVal)}
                    </strong>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-3 text-xs text-slate-500">
                This model ran with its default calibrated parameters.
              </p>
            )}
          </div>

          {/* Generalization & Fit Diagnostics */}
          {genAudit && (
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                <Cpu size={17} className="text-indigo-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Generalization & Bias-Variance Tradeoff Diagnosis
                </h3>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
                  <span className="text-[10px] font-mono uppercase text-slate-500">
                    Training Score
                  </span>
                  <strong className="mt-1 block font-mono text-base font-bold text-slate-900">
                    {formatScore(genAudit.train_score)}
                  </strong>
                  <small className="text-[10px] text-slate-400">Full train partition</small>
                </div>
                <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
                  <span className="text-[10px] font-mono uppercase text-slate-500">
                    CV Mean Score
                  </span>
                  <strong className="mt-1 block font-mono text-base font-bold text-slate-900">
                    {formatScore(genAudit.validation_score)}
                  </strong>
                  <small className="text-[10px] text-slate-400">Cross-validation</small>
                </div>
                <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
                  <span className="text-[10px] font-mono uppercase text-slate-500">
                    Generalization Gap
                  </span>
                  <strong className="mt-1 block font-mono text-base font-bold text-indigo-600">
                    {formatScore(genAudit.generalization_gap)}
                  </strong>
                  <small className="text-[10px] text-slate-400">{genAudit.gap_percentage}% divergence</small>
                </div>
                <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
                  <span className="text-[10px] font-mono uppercase text-slate-500">Fit Status</span>
                  <div className="mt-1">
                    <span
                      className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${
                        genAudit.status === 'well_fitted'
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}
                    >
                      {genAudit.status.replace('_', ' ')}
                    </span>
                  </div>
                  <small className="text-[10px] text-slate-400">{genAudit.risk_level} risk</small>
                </div>
              </div>

              <p className="mt-3 text-xs text-slate-600 leading-relaxed">{genAudit.explanation}</p>
            </div>
          )}

          {/* Optuna Parameter Importance and Trial History */}
          {(selectedImportances.length > 0 || selectedHistory.length > 0) && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {selectedImportances.length > 0 && (
                <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                  <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                    <Gauge size={17} className="text-indigo-600" />
                    <h3 className="text-sm font-bold text-slate-900">
                      Hyperparameter Importance (TPE Sensitivity)
                    </h3>
                  </div>
                  <div className="mt-4 space-y-3">
                    {selectedImportances.map(([name, value]) => (
                      <div key={name}>
                        <div className="flex justify-between text-xs">
                          <span className="font-mono text-slate-700">{name}</span>
                          <span className="font-mono font-bold text-slate-900">{formatScore(value)}</span>
                        </div>
                        <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-100">
                          <div
                            className="h-full rounded-full bg-indigo-600"
                            style={{ width: `${Math.max(3, Number(value) * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {selectedHistory.length > 0 && (
                <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                  <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                    <TrendingUp size={17} className="text-indigo-600" />
                    <h3 className="text-sm font-bold text-slate-900">
                      Optuna Trial Convergence History
                    </h3>
                  </div>
                  <div className="mt-4 flex h-32 items-end gap-1.5 rounded-lg border border-slate-100 bg-slate-50/50 p-3">
                    {selectedHistory.map((trial) => {
                      const val = Number(trial.value)
                      const isComplete = trial.state === 'COMPLETE' && Number.isFinite(val)
                      const heightPercent = isComplete
                        ? `${Math.max(6, ((val - lowVal) / (highVal - lowVal || 1)) * 100)}%`
                        : '4px'
                      return (
                        <div
                          key={trial.number}
                          title={`Trial #${trial.number}: ${trial.state} · ${formatScore(val)}`}
                          className={`flex-1 rounded-t transition-all ${
                            isComplete ? 'bg-indigo-600 hover:bg-indigo-700' : 'bg-slate-300'
                          }`}
                          style={{ height: heightPercent }}
                        />
                      )
                    })}
                  </div>
                  <p className="mt-2 text-[11px] text-slate-500 font-mono">
                    {validTrials.length} completed trials · evaluated under budget
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── TAB 3: OPTIMIZER COMPARISON (OPTUNA VS OTHERS) ── */}
      {activeTab === 'optimizers' && (
        <div className="mt-6 space-y-6">
          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-2 border-b border-slate-100 pb-4">
              <Flame size={20} className="text-amber-500" />
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Controlled Optimizer Benchmarks: Optuna TPE vs. GridSearchCV vs. RandomizedSearchCV
                </h3>
                <p className="text-xs text-slate-500">
                  Evaluated across identical dataset partitions, shared search budgets, and multi-seed folds.
                </p>
              </div>
            </div>

            {/* Rendered Seaborn / Matplotlib Visualizations */}
            <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
              <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-4">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono">
                  Figure 1: Optimizer Test Scores (Mean ± SD)
                </h4>
                <p className="text-xs text-slate-500 mb-3">
                  Publication chart generated with Matplotlib & Seaborn
                </p>
                <img
                  src={getOptimizerComparisonChartUrl()}
                  alt="Optimizer Comparison Chart"
                  className="w-full rounded-lg border border-slate-200 bg-white shadow-sm"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none'
                  }}
                />
              </div>

              <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-4">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono">
                  Figure 2: Master Real-World Multi-Task Evaluation
                </h4>
                <p className="text-xs text-slate-500 mb-3">
                  Classification, Regression, Clustering, and Anomaly Detection benchmarks
                </p>
                <img
                  src={getAllModelsChartUrl()}
                  alt="Master Model Evaluation Chart"
                  className="w-full rounded-lg border border-slate-200 bg-white shadow-sm"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none'
                  }}
                />
              </div>
            </div>

            {/* Key Findings Card */}
            <div className="mt-6 rounded-xl border border-slate-200 bg-slate-50 p-4">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono flex items-center gap-1.5">
                <Zap size={14} className="text-indigo-600" /> Key Benchmark Insights
              </h4>
              <ul className="mt-2 space-y-2 text-xs text-slate-600">
                <li className="flex items-start gap-2">
                  <span className="font-bold text-indigo-600 font-mono">1.</span>
                  <span>
                    <strong>Optuna TPE</strong> consistently discovers high-performing configurations in sparse non-convex hyperparameter landscapes by modelling probability distributions rather than brute-force grid enumeration.
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-bold text-indigo-600 font-mono">2.</span>
                  <span>
                    <strong>GridSearchCV</strong> evaluates a rigid uniform mesh that scales exponentially with parameter count, spending budget on non-impactful dimensions.
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-bold text-indigo-600 font-mono">3.</span>
                  <span>
                    <strong>RandomizedSearchCV</strong> samples broadly across parameter spaces but does not exploit previously successful regions.
                  </span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 4: HOLDOUT METRICS & EXPLAINABILITY ── */}
      {activeTab === 'holdout' && (
        <div className="mt-6 space-y-6">
          {/* Holdout Metrics Grid */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
              <CheckCircle2 size={17} className="text-teal-600" />
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  Holdout Validation Metrics (Unseen Test Partition)
                </h3>
                <p className="text-xs text-slate-500">
                  Computed on held-out samples that were never used for model fitting or hyperparameter selection
                </p>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {metrics.map(([name, value]) => (
                <div
                  key={name}
                  className="rounded-lg border border-slate-100 bg-slate-50/70 p-3.5 transition hover:bg-slate-100/60"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono text-slate-700 capitalize font-medium">
                      {name.replaceAll('_', ' ')}
                    </span>
                    <strong className="font-mono text-sm font-bold text-slate-900">
                      {formatScore(value)}
                    </strong>
                  </div>
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-teal-600"
                      style={{
                        width: `${Math.max(
                          4,
                          Math.min(100, (Math.abs(Number(value)) / (maxMetric || 1)) * 100)
                        )}%`,
                      }}
                    />
                  </div>
                  <p className="mt-2 text-[10px] text-slate-500 leading-tight">
                    {metricDefinitions[name] || 'Measured holdout test performance.'}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Feature Importance & Explainability */}
          {importance.length > 0 && (
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                <Sparkles size={17} className="text-indigo-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Global Holdout Feature Importance (Permutation Impact)
                </h3>
              </div>
              <div className="mt-4 space-y-3">
                {importance.slice(0, 10).map((item) => (
                  <div key={item.feature}>
                    <div className="flex justify-between text-xs">
                      <span className="font-mono font-medium text-slate-700">{item.feature}</span>
                      <span className="font-mono font-bold text-slate-900">
                        {formatScore(item.importance_mean)}{' '}
                        <span className="text-[10px] text-slate-400">
                          ±{formatScore(item.importance_std)}
                        </span>
                      </span>
                    </div>
                    <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className="h-full rounded-full bg-indigo-600"
                        style={{
                          width: `${Math.max(
                            3,
                            (Math.abs(item.importance_mean) /
                              Math.max(1e-9, ...importance.map((entry) => Math.abs(entry.importance_mean)))) *
                              100
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Confusion Matrix if applicable */}
          {confusionSize > 0 && (
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="text-sm font-bold text-slate-900 mb-3">Holdout Confusion Matrix</h3>
              <div
                className="inline-grid gap-1.5"
                style={{ gridTemplateColumns: `repeat(${confusionSize}, minmax(4rem, 1fr))` }}
              >
                {confusion.flatMap((row, r) =>
                  row.map((val, c) => (
                    <div
                      key={`${r}-${c}`}
                      className="rounded-lg p-3 text-center border border-slate-200"
                      style={{
                        backgroundColor:
                          r === c
                            ? `rgba(13, 148, 136, ${Math.min(0.85, 0.15 + (val / Math.max(1, ...confusion.flat())) * 0.7)})`
                            : `rgba(244, 63, 94, ${Math.min(0.4, 0.05 + (val / Math.max(1, ...confusion.flat())) * 0.35)})`,
                        color: r === c ? '#0f766e' : '#be123c',
                      }}
                    >
                      <strong className="block text-base font-bold font-mono">{val}</strong>
                      <span className="text-[10px] font-mono opacity-80 block">
                        Act {r} / Pred {c}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
