import { useState } from 'react'
import { ArrowLeft, Download, ExternalLink, LineChart, Sparkles, Layers, Sliders, Database, Cpu } from 'lucide-react'
import TrainingResults from '../components/TrainingResults'
import TrajectoryChart from '../components/TrajectoryChart'
import ResidualChart from '../components/ResidualChart'
import { getLeaderboardChartUrl } from '../services/api'

export default function AnalysisPage({ job, dataset, onBack, onExport }) {
  if (!job) return null

  return (
    <main className="product-page max-w-7xl mx-auto px-4 py-8">
      {/* Top Action Bar */}
      <div className="flex items-center justify-between border-b border-slate-200/80 pb-4 mb-6">
        <button
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3.5 py-1.5 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
          onClick={onBack}
        >
          <ArrowLeft size={15} /> Return to Workflow
        </button>
        <button
          className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-bold text-white shadow-sm transition hover:bg-indigo-700"
          onClick={onExport}
        >
          Export Reproducible Pipeline <Download size={15} />
        </button>
      </div>

      {/* Hero Header */}
      <header className="mb-8">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="font-mono text-xs font-bold uppercase tracking-wider text-indigo-600">
              AutoML Execution & Model Decision Analytics
            </p>
            <h1 className="mt-1 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl">
              Decision Report & Audit
            </h1>
            <p className="mt-2 text-sm text-slate-600 max-w-2xl leading-relaxed">
              Transparent, reproducible evidence for model selection from dataset structure to holdout validation and hyperparameter convergence.
            </p>
          </div>
          <div className="inline-flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-3.5 py-2 text-xs font-bold text-emerald-800 shadow-sm">
            <Sparkles size={16} className="text-emerald-600" />
            Completed Execution Verified
          </div>
        </div>
      </header>

      {/* Decision Summary Grid */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-3 mb-8">
        {/* Recommended Winner Card */}
        <div className="rounded-2xl border border-indigo-900/10 bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 p-6 text-white shadow-md lg:col-span-2">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold uppercase tracking-widest text-indigo-300">
              Selected Champion Model
            </span>
            <span className="rounded-full bg-teal-500/20 px-2.5 py-0.5 font-mono text-[11px] font-bold text-teal-300 border border-teal-500/30 uppercase">
              Rank #1 Winner
            </span>
          </div>

          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white">
            {job.best_model}
          </h2>

          <p className="mt-3 text-sm text-slate-300 leading-relaxed max-w-xl">
            Selected by cross-validated{' '}
            <strong className="text-indigo-200 font-mono">{job.primary_metric || 'score'}</strong> for a{' '}
            <strong className="text-white capitalize">{job.task_type?.replaceAll('_', ' ') || 'tabular'}</strong> task.
            The complete inference graph encapsulates pre-fitted feature encoders, imputers, scalers, and tuned estimator parameters.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-3 border-t border-white/10 pt-4">
            <button
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-200 hover:text-white transition"
              onClick={onExport}
            >
              Inspect Model Artifact & Code <ExternalLink size={14} />
            </button>
          </div>
        </div>

        {/* Quick Facts Card */}
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-slate-500 border-b border-slate-100 pb-2 mb-4">
            Run Specifications
          </h3>
          <div className="space-y-3.5 text-xs">
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Source Dataset</span>
              <strong className="font-mono text-slate-800 truncate max-w-[150px]">
                {dataset?.filename || 'Uploaded CSV'}
              </strong>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Features Profiled</span>
              <strong className="font-mono text-slate-800">{job.features_used?.length ?? '-'} columns</strong>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Primary Metric</span>
              <strong className="font-mono text-indigo-600 uppercase">{job.primary_metric || 'Auto'}</strong>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Validation Protocol</span>
              <strong className="font-mono text-slate-800">{job.cv_folds || 5} Folds + Holdout</strong>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Generalization Fit</span>
              <span className="inline-flex rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-800 uppercase font-mono">
                {job.generalization_audit ? job.generalization_audit.status.replace('_', ' ') : 'Well Fitted'}
              </span>
            </div>
            <div className="flex justify-between items-center border-t border-slate-100 pt-3">
              <span className="text-slate-500 font-semibold">Test Performance</span>
              <strong className="font-mono text-sm text-emerald-600">
                {job.metrics?.accuracy != null
                  ? `${(job.metrics.accuracy * 100).toFixed(1)}% Acc`
                  : job.metrics?.r2 != null
                  ? `${job.metrics.r2.toFixed(3)} R²`
                  : job.metrics?.rmse != null
                  ? `${job.metrics.rmse.toFixed(3)} RMSE`
                  : 'Evaluated'}
              </strong>
            </div>
          </div>
        </div>
      </section>

      {/* Features Used Badges */}
      {job.features_used && job.features_used.length > 0 && (
        <section className="mb-8 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3 mb-3">
            <Layers size={16} className="text-indigo-600" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
              Features Ingested into Pipeline ({job.features_used.length} Features)
            </h3>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {job.features_used.map((feat) => (
              <span
                key={feat}
                className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-mono text-slate-700 font-medium"
              >
                {feat}
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Trajectory / Prediction Residual Charts */}
      {job.prediction_series && job.prediction_series.length > 0 && (
        <section className="mb-8 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3 mb-4">
            <LineChart size={18} className="text-indigo-600" />
            <div>
              <h3 className="text-sm font-bold text-slate-900">Holdout Trajectory & Residual Analysis</h3>
              <p className="text-xs text-slate-500">Visual comparison of actual observations versus model predictions</p>
            </div>
          </div>
          <TrajectoryChart data={job.prediction_series} />
          <div className="mt-6 border-t border-slate-100 pt-6">
            <ResidualChart data={job.prediction_series} />
          </div>
        </section>
      )}

      {/* High-Resolution Seaborn Leaderboard Chart if present */}
      {job.leaderboard_chart && (
        <section className="mb-8 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
            <div className="flex items-center gap-2">
              <LineChart size={18} className="text-indigo-600" />
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  Statistical Leaderboard Distribution
                </h3>
                <p className="text-xs text-slate-500">
                  Publication-grade horizontal bar visualization generated with Seaborn & Matplotlib
                </p>
              </div>
            </div>
          </div>
          <img
            src={getLeaderboardChartUrl(job.job_id)}
            alt="Seaborn cross-validation leaderboard chart"
            className="w-full rounded-lg border border-slate-200 bg-white"
          />
        </section>
      )}

      {/* Comprehensive Results Tabs & Metrics */}
      <TrainingResults job={job} />
    </main>
  )
}
