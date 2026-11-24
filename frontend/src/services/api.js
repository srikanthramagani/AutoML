import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

const apiKey = import.meta.env.VITE_API_KEY
if (apiKey) {
  api.defaults.headers.common['X-API-Key'] = apiKey
}

export const uploadDataset = (file) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/upload', form)
}

export const getDatasetProfile = (datasetId) => api.get(`/datasets/${datasetId}/profile`)
export const startTraining = (payload) => api.post('/training/start', payload)
export const startMultimodalTraining = (payload) => api.post('/multimodal/train', payload)
export const runForecast = (payload) => api.post('/forecast/run', payload)
export const downloadForecastUrl = (forecastId) => `/api/forecasts/${forecastId}/download`
export const predictForecast = (forecastId, file, horizon) => {
  const form = new FormData()
  if (file) form.append('history_file', file)
  if (horizon) form.append('horizon', String(horizon))
  return api.post(`/forecasts/${forecastId}/predict`, form)
}
export const uploadPredictionCsv = (modelId, file) => {
  const form = new FormData(); form.append('file', file)
  return api.post(`/models/${modelId}/predict-upload`, form)
}
export const getTrainingStatus = (jobId) => api.get(`/training/${jobId}`)
export const getModelMetadata = (modelId) => api.get(`/models/${modelId}`)
export const getModelDownloadUrl = (modelId) => `/api/models/${modelId}/download`
export const getLeaderboardChartUrl = (modelId) => `/api/models/${modelId}/leaderboard-chart`
export const getOptimizerComparisonChartUrl = () => '/api/experiments/optimizer-comparison-chart'
export const getAllModelsChartUrl = () => '/api/experiments/all-models-chart'
export const getExperimentsMetricsSummary = () => api.get('/experiments/metrics-summary')
