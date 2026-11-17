import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const uploadDataset = (file) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/upload', form)
}

export const getDatasetProfile = (datasetId) => api.get(`/datasets/${datasetId}/profile`)
export const startTraining = (payload) => api.post('/training/start', payload)
export const getTrainingStatus = (jobId) => api.get(`/training/${jobId}`)
export const getModelMetadata = (modelId) => api.get(`/models/${modelId}`)
export const getModelDownloadUrl = (modelId) => `/api/models/${modelId}/download`
export const getLeaderboardChartUrl = (modelId) => `/api/models/${modelId}/leaderboard-chart`
