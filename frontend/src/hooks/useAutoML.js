import { useEffect, useState } from 'react'
import { getTrainingStatus, startTraining, uploadDataset } from '../services/api'

function getErrorMessage(requestError, fallback) {
  const detail = requestError.response?.data?.detail
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join('; ')
  if (typeof detail === 'object' && detail !== null) return detail.msg || fallback
  return detail || requestError.response?.data?.error || fallback
}

export function useAutoML() {
  const [dataset, setDataset] = useState(null)
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')

  async function upload(file) {
    setError('')
    setJob(null)
    try {
      const response = await uploadDataset(file)
      setDataset(response.data)
      return response.data
    } catch (requestError) {
      setError(getErrorMessage(requestError, 'Dataset upload failed.'))
      throw requestError
    }
  }

  async function train(configuration) {
    setError('')
    try {
      const response = await startTraining({ ...configuration, dataset_id: dataset?.id })
      setJob(response.data)
      return response.data
    } catch (requestError) {
      setError(getErrorMessage(requestError, 'Training could not be started.'))
      throw requestError
    }
  }

  useEffect(() => {
    if (!job?.job_id || !['queued', 'running'].includes(job.status)) return undefined
    const timer = window.setInterval(async () => {
      try {
        const response = await getTrainingStatus(job.job_id)
        setJob(response.data)
      } catch (requestError) {
        setError(getErrorMessage(requestError, 'Could not read training status.'))
      }
    }, 1500)
    return () => window.clearInterval(timer)
  }, [job?.job_id, job?.status])

  return { dataset, job, error, upload, train }
}
