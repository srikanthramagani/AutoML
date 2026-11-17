import { Download, FileDown } from 'lucide-react'
import { getModelDownloadUrl } from '../services/api'

export default function ModelDownload({ job }) {
  return <section className="download-panel" aria-label="Model download"><div><p className="eyebrow">05 / export</p><h3>Take the trained pipeline with you</h3><p>Download the complete preprocessing and {job.best_model} pipeline as a compressed Joblib-compatible `.pkl` artifact.</p></div><a className="download-action" href={getModelDownloadUrl(job.job_id)} download><FileDown size={18} /> Download trained model <Download size={15} /></a></section>
}
