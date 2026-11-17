import { Braces, CheckCircle2, Database, Filter, GitBranch, Hash, Layers3, Table2 } from 'lucide-react'

const transformations = [
  ['Rows with a missing target', 'Removed before feature/target split so evaluation uses known outcomes.', Filter],
  ['Missing numeric values', 'Filled with the training-set median inside the pipeline.', Hash],
  ['Missing categorical values', 'Filled with the most frequent training-set category.', Table2],
  ['Categorical features', 'Automatically encoded with one-hot or ordinal encoding based on cardinality.', Braces],
  ['Numeric features', 'Standard-scaled after imputation and encoding.', Layers3],
  ['Numeric outliers', 'Clipped using the configured Tukey IQR handler.', GitBranch],
]

function display(value) {
  return value === null || value === undefined ? '-' : typeof value === 'number' ? Number(value.toFixed(4)) : value
}

export default function DatasetProfile({ dataset, keptIdentifiers = [], onKeepIdentifier }) {
  const columns = dataset.column_details || []
  const identifierCandidates = columns.filter((column) => column.identifier_candidate)
  return <section className="dataset-profile" aria-label="Dataset profile">
    <div className="profile-heading"><div><p className="eyebrow">02 / dataset profile</p><h3>What arrived, before modeling</h3></div><Database size={19} className="muted-icon" /></div>
    <div className="profile-stats"><div><strong>{dataset.rows?.toLocaleString() || '-'}</strong><span>rows</span></div><div><strong>{dataset.columns || '-'}</strong><span>columns</span></div><div><strong>{dataset.numeric_columns || 0}</strong><span>numeric</span></div><div><strong>{dataset.categorical_columns || 0}</strong><span>categorical</span></div><div><strong>{dataset.missing_pct ?? 0}%</strong><span>missing cells</span></div><div><strong>{dataset.duplicates || 0}</strong><span>duplicates</span></div></div>
    <div className="profile-target"><span>Inferred target</span><strong>{dataset.target || 'Select a target column'}</strong><small>The final CSV column is used by default. Change it below when your dataset has a different outcome column.</small></div>
    {identifierCandidates.length > 0 && <div className="identifier-review"><div className="analysis-title"><Database size={15} /><span>Identifier and leakage review</span></div><p>Confirmed identifier/index columns are excluded before splitting by default. Generic high-cardinality columns are shown for review and remain available unless you choose otherwise.</p>{identifierCandidates.map((column) => <label className="identifier-option" key={column.name}><input type="checkbox" checked={keptIdentifiers.includes(column.name)} onChange={(event) => onKeepIdentifier?.(column.name, event.target.checked)} /><span><strong>{column.name}</strong><small>{column.identifier_excluded_by_default && !keptIdentifiers.includes(column.name) ? 'Excluded by default' : 'Kept as a feature'} · {column.identifier_reason}</small></span></label>)}</div>}
    <div className="column-table-wrap"><table className="column-table"><thead><tr><th>Column</th><th>Type</th><th>Data kind</th><th>Missing</th><th>Unique</th><th>Summary / sample</th></tr></thead><tbody>{columns.map((column) => <tr key={column.name}><td><strong>{column.name}</strong></td><td>{column.dtype}</td><td>{column.inferred_type}</td><td>{column.missing} ({column.missing_pct}%)</td><td>{column.unique}</td><td>{column.mean !== null && column.mean !== undefined ? `mean ${display(column.mean)} · ${display(column.min)} to ${display(column.max)}` : column.top_values?.join(', ') || '-'}</td></tr>)}</tbody></table></div>
    <div className="transform-panel"><div className="analysis-title"><CheckCircle2 size={15} /><span>Transformation pipeline applied during training</span></div><div className="transform-grid">{transformations.map(([name, detail, Icon]) => <div className="transform-step" key={name}><Icon size={16} /><div><strong>{name}</strong><p>{detail}</p></div></div>)}</div></div>
  </section>
}
