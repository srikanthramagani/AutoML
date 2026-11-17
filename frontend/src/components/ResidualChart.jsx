import React from 'react'

export default function ResidualChart({ data = [] }) {
  if (!data || data.length === 0) return null

  const width = 800
  const height = 180
  const padding = { top: 20, right: 30, bottom: 30, left: 55 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  const residuals = data.map((d) => d.residual || 0)
  const maxAbsResidual = Math.max(...residuals.map(Math.abs), 1e-4)

  const getX = (idx) => padding.left + (idx / Math.max(1, data.length - 1)) * chartWidth
  const zeroY = padding.top + chartHeight / 2
  const getY = (res) => zeroY - (res / maxAbsResidual) * (chartHeight / 2)

  return (
    <div className="residual-card">
      <div className="trajectory-header">
        <div>
          <p className="card-kicker">Model Error Distribution</p>
          <h3>Residual Variance (Actual - Predicted)</h3>
        </div>
        <span className="zero-line-label">Center: 0.00 Error Baseline</span>
      </div>

      <div className="svg-container">
        <svg viewBox={`0 0 ${width} ${height}`} className="trajectory-svg">
          {/* Zero baseline */}
          <line x1={padding.left} y1={zeroY} x2={width - padding.right} y2={zeroY} stroke="#718096" strokeWidth="1.5" strokeDasharray="4 2" />

          {/* Upper & Lower bound lines */}
          <line x1={padding.left} y1={padding.top} x2={width - padding.right} y2={padding.top} stroke="#edf2f7" />
          <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#edf2f7" />

          {/* Y Axis labels */}
          <text x={padding.left - 8} y={padding.top + 10} textAnchor="end" fontSize="10" fill="#a0aec0" fontFamily="'DM Mono', monospace">
            +{maxAbsResidual.toFixed(2)}
          </text>
          <text x={padding.left - 8} y={zeroY + 4} textAnchor="end" fontSize="10" fill="#718096" fontFamily="'DM Mono', monospace">
            0.0
          </text>
          <text x={padding.left - 8} y={height - padding.bottom} textAnchor="end" fontSize="10" fill="#a0aec0" fontFamily="'DM Mono', monospace">
            -{maxAbsResidual.toFixed(2)}
          </text>

          {/* Stem & Point plots */}
          {data.map((d, i) => {
            const cx = getX(i)
            const cy = getY(d.residual || 0)
            const isPositive = (d.residual || 0) >= 0

            return (
              <g key={i}>
                <line x1={cx} y1={zeroY} x2={cx} y2={cy} stroke={isPositive ? '#319795' : '#e53e3e'} strokeWidth="1.5" strokeOpacity="0.75" />
                <circle cx={cx} cy={cy} r="3" fill={isPositive ? '#319795' : '#e53e3e'} />
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}
