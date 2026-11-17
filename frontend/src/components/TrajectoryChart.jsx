import React, { useState } from 'react'
import { Activity, TrendingUp } from 'lucide-react'

export default function TrajectoryChart({ data = [] }) {
  const [hoveredPoint, setHoveredPoint] = useState(null)

  if (!data || data.length === 0) {
    return (
      <div className="empty-chart">
        <Activity size={24} className="muted-icon" />
        <p>No continuous prediction series available for this task.</p>
      </div>
    )
  }

  // Dimensions
  const width = 800
  const height = 260
  const padding = { top: 25, right: 30, bottom: 35, left: 55 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  const actuals = data.map((d) => d.actual)
  const predicteds = data.map((d) => d.predicted)
  const allValues = [...actuals, ...predicteds]

  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)
  const range = maxVal - minVal || 1.0

  const getX = (idx) => padding.left + (idx / Math.max(1, data.length - 1)) * chartWidth
  const getY = (val) => padding.top + chartHeight - ((val - minVal) / range) * chartHeight

  // Construct SVG paths
  const actualPoints = data.map((d, i) => `${getX(i).toFixed(1)},${getY(d.actual).toFixed(1)}`).join(' ')
  const predictedPoints = data.map((d, i) => `${getX(i).toFixed(1)},${getY(d.predicted).toFixed(1)}`).join(' ')

  return (
    <div className="trajectory-card">
      <div className="trajectory-header">
        <div>
          <p className="card-kicker">Holdout Trajectory & Forecast Fit</p>
          <h3>Actual vs. Model Predicted Sequence</h3>
        </div>
        <div className="trajectory-legend">
          <span className="legend-item actual">
            <span className="legend-dot" /> Actual Ground Truth
          </span>
          <span className="legend-item predicted">
            <span className="legend-dot" /> Model Prediction (ŷ)
          </span>
        </div>
      </div>

      <div className="svg-container">
        <svg viewBox={`0 0 ${width} ${height}`} className="trajectory-svg">
          {/* Y Axis Gridlines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
            const y = padding.top + chartHeight * (1 - ratio)
            const val = minVal + ratio * range
            return (
              <g key={ratio}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#e4eaf3" strokeDasharray="3 3" />
                <text x={padding.left - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#8898aa" fontFamily="'DM Mono', monospace">
                  {val.toFixed(1)}
                </text>
              </g>
            )
          })}

          {/* Actual Line */}
          <polyline fill="none" stroke="#2259ce" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" points={actualPoints} />

          {/* Predicted Line */}
          <polyline fill="none" stroke="#f26430" strokeWidth="2" strokeDasharray="4 3" strokeLinecap="round" strokeLinejoin="round" points={predictedPoints} />

          {/* Hover interactive markers */}
          {data.map((d, i) => {
            const cx = getX(i)
            const cyAct = getY(d.actual)
            const cyPred = getY(d.predicted)
            const isHovered = hoveredPoint && hoveredPoint.index === i

            return (
              <g key={i} onMouseEnter={() => setHoveredPoint({ ...d, index: i, x: cx, yAct: cyAct, yPred: cyPred })} onMouseLeave={() => setHoveredPoint(null)}>
                <circle cx={cx} cy={cyAct} r={isHovered ? 5 : 2} fill="#2259ce" className="point-dot" />
                <circle cx={cx} cy={cyPred} r={isHovered ? 5 : 2} fill="#f26430" className="point-dot" />
                {/* Transparent hit area */}
                <rect x={cx - 5} y={padding.top} width="10" height={chartHeight} fill="transparent" cursor="pointer" />
              </g>
            )
          })}

          {/* Tooltip render */}
          {hoveredPoint && (
            <g transform={`translate(${Math.min(hoveredPoint.x, width - 130)}, ${padding.top + 10})`}>
              <rect width="125" height="52" rx="6" fill="#1b2533" fillOpacity="0.9" />
              <text x="10" y="18" fill="#a0aec0" fontSize="10" fontFamily="'DM Mono', monospace">
                Sample #{hoveredPoint.index + 1}
              </text>
              <text x="10" y="32" fill="#63b3ed" fontSize="11" fontWeight="600" fontFamily="'DM Mono', monospace">
                Actual: {hoveredPoint.actual}
              </text>
              <text x="10" y="46" fill="#fbd38d" fontSize="11" fontWeight="600" fontFamily="'DM Mono', monospace">
                Predicted: {hoveredPoint.predicted}
              </text>
            </g>
          )}
        </svg>
      </div>
      <div className="trajectory-footer">
        <span>X-axis: Sequential test horizon ({data.length} sample points)</span>
        <span>
          Max Deviation: {Math.max(...data.map((d) => Math.abs(d.residual || 0))).toFixed(3)}
        </span>
      </div>
    </div>
  )
}
