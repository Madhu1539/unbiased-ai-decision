import React from 'react'

/**
 * ConfusionMatrix  —  Renders a colour-coded confusion matrix.
 *
 * For 2×2 binary matrices it additionally labels each cell as
 * TP / FP / FN / TN with a colour-coded badge and shows a
 * full breakdown legend below the table.
 *
 * Props:
 *   matrix   number[][]   2-D array from the API  [[TN,FP],[FN,TP]]
 *   labels   string[]     Class labels (optional)  e.g. ['No','Yes']
 *   tp       number       (optional, auto-derived if matrix is 2×2)
 *   fp       number       (optional)
 *   tn       number       (optional)
 *   fn       number       (optional)
 */
export default function ConfusionMatrix({ matrix, labels, tp, fp, tn, fn }) {
  if (!matrix || matrix.length === 0) return null

  const n = matrix.length
  const defaultLabels = Array.from({ length: n }, (_, i) => `Class ${i}`)
  const cls = labels || defaultLabels

  const isBinary = n === 2

  // Auto-derive TP/FP/TN/FN from the 2×2 matrix if not passed as props
  // sklearn layout: [[TN, FP], [FN, TP]]
  const _tn = tn ?? (isBinary ? matrix[0][0] : null)
  const _fp = fp ?? (isBinary ? matrix[0][1] : null)
  const _fn = fn ?? (isBinary ? matrix[1][0] : null)
  const _tp = tp ?? (isBinary ? matrix[1][1] : null)

  const maxVal = Math.max(...matrix.flat(), 1)
  const alpha = (v) => (Math.round((v / maxVal) * 255) / 255).toFixed(2)

  // Label badge for each cell (only for 2×2)
  const cellLabel = (i, j) => {
    if (!isBinary) return null
    const map = { '0,0': 'TN', '0,1': 'FP', '1,0': 'FN', '1,1': 'TP' }
    const key = `${i},${j}`
    const label = map[key]
    if (!label) return null
    const colours = {
      TP: 'bg-indigo-500/30 text-indigo-300 border border-indigo-500/40',
      TN: 'bg-emerald-500/30 text-emerald-300 border border-emerald-500/40',
      FP: 'bg-rose-500/30 text-rose-300 border border-rose-500/40',
      FN: 'bg-amber-500/30 text-amber-300 border border-amber-500/40',
    }
    return (
      <span className={`inline-block text-[9px] font-bold px-1 py-0.5 rounded mt-1 ${colours[label]}`}>
        {label}
      </span>
    )
  }

  return (
    <div className="space-y-4">
      {/* Note */}
      <p className="text-xs text-slate-500 italic">
        Metrics update based on selected threshold.
        Rows = Actual class · Columns = Predicted class
      </p>

      <div className="overflow-x-auto">
        <table className="mx-auto border-collapse text-sm">
          <thead>
            <tr>
              <th className="p-2 text-slate-500 text-xs">Actual ↓ / Pred →</th>
              {cls.map((c, j) => (
                <th key={j} className="p-2 text-center text-slate-400 text-xs font-semibold min-w-[80px]">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="p-2 text-slate-400 text-xs font-semibold text-right pr-3">
                  {cls[i]}
                </td>
                {row.map((val, j) => {
                  const isDiag = i === j
                  const a = alpha(val)
                  return (
                    <td
                      key={j}
                      className="p-2 text-center rounded-lg m-0.5 transition-all"
                      style={{
                        backgroundColor: isDiag
                          ? `rgba(99,102,241,${a})`
                          : `rgba(239,68,68,${Number(a) * 0.7})`,
                        color: Number(a) > 0.5 ? '#fff' : '#94a3b8',
                        border: isDiag
                          ? '2px solid rgba(99,102,241,0.5)'
                          : '1px solid rgba(51,65,85,0.5)',
                      }}
                    >
                      <div className="flex flex-col items-center gap-0.5">
                        <span className="font-bold text-base">{val}</span>
                        {cellLabel(i, j)}
                      </div>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>



      {/* Generic legend for N×N */}
      {!isBinary && (
        <div className="flex gap-4 justify-center mt-1 text-xs text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-brand-600 inline-block" /> Correct
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-danger-500 inline-block" /> Incorrect
          </span>
        </div>
      )}
    </div>
  )
}
