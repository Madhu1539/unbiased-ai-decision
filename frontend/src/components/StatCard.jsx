import React from 'react'
import { TrendingUp } from 'lucide-react'

/**
 * StatCard  —  Displays a single KPI metric with trend indicator.
 *
 * Props:
 *   label      string   – Metric name
 *   value      string   – Formatted metric value
 *   icon       node     – Lucide icon element
 *   color      string   – Tailwind text color class (default: text-brand-400)
 *   trend      number   – Percentage change (optional)
 *   subtitle   string   – Extra context below value (optional)
 */
export default function StatCard({ label, value, icon, color = 'text-brand-400', trend, subtitle }) {
  return (
    <div className="card flex flex-col gap-3 animate-slide-up hover:border-brand-500/40 transition-all duration-300">
      <div className="flex items-start justify-between">
        <div className={`w-10 h-10 rounded-xl bg-surface-700 flex items-center justify-center ${color}`}>
          {icon}
        </div>
        {trend !== undefined && (
          <span className={`text-xs font-semibold flex items-center gap-0.5
            ${trend >= 0 ? 'text-accent-400' : 'text-danger-400'}`}>
            <TrendingUp size={12} className={trend < 0 ? 'rotate-180' : ''} />
            {Math.abs(trend).toFixed(1)}%
          </span>
        )}
      </div>
      <div>
        <p className="text-2xl font-bold text-white">{value ?? '—'}</p>
        <p className="text-sm text-slate-400 mt-0.5">{label}</p>
        {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
      </div>
    </div>
  )
}
