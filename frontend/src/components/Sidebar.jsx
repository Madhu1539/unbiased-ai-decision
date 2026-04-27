import React from 'react'
import {
  Upload, Sliders, BarChart2, Layers, Cpu, Play,
  TrendingUp, ShieldCheck, Crosshair, FileText,
  Brain, ChevronRight, LayoutDashboard, Sparkles, GitFork, Scale, X,
} from 'lucide-react'

const NAV_ITEMS = [
  { id: 'dashboard',   label: 'Dashboard',            icon: LayoutDashboard },
  { id: 'upload',      label: 'Data Upload',          icon: Upload },
  { id: 'eda',         label: 'EDA',                  icon: BarChart2 },
  { id: 'cleaning',    label: 'Data Cleaning',        icon: Sparkles },
  { id: 'preprocess',  label: 'Preprocessing',        icon: Sliders },
  { id: 'split',       label: 'Split Data',           icon: GitFork },
  { id: 'features',    label: 'Feature Engineering',  icon: Layers },
  { id: 'imbalance',   label: 'Class Imbalance',      icon: Scale },
  { id: 'models',      label: 'Model Selection',      icon: Cpu },
  { id: 'training',    label: 'Training',             icon: Play },
  { id: 'evaluation',  label: 'Evaluation',           icon: TrendingUp },
  { id: 'bias',        label: 'Bias & Fairness',      icon: ShieldCheck },
  { id: 'predictions', label: 'Predictions',          icon: Crosshair },
  { id: 'reports',     label: 'Reports',              icon: FileText },
]

export default function Sidebar({ active, onNavigate, sidebarOpen, setSidebarOpen }) {
  return (
    <aside
      className={`
        fixed inset-y-0 left-0 z-40 w-64 flex flex-col flex-shrink-0
        bg-surface-800 border-r border-surface-700
        transition-transform duration-300 ease-in-out
        lg:relative lg:translate-x-0 lg:z-auto
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}
    >
      {/* Logo */}
      <div className="flex items-center justify-between gap-3 px-6 py-5 border-b border-surface-700">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-brand flex items-center justify-center shadow-lg flex-shrink-0">
            <Brain size={18} className="text-white" />
          </div>
          <div>
            <p className="font-bold text-white text-sm leading-tight">Unbiased AI</p>
            <p className="text-xs text-slate-400">Decision Dashboard</p>
          </div>
        </div>

        {/* Close button — mobile only */}
        <button
          className="lg:hidden w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-surface-700 transition-colors flex-shrink-0"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close sidebar"
        >
          <X size={16} />
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 space-y-0.5 overflow-y-auto">
        <p className="px-4 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">
          ML Pipeline
        </p>
        {NAV_ITEMS.map(({ id, label, icon: Icon }, idx) => {
          const isActive = active === id
          return (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-none text-sm font-medium transition-all duration-150
                ${isActive
                  ? 'nav-active'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-surface-700/50'
                }`}
            >
              <span className={`flex items-center justify-center w-7 h-7 rounded-lg
                ${isActive ? 'bg-brand-600/30 text-brand-400' : 'text-slate-500'}`}>
                <Icon size={15} />
              </span>
              <span className="flex-1 text-left">{label}</span>
              <span className={`text-xs px-1.5 py-0.5 rounded-md font-mono
                ${isActive ? 'text-brand-400 bg-brand-900/50' : 'text-slate-600'}`}>
                {String(idx + 1).padStart(2, '0')}
              </span>
            </button>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-surface-700">
        <div className="glass rounded-xl p-3 text-xs text-slate-400">
          <p className="font-semibold text-slate-300 mb-1">💡 Pipeline Order</p>
          <p>Preprocess on full data → Split → Feature Eng → Imbalance → Train. Scaling happens inside Training only.</p>
        </div>
      </div>
    </aside>
  )
}
