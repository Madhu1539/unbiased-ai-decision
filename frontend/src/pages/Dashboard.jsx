import React, { useState, useEffect } from 'react'
import {
  Brain, Upload, Sliders, BarChart2, Layers, Cpu,
  Play, TrendingUp, ShieldCheck, Crosshair, FileText,
  ArrowRight, CheckCircle, Clock, Zap, Activity,
  Sparkles, Database, Code2, GitFork, Scale,
} from 'lucide-react'
import { useThemeContext } from '../context/ThemeContext'

const PIPELINE_STEPS = [
  { id: 'upload',      label: 'Data Upload',          icon: Upload,      desc: 'Import CSV & select target',        color: 'text-blue-400',   hoverBg: 'group-hover:bg-blue-500/15'   },
  { id: 'eda',         label: 'EDA',                  icon: BarChart2,   desc: 'Explore patterns (no transforms)',   color: 'text-cyan-400',   hoverBg: 'group-hover:bg-cyan-500/15'   },
  { id: 'cleaning',    label: 'Data Cleaning',        icon: Sparkles,    desc: 'Deduplication & type fixes',         color: 'text-pink-400',   hoverBg: 'group-hover:bg-pink-500/15'   },
  { id: 'preprocess',  label: 'Preprocessing',        icon: Sliders,     desc: 'Clean & transform (train only)',     color: 'text-violet-400', hoverBg: 'group-hover:bg-violet-500/15' },
  { id: 'split',       label: 'Split Data',           icon: GitFork,     desc: 'Train / Test split before fitting',  color: 'text-yellow-400', hoverBg: 'group-hover:bg-yellow-500/15' },
  { id: 'features',    label: 'Feature Engineering',  icon: Layers,      desc: 'Fit on train, apply to test',        color: 'text-emerald-400',hoverBg: 'group-hover:bg-emerald-500/15'},
  { id: 'imbalance',   label: 'Class Imbalance',      icon: Scale,       desc: 'Balance classes (train only)',       color: 'text-indigo-400', hoverBg: 'group-hover:bg-indigo-500/15' },
  { id: 'models',      label: 'Model Selection',      icon: Cpu,         desc: 'Choose algorithm',                   color: 'text-orange-400', hoverBg: 'group-hover:bg-orange-500/15' },
  { id: 'training',    label: 'Training',             icon: Play,        desc: 'Fit model on training data',         color: 'text-rose-400',   hoverBg: 'group-hover:bg-rose-500/15'   },
  { id: 'evaluation',  label: 'Evaluation',           icon: TrendingUp,  desc: 'Cross-val + final test metrics',     color: 'text-brand-400',  hoverBg: 'group-hover:bg-brand-500/15'  },
  { id: 'bias',        label: 'Bias & Fairness',      icon: ShieldCheck, desc: 'Detect algorithmic bias',            color: 'text-purple-400', hoverBg: 'group-hover:bg-purple-500/15' },
  { id: 'predictions', label: 'Predictions',          icon: Crosshair,   desc: 'Run on unseen data',                 color: 'text-teal-400',   hoverBg: 'group-hover:bg-teal-500/15'   },
  { id: 'reports',     label: 'Reports',              icon: FileText,    desc: 'Export results & report',            color: 'text-amber-400',  hoverBg: 'group-hover:bg-amber-500/15'  },
]

const FEATURES = [
  {
    icon: <ShieldCheck size={22} />,
    title: 'Bias Detection',
    desc: 'Disparate Impact, Demographic Parity, and Equalized Odds analysis across protected groups.',
    gradient: 'from-rose-500/15 via-orange-500/8 to-transparent',
    border: 'border-rose-400/30',
    iconBg: 'bg-rose-500/15',
    iconColor: 'text-rose-500',
    dot: 'bg-rose-400',
  },
  {
    icon: <Activity size={22} />,
    title: 'Full ML Pipeline',
    desc: 'End-to-end workflow from data upload through preprocessing, training, evaluation, and prediction.',
    gradient: 'from-indigo-500/15 via-blue-500/8 to-transparent',
    border: 'border-indigo-400/30',
    iconBg: 'bg-indigo-500/15',
    iconColor: 'text-indigo-500',
    dot: 'bg-indigo-400',
  },
  {
    icon: <BarChart2 size={22} />,
    title: 'Interactive Visualisations',
    desc: 'Histograms, correlation heatmaps, ROC curves, confusion matrices, and scatter plots.',
    gradient: 'from-emerald-500/15 via-teal-500/8 to-transparent',
    border: 'border-emerald-400/30',
    iconBg: 'bg-emerald-500/15',
    iconColor: 'text-emerald-600',
    dot: 'bg-emerald-400',
  },
  {
    icon: <Zap size={22} />,
    title: 'Multiple Algorithms',
    desc: 'Random Forest, XGBoost, SVM, KNN, Gradient Boosting, Logistic/Linear Regression, and more.',
    gradient: 'from-amber-500/15 via-yellow-500/8 to-transparent',
    border: 'border-amber-400/30',
    iconBg: 'bg-amber-500/15',
    iconColor: 'text-amber-600',
    dot: 'bg-amber-400',
  },
]

const STATS = [
  { icon: <Database size={14} />, label: '10+ Algorithms' },
  { icon: <Code2    size={14} />, label: 'FastAPI + React' },
  { icon: <Sparkles size={14} />, label: 'Bias-Aware ML'  },
]

// ── Light-mode hero ──────────────────────────────────────────────────────────
function HeroLight({ onNavigate, time }) {
  return (
    <div className="relative overflow-hidden rounded-3xl border border-indigo-200/70 bg-white"
      style={{ boxShadow: '0 4px 32px rgba(55,48,163,0.10), 0 1px 6px rgba(55,48,163,0.06)' }}>

      {/* Top accent gradient strip */}
      <div className="absolute inset-x-0 top-0 h-1.5 rounded-t-3xl"
        style={{ background: 'linear-gradient(90deg, #4f46e5 0%, #818cf8 50%, #34d399 100%)' }} />

      {/* Background geometric decoration */}
      <div className="absolute top-0 right-0 w-96 h-96 rounded-full opacity-40 -translate-y-1/2 translate-x-1/3"
        style={{ background: 'radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%)' }} />
      <div className="absolute bottom-0 left-1/4 w-72 h-72 rounded-full opacity-30 translate-y-1/2"
        style={{ background: 'radial-gradient(circle, rgba(16,185,129,0.10) 0%, transparent 70%)' }} />

      {/* Grid pattern overlay */}
      <div className="absolute inset-0 opacity-[0.025]"
        style={{ backgroundImage: 'linear-gradient(#4f46e5 1px, transparent 1px), linear-gradient(90deg, #4f46e5 1px, transparent 1px)', backgroundSize: '40px 40px' }} />

      <div className="relative z-10 p-8 md:p-10">
        {/* Brand row */}
        <div className="flex items-center gap-4 mb-5">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg flex-shrink-0"
            style={{ background: 'linear-gradient(135deg, #4f46e5 0%, #6366f1 60%, #818cf8 100%)', boxShadow: '0 4px 16px rgba(99,102,241,0.35)' }}>
            <Brain size={26} className="text-white" />
          </div>
          <div>
            <h1 className="text-2xl md:text-3xl font-bold leading-tight"
              style={{ color: '#0d1b3e' }}>
              Unbiased AI Decision Dashboard
            </h1>
            <p className="text-sm mt-0.5 font-medium"
              style={{ color: '#4f46e5' }}>
              End-to-end ML pipeline with transparency & fairness
            </p>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm max-w-2xl leading-relaxed mb-6"
          style={{ color: '#3d4f7c' }}>
          Upload your dataset, preprocess it, explore patterns, train models, evaluate performance,
          detect algorithmic bias, and export professional reports — all in one place.
        </p>

        {/* Stat badges */}
        <div className="flex flex-wrap gap-2 mb-7">
          {STATS.map(({ icon, label }) => (
            <span key={label}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border"
              style={{ background: 'rgba(79,70,229,0.07)', borderColor: 'rgba(79,70,229,0.2)', color: '#3730a3' }}>
              {icon}{label}
            </span>
          ))}
        </div>

        {/* CTA row */}
        <div className="flex flex-wrap gap-3 items-center">
          <button
            onClick={() => onNavigate('upload')}
            className="btn-primary text-sm"
          >
            <Upload size={16} /> Get Started — Upload Data
          </button>

          <div className="flex items-center gap-2 px-4 py-2 rounded-xl border text-xs font-medium"
            style={{ background: '#f0f2f9', borderColor: '#d1d9ee', color: '#3d4f7c' }}>
            <Clock size={13} className="text-indigo-400" />
            {time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            <span className="mx-0.5 opacity-50">·</span>
            {time.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Dark-mode hero (original look preserved) ─────────────────────────────────
function HeroDark({ onNavigate, time }) {
  return (
    <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-brand-600/30 via-surface-800 to-accent-500/20 border border-brand-500/20 p-8 md:p-10">
      <div className="relative z-10">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-brand flex items-center justify-center shadow-lg shadow-brand-500/30">
            <Brain size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-white leading-tight">
              Unbiased AI Decision Dashboard
            </h1>
            <p className="text-sm text-slate-400 mt-0.5">
              End-to-end ML pipeline with transparency & fairness
            </p>
          </div>
        </div>

        <p className="text-slate-300 text-sm max-w-2xl leading-relaxed mt-4">
          Upload your dataset, preprocess it, explore patterns, train models, evaluate performance,
          detect algorithmic bias, and export professional reports — all in one place.
        </p>

        <div className="flex flex-wrap gap-3 mt-6">
          <button onClick={() => onNavigate('upload')} className="btn-primary text-sm">
            <Upload size={16} /> Get Started — Upload Data
          </button>
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-700/60 border border-surface-600 text-xs text-slate-400">
            <Clock size={13} />
            {time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            <span className="mx-1">·</span>
            {time.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })}
          </div>
        </div>
      </div>
      <div className="absolute top-0 right-0 w-80 h-80 rounded-full bg-brand-500/10 blur-3xl -translate-y-1/2 translate-x-1/3" />
      <div className="absolute bottom-0 left-1/3 w-60 h-60 rounded-full bg-accent-500/10 blur-3xl translate-y-1/2" />
    </div>
  )
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function Dashboard({ onNavigate }) {
  const [time, setTime] = useState(new Date())
  const { theme } = useThemeContext()
  const isLight = theme === 'light'

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 60000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="space-y-8 animate-fade-in">

      {/* ── Hero ── */}
      {isLight
        ? <HeroLight onNavigate={onNavigate} time={time} />
        : <HeroDark  onNavigate={onNavigate} time={time} />
      }

      {/* ── Pipeline Steps ── */}
      <div>
        <h2 className="section-title">ML Pipeline Workflow</h2>
        <p className="section-subtitle">Follow these steps in order for the best results</p>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {PIPELINE_STEPS.map(({ id, label, icon: Icon, desc, color, hoverBg }, idx) => (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              className="group card text-center py-5 px-3 cursor-pointer relative overflow-hidden"
              style={{ transition: 'box-shadow 0.25s ease, transform 0.2s ease, border-color 0.2s ease' }}
            >
              {/* hover bg wash */}
              <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300
                bg-gradient-to-br from-brand-500/5 to-accent-500/5`} />

              <div className="relative z-10">
                <div className={`w-10 h-10 rounded-xl bg-surface-700 ${hoverBg}
                  flex items-center justify-center mx-auto mb-3 transition-all duration-300`}>
                  <Icon size={18} className={`text-slate-400 group-hover:${color} transition-colors`} />
                </div>
                <span className="text-[10px] font-mono text-slate-600 block mb-1">
                  STEP {String(idx + 1).padStart(2, '0')}
                </span>
                <p className="text-sm font-semibold text-white mb-1">{label}</p>
                <p className="text-xs text-slate-500 leading-snug">{desc}</p>
              </div>
              <ArrowRight size={14}
                className={`absolute top-3 right-3 text-slate-700 group-hover:${color} transition-colors`} />
            </button>
          ))}
        </div>
      </div>

      {/* ── Feature Cards ── */}
      <div>
        <h2 className="section-title">Key Capabilities</h2>
        <p className="section-subtitle">What makes this dashboard powerful</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {FEATURES.map(({ icon, title, desc, gradient, border, iconBg, iconColor, dot }) => (
            <div
              key={title}
              className={`card bg-gradient-to-br ${gradient} ${border} group cursor-default`}
              style={{ transition: 'box-shadow 0.25s ease, transform 0.2s ease, border-color 0.2s ease' }}
            >
              <div className="flex items-start gap-4">
                <div className={`w-11 h-11 rounded-2xl ${iconBg} ${iconColor}
                  flex items-center justify-center flex-shrink-0
                  group-hover:scale-110 transition-transform duration-300`}>
                  {icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`w-1.5 h-1.5 rounded-full ${dot} flex-shrink-0`} />
                    <h3 className="font-semibold text-white">{title}</h3>
                  </div>
                  <p className="text-sm text-slate-400 leading-relaxed">{desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Quick Start Guide ── */}
      <div className="card">
        <h3 className="font-semibold text-white mb-5 flex items-center gap-2">
          <Zap size={16} className="text-brand-400" /> Quick Start Guide
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            {
              step: '01', title: 'Upload & Explore',
              desc: 'Upload a CSV, select your target column, then use EDA to understand distributions and patterns — no transformations at this stage.',
              action: 'upload', actionLabel: 'Upload Data',
              accent: '#4f46e5',
            },
            {
              step: '02', title: 'Engineer & Preprocess',
              desc: 'After splitting train/test, apply Feature Engineering then Preprocessing (missing values, outliers, encoding, scaling) — fit on train only.',
              action: 'features', actionLabel: 'Feature Engineering',
              accent: '#06b6d4',
            },
            {
              step: '03', title: 'Train & Evaluate',
              desc: 'Select a model, handle class imbalance (train only), review cross-validation + test metrics, check for bias, and export a report.',
              action: 'models', actionLabel: 'Choose Model',
              accent: '#34d399',
            },
          ].map(({ step, title, desc, action, actionLabel, accent }) => (
            <div key={step}
              className="bg-surface-700/50 rounded-2xl p-5 border border-surface-600 flex flex-col relative overflow-hidden group cursor-pointer"
              onClick={() => onNavigate(action)}
              style={{ transition: 'box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease' }}>
              {/* left accent bar */}
              <div className="absolute left-0 inset-y-0 w-1 rounded-l-2xl opacity-70 group-hover:opacity-100 transition-opacity"
                style={{ background: accent }} />

              <span className="text-xs font-mono font-bold mb-2 ml-2" style={{ color: accent }}>STEP {step}</span>
              <h4 className="font-semibold text-white mb-2 ml-2">{title}</h4>
              <p className="text-xs text-slate-400 leading-relaxed flex-1 ml-2">{desc}</p>
              <button
                onClick={(e) => { e.stopPropagation(); onNavigate(action) }}
                className="mt-4 ml-2 text-xs font-semibold flex items-center gap-1.5 transition-all w-fit group-hover:gap-2.5"
                style={{ color: accent }}>
                {actionLabel} <ArrowRight size={12} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* ── Tech Stack ── */}
      <div className="flex flex-wrap items-center justify-center gap-3 py-4 text-xs text-slate-600">
        {['React 18', 'Vite', 'Tailwind CSS', 'Recharts', 'FastAPI', 'scikit-learn', 'Pandas', 'SQLite'].map(tech => (
          <span key={tech}
            className="px-3 py-1.5 rounded-lg bg-surface-800 border border-surface-700 transition-all duration-200 hover:border-brand-500/40 hover:text-brand-400 cursor-default">
            {tech}
          </span>
        ))}
      </div>
    </div>
  )
}
