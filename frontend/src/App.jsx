import React, { useState } from 'react'
import { ThemeProvider } from './context/ThemeContext'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import { ArrowLeft, ArrowRight, ChevronRight } from 'lucide-react'

// Pages
import Dashboard          from './pages/Dashboard'
import DataUpload         from './pages/DataUpload'
import EDA                from './pages/EDA'
import BasicDataCleaning  from './pages/BasicDataCleaning'
import SplitData          from './pages/SplitData'
import FeatureEngineering from './pages/FeatureEngineering'
import Preprocessing      from './pages/Preprocessing'
import ClassImbalance     from './pages/ClassImbalance'
import ModelSelection     from './pages/ModelSelection'
import Training           from './pages/Training'
import Evaluation         from './pages/Evaluation'
import BiasDetection      from './pages/BiasDetection'
import Predictions        from './pages/Predictions'
import Reports            from './pages/Reports'

// Ordered pipeline — order matters for next/prev logic
const PIPELINE = [
  { id: 'dashboard',  label: 'Dashboard',            component: Dashboard          },
  { id: 'upload',     label: 'Data Upload',           component: DataUpload         },
  { id: 'eda',        label: 'EDA',                   component: EDA                },
  { id: 'cleaning',   label: 'Basic Data Cleaning',   component: BasicDataCleaning  },
  { id: 'preprocess', label: 'Preprocessing',         component: Preprocessing      },
  { id: 'split',      label: 'Split Data',            component: SplitData          },
  { id: 'features',   label: 'Feature Engineering',   component: FeatureEngineering },
  { id: 'imbalance',  label: 'Class Imbalance',       component: ClassImbalance     },
  { id: 'models',     label: 'Model Selection',       component: ModelSelection     },
  { id: 'training',   label: 'Training',              component: Training           },
  { id: 'evaluation', label: 'Evaluation',            component: Evaluation         },
  { id: 'bias',       label: 'Bias & Fairness',       component: BiasDetection      },
  { id: 'predictions',label: 'Predictions',           component: Predictions        },
  { id: 'reports',    label: 'Reports',               component: Reports            },
]

const PAGE_MAP = Object.fromEntries(PIPELINE.map(p => [p.id, p.component]))

// ── Page Navigator bar ────────────────────────────────────────────────────────
function PageNavigator({ currentId, onNavigate }) {
  const idx     = PIPELINE.findIndex(p => p.id === currentId)
  const prev    = PIPELINE[idx - 1]
  const next    = PIPELINE[idx + 1]
  const total   = PIPELINE.length

  // Don't render on Dashboard (idx 0) — it has its own Get Started CTA
  if (idx <= 0) return null

  return (
    <div className="page-navigator">
      {/* ── Prev button ── */}
      <button
        onClick={() => onNavigate(prev.id)}
        className="nav-prev-btn"
        title={`Back to ${prev.label}`}
      >
        <ArrowLeft size={15} />
        <span className="nav-btn-label">{prev.label}</span>
      </button>

      {/* ── Step dots ── */}
      <div className="nav-steps">
        {PIPELINE.slice(1).map((step, i) => {
          const stepIdx = i + 1
          const isActive  = stepIdx === idx
          const isDone    = stepIdx < idx
          return (
            <button
              key={step.id}
              onClick={() => onNavigate(step.id)}
              title={`Step ${stepIdx}: ${step.label}`}
              className={`nav-dot ${isActive ? 'nav-dot-active' : isDone ? 'nav-dot-done' : 'nav-dot-idle'}`}
            />
          )
        })}
        <span className="nav-step-label">
          Step {idx} of {total - 1} — {PIPELINE[idx].label}
        </span>
      </div>

      {/* ── Next button ── */}
      {next ? (
        <button
          onClick={() => onNavigate(next.id)}
          className="nav-next-btn"
          title={`Continue to ${next.label}`}
        >
          <span className="nav-btn-label">{next.label}</span>
          <ArrowRight size={15} />
        </button>
      ) : (
        /* Last page — muted placeholder to keep layout balanced */
        <div className="nav-end-badge">
          <ChevronRight size={13} />
          Pipeline Complete
        </div>
      )}
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [activePage,     setActivePage]     = useState('dashboard')
  const [selectedModels, setSelectedModels] = useState([])   // always an array

  const PageComponent = PAGE_MAP[activePage] || Dashboard

  return (
    <ThemeProvider>
      <div className="flex h-screen bg-surface-900 dark:bg-surface-900 overflow-hidden transition-colors duration-300">
        <Sidebar active={activePage} onNavigate={setActivePage} />

        <div className="flex-1 flex flex-col overflow-hidden">
          <Header page={activePage} />

          <main className="flex-1 overflow-y-auto p-6">
            <div className="max-w-7xl mx-auto">
              <PageComponent
                onModelSelect={(nameOrArray) => {
                  // Normalise: single string or array → always store as array
                  const models = Array.isArray(nameOrArray)
                    ? nameOrArray.filter(Boolean)
                    : nameOrArray ? [nameOrArray] : []
                  setSelectedModels(models)
                  setActivePage('training')
                }}
                selectedModels={selectedModels}
                selectedModel={selectedModels[0] ?? null}  // backward compat
                onNavigate={setActivePage}
              />
            </div>
          </main>

          {/* ── Page Navigator ── */}
          <PageNavigator currentId={activePage} onNavigate={setActivePage} />

          {/* ── Footer ── */}
          <footer className="bg-surface-800 dark:bg-surface-800 border-t border-surface-700 px-6 py-2 flex items-center justify-between transition-colors duration-300">
            <p className="text-xs text-slate-500">
              Unbiased AI Decision Dashboard &copy; {new Date().getFullYear()}
            </p>
            <div className="flex items-center gap-3 text-xs text-slate-600">
              <span>FastAPI Backend</span>
              <span>·</span>
              <span>React + Vite Frontend</span>
              <span>·</span>
              <span>scikit-learn ML</span>
            </div>
          </footer>
        </div>

        {/* Background ambient glow */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
          <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-brand-600/10 blur-3xl" />
          <div className="absolute -bottom-40 -right-40 w-96 h-96 rounded-full bg-accent-500/10 blur-3xl" />
        </div>
      </div>
    </ThemeProvider>
  )
}
