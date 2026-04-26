import React from 'react'
import { Bell, Zap } from 'lucide-react'
import ThemeToggle from './ThemeToggle'

const PAGE_TITLES = {
  dashboard:   { title: 'Dashboard',            subtitle: 'Overview of your ML pipeline and key capabilities' },
  upload:      { title: 'Data Upload',          subtitle: 'Import your dataset and set the target column' },
  eda:         { title: 'Exploratory Analysis', subtitle: 'Visualise distributions, correlations, and patterns' },
  cleaning:    { title: 'Basic Data Cleaning',  subtitle: 'Manually remove duplicates, fix types, and drop irrelevant columns' },
  split:       { title: 'Split Data',           subtitle: 'Divide dataset into train & test sets before any model fitting' },
  features:    { title: 'Feature Engineering',  subtitle: 'Rank and select the most important features' },
  preprocess:  { title: 'Preprocessing',        subtitle: 'Fit on train data only — missing values, outliers, encoding, scaling' },
  models:      { title: 'Model Selection',     subtitle: 'Choose and configure the right algorithm' },
  training:    { title: 'Model Training',      subtitle: 'Train the model and monitor progress' },
  evaluation:  { title: 'Evaluation',          subtitle: 'Assess performance with detailed metrics' },
  bias:        { title: 'Bias & Fairness',     subtitle: 'Detect and mitigate algorithmic bias' },
  predictions: { title: 'Predictions',         subtitle: 'Explore predictions on test data' },
  reports:     { title: 'Reports',             subtitle: 'Export your analysis and model artefacts' },
}

export default function Header({ page }) {
  const info = PAGE_TITLES[page] || { title: 'Dashboard', subtitle: '' }
  return (
    <header className="h-16 bg-surface-800/80 dark:bg-surface-800/80 light:bg-white/90 backdrop-blur border-b border-surface-700 dark:border-surface-700 flex items-center px-6 gap-4 sticky top-0 z-10 transition-colors duration-300">
      {/* Page info */}
      <div className="flex-1">
        <h1 className="text-lg font-bold text-white dark:text-white leading-tight">{info.title}</h1>
        <p className="text-xs text-slate-400 dark:text-slate-400">{info.subtitle}</p>
      </div>

      {/* Status pill */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-accent-500/10 border border-accent-500/20">
        <span className="w-1.5 h-1.5 rounded-full bg-accent-400 animate-pulse-slow" />
        <span className="text-xs font-semibold text-accent-400">API Connected</span>
      </div>

      {/* Theme Toggle */}
      <ThemeToggle />

      {/* Notification Bell */}
      <button
        id="header-notification-btn"
        aria-label="Notifications"
        className="w-9 h-9 rounded-xl bg-surface-700 hover:bg-surface-600 flex items-center justify-center transition-colors text-slate-400 hover:text-white"
      >
        <Bell size={16} />
      </button>

      {/* Powered by AI badge */}
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-gradient-brand text-white text-xs font-semibold shadow-lg">
        <Zap size={13} />
        Powered by AI
      </div>
    </header>
  )
}
