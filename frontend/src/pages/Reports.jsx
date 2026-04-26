import React, { useState } from 'react'
import { FileText, Download, Database, FileBarChart, CheckCircle, AlertCircle } from 'lucide-react'
import { downloadCSV, downloadPDF } from '../services/api'

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

export default function Reports() {
  const [csvStatus, setCsvStatus] = useState('idle')
  const [pdfStatus, setPdfStatus] = useState('idle')
  const [error, setError]         = useState(null)

  const handleCSV = async () => {
    setCsvStatus('loading'); setError(null)
    try {
      const res = await downloadCSV()
      triggerDownload(new Blob([res.data], { type: 'text/csv' }), 'processed_dataset.csv')
      setCsvStatus('done')
    } catch {
      setError('CSV download failed.'); setCsvStatus('idle')
    }
  }

  const handlePDF = async () => {
    setPdfStatus('loading'); setError(null)
    try {
      const res = await downloadPDF()
      triggerDownload(new Blob([res.data], { type: 'application/pdf' }), 'evaluation_report.pdf')
      setPdfStatus('done')
    } catch {
      setError('PDF download failed. Make sure a model is trained.'); setPdfStatus('idle')
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="section-title">Reports & Exports</h2>
        <p className="section-subtitle">Download your processed data and model evaluation reports</p>
      </div>

      {error && <div className="card border-danger-500/30 flex items-center gap-3 text-danger-400"><AlertCircle size={18}/>{error}</div>}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* CSV Export */}
        <div className="card space-y-4 hover:border-brand-500/40 transition-all">
          <div className="w-12 h-12 rounded-2xl bg-accent-500/20 flex items-center justify-center">
            <Database size={22} className="text-accent-400" />
          </div>
          <div>
            <h3 className="font-bold text-white text-lg">Processed Dataset</h3>
            <p className="text-sm text-slate-400 mt-1">
              Download the cleaned and preprocessed dataset as a CSV file, ready for use in other tools.
            </p>
          </div>
          <ul className="text-xs text-slate-500 space-y-1">
            <li className="flex items-center gap-2"><CheckCircle size={11} className="text-accent-400"/> Missing values handled</li>
            <li className="flex items-center gap-2"><CheckCircle size={11} className="text-accent-400"/> Categorical columns encoded</li>
            <li className="flex items-center gap-2"><CheckCircle size={11} className="text-accent-400"/> Outliers removed (if selected)</li>
          </ul>
          <button onClick={handleCSV} disabled={csvStatus === 'loading'} className="btn-primary w-full justify-center">
            {csvStatus === 'loading' ? <><div className="spinner"/>Preparing…</> :
             csvStatus === 'done'    ? <><CheckCircle size={16}/>Downloaded!</> :
             <><Download size={16}/>Download CSV</>}
          </button>
        </div>

        {/* PDF Report */}
        <div className="card space-y-4 hover:border-brand-500/40 transition-all">
          <div className="w-12 h-12 rounded-2xl bg-brand-500/20 flex items-center justify-center">
            <FileBarChart size={22} className="text-brand-400" />
          </div>
          <div>
            <h3 className="font-bold text-white text-lg">Evaluation Report</h3>
            <p className="text-sm text-slate-400 mt-1">
              A professional PDF report summarising model configuration, performance metrics, and fairness findings.
            </p>
          </div>
          <ul className="text-xs text-slate-500 space-y-1">
            <li className="flex items-center gap-2"><CheckCircle size={11} className="text-brand-400"/> Model metadata</li>
            <li className="flex items-center gap-2"><CheckCircle size={11} className="text-brand-400"/> Performance metrics table</li>
            <li className="flex items-center gap-2"><CheckCircle size={11} className="text-brand-400"/> Generated timestamp</li>
          </ul>
          <button onClick={handlePDF} disabled={pdfStatus === 'loading'} className="btn-primary w-full justify-center">
            {pdfStatus === 'loading' ? <><div className="spinner"/>Generating PDF…</> :
             pdfStatus === 'done'    ? <><CheckCircle size={16}/>Downloaded!</> :
             <><Download size={16}/>Download PDF Report</>}
          </button>
        </div>
      </div>

      {/* Pipeline summary */}
      <div className="card">
        <h3 className="font-semibold text-white mb-5 flex items-center gap-2">
          <FileText size={16} className="text-brand-400"/> Pipeline Summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-0">
          {[
            { step: '01', label: 'Upload',       color: 'text-brand-400' },
            { step: '02', label: 'Preprocess',   color: 'text-accent-400' },
            { step: '03', label: 'EDA',           color: 'text-warn-400' },
            { step: '04', label: 'Features',      color: 'text-brand-300' },
            { step: '05', label: 'Train & Eval',  color: 'text-accent-300' },
          ].map(({ step, label, color }, i, arr) => (
            <React.Fragment key={step}>
              <div className="flex flex-col items-center gap-2 py-4">
                <div className={`w-10 h-10 rounded-xl bg-surface-700 flex items-center justify-center font-bold text-sm ${color}`}>
                  {step}
                </div>
                <p className="text-xs text-slate-400 text-center">{label}</p>
              </div>
              {i < arr.length - 1 && (
                <div className="hidden md:flex items-center justify-center text-slate-600 mb-6">→</div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Insights */}
      <div className="card">
        <h3 className="font-semibold text-white mb-4">💡 Recommendations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { title: 'Improve Accuracy', points: ['Try ensemble methods like Random Forest or XGBoost.', 'Increase training data via data augmentation.', 'Fine-tune hyperparameters using cross-validation.'] },
            { title: 'Reduce Bias', points: ['Use resampling (SMOTE) to balance protected groups.', 'Apply re-weighting during training.', 'Monitor fairness metrics iteratively after each run.'] },
          ].map(({ title, points }) => (
            <div key={title} className="bg-surface-700/50 rounded-xl p-4 border border-surface-600">
              <p className="font-semibold text-brand-300 mb-2">{title}</p>
              <ul className="space-y-1.5">
                {points.map(p => (
                  <li key={p} className="text-xs text-slate-400 flex items-start gap-2">
                    <span className="text-brand-400 mt-0.5">•</span> {p}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
