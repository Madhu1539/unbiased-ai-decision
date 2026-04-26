import React, { useState, useEffect, useCallback } from 'react'
import {
  ShieldCheck, AlertTriangle, CheckCircle, RefreshCw,
  AlertCircle, XCircle, Ban, Shield, HelpCircle, Users,
  Sparkles, Loader2, ChevronDown, ChevronUp, Zap,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Cell,
} from 'recharts'
import { getProtectedAttributes, analyzeBias, generateFairnessAudit } from '../services/api'

const COLORS = ['#6366f1', '#34d399', '#f97316', '#f43f5e', '#06b6d4', '#8b5cf6']

// ── Column category styling ───────────────────────────────────────────
const CATEGORY_STYLE = {
  SENSITIVE: {
    badge     : 'bg-green-500/15 text-green-400 border-green-500/30',
    icon      : CheckCircle,
    iconColor : 'text-green-400',
    label     : 'Protected',
  },
  POTENTIALLY_SENSITIVE: {
    badge     : 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
    icon      : AlertTriangle,
    iconColor : 'text-yellow-400',
    label     : 'Proxy',
  },
  NON_SENSITIVE: {
    badge     : 'bg-red-500/15 text-red-400 border-red-500/30',
    icon      : Ban,
    iconColor : 'text-red-400',
    label     : 'Excluded',
  },
  UNKNOWN: {
    badge     : 'bg-slate-500/15 text-slate-400 border-slate-500/30',
    icon      : HelpCircle,
    iconColor : 'text-slate-400',
    label     : 'Unknown',
  },
}

// ── Severity config (for verdict card) ───────────────────────────────
const SEVERITY = {
  fair    : { label: 'Fair',     border: 'border-green-500/40',  bg: 'bg-green-500/10',  text: 'text-green-400',  bar: 'bg-green-500'  },
  mild    : { label: 'Mild',     border: 'border-yellow-500/40', bg: 'bg-yellow-500/10', text: 'text-yellow-400', bar: 'bg-yellow-500' },
  moderate: { label: 'Moderate', border: 'border-orange-500/40', bg: 'bg-orange-500/10', text: 'text-orange-400', bar: 'bg-orange-500' },
  severe  : { label: 'Severe',   border: 'border-red-500/40',    bg: 'bg-red-500/10',    text: 'text-red-400',    bar: 'bg-red-500'    },
}

// ── Risk level config for AI audit badge ────────────────────────────
const RISK_CONFIG = {
  HIGH   : { cls: 'border-red-500/40 bg-red-500/10 text-red-400',       dot: 'bg-red-400'    },
  MEDIUM : { cls: 'border-yellow-500/40 bg-yellow-500/10 text-yellow-400', dot: 'bg-yellow-400' },
  LOW    : { cls: 'border-green-500/40 bg-green-500/10 text-green-400',  dot: 'bg-green-400'  },
}

// ── Parse Gemini / fallback output into structured sections ───────────
function parseAuditText(text) {
  if (!text) return {}
  const result = {}

  // ── New strict auditor format ─────────────────────────────────────
  // Fields: Bias Summary / Fairness Risk Level / Reasoning / Actionable Fixes
  const bsMatch = text.match(/^Bias Summary:\s*(.+)/im)
  if (bsMatch) result.bias_summary = bsMatch[1].trim()

  const frlMatch = text.match(/^Fairness Risk Level:\s*(.+)/im)
  if (frlMatch) result.fairness_risk = frlMatch[1].trim()

  const reasonMatch = text.match(/^Reasoning:\s*([\s\S]*?)(?=\n\s*Actionable Fixes:|$)/im)
  if (reasonMatch) result.reasoning = reasonMatch[1].trim()

  const fixesMatch = text.match(/Actionable Fixes:\s*\n([\s\S]*?)$/i)
  if (fixesMatch) result.fixes = fixesMatch[1].trim()

  // ── Legacy formats fallback ───────────────────────────────────────
  if (!Object.keys(result).length) {
    // Previous format: Performance / Bias / Reason / Suggestions / Fairness Score
    const perfMatch = text.match(/^Performance:\s*(.+)/im)
    if (perfMatch) result.bias_summary = `Performance: ${perfMatch[1].trim()}`

    const biasMatch = text.match(/^Bias:\s*(.+)/im)
    const reaMatch  = text.match(/^Reason:\s*(.+)/im)
    if (biasMatch) {
      result.bias_summary = [
        result.bias_summary,
        `Bias: ${biasMatch[1].trim()}`,
        reaMatch ? `Reason: ${reaMatch[1].trim()}` : '',
      ].filter(Boolean).join('  |  ')
    }
    const sugMatch = text.match(/Suggestions:\s*\n([\s\S]*?)(?=\n\s*Fairness Score:|$)/i)
    if (sugMatch) result.fixes = sugMatch[1].trim()

    const fsMatch = text.match(/^Fairness Score:\s*(.+)/im)
    if (fsMatch) result.fairness_risk = fsMatch[1].trim()
  }

  // Final fallback: raw text
  if (!Object.keys(result).length) result.bias_summary = text.trim()
  return result
}

// ── Render a bullet-style section from text ───────────────────────────
function AuditSection({ title, content, icon: Icon, color = 'text-brand-400' }) {
  if (!content) return null
  const lines = content
    .split('\n')
    .map(l => l.trim())
    .filter(Boolean)

  return (
    <div className="space-y-2">
      <h4 className={`text-sm font-bold flex items-center gap-2 ${color}`}>
        <Icon size={14} />{title}
      </h4>
      <div className="space-y-1 pl-1">
        {lines.map((line, i) => {
          const clean = line.replace(/^[-•*]\s*/, '')
          const isBullet = /^[-•*]/.test(line) || lines.length > 1
          return isBullet ? (
            <div key={i} className="flex items-start gap-2 text-sm text-slate-300">
              <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-slate-500 flex-shrink-0" />
              <span className="leading-relaxed">{clean}</span>
            </div>
          ) : (
            <p key={i} className="text-sm text-slate-300 leading-relaxed">{clean}</p>
          )
        })}
      </div>
    </div>
  )
}

// ── Gemini AI Insights Panel ──────────────────────────────────────────
function GeminiInsightsPanel({ audit, loading, error }) {
  const [collapsed, setCollapsed] = useState(false)

  if (!loading && !audit && !error) return null

  // Extract risk level keyword from the risk section text
  const riskText = (audit?.fairness_risk || '').toUpperCase()
  const riskLevel = ['HIGH', 'MEDIUM', 'LOW'].find(r => riskText.includes(r)) || null
  const riskCfg   = riskLevel ? RISK_CONFIG[riskLevel] : null

  return (
    <div
      id="gemini-insights-panel"
      className="card border border-brand-500/30 bg-gradient-to-br from-brand-500/5 to-accent-500/5 animate-fade-in-up"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-gradient-to-br from-brand-500/20 to-accent-500/20 border border-brand-500/20">
            <Sparkles size={18} className="text-brand-400" />
          </div>
          <div>
            <h3 className="font-bold text-white text-base flex items-center gap-2">
              AI Fairness Audit
              <span className="text-[10px] font-semibold uppercase tracking-widest px-2 py-0.5
                rounded-full bg-brand-500/15 border border-brand-500/25 text-brand-400">
                Gemini 2.0 Flash Lite
              </span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">Powered by Google Gemini — analysis based solely on provided metrics</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {riskCfg && (
            <span className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-bold uppercase tracking-wide ${riskCfg.cls}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${riskCfg.dot}`} />
              {riskLevel} Risk
            </span>
          )}
          {audit && (
            <button
              onClick={() => setCollapsed(c => !c)}
              className="p-1.5 rounded-lg hover:bg-surface-700/50 text-slate-400 hover:text-white transition-colors"
              title={collapsed ? 'Expand' : 'Collapse'}
            >
              {collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
            </button>
          )}
        </div>
      </div>

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-3">
          <div className="flex items-center gap-3 text-brand-400 text-sm mb-4">
            <Loader2 size={16} className="animate-spin" />
            <span>Gemini is analysing your model metrics for bias and fairness…</span>
          </div>
          {[120, 90, 160, 80, 110].map((w, i) => (
            <div key={i} className="skeleton h-3 rounded" style={{ width: `${w}px`, maxWidth: '100%' }} />
          ))}
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="flex items-start gap-3 p-4 rounded-xl border border-red-500/30 bg-red-500/8 text-red-400">
          <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-semibold text-sm mb-0.5">Audit failed</p>
            <p className="text-xs leading-relaxed text-red-300">{error}</p>
          </div>
        </div>
      )}

      {/* Audit sections */}
      {audit && !loading && !collapsed && (
        <div className="space-y-5 divide-y divide-surface-700/60">

          {/* Bias Summary */}
          {audit.bias_summary && (
            <AuditSection
              title="Bias Summary"
              content={audit.bias_summary}
              icon={ShieldCheck}
              color="text-brand-400"
            />
          )}

          {/* Fairness Risk Level */}
          {audit.fairness_risk && (
            <div className="pt-4">
              <AuditSection
                title="Fairness Risk Level"
                content={audit.fairness_risk}
                icon={Zap}
                color={riskCfg?.cls?.split(' ').find(c => c.startsWith('text-')) || 'text-slate-300'}
              />
            </div>
          )}

          {/* Reasoning */}
          {audit.reasoning && (
            <div className="pt-4">
              <AuditSection
                title="Reasoning"
                content={audit.reasoning}
                icon={HelpCircle}
                color="text-orange-400"
              />
            </div>
          )}

          {/* Actionable Fixes */}
          {audit.fixes && (
            <div className="pt-4">
              <AuditSection
                title="Actionable Fixes"
                content={audit.fixes}
                icon={CheckCircle}
                color="text-green-400"
              />
            </div>
          )}

        </div>
      )}

      {/* Collapsed pill */}
      {audit && !loading && collapsed && (
        <p className="text-xs text-slate-500 italic">Audit hidden — click the expand button to view insights.</p>
      )}
    </div>
  )
}

// ── Feature Analysis Panel ────────────────────────────────────────────
function FeatureAnalysisPanel({ analysis }) {
  const [expanded, setExpanded] = useState(false)
  if (!analysis) return null

  const { sensitive_columns = [], potentially_sensitive_columns = [],
          non_sensitive_columns = [], unknown_columns = [],
          column_details = [] } = analysis

  const total = column_details.length
  const catCount = {
    SENSITIVE            : sensitive_columns.length,
    POTENTIALLY_SENSITIVE: potentially_sensitive_columns.length,
    NON_SENSITIVE        : non_sensitive_columns.length,
    UNKNOWN              : unknown_columns.length,
  }

  return (
    <div className="card border border-surface-600">
      {/* Summary row */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <Shield size={16} className="text-brand-400" />
          Feature Sensitivity Analysis
          <span className="text-slate-500 text-xs font-normal">({total} columns scanned)</span>
        </h3>
        <button
          onClick={() => setExpanded(e => !e)}
          className="text-xs text-brand-400 hover:text-brand-300 underline"
        >
          {expanded ? 'Hide details' : 'Show all columns'}
        </button>
      </div>

      {/* Category chips */}
      <div className="flex flex-wrap gap-2 mb-4">
        {[
          { cat: 'SENSITIVE',             count: catCount.SENSITIVE },
          { cat: 'POTENTIALLY_SENSITIVE', count: catCount.POTENTIALLY_SENSITIVE },
          { cat: 'NON_SENSITIVE',         count: catCount.NON_SENSITIVE },
          { cat: 'UNKNOWN',               count: catCount.UNKNOWN },
        ].filter(x => x.count > 0).map(({ cat, count }) => {
          const s   = CATEGORY_STYLE[cat]
          const Ico = s.icon
          return (
            <span key={cat} className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-semibold ${s.badge}`}>
              <Ico size={11} /> {s.label}: {count}
            </span>
          )
        })}
      </div>

      {/* Expanded column detail table */}
      {expanded && column_details.length > 0 && (
        <div className="space-y-2 mt-2 max-h-72 overflow-y-auto pr-1">
          {column_details.map(col => {
            const s   = CATEGORY_STYLE[col.category] || CATEGORY_STYLE.UNKNOWN
            const Ico = s.icon
            return (
              <div
                key={col.column}
                className="flex items-start gap-3 p-3 rounded-xl bg-surface-800/60 border border-surface-700"
              >
                <Ico size={14} className={`${s.iconColor} mt-0.5 flex-shrink-0`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-white text-sm font-semibold">{col.column}</span>
                    <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold uppercase ${s.badge}`}>
                      {s.label}
                    </span>
                    {col.type !== 'unknown' && col.type !== 'non_sensitive' && (
                      <span className="text-slate-500 text-[10px] capitalize">{col.type.replace(/_/g, ' ')}</span>
                    )}
                    <span className="text-slate-600 text-[10px] ml-auto flex-shrink-0">
                      {col.unique_values} unique values
                    </span>
                  </div>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">{col.reason}</p>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Fairness Blocked Card ─────────────────────────────────────────────
function FairnessBlockedCard({ message }) {
  return (
    <div className="card border-2 border-slate-600/50 bg-slate-800/30">
      <div className="flex items-start gap-4">
        <div className="p-3 rounded-xl bg-slate-700/50 flex-shrink-0">
          <Ban size={28} className="text-slate-400" />
        </div>
        <div>
          <p className="text-lg font-bold text-slate-300 mb-1">
            Fairness Analysis Not Applicable
          </p>
          <p className="text-slate-400 text-sm leading-relaxed max-w-xl">{message}</p>
          <div className="mt-3 flex items-start gap-2 text-xs text-slate-500 bg-slate-900/50 rounded-lg p-3 border border-slate-700">
            <AlertTriangle size={12} className="text-yellow-500 mt-0.5 flex-shrink-0" />
            <span>
              Fairness analysis requires at least one <strong className="text-slate-300">protected attribute</strong> —
              a column directly representing human identity (gender, age, race, religion, nationality, or disability).
              Business metrics, medical measurements, and ID columns are not valid for fairness analysis.
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Verdict Card ──────────────────────────────────────────────────────
function VerdictCard({ verdict }) {
  if (!verdict) return null
  const sev   = SEVERITY[verdict.severity] || SEVERITY.fair
  const score = verdict.fairness_score ?? 100
  const all   = [...(verdict.failed_metrics || []), ...(verdict.passed_metrics || [])]

  return (
    <div className={`card border-2 ${sev.border} ${sev.bg}`}>
      <div className="flex items-center justify-between flex-wrap gap-4 mb-5">
        <div className="flex items-center gap-4">
          {verdict.is_biased
            ? <XCircle size={40} className="text-red-400 flex-shrink-0" />
            : <ShieldCheck size={40} className="text-green-400 flex-shrink-0" />}
          <div>
            <p className={`text-2xl font-bold ${sev.text}`}>{verdict.verdict}</p>
            <p className="text-slate-300 text-sm mt-0.5 max-w-xl">{verdict.explanation}</p>
          </div>
        </div>
        <div className="flex flex-col items-center gap-1 min-w-[90px]">
          <div className={`text-3xl font-black ${sev.text}`}>{score}%</div>
          <div className="text-slate-400 text-xs font-medium">Fairness Score</div>
          <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${sev.bar}`}
              style={{ width: `${score}%` }}
            />
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 mb-5">
        <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${sev.border} ${sev.text}`}>
          {sev.label} Bias
        </span>
        <span className="text-slate-500 text-xs">
          {verdict.failed_metrics?.length ?? 0} of {verdict.total_checks ?? 0} checks failed
        </span>
      </div>

      {all.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {all.map(m => {
            const pass = m.status === 'pass'
            return (
              <div
                key={m.metric}
                className={`rounded-xl border p-3 ${
                  pass ? 'border-green-500/20 bg-green-500/5' : 'border-red-500/20 bg-red-500/5'
                }`}
              >
                <div className={`flex items-center gap-2 font-semibold text-sm mb-1 ${pass ? 'text-green-400' : 'text-red-400'}`}>
                  {pass ? <CheckCircle size={14} /> : <XCircle size={14} />}
                  {m.metric}
                </div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-slate-400 text-xs">Value</span>
                  <span className={`text-xs font-mono font-bold ${pass ? 'text-green-300' : 'text-red-300'}`}>
                    {m.value ?? 'N/A'}
                  </span>
                </div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-slate-400 text-xs">Threshold</span>
                  <span className="text-slate-300 text-xs font-mono">{m.threshold}</span>
                </div>
                <p className="text-slate-500 text-xs leading-relaxed">{m.description}</p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────
export default function BiasDetection() {
  const [attributes, setAttributes]   = useState([])
  const [analysis, setAnalysis]       = useState(null)   // full detection result
  const [selected, setSelected]       = useState('')
  const [report, setReport]           = useState(null)
  const [verdict, setVerdict]         = useState(null)
  const [insights, setInsights]       = useState([])
  const [loading, setLoading]         = useState(false)
  const [attrLoading, setAttrLoading] = useState(true)
  const [error, setError]             = useState(null)
  const [attrError, setAttrError]     = useState(null)

  // ── Gemini AI audit state ─────────────────────────────────────────
  const [auditLoading, setAuditLoading] = useState(false)
  const [auditResult,  setAuditResult]  = useState(null)   // parsed sections
  const [auditError,   setAuditError]   = useState(null)
  const [auditVisible, setAuditVisible] = useState(false)

  // ── Fetch feature analysis on mount ──────────────────────────────
  const loadAttributes = useCallback(async () => {
    setAttrLoading(true)
    setAttrError(null)
    try {
      const res = await getProtectedAttributes()
      console.log('[BiasDetection] API response:', res.data)

      const data  = res.data
      const attrs = data.eligible_columns || data.protected_attributes || data.columns || []

      console.log('[BiasDetection] Eligible columns:', attrs)
      console.log('[BiasDetection] fairness_applicable:', data.fairness_applicable)

      setAttributes(attrs)
      setAnalysis(data)
      if (attrs.length > 0) setSelected(attrs[0])
    } catch (e) {
      const msg = e.response?.data?.detail || e.message || 'Failed to load attributes.'
      console.error('[BiasDetection] Attribute fetch error:', msg, e)
      setAttrError(msg)
    } finally {
      setAttrLoading(false)
    }
  }, [])

  useEffect(() => { loadAttributes() }, [loadAttributes])

  // ── Run analysis ──────────────────────────────────────────────────
  const runAnalysis = async () => {
    if (!selected) return
    setLoading(true)
    setError(null)
    setReport(null)
    setVerdict(null)
    try {
      const res = await analyzeBias({ protected_attribute: selected })
      console.log('[BiasDetection] Analysis result:', res.data)
      setReport(res.data.report)
      setInsights(res.data.insights || [])
      setVerdict(res.data.verdict || null)
    } catch (e) {
      setError(e.response?.data?.detail || 'Bias analysis failed. Train a model first.')
    } finally {
      setLoading(false)
    }
  }

  // ── Build payload + call Gemini audit ──────────────────────────
  const buildAuditPayload = () => {
    const payload = {}
    if (report) {
      // Group accuracy
      if (report.group_accuracy?.group_accuracy) {
        payload.group_metrics = Object.fromEntries(
          Object.entries(report.group_accuracy.group_accuracy).map(
            ([g, d]) => [g, { accuracy: d.accuracy }]
          )
        )
      }
      // Demographic parity
      if (report.demographic_parity) {
        payload.demographic_parity_difference =
          report.demographic_parity.demographic_parity_difference
        payload.group_positive_rates =
          report.demographic_parity.group_positive_rates
      }
      // Disparate impact
      if (report.disparate_impact?.groups) {
        payload.disparate_impact_groups = Object.fromEntries(
          Object.entries(report.disparate_impact.groups).map(
            ([g, d]) => [g, { disparate_impact: d.disparate_impact, biased: d.biased }]
          )
        )
      }
    }
    if (verdict) {
      payload.verdict = verdict.verdict
      payload.fairness_score = verdict.fairness_score
      payload.severity = verdict.severity
      payload.is_biased = verdict.is_biased
      payload.failed_metrics = verdict.failed_metrics
      payload.passed_metrics = verdict.passed_metrics
    }
    if (selected) payload.protected_attribute = selected
    return payload
  }

  const runAudit = async () => {
    setAuditLoading(true)
    setAuditError(null)
    setAuditResult(null)
    setAuditVisible(true)

    // Scroll to the panel after a short tick
    setTimeout(() => {
      document.getElementById('gemini-insights-panel')
        ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 200)

    try {
      const payload = buildAuditPayload()
      const res = await generateFairnessAudit(payload)
      const data = res.data

      if (data.status === 'error') {
        setAuditError(data.message || 'Gemini returned an error.')
      } else {
        // data.audit is the raw text from Gemini — parse it into sections
        const parsed = parseAuditText(data.audit)
        setAuditResult(parsed)
      }
    } catch (e) {
      const msg =
        e.response?.data?.message ||
        e.response?.data?.detail ||
        e.message ||
        'Failed to reach the AI audit endpoint.'
      setAuditError(msg)
    } finally {
      setAuditLoading(false)
    }
  }

  // ── Derived chart data ────────────────────────────────────────────
  const diGroups    = report?.disparate_impact?.groups || {}
  const diChartData = Object.entries(diGroups).map(([g, d]) => ({
    group: g, disparate_impact: d.disparate_impact ?? 0,
    positive_rate: d.positive_rate, biased: d.biased,
  }))
  const accData      = report?.group_accuracy?.group_accuracy || {}
  const accChartData = Object.entries(accData).map(([g, d]) => ({
    group: g, accuracy: d.accuracy ?? 0, count: d.count,
  }))
  const dpRates     = report?.demographic_parity?.group_positive_rates || {}
  const dpChartData = Object.entries(dpRates).map(([g, rate]) => ({ group: g, rate }))
  const dpDiff      = report?.demographic_parity?.demographic_parity_difference

  // ── Get category style for selected column ────────────────────────
  const selectedDetail = analysis?.column_details?.find(c => c.column === selected)
  const selectedStyle  = selectedDetail ? CATEGORY_STYLE[selectedDetail.category] : null

  const fairnessApplicable = analysis?.fairness_applicable

  return (
    <div className="space-y-6 animate-fade-in">
      {/* ── Header ── */}
      <div>
        <h2 className="section-title">Bias &amp; Fairness Analysis</h2>
        <p className="section-subtitle">Detect and understand algorithmic bias across demographic groups</p>
      </div>

      {/* ── Controls ── */}
      <div className="card flex flex-wrap items-end gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="block text-sm text-slate-300 mb-2 font-medium">
            Protected Attribute
          </label>

          {/* Loading */}
          {attrLoading && (
            <div className="select flex items-center gap-2 text-slate-400">
              <div className="spinner w-4 h-4" />
              <span>Detecting sensitive features…</span>
            </div>
          )}

          {/* Error */}
          {attrError && !attrLoading && (
            <div className="flex items-center gap-2 text-red-400 text-sm mb-2">
              <AlertCircle size={14} />
              <span>{attrError}</span>
              <button onClick={loadAttributes} className="text-brand-400 underline ml-1">Retry</button>
            </div>
          )}

          {/* No dataset */}
          {!attrLoading && !attrError && !analysis && (
            <div className="flex items-center gap-2 text-slate-400 text-sm py-2">
              <AlertCircle size={14} />
              <span>No columns available. Upload a dataset first.</span>
            </div>
          )}

          {/* Fairness NOT applicable — dropdown hidden */}
          {!attrLoading && analysis && fairnessApplicable === false && (
            <div className="flex items-center gap-2 text-slate-500 text-sm py-2 italic">
              <Ban size={14} />
              <span>Analysis blocked — no valid protected attributes found.</span>
            </div>
          )}

          {/* Fairness IS applicable — show dropdown */}
          {!attrLoading && analysis && fairnessApplicable !== false && attributes.length > 0 && (
            <>
              <select
                value={selected}
                onChange={e => setSelected(e.target.value)}
                className="select"
              >
                <option value="">-- Select protected attribute --</option>
                {/* Directly sensitive: SENSITIVE category */}
                {attributes.filter(a => {
                  const d = analysis.column_details?.find(c => c.column === a)
                  return d?.category === 'SENSITIVE'
                }).length > 0 && (
                  <optgroup label="✅ Protected Attributes (Recommended)">
                    {attributes
                      .filter(a => {
                        const d = analysis.column_details?.find(c => c.column === a)
                        return d?.category === 'SENSITIVE'
                      })
                      .map(a => {
                        const d = analysis.column_details?.find(c => c.column === a)
                        return (
                          <option key={a} value={a}>
                            {a} — {d?.type?.replace(/_/g, ' ')} ({d?.unique_values} values)
                          </option>
                        )
                      })}
                  </optgroup>
                )}
                {/* Potentially sensitive */}
                {attributes.filter(a => {
                  const d = analysis.column_details?.find(c => c.column === a)
                  return d?.category === 'POTENTIALLY_SENSITIVE'
                }).length > 0 && (
                  <optgroup label="⚠️ Proxy Attributes (Use with caution)">
                    {attributes
                      .filter(a => {
                        const d = analysis.column_details?.find(c => c.column === a)
                        return d?.category === 'POTENTIALLY_SENSITIVE'
                      })
                      .map(a => {
                        const d = analysis.column_details?.find(c => c.column === a)
                        return (
                          <option key={a} value={a}>
                            {a} — {d?.type?.replace(/_/g, ' ')} ({d?.unique_values} values)
                          </option>
                        )
                      })}
                  </optgroup>
                )}
              </select>

              {/* Selected column info */}
              {selected && selectedDetail && (
                <div className={`mt-2 p-2 rounded-lg border text-xs flex items-start gap-2
                  ${selectedStyle?.badge || 'border-slate-600 bg-slate-800/30 text-slate-400'}`}>
                  {selectedDetail.category === 'POTENTIALLY_SENSITIVE' && (
                    <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
                  )}
                  {selectedDetail.category === 'SENSITIVE' && (
                    <CheckCircle size={12} className="mt-0.5 flex-shrink-0" />
                  )}
                  <span>{selectedDetail.reason}</span>
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadAttributes}
            disabled={attrLoading}
            title="Re-scan columns"
            className="btn-secondary text-sm px-3"
          >
            <RefreshCw size={14} className={attrLoading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={runAnalysis}
            disabled={loading || !selected || fairnessApplicable === false}
            className="btn-primary"
          >
            {loading
              ? <><div className="spinner" />Analysing…</>
              : <><ShieldCheck size={16} />Run Analysis</>}
          </button>
          <button
            id="btn-generate-ai-insights"
            onClick={runAudit}
            disabled={auditLoading}
            title="Generate Gemini AI fairness insights"
            className="btn-primary"
            style={{ background: 'linear-gradient(135deg,#7c3aed 0%,#6366f1 55%,#818cf8 100%)', boxShadow: '0 2px 14px rgba(124,58,237,0.4)' }}
          >
            {auditLoading
              ? <><Loader2 size={15} className="animate-spin" />Auditing…</>
              : <><Sparkles size={15} />Generate AI Insights</>}
          </button>
        </div>
      </div>

      {/* ── Feature sensitivity analysis panel ── */}
      {analysis && <FeatureAnalysisPanel analysis={analysis} />}

      {/* ── Fairness blocked message ── */}
      {analysis && fairnessApplicable === false && (
        <FairnessBlockedCard message={analysis.message} />
      )}

      {/* ── Analysis error ── */}
      {error && (
        <div className="card border-red-500/30 flex items-center gap-3 text-red-400">
          <AlertCircle size={18} />{error}
        </div>
      )}

      {/* ── Verdict Card ── */}
      {verdict && <VerdictCard verdict={verdict} />}

      {/* ── Insights ── */}
      {insights.length > 0 && (
        <div className="card space-y-2">
          <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
            <AlertTriangle size={16} className="text-yellow-400" /> Insights &amp; Alerts
          </h3>
          {insights.map((ins, i) => {
            const isWarn = ins.startsWith('⚠️')
            return (
              <div key={i} className={`flex items-start gap-3 p-3 rounded-xl border text-sm
                ${isWarn
                  ? 'border-yellow-500/30 bg-yellow-500/10 text-yellow-300'
                  : 'border-green-500/30 bg-green-500/10 text-green-300'}`}>
                {isWarn
                  ? <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
                  : <CheckCircle   size={16} className="mt-0.5 flex-shrink-0" />}
                <span>{ins}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* ── Gemini AI Insights Panel ── */}
      {auditVisible && (
        <GeminiInsightsPanel
          audit={auditResult}
          loading={auditLoading}
          error={auditError}
        />
      )}

      {/* ── Charts ── */}
      {report && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {diChartData.length > 0 && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-white flex items-center gap-2">
                  <ShieldCheck size={16} className="text-brand-400" /> Disparate Impact Ratio
                </h3>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <span className="w-3 h-0.5 bg-red-400 inline-block" /> Threshold (0.80)
                </div>
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={diChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="group" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis domain={[0, 1.5]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 12, color: '#e2e8f0' }}
                    formatter={v => v?.toFixed?.(3) ?? v} />
                  <Bar dataKey="disparate_impact" radius={[6, 6, 0, 0]}>
                    {diChartData.map((d, i) => <Cell key={i} fill={d.biased ? '#ef4444' : '#34d399'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="text-xs text-slate-500 mt-2 text-center">Bars below 0.80 (red) indicate potential bias (80% Rule)</p>
            </div>
          )}

          {accChartData.length > 0 && (
            <div className="card">
              <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                <Users size={16} className="text-green-400" /> Accuracy by Group
              </h3>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={accChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="group" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 12, color: '#e2e8f0' }}
                    formatter={v => v?.toFixed?.(3) ?? v} />
                  <Bar dataKey="accuracy" radius={[6, 6, 0, 0]}>
                    {accChartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {dpChartData.length > 0 && (
            <div className="card lg:col-span-2">
              <div className="flex items-start justify-between mb-4">
                <h3 className="font-semibold text-white flex items-center gap-2">
                  <ShieldCheck size={16} className="text-brand-400" />
                  Demographic Parity — Positive Prediction Rate
                </h3>
                {dpDiff != null && (
                  <span className={`badge ${dpDiff > 0.1 ? 'badge-warn' : 'badge-success'}`}>
                    Δ = {dpDiff.toFixed(3)}
                  </span>
                )}
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={dpChartData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical horizontal={false} />
                  <XAxis type="number" domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis type="category" dataKey="group" width={100} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 12, color: '#e2e8f0' }}
                    formatter={v => `${(v * 100).toFixed(1)}%`} />
                  <Bar dataKey="rate" radius={[0, 6, 6, 0]}>
                    {dpChartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* ── Fairness Reference ── */}
      <div className="card">
        <h3 className="font-semibold text-white mb-4">📚 Fairness Metrics Reference</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { name: 'Disparate Impact', formula: 'P(Ŷ=1|unprivileged) / P(Ŷ=1|privileged)', threshold: '≥ 0.80 (80% Rule)', desc: 'Ratio of positive prediction rates. Values <0.8 indicate potential bias.' },
            { name: 'Demographic Parity', formula: 'max(P(Ŷ=1|g)) − min(P(Ŷ=1|g))', threshold: '< 0.10', desc: 'Difference in positive rates across groups. Smaller is fairer.' },
            { name: 'Equalized Odds', formula: 'TPR per group', threshold: 'Equal across groups', desc: 'True positive rates should be similar for all groups.' },
          ].map(({ name, formula, threshold, desc }) => (
            <div key={name} className="bg-surface-700/50 rounded-xl p-4 border border-surface-600">
              <p className="font-semibold text-brand-300 mb-1">{name}</p>
              <p className="font-mono text-xs text-slate-400 mb-2 bg-surface-900 rounded-lg px-2 py-1">{formula}</p>
              <p className="text-xs text-green-400 mb-1.5">Threshold: {threshold}</p>
              <p className="text-xs text-slate-500">{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
