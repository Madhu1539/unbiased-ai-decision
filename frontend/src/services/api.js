/**
 * api.js  —  Axios API service layer
 *
 * Performance fixes:
 *  - Default timeout raised to 120 s (2 min) for general calls
 *  - trainModel timeout raised to 600 s (10 min) for large-dataset training
 *  - Request/response interceptors for unified error logging
 *  - Named axios instances so timeouts are per-endpoint category
 */
import axios from 'axios'

const BASE = '/api'

// ── General-purpose client (2-minute timeout) ─────────────────────────
const api = axios.create({
  baseURL: BASE,
  timeout: 120_000,   // 2 minutes — enough for EDA, evaluation, preprocessing
})

// ── Long-running client for model training (10-minute timeout) ────────
const trainApi = axios.create({
  baseURL: BASE,
  timeout: 600_000,   // 10 minutes — covers SVM/RandomForest on 50k rows
})

// ── Response interceptor: log slow/failed requests ───────────────────
const _addInterceptors = (instance) => {
  instance.interceptors.request.use((config) => {
    config.metadata = { startTime: Date.now() }
    return config
  })
  instance.interceptors.response.use(
    (response) => {
      const ms = Date.now() - (response.config.metadata?.startTime ?? 0)
      if (ms > 5000) {
        console.warn(`[API] Slow response: ${response.config.url} took ${ms} ms`)
      }
      return response
    },
    (error) => {
      const ms = Date.now() - (error.config?.metadata?.startTime ?? 0)
      const url = error.config?.url ?? 'unknown'
      if (error.code === 'ECONNABORTED') {
        console.error(`[API] TIMEOUT after ${ms} ms — ${url}`)
      } else {
        console.error(`[API] Error on ${url}:`, error.response?.data?.detail ?? error.message)
      }
      return Promise.reject(error)
    }
  )
}

_addInterceptors(api)
_addInterceptors(trainApi)

// ─── Upload ──────────────────────────────────────────────────────────
export const uploadCSV = (file) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/upload/csv', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 180_000,   // 3 min for very large CSV uploads
  })
}

export const setTargetColumn = (target_column) =>
  api.post('/upload/target', { target_column })

export const getPreview = () => api.get('/upload/preview')

// ─── Basic Data Cleaning ─────────────────────────────────────────────
export const analyzeDataset       = () => api.get('/clean/analyze')
export const removeDuplicates     = () => api.post('/clean/duplicates')
export const fixDataTypes         = (conversions) => api.post('/clean/fix-dtypes', { conversions })
export const dropColumns          = (columns) => api.post('/clean/drop-columns', { columns })
export const getCleaningPreview   = () => api.get('/clean/preview')

// ─── Split ───────────────────────────────────────────────────────────
export const analyzeSplit  = ()       => api.get('/split/analyze')
export const performSplit  = (params) => api.post('/split', params)
export const getSplitStatus= ()       => api.get('/split/status')

// ─── Preprocessing (leakage-free sklearn Pipeline) ────────────────────
export const analyzePreprocessing  = ()       => api.get('/preprocess/analyze')
export const applyPreprocessing    = (params) => api.post('/preprocess/apply', params, { timeout: 300_000 })
export const previewPreprocessing  = ()       => api.get('/preprocess/preview')
export const getPreprocessingStatus= ()       => api.get('/preprocess/status')
// Legacy alias
export const runPreprocessing      = (params) => api.post('/preprocess/apply', params, { timeout: 300_000 })

// ─── EDA ─────────────────────────────────────────────────────────────
export const getSummary          = () => api.get('/eda/summary')
export const getHistogram        = (column, bins = 20) =>
  api.get('/eda/histogram', { params: { column, bins } })
export const getCorrelation      = () => api.get('/eda/correlation')
export const getValueCounts      = () => api.get('/eda/valuecounts')
export const getFeatureVsTarget  = (feature) =>
  api.get('/eda/feature-vs-target', { params: { feature } })
export const getClassDistribution = () => api.get('/eda/class-distribution')

// ─── Features (leakage-free, train-only) ─────────────────────────────
export const getFEStatus      = ()       => api.get('/features/status')
export const getFEAnalysis    = ()       => api.get('/features/analyze', { timeout: 180_000 })
export const createFormula    = (params) => api.post('/features/formula', params)
export const runFESelect      = (params) => api.post('/features/select', params)
export const getFECorrelation = ()       => api.get('/features/correlation')
export const getFEPreview     = ()       => api.get('/features/preview')
export const undoFE           = ()       => api.post('/features/undo')
export const resetFE          = ()       => api.post('/features/reset')
// Legacy aliases kept for any other pages that still reference them
export const getFeatureImportance    = () => api.get('/features/importance')
export const selectFeatures          = (params) => api.post('/features/select', params)
export const getFeatureAnalysis      = () => api.get('/features/analyze', { timeout: 180_000 })
export const getDatasetInfo          = () => api.get('/features/dataset-info')

// ─── Training (uses long-timeout client) ─────────────────────────────
export const getAvailableModels       = () => api.get('/train/models')
export const getModelRecommendations  = () => api.get('/train/recommendations')
export const trainModel               = (params) => trainApi.post('/train', params)
export const trainMultiModel          = (params) => trainApi.post('/train/multi', params)
export const analyzeTrainingImbalance = (params) =>
  api.post('/train/imbalance-analysis', params, { timeout: 120_000 })

// ─── Class Imbalance (standalone pipeline step) ──────────────────────
export const analyzeImbalance   = ()       => api.get('/imbalance/analyze', { timeout: 120_000 })
export const previewImbalance   = (params) => api.post('/imbalance/preview', params, { timeout: 120_000 })
export const confirmImbalance   = (params) => api.post('/imbalance/confirm', params)
export const getImbalanceStatus = ()       => api.get('/imbalance/status')


// ─── Evaluation ──────────────────────────────────────────────────────
export const getEvaluation = () =>
  api.get('/evaluate', { timeout: 120_000 })

/**
 * Live metric recomputation at a given threshold (API fallback).
 * Used only when the frontend probability cache is unavailable.
 *
 * @param {number}          threshold  0.0 → 1.0  (default 0.5)
 * @param {boolean}         debug      if true, backend logs TP/FP/TN/FN
 * @param {AbortSignal|null} signal    AbortController.signal for cancellation
 */
export const getLiveMetrics = (threshold = 0.5, debug = false, signal = null) =>
  api.get('/evaluate/metrics', {
    params : { threshold, debug },
    timeout: 60_000,
    ...(signal ? { signal } : {}),
  })

/**
 * Fetch y_prob + y_test_enc once for the frontend local metric engine.
 * Returns a stratified sample (≤10k) for large datasets.
 * The ROC curve uses the full dataset regardless.
 */
export const getProbabilities = () =>
  api.get('/evaluate/probabilities', { timeout: 90_000 })

// ─── Threshold Optimisation ───────────────────────────────────────────
// strategy: 'auto' | 'f1' | 'recall_priority'
export const getOptimalThreshold = (strategy = 'auto') =>
  api.get('/evaluate/threshold', { params: { strategy }, timeout: 120_000 })

export const applyThreshold = (threshold, strategy = 'auto') =>
  api.post('/evaluate/threshold/apply', { threshold, strategy }, { timeout: 120_000 })

export const getErrorAnalysis = (limit = 100) =>
  api.get('/evaluate/error-analysis', { params: { limit }, timeout: 90_000 })


// ─── Bias ────────────────────────────────────────────────────────────
export const getProtectedAttributes = () => api.get('/bias/attributes')
export const analyzeBias            = (params) =>
  api.post('/bias/analyze', params, { timeout: 180_000 })

// ─── Gemini AI Fairness Audit ─────────────────────────────────────────
export const generateFairnessAudit = (payload) =>
  api.post('/fairness-audit', payload, { timeout: 120_000 })

// ─── Predictions ─────────────────────────────────────────────────────
export const getBatchPredictions = () =>
  api.get('/predict/batch', { timeout: 180_000 })
export const predictSingle = (input_data) =>
  api.post('/predict/single', { input_data })

// ─── Reports ─────────────────────────────────────────────────────────
export const downloadCSV = () =>
  api.get('/reports/download/csv', { responseType: 'blob', timeout: 180_000 })
export const downloadPDF = () =>
  api.get('/reports/download/pdf', { responseType: 'blob', timeout: 180_000 })

export default api
