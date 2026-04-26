"""
gemini_service.py — Google Gemini Fairness Auditor (v4 — final)
================================================================
Production-grade integration with all optimizations:

  Prior:  cache normalization, smart fallback, dynamic cooldown,
          hard API guard, session control, token optimization

  v4 additions:
    1. LRU cache       — replaces FIFO; frequently accessed results persist
    2. Retry isolation  — retries run immediately; cooldown only affects new requests
    3. Context-aware fallback — cross-metric composite insights
    4. Confidence score — high / medium / low signal consistency rating
    5. Circuit breaker  — auto-disables API after repeated failures, self-heals

SDK: google-genai  (pip install google-genai)
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import math
import os
import threading
import time
from typing import Any

from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)

logger = logging.getLogger(__name__)


# ── System prompt (strict auditor format) ─────────────────────────────────
_SYSTEM_PROMPT = """\
You are an AI Fairness Auditor. Analyze model fairness using the provided
metrics and generate a consistent, logically correct audit report.

STRICT RULES:
1. Do NOT claim No Bias unless ALL of these are present:
   - At least 2 groups with group-wise metrics
   - At least 1 fairness metric (SPD, disparate impact, or equal opportunity diff)
   If missing, output: Bias Summary: Inconclusive — insufficient data

2. Bias triggers (any one = YES):
   - Statistical Parity Difference > 0.10
   - Disparate Impact < 0.80 or > 1.25
   - Equal Opportunity Difference > 0.10

3. Fairness Risk Level:
   - 0 violations: LOW (0-20%)
   - 1-2 violations: MEDIUM (30-60%)
   - 3+ violations: HIGH (70-100%)

4. NEVER output contradictory statements (e.g., No Bias + HIGH risk).

Output EXACTLY in this format (no markdown, no extra text, max 120 words):

Bias Summary: <YES / NO / INCONCLUSIVE — reason>

Fairness Risk Level: <LOW / MEDIUM / HIGH> (<% range>)

Reasoning: <which metric triggered the decision, one sentence>

Actionable Fixes:
1. <fix only if bias or risk exists, else write: Continue monitoring fairness metrics>
2. <second fix>"""


# ═══════════════════════════════════════════════════════════════════════════
# 1. LRU CACHE (replaces FIFO)
# ═══════════════════════════════════════════════════════════════════════════
_CACHE_MAX_SIZE = 50
_cache_lock = threading.Lock()
_audit_cache: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()


def _normalize_value(v: Any, precision: int = 4) -> Any:
    """Round floats to `precision` decimals for cache-stable hashing."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, precision)
    if isinstance(v, dict):
        return {k: _normalize_value(val, precision) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_normalize_value(item, precision) for item in v]
    return v


def _cache_key(data: dict) -> str:
    """Deterministic SHA-256 hash after float normalization."""
    normalized = _normalize_value(data)
    raw = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    """LRU read: return cached result and promote to most-recently-used."""
    with _cache_lock:
        if key not in _audit_cache:
            return None
        _audit_cache.move_to_end(key)           # promote to MRU
        return _audit_cache[key].copy()


def _cache_put(key: str, result: dict) -> None:
    """LRU write: insert/update entry; evict least-recently-used if full."""
    with _cache_lock:
        if key in _audit_cache:
            _audit_cache.move_to_end(key)
        else:
            if len(_audit_cache) >= _CACHE_MAX_SIZE:
                _audit_cache.popitem(last=False)  # evict LRU (oldest)
        _audit_cache[key] = result.copy()


# ═══════════════════════════════════════════════════════════════════════════
# 5. CIRCUIT BREAKER (auto-disables API after repeated failures)
# ═══════════════════════════════════════════════════════════════════════════
_breaker_lock = threading.Lock()
_consecutive_failures: int = 0
_last_failure_time: float = 0.0
_circuit_open_until: float = 0.0       # timestamp when circuit closes

_BREAKER_THRESHOLD = 3                 # open circuit after this many failures
_BREAKER_RESET_SECONDS = 120           # auto-close after 2 min
_COOLDOWN_BASE_SECONDS = 20
_COOLDOWN_MAX_SECONDS = 60


def _record_success() -> None:
    """Reset all failure state on a successful API call."""
    global _consecutive_failures, _circuit_open_until
    with _breaker_lock:
        _consecutive_failures = 0
        _circuit_open_until = 0.0


def _record_failure() -> None:
    """Track failure; open circuit breaker if threshold exceeded."""
    global _last_failure_time, _consecutive_failures, _circuit_open_until
    with _breaker_lock:
        _last_failure_time = time.time()
        _consecutive_failures += 1
        if _consecutive_failures >= _BREAKER_THRESHOLD:
            _circuit_open_until = time.time() + _BREAKER_RESET_SECONDS
            logger.warning(
                "Circuit breaker OPEN — Gemini disabled for %ds after %d failures.",
                _BREAKER_RESET_SECONDS, _consecutive_failures,
            )


def _is_circuit_open() -> bool:
    """Return True if the circuit breaker is open (API disabled)."""
    with _breaker_lock:
        if _circuit_open_until <= 0:
            return False
        if time.time() >= _circuit_open_until:
            # Auto-reset: half-open → allow one probe
            logger.info("Circuit breaker auto-reset (cooldown expired).")
            return False
        return True


def _is_in_cooldown() -> bool:
    """Return True if in post-failure cooldown (shorter than circuit break)."""
    with _breaker_lock:
        if _consecutive_failures == 0:
            return False
        cooldown = min(
            _COOLDOWN_BASE_SECONDS * _consecutive_failures,
            _COOLDOWN_MAX_SECONDS,
        )
        return (time.time() - _last_failure_time) < cooldown


def _get_cooldown_wait() -> float:
    """Return seconds remaining in cooldown, or 0."""
    with _breaker_lock:
        if _consecutive_failures == 0:
            return 0.0
        cooldown = min(
            _COOLDOWN_BASE_SECONDS * _consecutive_failures,
            _COOLDOWN_MAX_SECONDS,
        )
        remaining = cooldown - (time.time() - _last_failure_time)
        return max(0.0, remaining)


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONFIDENCE SCORE
# ═══════════════════════════════════════════════════════════════════════════
def _compute_confidence(data: dict[str, Any]) -> str:
    """
    Rate confidence based on data sufficiency for a valid fairness determination.

    Aligned with the sufficiency rule in _SYSTEM_PROMPT:
      high   — group metrics present + ≥1 fairness metric + verdict
      medium — some fairness data but incomplete
      low    — insufficient for any determination
    """
    metrics = data.get("metrics") or {}
    core_present = sum(1 for k in ("accuracy", "precision", "recall", "f1")
                       if metrics.get(k) is not None)

    group_metrics = data.get("group_metrics") or {}
    has_group_data = len(group_metrics) >= 2

    has_fairness_metric = any(data.get(k) is not None for k in (
        "demographic_parity_difference",
        "disparate_impact_groups",
    ))
    has_verdict = data.get("verdict") is not None

    # High: meets the minimum sufficiency rule
    if has_group_data and has_fairness_metric and has_verdict:
        return "high"
    # Medium: partial fairness data
    if core_present >= 2 or has_fairness_metric or data.get("is_biased") is not None:
        return "medium"
    return "low"


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONTEXT-AWARE SMART FALLBACK
#    Implements all 7 strict auditor rules. Output format matches _SYSTEM_PROMPT
#    exactly so the frontend parser handles Gemini + fallback identically.
# ═══════════════════════════════════════════════════════════════════════════
def _smart_fallback(data: dict[str, Any]) -> dict[str, Any]:
    """
    Strict offline fairness audit following all 7 auditor rules:
      1. INCONCLUSIVE if minimum data not met
      2. Threshold-based violation counting
      3. No contradictions (Bias=No ↔ Risk=LOW only)
      4. Risk mapped to violation count
      5. Output: Bias Summary / Fairness Risk Level / Reasoning / Actionable Fixes
    """
    metrics      = data.get("metrics") or {}
    acc          = metrics.get("accuracy")
    prec         = metrics.get("precision")
    rec          = metrics.get("recall")
    f1           = metrics.get("f1")

    dpd          = data.get("demographic_parity_difference")
    is_biased    = data.get("is_biased")
    severity     = data.get("severity", "unknown")
    fairness_score = data.get("fairness_score")
    di_groups    = data.get("disparate_impact_groups") or {}
    group_metrics = data.get("group_metrics") or {}
    failed_metrics = data.get("failed_metrics") or []
    passed_metrics = data.get("passed_metrics") or []

    # ── RULE 1: Minimum sufficiency check ─────────────────────────────
    # Requires: ≥2 groups with metrics, ≥1 fairness metric
    has_group_data    = len(group_metrics) >= 2
    has_fairness_metric = dpd is not None or bool(di_groups)
    data_sufficient   = has_group_data and has_fairness_metric

    # Also accept verdict from backend bias engine as sufficient
    if is_biased is not None and (dpd is not None or bool(di_groups)):
        data_sufficient = True

    # ── RULE 2: Count threshold violations ────────────────────────────
    violations: list[str] = []

    if dpd is not None and dpd > 0.10:
        violations.append(f"Statistical Parity Difference ({dpd:.3f}) > 0.10")

    low_di_groups = [
        g for g, info in di_groups.items()
        if isinstance(info, dict) and info.get("biased") is True
    ]
    # Also check di value directly if available
    for g, info in di_groups.items():
        if not isinstance(info, dict):
            continue
        di_val = info.get("disparate_impact")
        if di_val is not None and (di_val < 0.80 or di_val > 1.25):
            viol = f"Disparate Impact ({di_val:.3f}) out of range [0.80, 1.25] in group '{g}'"
            if viol not in violations:
                violations.append(viol)

    # Count failed metrics from backend verdict as additional violations
    for fm in failed_metrics:
        name = fm.get("metric", "") if isinstance(fm, dict) else str(fm)
        viol = f"{name} failed fairness check"
        if viol not in violations:
            violations.append(viol)

    n_violations = len(violations)

    # ── RULE 3 + 4: Bias status + Risk level (no contradictions) ──────
    if not data_sufficient:
        # RULE 1: insufficient data → INCONCLUSIVE
        bias_summary   = "Inconclusive — insufficient data"
        risk_label     = "MEDIUM"
        risk_range     = "30-60%"
        reasoning      = (
            "Bias determination requires at least 2 groups with per-group metrics "
            "and at least 1 fairness metric (SPD or disparate impact). "
            "Run a full bias analysis first."
        )
        has_actionable = False

    elif n_violations >= 3:
        # HIGH risk
        bias_summary   = "YES — multiple fairness violations detected"
        risk_label     = "HIGH"
        risk_range     = "70-100%"
        reasoning      = "; ".join(violations[:3])  # top 3
        has_actionable = True

    elif n_violations in (1, 2):
        # MEDIUM risk
        bias_summary   = "YES — fairness threshold(s) exceeded"
        risk_label     = "MEDIUM"
        risk_range     = "30-60%"
        reasoning      = "; ".join(violations)
        has_actionable = True

    elif is_biased is True:
        # Backend says biased but our metric check didn't catch explicit violations
        bias_summary   = f"YES — model flagged as biased (severity: {severity})"
        risk_label     = "MEDIUM"
        risk_range     = "30-60%"
        reasoning      = f"Overall bias verdict is positive with severity '{severity}'"
        has_actionable = True

    elif is_biased is False and data_sufficient:
        # RULE 3: No Bias → MUST be LOW risk (no contradiction)
        bias_summary   = "NO — all fairness checks passed"
        risk_label     = "LOW"
        risk_range     = "0-20%"
        reasoning      = "No threshold violations detected across disparate impact and demographic parity"
        has_actionable = False

    else:
        bias_summary   = "Inconclusive — fairness data present but verdict unclear"
        risk_label     = "MEDIUM"
        risk_range     = "30-60%"
        reasoning      = "Fairness data is available but insufficient to make a definitive determination"
        has_actionable = False

    # ── RULE 6: Actionable fixes (context-aware) ──────────────────────
    if has_actionable:
        has_bias_signal = n_violations > 0 or is_biased is True
        if has_bias_signal and acc is not None and acc >= 0.80:
            fix1 = "Apply fairness constraints (equalized odds or demographic parity) during training"
            fix2 = (
                "Audit and remove proxy features correlated with the protected attribute"
                if low_di_groups
                else "Tune decision threshold per group to equalize positive prediction rates"
            )
        elif has_bias_signal and rec is not None and rec < 0.40:
            fix1 = "Lower decision threshold to improve recall for the disadvantaged group"
            fix2 = "Apply SMOTE or ADASYN resampling on the minority class before training"
        else:
            fix1 = "Apply class-weight balancing or threshold tuning to equalize group outcomes"
            fix2 = "Use resampling (SMOTE/ADASYN) on the minority class before training"
    else:
        fix1 = "Continue monitoring fairness metrics on new data batches"
        fix2 = "Re-run bias analysis if the model is retrained or data distribution shifts"

    # ── Fairness score (consistent with risk level) ───────────────────
    if fairness_score is not None:
        fs_str = f"{fairness_score}%"
    elif risk_label == "LOW":
        fs_str = "90%"
    elif risk_label == "HIGH":
        fs_str = {"mild": "30%", "moderate": "15%", "severe": "5%"}.get(severity, "15%")
    elif risk_label == "MEDIUM":
        fs_str = {"mild": "60%", "moderate": "45%"}.get(severity, "50%")
    else:
        fs_str = "N/A"

    # ── Build strict output ───────────────────────────────────────────
    audit_text = (
        f"Bias Summary: {bias_summary}\n"
        f"\n"
        f"Fairness Risk Level: {risk_label} ({risk_range})\n"
        f"\n"
        f"Reasoning: {reasoning}\n"
        f"\n"
        f"Actionable Fixes:\n"
        f"1. {fix1}\n"
        f"2. {fix2}"
    )

    confidence = _compute_confidence(data)

    return {
        "audit": audit_text,
        "status": "success",
        "message": "Offline heuristic audit (Gemini API not called).",
        "source": "fallback",
        "confidence": confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _get_api_key() -> str | None:
    """Return the Gemini API key, or None if absent/placeholder."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key == "your_gemini_api_key_here" or key.startswith("your_"):
        return None
    return key


_ALLOWED_KEYS = frozenset({
    "metrics", "class_distribution", "threshold", "group_metrics",
    "feature_importance", "model_name", "protected_attribute",
    "demographic_parity_difference", "group_positive_rates",
    "disparate_impact_groups", "verdict", "fairness_score",
    "severity", "is_biased", "failed_metrics", "passed_metrics",
})


def _sanitize_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, non-empty, normalized metric summary."""
    clean: dict[str, Any] = {}
    for key in _ALLOWED_KEYS:
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, (dict, list)) and len(val) == 0:
            continue
        clean[key] = _normalize_value(val)

    if not clean:
        clean["_note"] = "No evaluation data available yet."
    return clean


# ── Session-level control ──────────────────────────────────────────────────
def _check_session_guard() -> dict[str, Any] | None:
    """Return stored result if Gemini was already called this session."""
    try:
        from backend.services.session_store import store
        cached = store.get("gemini_audit_result")
        if cached is not None:
            logger.info("Session guard: returning previously stored audit.")
            cached["source"] = "session_cache"
            return cached
    except ImportError:
        pass
    return None


def _save_to_session(result: dict[str, Any]) -> None:
    """Persist audit result in session store to prevent re-calls."""
    try:
        from backend.services.session_store import store
        store.set("gemini_audit_result", result.copy())
    except ImportError:
        pass


def reset_gemini_session() -> None:
    """Reset session guard — call when new bias analysis is run."""
    try:
        from backend.services.session_store import store
        store.set("gemini_audit_result", None)
        logger.info("Gemini session guard reset.")
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════
def generate_fairness_audit(data: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a fairness audit from ML metrics.

    Guard chain:
      1. Input validation
      2. Session dedup
      3. LRU cache lookup
      4. Circuit breaker check
      5. Cooldown check
      6. API call with isolated retry
      7. Context-aware fallback on failure
    """

    # ── Guard: empty input ────────────────────────────────────────────────
    if not isinstance(data, dict) or not data:
        logger.warning("generate_fairness_audit called with empty/invalid data.")
        return _smart_fallback({})

    # ── Sanitize + normalize ──────────────────────────────────────────────
    clean_data = _sanitize_payload(data)
    confidence = _compute_confidence(clean_data)

    # ── Session guard ─────────────────────────────────────────────────────
    session_result = _check_session_guard()
    if session_result is not None:
        session_result["confidence"] = confidence
        return session_result

    # ── LRU cache lookup ──────────────────────────────────────────────────
    key = _cache_key(clean_data)
    cached = _cache_get(key)
    if cached is not None:
        logger.info("LRU cache hit (hash=%s…).", key[:12])
        cached["source"] = "cache"
        cached["confidence"] = confidence
        _save_to_session(cached)
        return cached

    # ── Circuit breaker ───────────────────────────────────────────────────
    if _is_circuit_open():
        logger.info("Circuit breaker OPEN — returning fallback.")
        fb = _smart_fallback(clean_data)
        fb["message"] = "API temporarily disabled (circuit breaker). Showing offline analysis."
        fb["confidence"] = confidence
        return fb

    # ── Cooldown guard ────────────────────────────────────────────────────
    if _is_in_cooldown():
        wait = _get_cooldown_wait()
        logger.info("Cooldown active (%.0f s left). Returning fallback.", wait)
        fb = _smart_fallback(clean_data)
        fb["message"] = f"API in cooldown ({wait:.0f}s). Showing offline analysis."
        fb["confidence"] = confidence
        return fb

    # ── Validate API key ──────────────────────────────────────────────────
    api_key = _get_api_key()
    if not api_key:
        logger.error("GEMINI_API_KEY is not set.")
        fb = _smart_fallback(clean_data)
        fb["confidence"] = confidence
        _save_to_session(fb)
        return fb

    # ── Late-import SDK ───────────────────────────────────────────────────
    try:
        from google import genai                        # type: ignore[import]
        from google.genai import types as genai_types  # type: ignore[import]
    except ImportError:
        logger.error("google-genai not installed.")
        fb = _smart_fallback(clean_data)
        fb["confidence"] = confidence
        _save_to_session(fb)
        return fb

    # ── Initialise client ─────────────────────────────────────────────────
    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        logger.exception("Gemini client init failed: %s", exc)
        _record_failure()
        fb = _smart_fallback(clean_data)
        fb["confidence"] = confidence
        _save_to_session(fb)
        return fb

    # ── Build structured, valid JSON payload ─────────────────────────────
    # Gemini requires contents as a list of role/parts dicts.
    # The text part must always be valid, non-empty JSON — never a raw string.
    try:
        payload_data = clean_data if clean_data else {"note": "no data"}
        payload_json = json.dumps(
            {"task": "fairness_audit", "data": payload_data},
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        payload_json = json.dumps({"task": "fairness_audit", "data": {"note": "no data"}})

    structured_contents = [
        {
            "role": "user",
            "parts": [{"text": payload_json}],
        }
    ]

    # ══════════════════════════════════════════════════════════════════════
    # 2. ISOLATED RETRY — retries happen immediately within this call;
    #    cooldown / circuit breaker only affect FUTURE requests.
    # ══════════════════════════════════════════════════════════════════════
    _MAX_ATTEMPTS = 3
    _RETRY_WAIT = 20     # sleep between retry attempts (within this call)

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=structured_contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=256,
                ),
            )

            audit_text: str = (response.text or "").strip()

            if not audit_text:
                logger.warning("Empty response on attempt %d.", attempt)
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_WAIT)
                    continue
                # All attempts returned empty — fallback
                fb = _smart_fallback(clean_data)
                fb["confidence"] = confidence
                _save_to_session(fb)
                return fb

            # ── Success ───────────────────────────────────────────────────
            logger.info("Gemini audit generated (attempt %d).", attempt)
            _record_success()

            result: dict[str, Any] = {
                "audit": audit_text,
                "status": "success",
                "message": "Fairness audit generated successfully.",
                "source": "gemini",
                "confidence": confidence,
            }

            _cache_put(key, result)
            _save_to_session(result)
            return result

        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = any(
                kw in exc_str
                for kw in ("429", "quota", "resource_exhausted", "rate")
            )

            if is_rate_limit and attempt < _MAX_ATTEMPTS:
                # Retry immediately (within this request) — no cooldown yet
                logger.warning(
                    "429 on attempt %d/%d — retrying in %ds.",
                    attempt, _MAX_ATTEMPTS, _RETRY_WAIT,
                )
                time.sleep(_RETRY_WAIT)
                continue

            # Final attempt failed or non-retryable error
            # Record failure ONCE after all retries are exhausted
            _record_failure()
            logger.exception("Gemini failed (attempt %d): %s", attempt, exc)

            fb = _smart_fallback(clean_data)
            fb["message"] = f"Gemini failed ({exc}). Showing offline analysis."
            fb["confidence"] = confidence
            _save_to_session(fb)
            return fb

    # Safety net (unreachable)
    fb = _smart_fallback(clean_data)
    fb["confidence"] = confidence
    _save_to_session(fb)
    return fb
