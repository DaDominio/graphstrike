"""Platform policy compiler - extracts policy parameters from transparency reports.

Round 2: Dynamically derives detection thresholds from real platform policies.

Formula:
    θ_raw = (C_fn · π) / (C_fn · π + C_fp · (1 − π))
    θ*    = clamp(θ_raw / harm_weight, 0.01, 0.95)
    fp_penalty_weight = C_fp

    Action rule the threshold serves: "FLAG if risk_score ≥ θ*".
    θ_raw is the fraction of expected cost coming from missed fakes; higher
    C_fn or higher base_rate → lower threshold (flag more aggressively).
    harm_weight > 1 strict (lowers θ*); < 1 lenient (raises θ*).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
import argparse
import json
import re
import time
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from tavily import TavilyClient
    import requests
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("Warning: tavily-python not installed, using cached policies only")

# Import PlatformPolicy model from parent directory
import sys
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import importlib.util
from typing import List as TypingList

spec = importlib.util.spec_from_file_location("local_models", parent_dir / "models.py")
if spec and spec.loader:
    models_module = importlib.util.module_from_spec(spec)
    sys.modules["local_models"] = models_module
    spec.loader.exec_module(models_module)
    PlatformPolicy = models_module.PlatformPolicy
    PlatformPolicy.model_rebuild(_types_namespace={"List": TypingList})
else:
    raise ImportError("Failed to load models module")

# ================= CONFIG =================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

K = 6
MAX_CONTEXT_CHARS = 6000
RETRIES = 3
CACHE_TTL_DAYS = 30

CACHE_DIR = Path(__file__).parent.parent / "policy_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Generic default used when a platform has no hardcoded fallback.
GENERIC_FALLBACK = {
    "threshold": 0.35,
    "base_rate": 0.005,
    "fn_cost_signal": "high",
    "fp_cost_signal": "medium",
    "harm_weight": 1.0,
    "primary_enforcement_signal": "photo_reuse",
    "fp_penalty_weight": 0.5,
    "sources": [],
    "confidence": 0.5,
}

FALLBACK_POLICIES = {
    "Instagram": {
        "base_rate": 0.0017,
        "fn_cost_signal": "critical",
        "fp_cost_signal": "medium",
        "harm_weight": 1.2,
        "primary_enforcement_signal": "photo_reuse",
        "sources": ["https://transparency.meta.com/reports/"],
        "confidence": 0.85,
    },
    "Snapchat": {
        "base_rate": 0.008,
        "fn_cost_signal": "high",
        "fp_cost_signal": "low",
        "harm_weight": 0.7,
        "primary_enforcement_signal": "bio_template",
        "sources": ["https://snap.com/en-US/safety/"],
        "confidence": 0.82,
    },
}
# ==========================================

if TAVILY_AVAILABLE:
    tavily = TavilyClient(TAVILY_API_KEY)

# ================= EXTRACTION PROMPT =================
EXTRACTION_PROMPT = """
You are a policy analyst. Read the platform policy excerpt below and extract
exactly these parameters as a JSON object.

Parameters to extract:

1. base_rate (float 0.0–1.0) - The fraction of accounts on the platform that are fake
   impersonators at any given time (PREVALENCE, NOT enforcement rate).

   IMPORTANT DISAMBIGUATION:
   - "X% of accounts were removed", "X% were actioned", "X% impacted by enforcement"
     → that is ENFORCEMENT RATE. Do NOT use as base_rate.
   - "X% of accounts are estimated inauthentic", "X% confirmed fake on investigation"
     → closer to base_rate. Use cautiously.
   - If no prevalence estimate is found, return 0.005 as a safe default.
   - Typical real-world fake-account prevalence is 0.1% to 3%.
     If your extracted value exceeds 5%, it is almost certainly an enforcement
     rate. Return 0.005 instead.
2. fn_cost_signal ("low" | "medium" | "high" | "critical") - How bad is it to miss a fake account?
3. fp_cost_signal ("low" | "medium" | "high") - How bad is it to wrongly remove a real account?
4. harm_weight (float 0.5–2.0) - Does platform prioritize enforcement (>1.0) or creator protection (<1.0)?
5. primary_enforcement_signal (string) - Which signal does platform mention most: "photo_reuse", "bio_template", or "ip_cluster"?
6. policy_confidence (float 0.0–1.0) - How confident are you in these extractions?

Return ONLY valid JSON, no explanation.

Policy text:
{policy_text}
"""
# ==========================================


# =============== FETCH ===================
def fetch_contents(platform: str) -> tuple[list[str], list[str]]:
    """Fetch policy documents via Tavily search (generic, platform-agnostic)."""
    if not TAVILY_AVAILABLE:
        return [], []

    query = f"{platform} fake account content policy enforcement 2024 2025"

    try:
        res = tavily.search(query=query, search_depth="advanced", max_results=25)

        contents, sources = [], []
        for r in res.get("results", []):
            url = r.get("url", "")
            content = r.get("content")
            if isinstance(content, str) and len(content) > 200:
                contents.append(content.strip())
                sources.append(url)
            if len(contents) >= K:
                break
        return contents, sources
    except Exception as e:
        print(f"Tavily fetch error: {e}")
        return [], []


# =============== UTIL ===================
def build_context(contents: list[str]) -> str:
    return ("\n---\n".join(contents))[:MAX_CONTEXT_CHARS]


def clean_json(text: str) -> str:
    text = re.sub(r"```json|```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def call_groq(prompt: str) -> dict:
    if not TAVILY_AVAILABLE:
        return {}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    for _ in range(RETRIES):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            out = r.json()
            content = clean_json(out["choices"][0]["message"]["content"])
            return json.loads(content)
        except Exception as e:
            print(f"Groq API error (retrying): {e}")
            time.sleep(1)

    return {}


# =============== LOGIC ===================
FN_COST_MAP = {"low": 0.5, "medium": 1.0, "high": 2.0, "critical": 4.0}
FP_COST_MAP = {"low": 0.1, "medium": 0.5, "high": 1.5}


def sanitize_pi(pi: Any, platform: str = "?") -> float:
    # Fake-account PREVALENCE (base_rate) is typically 0.1–3%.
    # Enforcement/removal RATES are typically 5–30%.
    # The LLM often confuses the two when parsing transparency reports.
    # Upper clamp of 0.05 guards against this extraction artifact.
    # Lower clamp of 0.0005 avoids division instability near zero.
    if isinstance(pi, (int, float)):
        if pi <= 0:
            return 0.002
        raw = float(pi)
        if raw > 0.05:
            print(
                f"⚠ [{platform}] Extracted base_rate {raw:.4f} exceeds 0.05 — "
                f"likely an enforcement rate misread as prevalence. "
                f"Clamped to 0.05. Check extraction sources."
            )
        return max(0.0005, min(raw, 0.05))
    return 0.002


def compute_threshold(pi: float, fn_signal: str, fp_signal: str, harm_weight: float) -> tuple[float, float]:
    """Bayesian threshold for the action rule "FLAG if risk_score ≥ θ*".

    θ_raw = C_fn·π / [C_fn·π + C_fp·(1−π)]

    Interpretation: fraction of expected cost that comes from missed fakes.
    Higher C_fn or higher base_rate → higher θ_raw → lower threshold for
    our action rule "FLAG if risk_score ≥ θ*" because we are more willing
    to flag when misses are expensive.

    harm_weight > 1 means stricter enforcement → dividing lowers θ* further.
    harm_weight < 1 means lenient enforcement → dividing raises θ*.

    θ* = clamp(θ_raw / harm_weight, 0.01, 0.95)

    Returns (theta_star, fp_penalty_weight=C_fp).
    """
    C_fn = FN_COST_MAP.get(fn_signal, FN_COST_MAP["high"])
    C_fp = FP_COST_MAP.get(fp_signal, FP_COST_MAP["medium"])

    num = C_fn * pi
    den = C_fn * pi + C_fp * (1.0 - pi)
    theta_raw = num / den if den > 0 else 0.5

    hw = harm_weight if isinstance(harm_weight, (int, float)) and harm_weight > 0 else 1.0
    theta = theta_raw / hw
    theta = max(0.01, min(theta, 0.95))

    return theta, C_fp


# =============== SANITY CHECK ===================
def sanity_check_policy(policy: PlatformPolicy) -> list[str]:
    """Per-platform sanity warnings. Does NOT block compilation; surfaces issues
    so the operator can decide whether the policy is fit for evaluation.

    No cross-platform ordering is enforced — any platform can land anywhere on
    the [0.01, 0.95] θ* scale depending on its actual policy.
    """
    warnings: list[str] = []

    # 1. Threshold range.
    # Env's risk_score is typically 0.5–0.95 for true fakes and 0.1–0.4 for decoys.
    # θ* > 0.90 → almost nothing flagged. θ* < 0.005 → everything flagged.
    if policy.threshold > 0.90:
        warnings.append(
            f"threshold={policy.threshold:.3f} is very high — "
            f"agent will almost never flag. Check fn_cost extraction."
        )
    if policy.threshold < 0.005:
        warnings.append(
            f"threshold={policy.threshold:.3f} is very low — "
            f"agent will flag nearly everything. Check fp_cost extraction."
        )

    # 2. Base rate plausibility.
    if policy.base_rate > 0.05:
        warnings.append(
            f"base_rate={policy.base_rate:.4f} exceeds 5% — "
            f"likely an enforcement rate misread. Expected 0.001–0.03."
        )

    # 3. Confidence floor.
    if policy.confidence < 0.60:
        warnings.append(
            f"confidence={policy.confidence:.2f} is below 0.60 — "
            f"extraction quality is low. Used Fallback should be True."
        )

    # 4. Primary signal must be one of the known tool actions.
    valid_signals = {"photo_reuse", "bio_template", "ip_cluster", "behavior"}
    if policy.primary_enforcement_signal not in valid_signals:
        warnings.append(
            f"primary_signal='{policy.primary_enforcement_signal}' "
            f"is not a valid tool action. Defaulting to photo_reuse."
        )

    return warnings


def _print_sanity(policy: PlatformPolicy) -> None:
    warnings = sanity_check_policy(policy)
    if not warnings:
        print(f"✓ sanity_check[{policy.platform}]: 0 warnings — ready to use")
        return
    print(f"⚠ sanity_check[{policy.platform}]: {len(warnings)} warning(s) — review before eval")
    for w in warnings:
        print(f"   - {w}")


# =============== CACHING ===================
def _is_fresh(compiled_at: str) -> bool:
    if not compiled_at:
        return False
    try:
        ts = datetime.fromisoformat(compiled_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        return age < timedelta(days=CACHE_TTL_DAYS)
    except Exception:
        return False


def load_cached_policy(platform: str) -> Optional[PlatformPolicy]:
    cache_file = CACHE_DIR / f"{platform.lower()}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        if not _is_fresh(data.get("compiled_at", "")):
            print(f"⚠ Cached policy for {platform} is stale (>{CACHE_TTL_DAYS}d), recompiling")
            return None
        return PlatformPolicy(**data)
    except Exception as e:
        print(f"Failed to load cached policy for {platform}: {e}")
        return None


def save_policy_to_cache(policy: PlatformPolicy) -> None:
    cache_file = CACHE_DIR / f"{policy.platform.lower()}.json"
    try:
        cache_file.write_text(json.dumps(policy.model_dump(), indent=2))
        print(f"✓ Cached policy for {policy.platform} to {cache_file}")
    except Exception as e:
        print(f"Failed to cache policy: {e}")


# =============== FALLBACK BUILDER ===================
def build_fallback_policy(platform: str) -> PlatformPolicy:
    """Construct a fallback PlatformPolicy using hardcoded data (if known) or generic defaults."""
    data = FALLBACK_POLICIES.get(platform, {})
    merged = {**GENERIC_FALLBACK, **data}

    pi = sanitize_pi(merged["base_rate"], platform)
    theta, C_fp = compute_threshold(
        pi,
        merged["fn_cost_signal"],
        merged["fp_cost_signal"],
        merged["harm_weight"],
    )

    return PlatformPolicy(
        platform=platform,
        threshold=theta,
        base_rate=pi,
        fn_cost_signal=merged["fn_cost_signal"],
        fp_cost_signal=merged["fp_cost_signal"],
        harm_weight=merged["harm_weight"],
        primary_enforcement_signal=merged["primary_enforcement_signal"],
        fp_penalty_weight=C_fp,
        sources=list(merged.get("sources", [])),
        confidence=merged.get("confidence", 0.5),
        compiled_at=datetime.now(timezone.utc).isoformat(),
        used_fallback=True,
    )


# =============== MAIN PIPELINE ===================
def compile_policy(platform: str, use_cache: bool = True) -> PlatformPolicy:
    """Compile a platform policy. Works for ANY platform name."""

    if use_cache:
        cached = load_cached_policy(platform)
        if cached:
            print(f"✓ Using cached policy for {platform} (threshold={cached.threshold:.3f})")
            return cached

    if TAVILY_AVAILABLE:
        print(f"🔍 Compiling policy for {platform} from live sources...")
        contents, sources = fetch_contents(platform)

        if contents:
            context = build_context(contents)
            extracted = call_groq(EXTRACTION_PROMPT.replace("{policy_text}", context))

            if extracted:
                pi = sanitize_pi(extracted.get("base_rate"), platform)
                fn_signal = extracted.get("fn_cost_signal", "high")
                fp_signal = extracted.get("fp_cost_signal", "medium")
                weight = extracted.get("harm_weight", 1.0)
                confidence = extracted.get("policy_confidence", 0.0)
                primary_signal = extracted.get("primary_enforcement_signal") or "photo_reuse"
                if not isinstance(primary_signal, str) or not primary_signal.strip():
                    primary_signal = "photo_reuse"

                if fn_signal not in FN_COST_MAP:
                    fn_signal = "high"
                if fp_signal not in FP_COST_MAP:
                    fp_signal = "medium"
                if not isinstance(weight, (int, float)):
                    weight = 1.0
                if not isinstance(confidence, (int, float)):
                    confidence = 0.0

                theta, C_fp = compute_threshold(pi, fn_signal, fp_signal, weight)

                policy = PlatformPolicy(
                    platform=platform,
                    threshold=theta,
                    base_rate=pi,
                    fn_cost_signal=fn_signal,
                    fp_cost_signal=fp_signal,
                    harm_weight=weight,
                    primary_enforcement_signal=primary_signal,
                    fp_penalty_weight=C_fp,
                    sources=sources,
                    confidence=confidence,
                    compiled_at=datetime.now(timezone.utc).isoformat(),
                    used_fallback=False,
                )
                save_policy_to_cache(policy)
                print(f"✓ Compiled {platform} policy: threshold={policy.threshold:.3f}, primary_signal={policy.primary_enforcement_signal}")
                _print_sanity(policy)
                return policy
            else:
                print(f"⚠ Extraction failed for {platform}, using fallback")
        else:
            print(f"⚠ No sources found for {platform}, using fallback")

    print(f"⚠ Using fallback policy for {platform}")
    policy = build_fallback_policy(platform)
    save_policy_to_cache(policy)
    _print_sanity(policy)
    return policy


def get_policy(platform: str) -> PlatformPolicy:
    return compile_policy(platform, use_cache=True)


# =============== CLI ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile platform policy from transparency reports")
    parser.add_argument("--platform", "-p", default="Instagram", help="Platform name (any)")
    parser.add_argument("--use-cache", action="store_true", help="Reuse cached policy if fresh")
    args = parser.parse_args()

    platform = args.platform

    print(f"\n{'='*60}")
    print(f"Policy Compiler - {platform}")
    print(f"{'='*60}\n")

    policy = compile_policy(platform, use_cache=args.use_cache)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Platform: {policy.platform}")
    print(f"Threshold (θ*): {policy.threshold:.4f}")
    print(f"Base Rate (π): {policy.base_rate:.4%}")
    print(f"FN Cost Signal: {policy.fn_cost_signal}")
    print(f"FP Cost Signal: {policy.fp_cost_signal}")
    print(f"FP Penalty Weight (C_fp): {policy.fp_penalty_weight}")
    print(f"Harm Weight: {policy.harm_weight}")
    print(f"Primary Signal: {policy.primary_enforcement_signal}")
    print(f"Confidence: {policy.confidence:.2%}")
    print(f"Used Fallback: {policy.used_fallback}")
    print(f"Sources: {len(policy.sources)} URLs")
    for i, src in enumerate(policy.sources[:3], 1):
        print(f"  {i}. {src}")
    print(f"{'='*60}\n")
