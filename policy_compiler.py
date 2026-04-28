# pip install tavily-python requests

from tavily import TavilyClient
import requests
import json
import re
import time
import os
from dotenv import load_dotenv
load_dotenv()

# ================= CONFIG =================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

K = 6
MAX_CONTEXT_CHARS = 6000
RETRIES = 3
# ==========================================

tavily = TavilyClient(TAVILY_API_KEY)

# ================= YOUR PROMPT =================
EXTRACTION_PROMPT = """
You are a policy analyst. Read the platform policy excerpt below and extract 
exactly these parameters as a JSON object.

Parameters to extract:

1. base_rate (float 0.0–1.0)
2. fn_cost_signal ("low" | "medium" | "high" | "critical")
3. fp_cost_signal ("low" | "medium" | "high")
4. harm_weight (float 0.5–2.0)
5. primary_enforcement_signal (string)
6. policy_confidence (float 0.0–1.0)

Return ONLY valid JSON, no explanation.

Policy text:
{policy_text}
"""
# ==========================================


# =============== SOURCE FILTER ===================
def is_high_signal_source(url):
    allow_domains = [
        "meta.com",
        "transparency.meta.com",
        "about.meta.com",
        "help.instagram.com",
        "instagram.com/help",
        "instagram.com/legal"
    ]

    deny_patterns = [
        "blog", "how-to", "guide",
        "report-fake", "remove",
        "youtube", "tiktok", "reel", "/p/"
    ]

    if any(d in url for d in deny_patterns):
        return False

    return any(domain in url for domain in allow_domains)


# =============== FETCH ===================
def fetch_contents(query):
    res = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=25
    )

    contents, sources = [], []

    for r in res.get("results", []):
        url = r.get("url", "")
        content = r.get("content")

        if not is_high_signal_source(url):
            continue

        if isinstance(content, str) and len(content) > 200:
            contents.append(content.strip())
            sources.append(url)

        if len(contents) >= K:
            break

    return contents, sources


# =============== UTIL ===================
def build_context(contents):
    return ("\n---\n".join(contents))[:MAX_CONTEXT_CHARS]


def clean_json(text):
    text = re.sub(r"```json|```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    for _ in range(RETRIES):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            out = r.json()

            print("RAW:", out)

            content = out["choices"][0]["message"]["content"]
            content = clean_json(content)

            return json.loads(content)

        except Exception as e:
            print("Retry:", e)
            time.sleep(1)

    return {}


# =============== LOGIC ===================
def sanitize_pi(pi):
    # realistic prevalence bounds
    if isinstance(pi, (int, float)):
        if pi <= 0:
            return 0.002
        return max(0.0005, min(pi, 0.02))
    return 0.002


def map_costs(fn_signal, fp_signal, weight):
    fn_map = {
        "low": 100,
        "medium": 1000,
        "high": 5000,
        "critical": 20000
    }

    fp_map = {
        "low": 0.01,
        "medium": 0.1,
        "high": 1.0
    }

    C_fn = fn_map.get(fn_signal)
    C_fp = fp_map.get(fp_signal)

    if C_fn is not None and isinstance(weight, (int, float)):
        C_fn *= weight

    return C_fp, C_fn


def compute_theta_star(pi, C_fp, C_fn):
    if None in (pi, C_fp, C_fn):
        return None
    return 1 / (1 + (pi * C_fn) / ((1 - pi) * C_fp))


# 🔥 operational constraint layer
def apply_constraints(theta_star, pi):
    if theta_star is None:
        return None

    # theta_min = max(0.01, pi * 5)
    theta_min = max(0.01, pi * 3 + 0.01)
    theta_max = 0.5

    theta = max(theta_star, theta_min)
    theta = min(theta, theta_max)

    return theta


# =============== PIPELINE ===================
def run(platform):

    query = (
        platform + " fake accounts prevalence transparency report site:meta.com "
        + platform + " community guidelines fake accounts impersonation policy site:help.instagram.com "
        + platform + " authentic identity representation policy site:transparency.meta.com"
    )

    contents, sources = fetch_contents(query)

    if not contents:
        return {"error": "No high-quality policy sources found"}

    context = build_context(contents)

    extracted = call_groq(
        EXTRACTION_PROMPT.replace("{policy_text}", context)
    )

    pi = sanitize_pi(extracted.get("base_rate"))
    fn_signal = extracted.get("fn_cost_signal")
    fp_signal = extracted.get("fp_cost_signal")
    weight = extracted.get("harm_weight")
    confidence = extracted.get("policy_confidence")

    # 🔥 sanity guardrails
    if fn_signal not in ["low", "medium", "high", "critical"]:
        fn_signal = "high"

    if fn_signal == "low":
        fn_signal = "high"

    if fp_signal not in ["low", "medium", "high"]:
        fp_signal = "medium"

    if not isinstance(weight, (int, float)):
        weight = 1.0

    # compute
    C_fp, C_fn = map_costs(fn_signal, fp_signal, weight)

    theta_star = compute_theta_star(pi, C_fp, C_fn)
    theta_final = apply_constraints(theta_star, pi)

    return {
        "platform": platform,
        "pi": pi,
        "fn_signal": fn_signal,
        "fp_signal": fp_signal,
        "harm_weight": weight,
        "C_fp": C_fp,
        "C_fn": C_fn,
        "theta_star": theta_star,
        "theta_final": theta_final,
        "confidence": confidence,
        "sources": sources
    }


# =============== RUN ======================
if __name__ == "__main__":
    result = run("Instagram")
    print(json.dumps(result, indent=2))
