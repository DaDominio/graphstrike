"""LLM backend adapters shared across eval shims (HF router / AWS Bedrock)."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Callable


SYSTEM_PROMPT_DEFAULT = (
    "You are a careful fake-account moderator. Answer each question with "
    "exactly one lowercase token from the allowed list. No explanation."
)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def make_hf_caller(model_name: str, system_prompt: str = SYSTEM_PROMPT_DEFAULT) -> Callable[[str], str]:
    """Return a call_llm(prompt) -> str bound to the HF router for this model."""
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    token = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key=token)

    def _call(prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=32,
                )
                raw = (resp.choices[0].message.content or "").strip()
                return _strip_think(raw) or raw
            except Exception as e:
                if attempt == 2:
                    print(f"    [HF ERROR] {e}")
                    return ""
                time.sleep(2 * (attempt + 1))
        return ""

    return _call


def make_bedrock_caller(model_id: str, system_prompt: str = SYSTEM_PROMPT_DEFAULT) -> Callable[[str], str]:
    """Return a call_llm(prompt) -> str bound to AWS Bedrock for this model."""
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        raise RuntimeError("AWS_ACCESS_KEY_ID not set")

    import boto3
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    def _call(prompt: str) -> str:
        for attempt in range(3):
            try:
                if hasattr(client, "converse"):
                    resp = client.converse(
                        modelId=model_id,
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                        system=[{"text": system_prompt}],
                        inferenceConfig={"maxTokens": 32, "temperature": 0.2},
                    )
                    raw = resp["output"]["message"]["content"][0]["text"].strip()
                else:
                    body = json.dumps({
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 32,
                        "temperature": 0.2,
                    })
                    resp = client.invoke_model(
                        modelId=model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=body,
                    )
                    result = json.loads(resp["body"].read())
                    if "choices" in result:
                        raw = result["choices"][0]["message"]["content"].strip()
                    elif "content" in result:
                        c = result["content"]
                        raw = c[0].get("text", "").strip() if isinstance(c, list) else str(c).strip()
                    else:
                        raw = str(result).strip()
                return _strip_think(raw) or raw
            except Exception as e:
                if attempt == 2:
                    print(f"    [BEDROCK ERROR] {e}")
                    return ""
                time.sleep(2 * (attempt + 1))
        return ""

    return _call


def make_caller(backend: str, hf_model: str, bedrock_model: str) -> Callable[[str], str]:
    if backend == "bedrock":
        return make_bedrock_caller(bedrock_model)
    return make_hf_caller(hf_model)
