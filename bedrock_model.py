import boto3
import json
import os
from botocore.exceptions import ClientError
from typing import Any, List, Dict, Optional, Union

# --- Credentials setup ---
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Bedrock Marketplace model endpoint ARN (replace with your own)
MODEL_ID = "qwen.qwen3-next-80b-a3b"

# Build the Bedrock runtime client
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)


def invoke_qwen(
    prompt: str,
    system: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    images: Optional[List[Dict[str, Union[str, bytes]]]] = None
) -> str:
    """
    Invoke the Qwen VL model via Bedrock Converse API with optional image input.

    Args:
        prompt: Text prompt for the model.
        system: Optional system prompt.
        max_tokens: Maximum tokens to generate.
        images: Optional list of image dictionaries. Each dict must contain:
                - "bytes": raw image bytes (e.g., from open(file, "rb").read())
                  OR
                - "path": local file path to an image (will be read as raw bytes).
                - "format": image format string ("jpeg", "png", "gif", or "webp").

    Returns:
        Generated text response.
    """
    # Start building the content blocks for the user message
    content_blocks: List[Dict] = [{"text": prompt}]

    # Add image blocks if any are supplied
    if images:
        for img in images:
            # Determine raw bytes from either "bytes" or "path"
            if "bytes" in img:
                img_bytes = img["bytes"]
            elif "path" in img:
                with open(img["path"], "rb") as f:
                    img_bytes = f.read()
            else:
                raise ValueError(
                    "Each image dict must include either 'bytes' (raw bytes) "
                    "or 'path' (file path) plus a 'format' key."
                )

            if "format" not in img:
                raise ValueError("Each image dict must include a 'format' (e.g., 'jpeg').")

            # Build the image block as required by the Converse API
            image_block = {
                "image": {
                    "format": img["format"],
                    "source": {"bytes": img_bytes}   # raw bytes, NOT base64‑encoded
                }
            }
            content_blocks.append(image_block)

    # Assemble the final message
    messages = [{"role": "user", "content": content_blocks}]

    # Prepare the Converse API call
    kwargs = {
        "modelId": MODEL_ID,   # Marketplace endpoint ARN
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }
    }
    if system:
        kwargs["system"] = [{"text": system}]

    try:
        response = client.converse(**kwargs)
        output = response["output"]["message"]["content"][0]["text"]
        usage = response["usage"]
        print(f"[Tokens] in={usage['inputTokens']} out={usage['outputTokens']}")
        return output
    except ClientError as e:
        raise RuntimeError(f"Bedrock error: {e.response['Error']['Message']}")


def _parse_score_response(text: str) -> Dict[str, Any]:
    """Parse JSON object with score, issues, summary from model output."""
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "issues": ["Could not parse model JSON"], "summary": text[:400] if text else ""}
    score = d.get("score", 0)
    try:
        score_i = int(score)
    except (TypeError, ValueError):
        score_i = 0
    score_i = max(0, min(100, score_i))
    issues = d.get("issues")
    if issues is None:
        issues = []
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        issues = []
    issues = [str(x).strip() for x in issues if str(x).strip()]
    summary = str(d.get("summary", "") or "").strip()
    return {"score": score_i, "issues": issues[:12], "summary": summary}


def score_design_against_spec(image_bytes: bytes, spec: Optional[str]) -> Dict[str, Any]:
    """
    Score a UI screenshot (PNG bytes) against the product spec via the vision model.
    Returns dict: score (0-100), issues (list of str), summary (str).
    """
    system = (
        "You are a product design QA assistant. Compare the screenshot to the product spec. "
        "Return ONLY a single JSON object, no markdown, with keys: "
        'score (integer 0-100), issues (array of short strings, max 8 items), '
        'summary (one sentence). Be strict about spec mismatches.'
    )
    spec_block = (spec or "").strip() or (
        "(No spec was provided — score clarity, hierarchy, visual polish, and common UX patterns.)"
    )
    prompt = (
        "Evaluate this design for the following product spec.\n\n"
        f"SPEC:\n{spec_block}\n\n"
        "Output JSON only."
    )
    text = invoke_qwen(
        prompt=prompt,
        system=system,
        max_tokens=1024,
        temperature=0.2,
        images=[{"bytes": image_bytes, "format": "png"}],
    )
    return _parse_score_response(text)


# --- Example usage ---
if __name__ == "__main__":
    # 1️⃣ Text‑only call (same as before)
    text_result = invoke_qwen(
        prompt="Hello",
        # system="You are a content extractor. Return content as JSON list.",
    )
    print("\nText‑only result:")
    print(text_result)

