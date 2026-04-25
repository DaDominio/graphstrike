"""Strict JSON parser for policy completions.

Accepts ONLY a single JSON object with keys {"action", "reason"}. Returns
(action, reason, format_ok). When format is invalid, action == None.
"""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from .prompts import DP1_ALLOWED, DP2_ALLOWED

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_completion(text: str, decision_type: str) -> Tuple[Optional[str], str, bool]:
    """Return (action, reason, format_ok).

    `decision_type` ∈ {"dp1", "dp2"} restricts the allowed action set.
    A completion is considered well-formed iff it parses as a JSON object with
    string `action` and `reason` keys, and `action` ∈ allowed set for the dp.
    """
    allowed = DP1_ALLOWED if decision_type == "dp1" else DP2_ALLOWED
    raw = text.strip()
    # Strip common code-fence wrappers if the model used them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        m = _JSON_OBJECT_RE.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return None, "", False
    action = obj.get("action")
    reason = obj.get("reason", "")
    if not isinstance(action, str) or not isinstance(reason, str):
        return None, "", False
    action = action.strip().lower()
    if action not in allowed:
        return None, reason[:80], False
    return action, reason[:80], True
