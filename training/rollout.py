"""GRPO rollout adapter.

Drives a policy (HF model) against the live env, returning per-decision tuples
identical in shape to `build_dataset.py` rows. The same `_round2_runner._run_episode`
is reused — we only swap the `call_llm` callable to invoke our HF model.

This module exposes:
  - HFPolicy.generate(prompt) -> str
  - rollout_batch(policy, ...) -> list[dict] (decision tuples)
  - reward_for_tuple(t, format_ok) -> float

Designed so a TRL GRPOTrainer can call rollout_batch() each step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Dict, Optional

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_PARENT / "eval-models"))

from _round2_runner import _run_episode  # noqa: E402
from client import FakeGangEnvClient  # noqa: E402

from training.parse import parse_completion  # noqa: E402
from training.rewards import compute_reward, RewardBreakdown  # noqa: E402


class HFPolicy:
    """Lightweight HF causal-LM wrapper. Loaded lazily; safe to import on CPU."""

    def __init__(self, model_name: str, max_new_tokens: int = 80, temperature: float = 0.8):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()

    def generate(self, prompt: str) -> str:
        self._load()
        from training.prompts import build_chat
        chat = build_chat(prompt)
        enc = self._tok.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True)
        # Newer transformers returns a BatchEncoding (dict-like); older returns a Tensor.
        input_ids = enc["input_ids"] if hasattr(enc, "data") else enc
        input_ids = input_ids.to(self._model.device)
        import torch
        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self._tok.pad_token_id,
            )
        text = self._tok.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True)
        return text


def rollout_batch(
    call_llm: Callable[[str], str],
    base_url: str,
    platform: str,
    tasks: List[str],
    seeds: List[int],
    model_tag: str = "policy",
) -> List[Dict]:
    """Run one episode per (task, seed) and concatenate decision tuples.
    Each tuple gains: r_grader, r_step, r_format, r_total, format_ok, action, reason.
    """
    client = FakeGangEnvClient(base_url=base_url)
    rows: List[Dict] = []
    for task in tasks:
        for seed in seeds:
            log, tuples = _run_episode(
                client, model_tag, platform, task, seed, call_llm,
                collect_tuples=True,
            )
            for di, t in enumerate(tuples):
                action, reason, format_ok = parse_completion(t["completion"], t["decision_type"])
                rb = compute_reward(t.get("grader_score"), t.get("step_reward"), format_ok)
                rows.append({
                    **t,
                    "task": task, "seed": seed, "decision_index": di,
                    "action_parsed": action, "reason_parsed": reason,
                    "format_ok": format_ok,
                    **rb.as_dict(),
                })
    return rows


def reward_for_tuple(t: Dict) -> float:
    """Recompute the scalar reward from a tuple dict (idempotent helper)."""
    rb = compute_reward(t.get("grader_score"), t.get("step_reward"), bool(t.get("format_ok", False)))
    return rb.total
