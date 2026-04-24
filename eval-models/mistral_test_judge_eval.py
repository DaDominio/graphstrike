#!/usr/bin/env python3
"""Round 2 eval shim for mistral — delegates to shared _round2_runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure this directory is importable so _round2_runner / _llm_adapters resolve.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _round2_runner import run_evaluation, build_cli  # noqa: E402
from _llm_adapters import make_caller  # noqa: E402

DEFAULT_HF_MODEL      = os.getenv("MODEL_NAME", "mistral.ministral-3-8b-instruct")
DEFAULT_BEDROCK_MODEL = os.getenv("BEDROCK_MODEL_ID", "mistral.ministral-3-8b-instruct")


def main():
    parser = build_cli()
    parser.add_argument("--bedrock", action="store_true", help="Use AWS Bedrock instead of HF router")
    args = parser.parse_args()

    backend = "bedrock" if args.bedrock else "hf"
    model_display = f"Bedrock/{DEFAULT_BEDROCK_MODEL}" if backend == "bedrock" else DEFAULT_HF_MODEL
    call_llm = make_caller(backend, DEFAULT_HF_MODEL, DEFAULT_BEDROCK_MODEL)

    run_evaluation(
        model_name=model_display,
        call_llm=call_llm,
        platform=args.platform,
        base_url=args.url,
        tasks=args.tasks,
        seeds=args.seeds,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
