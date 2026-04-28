#!/usr/bin/env bash
# Local smoke test — verifies every wired-up piece BEFORE deploying to Spaces.
#
# Stage 1: pure-Python unit checks (no env, no GPU)
# Stage 2: connectivity check against remote env (HEAD /health)
# Stage 3: heuristic-teacher rollout against remote env (no model load, ~30s)
# Stage 4: full GRPO smoke against remote env using a tiny model (CPU OK, ~2-5min)
#
# Usage:
#   ./training/test_local.sh                  # stages 1-3
#   ./training/test_local.sh --with-grpo      # stages 1-4 (needs torch, ~5min)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENV_BASE_URL="${ENV_BASE_URL:-https://pandago-graphstrike-model-training.hf.space}"
WITH_GRPO=0
[ "${1:-}" = "--with-grpo" ] && WITH_GRPO=1

echo "=== Stage 1: unit checks ==="
python3 - <<'PY'
import sys, json
sys.path.insert(0, '.')
from training.parse import parse_completion
from training.rewards import compute_reward

cases = [
    (json.dumps({"action":"reverse_image_search","reason":"r"}), "dp1", True,  "reverse_image_search"),
    (json.dumps({"action":"flag","reason":"r"}),                  "dp2", True,  "flag"),
    (json.dumps({"action":"flag","reason":"r"}),                  "dp1", False, None),  # wrong set
    ("not json at all",                                           "dp1", False, None),
    ("```json\n"+json.dumps({"action":"skip","reason":"r"})+"\n```", "dp2", True, "skip"),
]
for text, dt, want_ok, want_action in cases:
    a, _, ok = parse_completion(text, dt)
    assert ok == want_ok and a == want_action, f"parse failed for {text!r} (dt={dt}): got ({a},{ok})"
print("  parse: 5/5 OK")

rb = compute_reward(grader_score=0.7, step_reward=2.0, format_ok=True)
# step is scaled by 0.1 then clipped to ±0.5; 2.0*0.1=0.2 (in range)
expected = 1.0*0.7 + 0.3*0.2 + 0.2*1.0
assert abs(rb.total - expected) < 1e-9, (rb, expected)
rb2 = compute_reward(grader_score=0.0, step_reward=20.0, format_ok=False)
# clipped to +0.5 → 0.3*0.5 = 0.15
assert abs(rb2.total - 0.15) < 1e-9, rb2
print(f"  reward: in-range={rb.total:.4f} clipped={rb2.total:.4f} OK")
PY

echo
echo "=== Stage 2: env connectivity ($ENV_BASE_URL) ==="
if curl -fsS "${ENV_BASE_URL}/health" >/dev/null; then
    echo "  /health: OK"
else
    echo "  /health: FAIL — env unreachable. Aborting." >&2
    exit 1
fi
curl -fsS "${ENV_BASE_URL}/tasks" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  tasks:', d.get('tasks'))"

echo
echo "=== Stage 3: heuristic rollout (1 episode) ==="
python3 - <<PY
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'eval-models')
from _round2_runner import _run_episode
from client import FakeGangEnvClient
from training.build_dataset import heuristic_teacher

client = FakeGangEnvClient(base_url="${ENV_BASE_URL}")
log, tuples = _run_episode(client, "smoke", "Instagram", "easy", 0,
                           heuristic_teacher(), collect_tuples=True)
print(f"  episode_id={log.episode_id} grader={log.grader_score} flagged={log.flagged}")
print(f"  decision tuples: {len(tuples)}  (sample dt={tuples[0]['decision_type'] if tuples else None})")
assert len(tuples) > 0, "no decision tuples produced — runner refactor broken?"
print("  rollout: OK")
PY

if [ "$WITH_GRPO" -eq 0 ]; then
    echo
    echo "stages 1-3 PASS. Re-run with --with-grpo to also test the trainer path."
    exit 0
fi

echo
echo "=== Stage 4: GRPO phase0 baseline (tiny model, CPU OK) ==="
python3 -m training.train_grpo --phase phase0 \
    --model "sshleifer/tiny-gpt2" \
    --platform Instagram --tasks easy --seeds 0 1 \
    --base-url "${ENV_BASE_URL}"

echo
echo "All stages PASS. Ready to deploy."
