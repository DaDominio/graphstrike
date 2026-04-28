#!/bin/bash
# GraphStrike Round 2 — Full System Check
# Run: bash check_round2.sh
# Assumes server is running at localhost:7860

BASE="http://localhost:7860"
echo "========================================"
echo "GRAPHSTRIKE ROUND 2 — SYSTEM CHECK"
echo "========================================"

# -------------------------------------------------------
# CHECK 1 — Server is alive
# -------------------------------------------------------
echo ""
echo "CHECK 1: Server health"
curl -s "$BASE/" | python3 -m json.tool 2>/dev/null || \
curl -s "$BASE/health" | python3 -m json.tool 2>/dev/null || \
echo "  (no health endpoint — checking /tasks instead)"

# -------------------------------------------------------
# CHECK 2 — Tasks endpoint shows Round 2 actions
# -------------------------------------------------------
echo ""
echo "CHECK 2: /tasks — must show all Round 2 action types"
curl -s "$BASE/tasks" | python3 -m json.tool

# -------------------------------------------------------
# CHECK 3 — Reset episode (instagram)
# -------------------------------------------------------
echo ""
echo "CHECK 3: /reset — instagram, easy task"
RESET=$(curl -s -X POST "$BASE/reset" \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}')
echo $RESET | python3 -m json.tool

# Extract session or episode info if present
EPISODE_ID=$(echo $RESET | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('episode_id',''))" 2>/dev/null)
echo "  Episode ID: $EPISODE_ID"

# -------------------------------------------------------
# CHECK 4 — GET_POLICY (lowercase — correct form)
# -------------------------------------------------------
echo ""
echo "CHECK 4: step GET_POLICY (lowercase) — must return threshold"
curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "get_policy"}' | python3 -m json.tool

# Extract a real account_id from the reset observation BEFORE the first inspect.
ACCOUNT_ID=$(echo $RESET | python3 -c "
import sys, json
d = json.load(sys.stdin)
obs = d.get('observation', d)
accounts = obs.get('visible_account_ids') or obs.get('visible_accounts') or []
if accounts:
    print(accounts[0] if isinstance(accounts[0], str) else accounts[0].get('account_id','acc_000'))
else:
    print('acc_000')
" 2>/dev/null)
echo "  Using account_id: $ACCOUNT_ID"

# -------------------------------------------------------
# CHECK 5 — INSPECT first visible account
# -------------------------------------------------------
echo ""
echo "CHECK 5: step inspect — first visible account ($ACCOUNT_ID)"
INSPECT=$(curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d "{\"action_type\": \"inspect\", \"account_id\": \"$ACCOUNT_ID\"}")
echo $INSPECT | python3 -m json.tool

# -------------------------------------------------------
# CHECK 6 — REVERSE_IMAGE_SEARCH
#           photo_reuse_score must be None before, filled after
# -------------------------------------------------------
echo ""
echo "CHECK 6: reverse_image_search — must populate photo_reuse_score"
echo "  Before call — check photo_reuse_score is hidden:"
curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d "{\"action_type\": \"inspect\", \"account_id\": \"$ACCOUNT_ID\"}" | \
  ACCOUNT_ID="$ACCOUNT_ID" python3 -c "
import sys, json, os
d = json.load(sys.stdin)
obs = d.get('observation', {})
target = os.environ['ACCOUNT_ID']
profile = next((a for a in obs.get('visible_accounts', []) if a.get('account_id') == target), {})
val = profile.get('photo_reuse_score', 'KEY_MISSING')
print(f'  photo_reuse_score = {val}')
print('  PASS — hidden before tool call' if (val in (None, 0.0)) else
      f'  WARN — already visible (={val})' if val != 'KEY_MISSING' else
      '  WARN — field not found in response')
"

echo "  After reverse_image_search:"
curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d "{\"action_type\": \"reverse_image_search\", \"account_id\": \"$ACCOUNT_ID\"}" | \
  ACCOUNT_ID="$ACCOUNT_ID" python3 -c "
import sys, json, os
d = json.load(sys.stdin)
obs = d.get('observation', {})
target = os.environ['ACCOUNT_ID']
profile = next((a for a in obs.get('visible_accounts', []) if a.get('account_id') == target), {})
val = profile.get('photo_reuse_score', 'KEY_MISSING')
reward = d.get('reward', 'N/A')
print(f'  photo_reuse_score = {val}')
print(f'  reward = {reward}')
print('  PASS — signal revealed' if val not in (None, 'KEY_MISSING') else
      '  FAIL — signal still hidden after tool call')
"

# -------------------------------------------------------
# CHECK 7 — ANALYZE_BIO
# -------------------------------------------------------
echo ""
echo "CHECK 7: analyze_bio — must populate bio_template_score"
curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d "{\"action_type\": \"analyze_bio\", \"account_id\": \"$ACCOUNT_ID\"}" | \
  ACCOUNT_ID="$ACCOUNT_ID" python3 -c "
import sys, json, os
d = json.load(sys.stdin)
obs = d.get('observation', {})
target = os.environ['ACCOUNT_ID']
profile = next((a for a in obs.get('visible_accounts', []) if a.get('account_id') == target), {})
val = profile.get('bio_template_score', 'KEY_MISSING')
reward = d.get('reward', 'N/A')
print(f'  bio_template_score = {val}')
print(f'  reward = {reward}')
print('  PASS' if val not in (None, 0.0, 'KEY_MISSING') else '  FAIL')
"

# -------------------------------------------------------
# CHECK 8 — CHECK_IP
# -------------------------------------------------------
echo ""
echo "CHECK 8: check_ip — must reveal ip_cluster, costs 2 steps"
curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d "{\"action_type\": \"check_ip\", \"account_id\": \"$ACCOUNT_ID\"}" | \
  ACCOUNT_ID="$ACCOUNT_ID" python3 -c "
import sys, json, os
d = json.load(sys.stdin)
obs = d.get('observation', {})
target = os.environ['ACCOUNT_ID']
profile = next((a for a in obs.get('visible_accounts', []) if a.get('account_id') == target), {})
# server exposes the cluster id via the message and shared_ip_count via the profile.
shared = profile.get('shared_ip_count', 'KEY_MISSING')
msg = d.get('message','')
reward = d.get('reward', 'N/A')
print(f'  shared_ip_count = {shared}')
print(f'  message excerpt: {msg[:120]}')
print(f'  reward = {reward}')
print('  PASS' if 'cluster' in msg.lower() or shared not in (None,'KEY_MISSING') else '  FAIL')
"

# -------------------------------------------------------
# CHECK 9 — GET_POLICY first-step bonus
# -------------------------------------------------------
echo ""
echo "CHECK 9: GET_POLICY at step 0 must give +0.20 reward"
echo "  Resetting fresh episode..."
curl -s -X POST "$BASE/reset" \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}' > /dev/null

curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "get_policy"}' | \
  python3 -c "
import sys, json
d = json.load(sys.stdin)
reward = d.get('reward', None)
msg = d.get('message', '')
threshold = None
try:
    import re
    m = re.search(r'threshold[=:\s]+([\d.]+)', str(d))
    if m: threshold = m.group(1)
except: pass
print(f'  reward = {reward}')
print(f'  threshold found = {threshold}')
print('  PASS — +0.20 bonus received' if reward and float(reward) >= 0.15 else
      f'  WARN — reward={reward}, expected ~0.20')
"

# -------------------------------------------------------
# CHECK 10 — Redundant tool call penalty
# -------------------------------------------------------
echo ""
echo "CHECK 10: Calling reverse_image_search twice must give -0.05 penalty"
curl -s -X POST "$BASE/reset" \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}' > /dev/null

# First call — should give normal reward
R1=$(curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d "{\"action_type\": \"reverse_image_search\", \"account_id\": \"$ACCOUNT_ID\"}" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('reward',0))")

# Second call on same account — should give -0.05
R2=$(curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d "{\"action_type\": \"reverse_image_search\", \"account_id\": \"$ACCOUNT_ID\"}" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('reward',0))")

echo "  First call reward:  $R1"
echo "  Second call reward: $R2"
python3 -c "
r1, r2 = $R1, $R2
print('  PASS — penalty applied on redundant call' if r2 < r1 else
      'FAIL — no penalty on redundant call (should be -0.05)')
"

# -------------------------------------------------------
# CHECK 11 — FLAG without revealed signals gives penalty
# -------------------------------------------------------
echo ""
echo "CHECK 11: FLAG without any revealed signals must give -0.15 penalty"
curl -s -X POST "$BASE/reset" \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}' > /dev/null

curl -s -X POST "$BASE/step" \
  -H "Content-Type: application/json" \
  -d "{\"action_type\": \"flag\", \"account_id\": \"$ACCOUNT_ID\"}" | \
  python3 -c "
import sys, json
d = json.load(sys.stdin)
reward = d.get('reward', None)
print(f'  reward = {reward}')
print('  PASS — penalty applied for flagging without evidence' if reward and float(reward) <= -0.10 else
      f'  WARN — reward={reward}, expected -0.15 penalty')
"

# -------------------------------------------------------
# CHECK 12 — Full episode end to end
# -------------------------------------------------------
echo ""
echo "CHECK 12: Full episode — reset, get_policy, tools, flag, submit"
python3 - <<'PYEOF'
import requests, json

BASE = "http://localhost:7860"

# Reset
r = requests.post(f"{BASE}/reset", json={"task": "easy"})
resp = r.json()
obs = resp.get("observation", resp)
accounts = obs.get("visible_account_ids") or obs.get("visible_accounts") or []
if accounts:
    acc = accounts[0] if isinstance(accounts[0], str) else accounts[0].get("account_id")
else:
    acc = "acc_000"
print(f"  Episode started. First account: {acc}")

steps = []

# Step 1: get_policy
r = requests.post(f"{BASE}/step", json={"action_type": "get_policy"})
d = r.json()
steps.append(("get_policy", d.get("reward")))

# Step 2: reverse_image_search
r = requests.post(f"{BASE}/step", json={"action_type": "reverse_image_search", "account_id": acc})
d = r.json()
steps.append(("reverse_image_search", d.get("reward")))

# Step 3: analyze_bio
r = requests.post(f"{BASE}/step", json={"action_type": "analyze_bio", "account_id": acc})
d = r.json()
steps.append(("analyze_bio", d.get("reward")))

# Step 4: flag
r = requests.post(f"{BASE}/step", json={"action_type": "flag", "account_id": acc})
d = r.json()
steps.append(("flag", d.get("reward")))

# Step 5: submit
r = requests.post(f"{BASE}/step", json={"action_type": "submit"})
d = r.json()
steps.append(("submit", d.get("reward")))
msg = d.get("message", "")
done = d.get("done", False)

print("\n  Action log:")
for action, reward in steps:
    print(f"    {action:<25} reward={reward}")

print(f"\n  Episode done: {done}")

# Check decision package in submit message
for keyword in ["Decision:", "policy_rationale", "evidence_summary", "flagged_accounts"]:
    found = keyword.lower() in msg.lower()
    print(f"  Decision package [{keyword}]: {'PASS' if found else 'MISSING'}")

# Check grader score
grader = d.get("grader_score") or d.get("score")
print(f"  Grader score: {grader}")

# Check structured decision_package field
dp = d.get("decision_package")
if dp:
    print(f"  decision_package keys: {sorted(dp.keys())}")
else:
    print("  decision_package: missing (expected after submit)")
PYEOF

# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------
echo ""
echo "========================================"
echo "CHECK COMPLETE"
echo ""
echo "Expected results:"
echo "  CHECK 1-3:  Server healthy, tasks listed, reset works"
echo "  CHECK 4:    get_policy returns threshold from compiled policy"
echo "  CHECK 5:    inspect works"
echo "  CHECK 6:    reverse_image_search reveals photo_reuse_score"
echo "  CHECK 7:    analyze_bio reveals bio_template_score"
echo "  CHECK 8:    check_ip reveals ip_cluster_signal"
echo "  CHECK 9:    get_policy at step 0 gives +0.20 reward"
echo "  CHECK 10:   redundant tool call gives -0.05 penalty"
echo "  CHECK 11:   flag without evidence gives -0.15 penalty"
echo "  CHECK 12:   full episode runs, decision package in submit"
echo ""
echo "If any check fails, fix that component before running eval scripts."
echo "========================================"
