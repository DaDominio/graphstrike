#!/bin/bash
# Update all eval-models files with Round 2 SYSTEM_PROMPT

EVAL_DIR="eval-models"

ROUND2_PROMPT='"""You are an AI detective finding 10 coordinated fake accounts in a social network.

━━━ ROUND 2: PLATFORM-ADAPTIVE DETECTION ━━━
Episodes run on specific platforms (Instagram/Snapchat) with different thresholds and enforcement priorities.

ACTIONS (reply with exactly ONE line):
- GET_POLICY              — get platform policy (FREE, call first!)
- INSPECT acc_XXXX        — reveal profile (costs 1 step)
- REVERSE_IMAGE_SEARCH acc_XXXX  — reveal photo_reuse_score (costs 1 step)
- ANALYZE_BIO acc_XXXX    — reveal bio_template_score (costs 1 step)
- CHECK_IP acc_XXXX       — reveal ip_cluster_id (costs 2 steps, expensive!)
- FLAG acc_XXXX           — mark as fake (FREE, no step cost, triggers suspect cascade)
- SUBMIT                  — end episode, get scored

DECISION RULES (Round 2, apply top-to-bottom):
1. First action of episode → GET_POLICY (learn platform threshold and primary signal)
2. If suspect_ids lists accounts you haven'"'"'t inspected → INSPECT the first one
3. If ANY profiled account has shared_ip_count >= 5 → CHECK_IP to confirm cluster, then FLAG
4. If photo_reuse_score or bio_template_score is 0.0 on suspicious accounts → use REVERSE_IMAGE_SEARCH or ANALYZE_BIO
5. If ANY profiled account has photo_reuse >= 0.50 AND bio_template >= 0.40 and hub < 0.70 → FLAG
6. If fake_risk_score >= platform_threshold and hub < 0.70 → FLAG
7. If uninspected visible accounts and steps > 3 → INSPECT the next one
8. If you have 10 flags OR steps <= 3 → SUBMIT

PLATFORM-SPECIFIC STRATEGIES:
- Instagram (threshold ~0.08, high FP penalty): Be precise, use REVERSE_IMAGE_SEARCH on borderline cases
- Snapchat (threshold ~0.74, low FP penalty): Be aggressive, flag when fake_risk >= 0.74

IMPORTANT:
- Hidden signals (photo_reuse, bio_template, ip_cluster) start as 0.0/None — use tools to reveal!
- GET_POLICY is FREE and shows platform threshold — always call first
- FLAG is FREE (costs 0 steps) — flag aggressively when you see suspicious signals
- CHECK_IP costs 2 steps (expensive) — only use when shared_ip_count >= 5
- hub_legitimacy_score > 0.70 means celebrity — do NOT flag

Reply with EXACTLY one line, nothing else:
GET_POLICY
REVERSE_IMAGE_SEARCH acc_XXXX
ANALYZE_BIO acc_XXXX
CHECK_IP acc_XXXX
FLAG acc_XXXX
INSPECT acc_XXXX
SUBMIT"""'

echo "Updating eval-models files with Round 2 SYSTEM_PROMPT..."

for file in "$EVAL_DIR"/*_test_judge_eval.py; do
    if [ "$file" = "$EVAL_DIR/qwen_test_judge_eval.py" ]; then
        echo "  ✓ Skipping $file (already updated)"
        continue
    fi

    echo "  📝 Updating $file..."

    # Create backup
    cp "$file" "$file.bak"

    # Use Python to replace SYSTEM_PROMPT
    python3 << EOF
import re

with open("$file", "r") as f:
    content = f.read()

# Find and replace SYSTEM_PROMPT
pattern = r'SYSTEM_PROMPT = """.*?"""'
replacement = f'SYSTEM_PROMPT = {$ROUND2_PROMPT}'

updated = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open("$file", "w") as f:
    f.write(updated)

print(f"    ✓ Updated $file")
EOF
done

echo ""
echo "✅ All eval files updated!"
echo ""
echo "Files updated:"
ls -1 "$EVAL_DIR"/*_test_judge_eval.py | grep -v qwen
echo ""
echo "Backups saved with .bak extension"
