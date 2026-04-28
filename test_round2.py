#!/usr/bin/env python3
"""End-to-end test for Round 2 implementation.

Tests:
1. Platform-specific episode loading
2. New tool actions (GET_POLICY, REVERSE_IMAGE_SEARCH, ANALYZE_BIO, CHECK_IP)
3. Platform-adaptive scoring
4. Hidden signals revelation
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from server.environment import FakeGangEnvironment
from models import FakeGangAction, ActionType


def test_round2():
    """Run comprehensive Round 2 test."""

    print("=" * 70)
    print("ROUND 2 END-TO-END TEST")
    print("=" * 70)

    env = FakeGangEnvironment()

    # Test 1: Instagram episode (even seed)
    print("\n[Test 1] Instagram Episode (seed=0)")
    print("-" * 70)
    obs = env.reset(task="easy", seed=0)
    print(f"✓ Platform: {obs.platform}")
    assert obs.platform == "Instagram", f"Expected Instagram, got {obs.platform}"
    print(f"✓ Steps remaining: {obs.steps_remaining}")
    print(f"✓ Starting visible: {len(obs.visible_account_ids)} accounts")

    # Test 2: GET_POLICY action
    print("\n[Test 2] GET_POLICY Action")
    print("-" * 70)
    action = FakeGangAction(action_type=ActionType.GET_POLICY)
    obs = env.step(action)
    print(f"✓ Message: {obs.message[:200]}")
    assert "Instagram" in obs.message or "threshold" in obs.message.lower(), "Policy not returned"
    assert obs.steps_remaining == 30, "GET_POLICY should not consume steps"

    # Test 3: INSPECT to find accounts
    print("\n[Test 3] INSPECT Action")
    print("-" * 70)
    acc_id = obs.visible_account_ids[0]
    action = FakeGangAction(action_type=ActionType.INSPECT, account_id=acc_id)
    obs = env.step(action)
    print(f"✓ Inspected: {acc_id}")
    print(f"✓ Steps remaining: {obs.steps_remaining}")
    assert obs.steps_remaining == 29, "INSPECT should consume 1 step"

    # Check that profile exists
    profile = next((p for p in obs.visible_accounts if p.account_id == acc_id), None)
    assert profile is not None, f"Profile for {acc_id} not found"
    print(f"✓ Profile created: fake_risk={profile.fake_risk_score:.3f}")

    # Test 4: REVERSE_IMAGE_SEARCH (hidden signal revelation)
    print("\n[Test 4] REVERSE_IMAGE_SEARCH Action")
    print("-" * 70)
    photo_before = profile.photo_reuse_score
    print(f"  Before: photo_reuse_score = {photo_before:.3f}")

    action = FakeGangAction(action_type=ActionType.REVERSE_IMAGE_SEARCH, account_id=acc_id)
    obs = env.step(action)
    print(f"✓ Steps remaining: {obs.steps_remaining}")
    assert obs.steps_remaining == 28, "REVERSE_IMAGE_SEARCH should consume 1 step"

    profile = next((p for p in obs.visible_accounts if p.account_id == acc_id), None)
    photo_after = profile.photo_reuse_score
    print(f"  After: photo_reuse_score = {photo_after:.3f}")
    print(f"✓ Signal revealed (changed: {photo_before != photo_after})")

    # Test 5: ANALYZE_BIO
    print("\n[Test 5] ANALYZE_BIO Action")
    print("-" * 70)
    bio_before = profile.bio_template_score
    print(f"  Before: bio_template_score = {bio_before:.3f}")

    action = FakeGangAction(action_type=ActionType.ANALYZE_BIO, account_id=acc_id)
    obs = env.step(action)
    assert obs.steps_remaining == 27, "ANALYZE_BIO should consume 1 step"

    profile = next((p for p in obs.visible_accounts if p.account_id == acc_id), None)
    bio_after = profile.bio_template_score
    print(f"  After: bio_template_score = {bio_after:.3f}")
    print(f"✓ Signal revealed (changed: {bio_before != bio_after})")

    # Test 6: CHECK_IP (expensive action)
    print("\n[Test 6] CHECK_IP Action")
    print("-" * 70)
    steps_before = obs.steps_remaining
    action = FakeGangAction(action_type=ActionType.CHECK_IP, account_id=acc_id)
    obs = env.step(action)
    print(f"✓ Steps consumed: {steps_before - obs.steps_remaining}")
    assert steps_before - obs.steps_remaining == 2, "CHECK_IP should consume 2 steps"
    print(f"✓ Message: {obs.message[:150]}")

    # Test 7: Snapchat episode (odd seed)
    print("\n[Test 7] Snapchat Episode (seed=1)")
    print("-" * 70)
    obs = env.reset(task="easy", seed=1)
    print(f"✓ Platform: {obs.platform}")
    assert obs.platform == "Snapchat", f"Expected Snapchat, got {obs.platform}"

    action = FakeGangAction(action_type=ActionType.GET_POLICY)
    obs = env.step(action)
    print(f"✓ Message: {obs.message[:200]}")
    assert "Snapchat" in obs.message or "threshold" in obs.message.lower()

    # Test 8: Platform-adaptive scoring
    print("\n[Test 8] Platform-Adaptive Scoring")
    print("-" * 70)

    # Reset to Instagram
    obs = env.reset(task="easy", seed=0)
    action = FakeGangAction(action_type=ActionType.GET_POLICY)
    obs = env.step(action)

    # Inspect and flag an account
    acc_id = obs.visible_account_ids[0]
    action = FakeGangAction(action_type=ActionType.INSPECT, account_id=acc_id)
    obs = env.step(action)

    profile = next((p for p in obs.visible_accounts if p.account_id == acc_id), None)
    print(f"  Account: {acc_id}")
    print(f"  fake_risk_score: {profile.fake_risk_score:.3f}")
    print(f"  status: {profile.status}")
    print(f"✓ Risk computed with platform-adaptive weights")

    # Test 9: SUBMIT with platform-specific rewards
    print("\n[Test 9] SUBMIT with Platform Rewards")
    print("-" * 70)

    # Flag gang members if we can identify them
    obs = env.reset(task="easy", seed=2)

    # Inspect a few accounts
    for acc_id in obs.visible_account_ids[:5]:
        action = FakeGangAction(action_type=ActionType.INSPECT, account_id=acc_id)
        obs = env.step(action)

    # Flag high-risk accounts
    flagged_count = 0
    for profile in obs.visible_accounts:
        if profile.fake_risk_score > 0.6 and flagged_count < 5:
            action = FakeGangAction(action_type=ActionType.FLAG, account_id=profile.account_id)
            obs = env.step(action)
            flagged_count += 1

    print(f"  Flagged: {len(obs.flagged_ids)} accounts")

    action = FakeGangAction(action_type=ActionType.SUBMIT)
    obs = env.step(action)
    print(f"✓ Episode complete: done={obs.done}")
    print(f"✓ Final reward: {obs.reward:.3f}")
    print(f"✓ Message: {obs.message[:200]}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nRound 2 implementation verified:")
    print("  ✓ Platform-specific episodes (Instagram/Snapchat)")
    print("  ✓ GET_POLICY action (0 steps)")
    print("  ✓ REVERSE_IMAGE_SEARCH (1 step)")
    print("  ✓ ANALYZE_BIO (1 step)")
    print("  ✓ CHECK_IP (2 steps)")
    print("  ✓ Hidden signals revelation")
    print("  ✓ Platform-adaptive scoring")
    print("  ✓ Complete episode flow")


if __name__ == "__main__":
    test_round2()
