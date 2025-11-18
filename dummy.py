#!/usr/bin/env python3
"""
Test script to verify OpenSpiel metadata extraction is working.

Usage:
    python dummy.py
"""

import sys
sys.path.insert(0, "/home/felipemello/OpenEnv/src")

from envs.openspiel_env.server.openspiel_environment import OpenSpielEnvironment
from envs.openspiel_env.models import OpenSpielAction

def test_direct_env():
    """Test using OpenSpielEnvironment directly (no HTTP server)."""
    print("=" * 60)
    print("TEST 1: Direct OpenSpielEnvironment (no server)")
    print("=" * 60)

    env = OpenSpielEnvironment(
        game_name="blackjack",
        agent_player=0,
        opponent_policy="random"
    )

    # Reset
    obs = env.reset()
    print(f"\n[DIRECT] Initial observation:")
    print(f"  legal_actions: {obs.legal_actions}")
    print(f"  metadata: {obs.metadata}")
    print(f"  done: {obs.done}")

    # Play one step
    if not obs.done:
        action_id = obs.legal_actions[0]
        action = OpenSpielAction(action_id=action_id, game_name="blackjack")
        obs = env.step(action)
        print(f"\n[DIRECT] After step 1:")
        print(f"  legal_actions: {obs.legal_actions}")
        print(f"  metadata: {obs.metadata}")
        print(f"  done: {obs.done}")


def test_http_env():
    """Test using OpenSpielEnv via HTTP client."""
    print("\n" + "=" * 60)
    print("TEST 2: OpenSpielEnv via HTTP (using server)")
    print("=" * 60)

    from envs.openspiel_env import OpenSpielEnv

    env = OpenSpielEnv(base_url="http://localhost:9000")
    # Bypass proxy
    env._http.trust_env = False

    try:
        # Reset
        result = env.reset()
        obs = result.observation
        print(f"\n[HTTP] Initial observation:")
        print(f"  legal_actions: {obs.legal_actions}")
        print(f"  metadata: {obs.metadata}")
        print(f"  done: {obs.done}")

        # Play one step
        if not obs.done:
            action_id = obs.legal_actions[0]
            action = OpenSpielAction(action_id=action_id, game_name="blackjack")
            result = env.step(action)
            obs = result.observation
            print(f"\n[HTTP] After step 1:")
            print(f"  legal_actions: {obs.legal_actions}")
            print(f"  metadata: {obs.metadata}")
            print(f"  done: {obs.done}")

        env.close()
    except Exception as e:
        print(f"\n[HTTP ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\nTesting OpenSpiel metadata extraction...\n")

    # Test 1: Direct environment (should work with our fix)
    test_direct_env()

    # Test 2: HTTP environment (depends on server having the fix)
    test_http_env()

    print("\n" + "=" * 60)
    print("COMPARISON:")
    print("=" * 60)
    print("If both tests show metadata with player_total and dealer_card,")
    print("then the server is using the updated code.")
    print("If only DIRECT test works, the server needs to be restarted.")
    print("=" * 60)


if __name__ == "__main__":
    main()
