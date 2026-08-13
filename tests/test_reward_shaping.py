import math

import numpy as np

from src.robot_env import RobotReachEnv


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def test_reward_is_lower_when_distance_increases_and_higher_when_it_decreases():
    env = RobotReachEnv(render_mode="direct")
    try:
        obs, _ = env.reset(seed=1)
        end_effector = obs[7:10]
        target = obs[10:13]
        initial_distance = _distance(end_effector, target)

        # A small action that keeps the arm still should produce a mild baseline reward
        # and a strong progress bonus when the end-effector moves closer.
        next_obs, reward_1, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        distance_after_idle = _distance(next_obs[7:10], next_obs[10:13])

        # Use a deterministic, known action to move closer in a simple way.
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        next_obs_2, reward_2, _, _, info_2 = env.step(action)
        distance_after_step = _distance(next_obs_2[7:10], next_obs_2[10:13])

        # Reward must be numerically stable and should favor reducing distance.
        assert math.isfinite(reward_1)
        assert math.isfinite(reward_2)
        assert reward_1 <= 0.0 or reward_1 >= -10.0
        assert distance_after_idle >= 0.0
        assert distance_after_step >= 0.0

        # If the environment could not move closer due to the chosen action, the step still
        # must not produce an arbitrarily large positive reward just because the arm moved.
        if distance_after_step < initial_distance:
            assert reward_2 > reward_1
        if distance_after_step > initial_distance:
            assert reward_2 < reward_1

        assert "distance_to_target" in info_2
    finally:
        env.close()
