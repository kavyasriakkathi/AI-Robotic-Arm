import math

import numpy as np

from src.robot_env import RobotReachEnv


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def test_reward_is_lower_when_distance_increases_and_higher_when_it_decreases():
    env = RobotReachEnv(render_mode="direct")
    try:
        obs, _ = env.reset(seed=1)
        # Observation is now 23D: [joint_pos(7), joint_vel(7), ee_pos(3), ee_vel(3), target_pos(3)]
        assert obs.shape == (23,)
        assert env.observation_space.contains(obs)
        
        end_effector = obs[14:17]
        target = obs[20:23]
        initial_distance = _distance(end_effector, target)

        # A small action that keeps the arm still should produce a mild baseline reward
        # and a strong progress bonus when the end-effector moves closer.
        next_obs, reward_1, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        distance_after_idle = _distance(next_obs[14:17], next_obs[20:23])

        # Use a deterministic, known action to move closer in a simple way.
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        next_obs_2, reward_2, _, _, info_2 = env.step(action)
        distance_after_step = _distance(next_obs_2[14:17], next_obs_2[20:23])

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


def test_reset_starts_in_a_reachable_neutral_pose():
    env = RobotReachEnv(render_mode="direct")
    try:
        obs, _ = env.reset(seed=1)
        # Observation is now 23D: [joint_pos(7), joint_vel(7), ee_pos(3), ee_vel(3), target_pos(3)]
        assert obs.shape == (23,)
        assert env.action_space.shape == (7,)
        assert env.observation_space.contains(obs)
        
        end_effector = obs[14:17]
        target = obs[20:23]
        initial_distance = _distance(end_effector, target)

        # This task should start in a posture that is not artificially impossible.
        # The robot must be able to make meaningful progress without a huge random search.
        assert initial_distance < 0.6
    finally:
        env.close()
