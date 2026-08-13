import numpy as np

from stable_baselines3 import PPO

from robot_env import RobotReachEnv


def main():
    """Tiny Stable-Baselines3 PPO smoke test without full training.

    This script verifies that:
    - the Gymnasium environment loads correctly,
    - the observation and action spaces match PPO expectations,
    - a PPO model can be instantiated,
    - a single policy forward pass works on real environment data.
    """
    env = RobotReachEnv(render_mode="direct")

    obs, info = env.reset(seed=0)
    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space.shape)
    print("Initial observation sample:", np.asarray(obs[:5], dtype=np.float32))
    print("Reset info:", info)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    print("Sample action:", np.asarray(action, dtype=np.float32))
    print("Step reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Step info keys:", sorted(step_info.keys()))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=8,
        batch_size=8,
        verbose=0,
        device="cpu",
    )
    print("PPO model created successfully.")

    policy_action, _ = model.predict(obs, deterministic=True)
    print("PPO policy action shape:", np.asarray(policy_action).shape)
    print("PPO policy action sample:", np.asarray(policy_action, dtype=np.float32))

    env.close()
    print("Smoke test complete without training or model saving.")


if __name__ == "__main__":
    main()
