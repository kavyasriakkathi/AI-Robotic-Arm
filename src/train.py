import os

import numpy as np

from stable_baselines3 import PPO

from robot_env import RobotReachEnv


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_robot_reach_v2")
TOTAL_TIMESTEPS = 25000
EVAL_STEPS = 60


def evaluate_model(model, env, max_steps=EVAL_STEPS):
    """Run a short evaluation on the saved PPO policy."""
    obs, _ = env.reset(seed=1)
    initial_distance = float(np.linalg.norm(obs[7:10] - obs[10:13]))

    total_reward = 0.0
    reached = False
    step_count = 0

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1

        if info.get("distance_to_target", float("inf")) <= env.distance_threshold:
            reached = True

        if terminated or truncated:
            break

    final_distance = float(np.linalg.norm(obs[7:10] - obs[10:13]))

    print("\nEvaluation summary")
    print(f"Initial distance to target: {initial_distance:.4f}")
    print(f"Final distance to target: {final_distance:.4f}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Steps: {step_count}")
    print(f"Target reached: {reached}")

    return {
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "total_reward": total_reward,
        "steps": step_count,
        "target_reached": reached,
    }


def main():
    """Train a small PPO agent for the robotic arm target-reaching task."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = RobotReachEnv(render_mode="direct")
    try:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=1024,
            batch_size=128,
            learning_rate=1e-4,
            gamma=0.99,
            verbose=1,
            device="cpu",
            policy_kwargs={"net_arch": [64, 64]},
        )
        print(f"\nTraining PPO for {TOTAL_TIMESTEPS} timesteps...")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
        print(f"\nSaving model to {MODEL_PATH}")
        model.save(MODEL_PATH)
    finally:
        train_env.close()

    eval_env = RobotReachEnv(render_mode="direct")
    try:
        model = PPO.load(MODEL_PATH, env=eval_env)
        evaluate_model(model, eval_env, max_steps=EVAL_STEPS)
    finally:
        eval_env.close()

    print(f"\nTraining command used: .\\.venv\\Scripts\\python.exe src/train.py")
    print(f"Number of timesteps: {TOTAL_TIMESTEPS}")
    print(f"Model file created: {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
