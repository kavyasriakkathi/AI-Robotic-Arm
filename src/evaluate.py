import os

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO

from robot_env import RobotReachEnv


MODEL_PATH = os.path.join("models", "ppo_robot_reach.zip")
GRAPH_PATH = os.path.join("results", "distance_vs_step.png")
MAX_EVAL_STEPS = 50


def evaluate_model():
    """Load the trained PPO model and evaluate it in the PyBullet robot environment.

    This script is intentionally simple and beginner-friendly. It does not retrain the
    model or change the state/action/reward design in RobotReachEnv.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")

    os.makedirs("results", exist_ok=True)

    env = RobotReachEnv(render_mode="human")
    model = PPO.load(MODEL_PATH, env=env)

    obs, _ = env.reset(seed=1)
    initial_distance = float(np.linalg.norm(obs[7:10] - obs[10:13]))

    distances = []
    rewards = []
    total_reward = 0.0
    reached = False
    step_count = 0

    print("\nStarting PPO evaluation in PyBullet GUI...")
    print(f"Initial distance to target: {initial_distance:.4f}")

    for step in range(MAX_EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        distance = float(np.linalg.norm(obs[7:10] - obs[10:13]))
        total_reward += float(reward)
        distances.append(distance)
        rewards.append(float(reward))
        step_count += 1

        print(
            f"Step {step + 1:02d} | distance={distance:.4f} | reward={reward:.4f}"
        )

        if info.get("distance_to_target", float("inf")) <= env.distance_threshold:
            reached = True

        if terminated or truncated:
            break

    final_distance = distances[-1] if distances else initial_distance

    print("\nEvaluation summary")
    print(f"Initial distance: {initial_distance:.4f}")
    print(f"Final distance: {final_distance:.4f}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Number of steps: {step_count}")
    print(f"Target reached: {reached}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(distances)), distances, marker="o", linewidth=2)
    plt.title("Distance to Target vs Step")
    plt.xlabel("Step")
    plt.ylabel("Distance to Target (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    plt.close()

    print(f"Graph saved to: {GRAPH_PATH}")
    env.close()

    return {
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "total_reward": total_reward,
        "steps": step_count,
        "target_reached": reached,
    }


if __name__ == "__main__":
    evaluate_model()
