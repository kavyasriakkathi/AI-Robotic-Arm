import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO

from robot_env import RobotReachEnv


DEFAULT_MODEL_PATH = os.path.join("models", "ppo_robot_reach_v8.zip")
DEFAULT_GRAPH_PATH = os.path.join("results", "ppo_robot_reach_v8_distance_vs_step.png")
MAX_EVAL_STEPS = 200


def evaluate_model(model_path=DEFAULT_MODEL_PATH, graph_path=DEFAULT_GRAPH_PATH):
    """Load a PPO model and evaluate it in the PyBullet robot environment.

    This script is intentionally simple and beginner-friendly. It does not retrain the
    model or change the state/action/reward design in RobotReachEnv.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    os.makedirs(os.path.dirname(graph_path) or ".", exist_ok=True)

    env = RobotReachEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset(seed=1)
    # Observation is now 23D: [joint_pos(7), joint_vel(7), ee_pos(3), ee_vel(3), target_pos(3)]
    initial_distance = float(np.linalg.norm(obs[14:17] - obs[20:23]))

    distances = []
    rewards = []
    total_reward = 0.0
    reached = False
    step_count = 0
    min_distance = float("inf")

    print("\nStarting PPO evaluation in PyBullet GUI...")
    print(f"Initial distance to target: {initial_distance:.4f}")

    for step in range(MAX_EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Use the new observation indices for 23D observation
        distance = float(np.linalg.norm(obs[14:17] - obs[20:23]))
        min_distance = min(min_distance, distance)
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
    min_distance = min_distance if np.isfinite(min_distance) else initial_distance

    print("\nEvaluation summary")
    print(f"Initial distance: {initial_distance:.4f}")
    print(f"Final distance: {final_distance:.4f}")
    print(f"Minimum distance: {min_distance:.4f}")
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
    plt.savefig(graph_path)
    plt.close()

    print(f"Graph saved to: {graph_path}")
    env.close()

    return {
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "min_distance": min_distance,
        "total_reward": total_reward,
        "steps": step_count,
        "target_reached": reached,
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    graph_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GRAPH_PATH
    evaluate_model(model_path=model_path, graph_path=graph_path)
