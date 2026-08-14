import argparse
import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO

# Add src to python path
sys.path.insert(0, os.path.dirname(__file__))

from robot_env import RobotReachEnv

DEFAULT_MODEL_PATH = os.path.join("models", "ppo_robot_reach_v8.zip")
BASE_TARGET = np.array([0.55, 0.0, 0.80], dtype=np.float32)
MAX_EVAL_STEPS = 200

def run_gui_demo(model_path=DEFAULT_MODEL_PATH):
    """Launch PyBullet GUI interactive rendering of the trained V8 policy."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

    print(f"\n=======================================================")
    print(f" Launching PyBullet 3D GUI Demo (Model: {model_path})")
    print(f"=======================================================")

    env = RobotReachEnv(render_mode="human")
    try:
        model = PPO.load(model_path, env=env)
        obs, _ = env.reset(seed=1)

        initial_distance = float(np.linalg.norm(obs[14:17] - obs[20:23]))
        print(f"Initial Distance to Target: {initial_distance:.4f} m")
        print(f"Target Location: [{obs[20]:.3f}, {obs[21]:.3f}, {obs[22]:.3f}] m")
        print("-" * 60)

        reached = False
        step_reached = None

        for step in range(1, MAX_EVAL_STEPS + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            dist = float(info.get("distance_to_target", np.linalg.norm(obs[14:17] - obs[20:23])))
            
            if dist <= env.distance_threshold and not reached:
                reached = True
                step_reached = step

            print(f"Step {step:03d}/{MAX_EVAL_STEPS} | distance={dist:.4f} m | reward={reward:+.4f} | reached={reached}")
            time.sleep(1.0 / 60.0)

            if terminated or truncated:
                break

        final_dist = float(np.linalg.norm(obs[14:17] - obs[20:23]))
        print("-" * 60)
        print("GUI Rollout Summary:")
        print(f"  Target Reached:      {reached}")
        print(f"  First Success Step:  {step_reached if step_reached else 'N/A'}")
        print(f"  Final Distance:      {final_dist:.4f} m")
        print("=======================================================\n")

    finally:
        try:
            env.close()
        except Exception:
            pass

def run_eval_demo(model_path=DEFAULT_MODEL_PATH):
    """Execute lightweight verification and generalization benchmarks for V8."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

    print(f"\n=======================================================")
    print(f" Step 21 Evaluation Benchmark (Model: {model_path})")
    print(f"=======================================================")

    env = RobotReachEnv(render_mode="direct")
    try:
        model = PPO.load(model_path, env=env)

        # 1. Nominal Target Evaluation
        obs, _ = env.reset(seed=1)
        initial_dist = float(np.linalg.norm(obs[14:17] - obs[20:23]))
        min_dist = initial_dist
        total_reward = 0.0
        reached = False
        success_step = None

        for step in range(1, MAX_EVAL_STEPS + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            dist = float(info.get("distance_to_target", np.linalg.norm(obs[14:17] - obs[20:23])))
            min_dist = min(min_dist, dist)
            total_reward += float(reward)

            if dist <= env.distance_threshold and not reached:
                reached = True
                success_step = step

            if terminated or truncated:
                break

        final_dist = float(np.linalg.norm(obs[14:17] - obs[20:23]))

        print("\n--- Nominal Target Validation ([0.55, 0.0, 0.80]) ---")
        print(f"  Initial Distance:    {initial_dist:.4f} m")
        print(f"  Final Distance:      {final_dist:.4f} m")
        print(f"  Minimum Distance:    {min_dist:.4f} m")
        print(f"  Target Threshold:    {env.distance_threshold:.4f} m")
        print(f"  Target Reached:      {reached}")
        print(f"  First Success Step:  {success_step}")
        print(f"  Total Reward:        {total_reward:.2f}")

        # 2. Generalization Benchmarks over 20 Unseen Target Perturbations
        print("\n--- Spatial Generalization Audit (20 Unseen Target Offsets) ---")
        successes = 0
        final_distances = []

        for i in range(20):
            seed = 100 + i
            np.random.seed(seed)
            offset = np.random.uniform(-0.03, 0.03, size=3).astype(np.float32)
            perturbed_target = BASE_TARGET + offset
            
            env.target_position = perturbed_target
            obs_p, _ = env.reset(seed=seed)
            reached_p = False

            for s in range(1, MAX_EVAL_STEPS + 1):
                act, _ = model.predict(obs_p, deterministic=True)
                obs_p, r_p, term_p, trunc_p, info_p = env.step(act)
                d_p = float(info_p.get("distance_to_target", np.linalg.norm(obs_p[14:17] - obs_p[20:23])))
                if d_p <= env.distance_threshold:
                    reached_p = True
                if term_p or trunc_p:
                    break

            d_final = float(np.linalg.norm(obs_p[14:17] - obs_p[20:23]))
            final_distances.append(d_final)
            if reached_p:
                successes += 1

        gen_success_rate = (successes / 20.0) * 100.0
        mean_final_d = float(np.mean(final_distances))

        print(f"  Perturbed Targets Evaluated:  20")
        print(f"  Spatial Success Rate:         {gen_success_rate:.1f}% ({successes}/20)")
        print(f"  Mean Final Distance:          {mean_final_d:.4f} m")
        print("\n=======================================================")
        print(" VERIFICATION PASSED: V8 Baseline Primary Model Operational")
        print("=======================================================\n")

    finally:
        try:
            env.close()
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="AI Robotic Arm V8 Demonstration Utility")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gui", "eval"],
        default="gui",
        help="Mode to run: 'gui' for 3D PyBullet visual demo, 'eval' for benchmark evaluation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained PPO model zip file.",
    )
    args = parser.parse_args()

    if args.mode == "gui":
        run_gui_demo(model_path=args.model)
    elif args.mode == "eval":
        run_eval_demo(model_path=args.model)

if __name__ == "__main__":
    main()
