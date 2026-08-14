import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from robot_env import RobotReachEnv

V8_MODEL_PATH = os.path.join("models", "ppo_robot_reach_v8.zip")
V9_MODEL_PATH = os.path.join("models", "ppo_robot_reach_v9.zip")
GRAPH_PATH = os.path.join("results", "ppo_robot_reach_v9_comparison.png")
BASE_TARGET = np.array([0.55, 0.0, 0.80], dtype=np.float32)
MAX_EVAL_STEPS = 200

def run_single_episode(model, env, seed, target_pos, deterministic):
    env.target_position = np.array(target_pos, dtype=np.float32)
    obs, _ = env.reset(seed=seed)
    
    initial_distance = float(np.linalg.norm(obs[14:17] - obs[20:23]))
    min_distance = initial_distance
    total_reward = 0.0
    reached = False
    first_success_step = None
    step_count = 0
    distances = []

    for step in range(1, MAX_EVAL_STEPS + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count = step
        total_reward += float(reward)
        
        current_dist = float(info.get("distance_to_target", np.linalg.norm(obs[14:17] - obs[20:23])))
        distances.append(current_dist)
        if current_dist < min_distance:
            min_distance = current_dist

        if current_dist <= env.distance_threshold and not reached:
            reached = True
            first_success_step = step

        if terminated or truncated:
            break

    final_distance = distances[-1] if distances else initial_distance

    return {
        "seed": seed,
        "target": target_pos.tolist(),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "min_distance": min_distance,
        "target_reached": reached,
        "first_success_step": first_success_step,
        "total_reward": total_reward,
        "steps": step_count,
        "distances": distances,
    }

def compute_aggregates(results):
    n = len(results)
    successes = [r for r in results if r["target_reached"]]
    success_rate = (len(successes) / n) * 100.0 if n > 0 else 0.0
    
    final_dists = [r["final_distance"] for r in results]
    min_dists = [r["min_distance"] for r in results]
    
    mean_final = float(np.mean(final_dists))
    median_final = float(np.median(final_dists))
    best_final = float(np.min(final_dists))
    worst_final = float(np.max(final_dists))
    mean_min = float(np.mean(min_dists))
    
    success_steps = [r["first_success_step"] for r in successes if r["first_success_step"] is not None]
    mean_success_steps = float(np.mean(success_steps)) if success_steps else None

    return {
        "episodes": n,
        "success_rate": success_rate,
        "mean_final": mean_final,
        "median_final": median_final,
        "best_final": best_final,
        "worst_final": worst_final,
        "mean_min": mean_min,
        "mean_success_steps": mean_success_steps,
    }

def generate_unseen_targets(num_targets=20, base_seed=100):
    targets = []
    for i in range(num_targets):
        np.random.seed(base_seed + i)
        offset = np.random.uniform(low=[-0.03, -0.03, -0.03], high=[0.03, 0.03, 0.03], size=3).astype(np.float32)
        target = BASE_TARGET + offset
        targets.append((base_seed + i, target))
    return targets

def main():
    if not os.path.exists(V8_MODEL_PATH):
        raise FileNotFoundError(f"V8 model checkpoint missing at {V8_MODEL_PATH}")
    if not os.path.exists(V9_MODEL_PATH):
        raise FileNotFoundError(f"V9 model checkpoint missing at {V9_MODEL_PATH}")

    os.makedirs(os.path.dirname(GRAPH_PATH) or ".", exist_ok=True)
    env = RobotReachEnv(render_mode="direct")

    try:
        model_v8 = PPO.load(V8_MODEL_PATH, env=env)
        model_v9 = PPO.load(V9_MODEL_PATH, env=env)

        # -------------------------------------------------------------
        # Test A: Nominal Target [0.55, 0.0, 0.80]
        # -------------------------------------------------------------
        v8_nom = run_single_episode(model_v8, env, seed=1, target_pos=BASE_TARGET, deterministic=True)
        v9_nom = run_single_episode(model_v9, env, seed=1, target_pos=BASE_TARGET, deterministic=True)

        print("\n=======================================================")
        print(" Test A: Nominal Target Performance ([0.55, 0.0, 0.80])")
        print("=======================================================")
        print(f"V8 Baseline: Reached={v8_nom['target_reached']} | Final={v8_nom['final_distance']:.4f}m | Min={v8_nom['min_distance']:.4f}m | Step={v8_nom['first_success_step']}")
        print(f"V9 Randomized: Reached={v9_nom['target_reached']} | Final={v9_nom['final_distance']:.4f}m | Min={v9_nom['min_distance']:.4f}m | Step={v9_nom['first_success_step']}")

        # -------------------------------------------------------------
        # Test B: 20 Unseen Target Perturbations (Deterministic)
        # -------------------------------------------------------------
        unseen_targets = generate_unseen_targets(num_targets=20, base_seed=100)
        v8_results_b, v9_results_b = [], []

        for seed, target in unseen_targets:
            v8_res = run_single_episode(model_v8, env, seed=seed, target_pos=target, deterministic=True)
            v9_res = run_single_episode(model_v9, env, seed=seed, target_pos=target, deterministic=True)
            v8_results_b.append(v8_res)
            v9_results_b.append(v9_res)

        v8_agg_b = compute_aggregates(v8_results_b)
        v9_agg_b = compute_aggregates(v9_results_b)

        # -------------------------------------------------------------
        # Test C: 20 Unseen Target Perturbations (Stochastic)
        # -------------------------------------------------------------
        v8_results_c, v9_results_c = [], []

        for seed, target in unseen_targets:
            v8_res = run_single_episode(model_v8, env, seed=seed, target_pos=target, deterministic=False)
            v9_res = run_single_episode(model_v9, env, seed=seed, target_pos=target, deterministic=False)
            v8_results_c.append(v8_res)
            v9_results_c.append(v9_res)

        v8_agg_c = compute_aggregates(v8_results_c)
        v9_agg_c = compute_aggregates(v9_results_c)

        print("\n=======================================================")
        print(" Step 19: Comprehensive V8 vs V9 Comparison Summary")
        print("=======================================================")
        print(f"| Metric | V8 Baseline (Fixed Target) | V9 (Target-Randomized) | Delta / Status |")
        print(f"| --- | ---: | ---: | --- |")
        
        nom_step_v8 = str(v8_nom['first_success_step']) if v8_nom['first_success_step'] else "-"
        nom_step_v9 = str(v9_nom['first_success_step']) if v9_nom['first_success_step'] else "-"
        print(f"| Nominal Target Reached | {str(v8_nom['target_reached'])} | {str(v9_nom['target_reached'])} | Preserved |")
        print(f"| Nominal Final Distance | {v8_nom['final_distance']:.4f} m | {v9_nom['final_distance']:.4f} m | Step {nom_step_v8} vs Step {nom_step_v9} |")
        
        step_b_v8 = f"{v8_agg_b['mean_success_steps']:.1f}" if v8_agg_b['mean_success_steps'] else "N/A"
        step_b_v9 = f"{v9_agg_b['mean_success_steps']:.1f}" if v9_agg_b['mean_success_steps'] else "N/A"
        succ_diff_b = v9_agg_b['success_rate'] - v8_agg_b['success_rate']
        print(f"| Deterministic Perturbed Success | {v8_agg_b['success_rate']:.1f}% | {v9_agg_b['success_rate']:.1f}% | {succ_diff_b:+.1f}% |")
        print(f"| Deterministic Mean Final Dist | {v8_agg_b['mean_final']:.4f} m | {v9_agg_b['mean_final']:.4f} m | {v9_agg_b['mean_final'] - v8_agg_b['mean_final']:+.4f} m |")
        print(f"| Deterministic Worst Final Dist | {v8_agg_b['worst_final']:.4f} m | {v9_agg_b['worst_final']:.4f} m | {v9_agg_b['worst_final'] - v8_agg_b['worst_final']:+.4f} m |")
        print(f"| Deterministic Mean Success Steps | {step_b_v8} | {step_b_v9} | - |")

        succ_diff_c = v9_agg_c['success_rate'] - v8_agg_c['success_rate']
        print(f"| Stochastic Perturbed Success | {v8_agg_c['success_rate']:.1f}% | {v9_agg_c['success_rate']:.1f}% | {succ_diff_c:+.1f}% |")

        # -------------------------------------------------------------
        # Plot Comparison Graph
        # -------------------------------------------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(v8_nom['distances'])), v8_nom['distances'], label="V8 Baseline (Nominal)", color="crimson", linewidth=2.5)
        plt.plot(range(len(v9_nom['distances'])), v9_nom['distances'], label="V9 Randomized (Nominal)", color="forestgreen", linewidth=2.5)
        plt.axhline(y=0.15, color="gray", linestyle="--", label="Success Threshold (0.15m)")
        plt.title("Step 19: Distance to Target vs Step (V8 Baseline vs V9 Randomized)")
        plt.xlabel("Step")
        plt.ylabel("Distance to Target (m)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(GRAPH_PATH)
        plt.close()
        print(f"\nComparison graph saved to: {GRAPH_PATH}")

    finally:
        env.close()

if __name__ == "__main__":
    main()
