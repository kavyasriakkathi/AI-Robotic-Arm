import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Add src to python path
sys.path.insert(0, os.path.dirname(__file__))

from robot_env import RobotReachEnv

MODEL_PATH = os.path.join("models", "ppo_robot_reach_v8.zip")
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

def print_table(title, results, agg):
    print(f"\n=======================================================")
    print(f" {title}")
    print(f"=======================================================")
    print(f"{'Seed':<5} | {'Target (X, Y, Z)':<25} | {'Init (m)':<8} | {'Final (m)':<9} | {'Min (m)':<8} | {'Reached':<7} | {'Step':<5} | {'Reward':<9}")
    print("-" * 90)
    for r in results:
        t_str = f"[{r['target'][0]:.3f}, {r['target'][1]:.3f}, {r['target'][2]:.3f}]"
        step_str = str(r['first_success_step']) if r['first_success_step'] is not None else "-"
        print(f"{r['seed']:<5} | {t_str:<25} | {r['initial_distance']:<8.4f} | {r['final_distance']:<9.4f} | {r['min_distance']:<8.4f} | {str(r['target_reached']):<7} | {step_str:<5} | {r['total_reward']:<9.2f}")
    print("-" * 90)
    print("Aggregate Statistics:")
    print(f"  Episodes Evaluated:       {agg['episodes']}")
    print(f"  Success Rate:             {agg['success_rate']:.1f}%")
    print(f"  Mean Final Distance:      {agg['mean_final']:.4f} m")
    print(f"  Median Final Distance:    {agg['median_final']:.4f} m")
    print(f"  Best Final Distance:      {agg['best_final']:.4f} m")
    print(f"  Worst Final Distance:     {agg['worst_final']:.4f} m")
    print(f"  Mean Minimum Distance:    {agg['mean_min']:.4f} m")
    if agg['mean_success_steps'] is not None:
        print(f"  Mean Steps to Success:    {agg['mean_success_steps']:.1f}")
    else:
        print(f"  Mean Steps to Success:    N/A (0 successes)")

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    env = RobotReachEnv(render_mode="direct")
    try:
        model = PPO.load(MODEL_PATH, env=env)

        # Test A: Deterministic Multi-Seed Validation (Seeds 1-10)
        results_a = []
        for seed in range(1, 11):
            res = run_single_episode(model, env, seed=seed, target_pos=BASE_TARGET, deterministic=True)
            results_a.append(res)
        agg_a = compute_aggregates(results_a)
        print_table("Test A: Deterministic Multi-Seed Validation (Seeds 1-10)", results_a, agg_a)

        # Test B: Stochastic Multi-Seed Validation (Seeds 1-10)
        results_b = []
        for seed in range(1, 11):
            res = run_single_episode(model, env, seed=seed, target_pos=BASE_TARGET, deterministic=False)
            results_b.append(res)
        agg_b = compute_aggregates(results_b)
        print_table("Test B: Stochastic Multi-Seed Validation (Seeds 1-10)", results_b, agg_b)

        # Test C: Target Perturbation Generalization (Seeds 1-10, reproducible random offsets)
        results_c = []
        for seed in range(1, 11):
            np.random.seed(seed)
            offset = np.random.uniform(-0.03, 0.03, size=3).astype(np.float32)
            perturbed_target = BASE_TARGET + offset
            res = run_single_episode(model, env, seed=seed, target_pos=perturbed_target, deterministic=True)
            results_c.append(res)
        agg_c = compute_aggregates(results_c)
        print_table("Test C: Target Perturbation Generalization (+/-3 cm Offsets)", results_c, agg_c)

        print("\nSummary Table:")
        print(f"| Evaluation | Episodes | Success Rate | Mean Final Distance | Best | Worst | Mean Success Steps |")
        print(f"| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        step_a_str = f"{agg_a['mean_success_steps']:.1f}" if agg_a['mean_success_steps'] is not None else "N/A"
        step_b_str = f"{agg_b['mean_success_steps']:.1f}" if agg_b['mean_success_steps'] is not None else "N/A"
        step_c_str = f"{agg_c['mean_success_steps']:.1f}" if agg_c['mean_success_steps'] is not None else "N/A"
        print(f"| Deterministic Seeds | {agg_a['episodes']} | {agg_a['success_rate']:.1f}% | {agg_a['mean_final']:.4f} m | {agg_a['best_final']:.4f} m | {agg_a['worst_final']:.4f} m | {step_a_str} |")
        print(f"| Stochastic Seeds | {agg_b['episodes']} | {agg_b['success_rate']:.1f}% | {agg_b['mean_final']:.4f} m | {agg_b['best_final']:.4f} m | {agg_b['worst_final']:.4f} m | {step_b_str} |")
        print(f"| Target Perturbation | {agg_c['episodes']} | {agg_c['success_rate']:.1f}% | {agg_c['mean_final']:.4f} m | {agg_c['best_final']:.4f} m | {agg_c['worst_final']:.4f} m | {step_c_str} |")

    finally:
        env.close()

if __name__ == "__main__":
    main()
