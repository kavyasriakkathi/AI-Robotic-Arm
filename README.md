# AI-Powered Robotic Arm Target Reaching Using Reinforcement Learning

A 7-DOF Kuka iiwa robotic arm simulation built with **PyBullet**, **Gymnasium**, and **Stable-Baselines3 (PPO)**. The agent learns continuous joint motor control to reach 3D spatial targets with millimeter-level precision.

---

## 📌 Project Overview & Problem Statement

Controlling multi-joint robotic manipulators to reach arbitrary spatial targets requires coordinating non-linear arm kinematics without fixed analytical inverse kinematics solver constraints. This project formulates 3D target reaching for a 7-DOF arm as a continuous Deep Reinforcement Learning (DRL) problem solved via **Proximal Policy Optimization (PPO)**.

### Objectives
1. **Kinematic Precision:** Reach spatial targets within a strict $0.15\text{ m}$ ($15\text{ cm}$) distance threshold.
2. **Reproducibility:** Achieve $100\%$ deterministic reaching stability across independent random seeds.
3. **Zero-Shot Generalization:** Generalize to unseen spatial target coordinates without retraining.

---

## 🛠️ Technology Stack

* **Physics Simulation:** PyBullet (`kuka_iiwa/model.urdf`)
* **Environment Interface:** Gymnasium (`gym.Env`)
* **Reinforcement Learning:** Stable-Baselines3 (`PPO`, `MlpPolicy`)
* **Numeric Computing & Visualization:** NumPy, Matplotlib
* **Testing:** PyTest
* **Language:** Python 3.12+

---

## 🦾 System Architecture & Specification

### 1. Robotic Arm & Physics Environment
* **Robot Model:** 7-DOF Kuka iiwa articulated arm with fixed base $\mathbf{p}_{\text{base}} = [0.0, 0.0, 0.0]^T$.
* **Initial Neutral Pose:** $\mathbf{q}_0 = [0.0, -0.6, 0.0, -1.4, 0.0, 1.0, 0.0]^T \text{ rad}$.
* **Nominal Target Position:** $\mathbf{p}_{\text{target}} = [0.55, 0.0, 0.80]^T \text{ m}$ (Red sphere visual/collision shape).
* **Episode Horizon:** `MAX_EVAL_STEPS = 200` steps.
* **Success Threshold:** $d \le 0.1500\text{ m}$ ($15\text{ cm}$).

### 2. State Space (23-Dimensional Continuous Box)
$$\mathbf{o} = [\mathbf{q}_{7}, \dot{\mathbf{q}}_{7}, \mathbf{p}_{\text{EE}, 3}, \mathbf{v}_{\text{EE}, 3}, \mathbf{p}_{\text{target}, 3}]^T \in \mathbb{R}^{23}$$
* $\mathbf{q}_{7}$: 7 joint positions (radians)
* $\dot{\mathbf{q}}_{7}$: 7 joint velocities (rad/s)
* $\mathbf{p}_{\text{EE}, 3}$: 3D end-effector Cartesian position (meters)
* $\mathbf{v}_{\text{EE}, 3}$: 3D end-effector Cartesian velocity (m/s)
* $\mathbf{p}_{\text{target}, 3}$: 3D target Cartesian position (meters)

### 3. Action Space (7-Dimensional Continuous Box)
$$\mathbf{a} \in [-1.0, 1.0]^7 \subset \mathbb{R}^7$$
Actions represent normalized continuous motor commands scaled by $0.2\text{ rad}$ per step to update target joint positions via PyBullet position control (`p.POSITION_CONTROL`, max force = $200.0\text{ N}\cdot\text{m}$).

### 4. Reward Shaping Engineering
The dense reward function balances progress, precision, directional movement, and termination:
$$R = 4.0 \cdot \Delta d - 0.5 \cdot d - 0.8 \cdot \max(0, -\Delta d) + 4.0 \cdot \max(0, 0.5 - d) + R_{\text{goal}}$$
Where:
* $\Delta d = d_{\text{prev}} - d_{\text{current}}$ (Relative progress toward target)
* $-0.5 \cdot d$ (Distance penalty encouraging rapid approach)
* $-0.8 \cdot \max(0, -\Delta d)$ (Penalty for moving away from target)
* $+4.0 \cdot \max(0, 0.5 - d)$ (Continuous precision bonus below $0.50\text{ m}$)
* $R_{\text{goal}} = +30.0$ when $d \le 0.15\text{ m}$ (Termination goal bonus)

---

## 📊 Validated V8 Benchmark Results

The primary benchmark model is **`models/ppo_robot_reach_v8.zip`** (trained for 100,000 timesteps).

| Benchmark Experiment | Episodes | Success Rate | Mean Final Distance | Best Final Distance | Worst Final Distance | Mean Steps to Success |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Target (Deterministic)** | 10 | **100.0%** | **0.1493 m** | **0.1493 m** | **0.1493 m** | **151.0** |
| **20 Unseen Target Perturbations** | 20 | **95.0%** | **0.1499 m** | **0.1486 m** | **0.1603 m** | **148.6** |
| **Stochastic Action Robustness** | 10 | **60.0%** | **0.1811 m** | **0.1459 m** | **0.2558 m** | **103.2** |

### Key Benchmark Highlights
* **Nominal Convergence:** V8 starts at an initial distance of `0.4334 m` and reliably reaches **`0.1493 m`** at **Step 151**.
* **Multi-Seed Reproducibility:** 100% deterministic success across independent environment seeds 1–10 with zero metric variance.
* **Zero-Shot Spatial Generalization:** 95.0% success rate across 20 unseen spatial target position perturbations ($\pm 3\text{ cm}$ offsets) with a mean final distance of `0.1499 m`.

---

## 🔬 V8 vs. V9 Target-Randomization Experiment

During Step 19, an experimental model **V9** (`models/ppo_robot_reach_v9.zip`) was trained with per-episode target randomization ($X \in [0.52, 0.58]$, $Y \in [-0.03, 0.03]$, $Z \in [0.77, 0.83]$) to test if domain randomization improved spatial generalization.

### Findings & Decision:
* **V9 Failed to Converge:** V9 achieved a 0% success rate on both nominal and perturbed targets (final distance $\approx 0.3017\text{ m}$). Target randomization without curriculum learning caused exploration collapse within 100k timesteps.
* **V8 Inherent Generalization:** Evaluation revealed that V8's 23D observation space (which explicitly includes target coordinates relative to the end-effector) already provides **95.0% zero-shot spatial generalization**.
* **Decision:** **V8 (`models/ppo_robot_reach_v8.zip`) is retained as the official primary model.** V9 is rejected.

---

## 📁 Repository Architecture

```
AI-Robotic-Arm/
├── src/
│   ├── robot_env.py                  # Gymnasium Environment & PyBullet Simulation
│   ├── train.py                      # SB3 PPO Training Script
│   ├── evaluate.py                   # Baseline Evaluation & Graph Script
│   ├── evaluate_generalization.py      # Multi-Seed & Target Perturbation Generalization Suite
│   ├── evaluate_v9_comparison.py      # V8 vs V9 Comparative Analysis Script
│   ├── demo.py                       # CLI GUI / Evaluation Entry Point
│   └── simulation.py                 # Standalone PyBullet Kinematics Inspector
├── tests/
│   └── test_reward_shaping.py        # PyTest Automated Test Suite (5 unit tests)
├── models/
│   └── ppo_robot_reach_v8.zip        # Primary Validated Benchmark PPO Model
├── results/
│   ├── ppo_robot_reach_v8_distance_vs_step.png
│   └── ppo_robot_reach_v9_comparison.png
├── requirements.txt                  # Python Project Dependencies
└── README.md                         # Documentation
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Python 3.12 virtual environment:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated PyTest Suite
Verify environment dynamics, observation/action shapes, and reward shaping integrity:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_reward_shaping.py -q
```

---

## 🎮 Running Demonstrations & Benchmarks

### 1. Interactive 3D PyBullet GUI Demo (`demo.py --mode gui`)
Launch real-time 60 FPS 3D PyBullet visual rendering of the trained V8 policy reaching the target:
```powershell
.\.venv\Scripts\python.exe src/demo.py --mode gui
```

### 2. Fast Evaluation Benchmark (`demo.py --mode eval`)
Run lightweight automated evaluation of nominal target performance and 20 unseen target perturbations:
```powershell
.\.venv\Scripts\python.exe src/demo.py --mode eval
```

### 3. Full Generalization Benchmark (`evaluate_generalization.py`)
Run comprehensive multi-seed deterministic, stochastic, and target perturbation experiments:
```powershell
.\.venv\Scripts\python.exe src/evaluate_generalization.py
```

### 4. Baseline Evaluation & Graph Generation (`evaluate.py`)
Execute single-seed evaluation and update distance vs step convergence curves:
```powershell
.\.venv\Scripts\python.exe src/evaluate.py
```

---

## 🔮 Limitations & Future Work

1. **Stochastic Action Variance:** Under stochastic action sampling (`deterministic=False`), success drops to 60%. Future iterations can reduce action Gaussian log standard deviation or implement entropy decay.
2. **Multi-Target Domain Randomization:** While V8 achieves 95% zero-shot generalization to $\pm 3\text{ cm}$ target perturbations, scaling to larger workspace regions ($> 10\text{ cm}$ target offsets) will benefit from **Curriculum Learning** (progressively introducing target noise) over longer training horizons ($300\text{k}\text{ timesteps}$).
