# AI-Powered Robotic Arm Target Reaching Using Reinforcement Learning

This project is being built step by step. It currently includes a PyBullet robot simulation, a simple Gymnasium environment, and a PPO smoke-test setup to confirm compatibility before any real training begins.

## Current status

- Basic project structure created
- PyBullet dependency installed
- Gymnasium environment created
- PPO smoke test prepared
- Full model training is intentionally not started yet

## Run the simulation

1. Open a terminal in the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the standalone simulation:
   ```bash
   python src/simulation.py
   ```
4. Or run the PPO smoke test:
   ```bash
   .\.venv\Scripts\python.exe src/train.py
   ```

## Project notes

- `src/simulation.py` is the standalone robot visualization and inspection script.
- `src/robot_env.py` is the Gymnasium environment for RL interactions.
- `src/train.py` is intentionally a small PPO compatibility smoke test, not full training.

## Next steps

The next steps will add more structured training logic and deeper RL behavior, but this checkpoint focuses only on environment compatibility and smoke testing.
