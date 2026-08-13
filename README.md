# AI-Powered Robotic Arm Target Reaching Using Reinforcement Learning

This project is being built step by step. This first step sets up the project folder structure and creates a minimal PyBullet simulation to confirm the environment works before adding the reinforcement learning logic.

## Current status

- Basic project structure created
- PyBullet dependency installed
- Minimal simulation with ground plane and robot is included
- RL training code is intentionally not added yet

## Run the simulation

1. Open a terminal in the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the simulation:
   ```bash
   python src/simulation.py
   ```

This will open the PyBullet GUI, load the ground plane, add a simple robotic arm, and keep the simulation running so you can see the robot.

## Next steps

The next steps will add the actual environment, reward function, and eventually the RL training code. This step is intentionally limited to setup and visualization only.
