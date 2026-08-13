import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


class RobotReachEnv(gym.Env):
    """A very small Gymnasium environment for a PyBullet robotic arm target-reaching task.

    This is intentionally beginner-friendly and designed for learning the RL loop.
    We are not training a model yet, and we are not using PPO or Stable-Baselines3.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode="human"):
        super().__init__()

        self.render_mode = render_mode
        self.physics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        self.ground_id = None
        self.robot_id = None
        self.target_id = None
        self.target_position = np.array([0.5, 0.0, 0.5], dtype=np.float32)
        self.distance_threshold = 0.15
        self.max_steps = 200
        self.step_count = 0
        self.last_distance = None
        self.joint_indices = list(range(7))

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7 + 3 + 3,),
            dtype=np.float32,
        )

        # Continuous action space for 7 robot joints.
        # We choose a bounded range of [-1, 1] and then scale it into joint motion.
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        self.joint_lower_limits = np.array([], dtype=np.float32)
        self.joint_upper_limits = np.array([], dtype=np.float32)

        self._load_scene()

    def _load_scene(self):
        """Build the basic environment: ground plane, robot, and target."""
        self.ground_id = p.loadURDF("plane.urdf")

        robot_start_position = [0, 0, 0.0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            robot_start_position,
            robot_start_orientation,
            useFixedBase=False,
        )

        self._create_target()
        self._collect_joint_limits()

        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0.75],
        )

    def _create_target(self):
        """Create a red sphere target at a fixed position in the world."""
        sphere_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[1, 0, 0, 1],
        )
        sphere_collision = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.05,
        )

        self.target_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.target_position.tolist(),
        )

    def _collect_joint_limits(self):
        """Read the robot's joint limits so actions can be safely scaled."""
        lower_limits = []
        upper_limits = []

        for joint_index in self.joint_indices:
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            lower_limits.append(float(joint_info[8]))
            upper_limits.append(float(joint_info[9]))

        self.joint_lower_limits = np.array(lower_limits, dtype=np.float32)
        self.joint_upper_limits = np.array(upper_limits, dtype=np.float32)

    def _get_joint_positions(self):
        """Get the current joint angles for all 7 robot joints."""
        joint_positions = [p.getJointState(self.robot_id, idx)[0] for idx in self.joint_indices]
        return np.array(joint_positions, dtype=np.float32)

    def _get_end_effector_position(self):
        """Return the end-effector position in world coordinates."""
        end_effector_index = p.getNumJoints(self.robot_id) - 1
        end_effector_state = p.getLinkState(
            self.robot_id,
            end_effector_index,
            computeForwardKinematics=True,
        )
        return np.array(end_effector_state[0], dtype=np.float32)

    def _get_observation(self):
        """Observation = [7 joint positions, 3 end-effector coordinates, 3 target coordinates]."""
        joint_positions = self._get_joint_positions()
        end_effector_position = self._get_end_effector_position()
        observation = np.concatenate(
            [
                joint_positions,
                end_effector_position,
                self.target_position,
            ]
        ).astype(np.float32)
        return observation

    def _apply_action(self, action):
        """Convert the continuous action into a valid joint-target update."""
        current_joint_positions = self._get_joint_positions()
        action = np.asarray(action, dtype=np.float32)

        # Scale the continuous action to a moderate movement range.
        scaled_action = action * 0.2
        desired_joint_positions = current_joint_positions + scaled_action
        desired_joint_positions = np.clip(
            desired_joint_positions,
            self.joint_lower_limits,
            self.joint_upper_limits,
        )

        for joint_index, target_position in zip(self.joint_indices, desired_joint_positions):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target_position),
                force=200.0,
            )

    def reset(self, *, seed=None, options=None):
        """Reset the environment and return an initial observation."""
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        self._load_scene()
        self.step_count = 0
        observation = self._get_observation()
        self.last_distance = float(np.linalg.norm(observation[7:10] - observation[10:13]))
        return observation, {}

    def step(self, action):
        """Advance the simulation by one action and return RL values."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}."
            )

        self._apply_action(action)
        p.stepSimulation()
        self.step_count += 1

        observation = self._get_observation()
        end_effector_position = self._get_end_effector_position()
        distance_to_target = float(np.linalg.norm(end_effector_position - self.target_position))

        # Dense shaping reward: reward progress toward the target directly.
        # A smaller distance should produce a higher reward, while moving away should be penalized.
        previous_distance = self.last_distance if self.last_distance is not None else distance_to_target
        progress = previous_distance - distance_to_target
        distance_penalty = 0.5 * distance_to_target
        away_penalty = 0.8 * max(0.0, -progress)

        # Keep the reward numerically stable and easy to understand for beginners.
        # The reward is small but dense: we want PPO to feel each useful movement.
        reward = 2.5 * progress - distance_penalty - away_penalty

        # Extra reward when the end-effector gets close to the target.
        if distance_to_target <= 0.5:
            reward += 2.0

        # Clear success bonus when the target is reached.
        if distance_to_target <= self.distance_threshold:
            reward += 30.0

        # Clip to a stable range so one bad action does not explode the learning signal.
        reward = float(np.clip(reward, -15.0, 35.0))
        self.last_distance = distance_to_target

        terminated = bool(distance_to_target <= self.distance_threshold)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "distance_to_target": distance_to_target,
            "end_effector_position": end_effector_position,
            "target_position": self.target_position.copy(),
        }

        return observation, float(reward), terminated, truncated, info

    def render(self):
        """Stub render method for Gymnasium compatibility."""
        return None

    def close(self):
        """Disconnect the PyBullet connection cleanly."""
        p.disconnect(self.physics_client)


if __name__ == "__main__":
    env = RobotReachEnv(render_mode="human")
    obs, info = env.reset()
    print("Observation shape:", env.observation_space.shape)
    print("Observation sample:", obs)
    action = env.action_space.sample()
    print("Action sample:", action)
    next_obs, reward, terminated, truncated, info = env.step(action)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)
    env.close()
