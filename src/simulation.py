import math
import time

import pybullet as p
import pybullet_data


def print_joint_information(robot_id):
    """Print each joint index and name so the robot structure is easy to inspect."""
    num_joints = p.getNumJoints(robot_id)
    print(f"\nRobot has {num_joints} joints.")

    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        joint_lower_limit = joint_info[8]
        joint_upper_limit = joint_info[9]
        joint_state = p.getJointState(robot_id, joint_index)
        joint_position = joint_state[0]

        print(
            f"Joint index {joint_index:>2} | "
            f"name: {joint_name:<20} | "
            f"type: {joint_type} | "
            f"position: {joint_position:.3f} rad | "
            f"limits: [{joint_lower_limit}, {joint_upper_limit}]"
        )

    end_effector_joint_index = num_joints - 1
    end_effector_link_name = p.getJointInfo(robot_id, end_effector_joint_index)[12].decode("utf-8")
    print(f"\nEnd-effector candidate: joint index {end_effector_joint_index}, link name {end_effector_link_name}")


def manual_joint_demo(robot_id):
    """Move a few joints through a small manual position-control demo."""
    # For this KUKA robot, joints 0 through 6 are the arm joints.
    joint_indices = list(range(7))

    for joint_index in joint_indices:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,
            force=200.0,
        )

    print("\nStarting a small manual joint-control demonstration...")
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time

        # These values create a gentle, controlled motion across several joints.
        target_positions = [
            0.25 * math.sin(elapsed * 1.3),
            0.20 * math.cos(elapsed * 1.1),
            -0.18 * math.sin(elapsed * 1.5),
            0.12 * math.cos(elapsed * 1.7),
            0.10 * math.sin(elapsed * 2.0),
            -0.08 * math.cos(elapsed * 2.1),
            0.15 * math.sin(elapsed * 1.9),
        ]

        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=[200.0] * len(joint_indices),
        )

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def main():
    """Open a minimal PyBullet GUI, inspect the robot, and then move it manually."""
    physics_client = p.connect(p.GUI)
    if physics_client < 0:
        raise RuntimeError("Could not connect to PyBullet GUI.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    ground_id = p.loadURDF("plane.urdf")
    print(f"Ground plane loaded with body ID: {ground_id}")

    robot_start_position = [0, 0, 0.0]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

    robot_id = p.loadURDF(
        "kuka_iiwa/model.urdf",
        robot_start_position,
        robot_start_orientation,
        useFixedBase=False,
    )
    print(f"Robot loaded with body ID: {robot_id}")

    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0, 0.75],
    )

    print_joint_information(robot_id)
    print("\nThe GUI is open. Watch the robot move as the demo begins.")
    print("Press Ctrl+C to stop the simulation.")

    manual_joint_demo(robot_id)


if __name__ == "__main__":
    main()
