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


def add_target(target_position):
    """Create a visible red sphere to represent the goal target in the scene."""
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.05,
        rgbaColor=[1, 0, 0, 1],
    )
    sphere_collision = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.05,
    )

    target_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=sphere_collision,
        baseVisualShapeIndex=sphere_visual,
        basePosition=target_position,
    )
    print(f"Target added at position {target_position} with body ID {target_id}")
    return target_id


def get_end_effector_position(robot_id):
    """Return the current end-effector position in world coordinates."""
    end_effector_index = p.getNumJoints(robot_id) - 1
    end_effector_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
    return end_effector_state[0]


def manual_joint_demo(robot_id, target_position):
    """Move a few joints through a small manual position-control demo while printing the target distance."""
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
    last_distance_print = 0.0

    while True:
        elapsed = time.time() - start_time

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

        end_effector_position = get_end_effector_position(robot_id)
        distance_to_target = math.dist(target_position, end_effector_position)

        if time.time() - last_distance_print >= 0.5:
            print(
                f"End-effector position: {end_effector_position} | "
                f"target position: {target_position} | "
                f"distance: {distance_to_target:.3f} m"
            )
            last_distance_print = time.time()

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def main():
    """Open a minimal PyBullet GUI, inspect the robot, add a target, and monitor the arm-to-target distance."""
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

    target_position = [0.5, 0.0, 0.5]
    target_id = add_target(target_position)
    print(f"Target body ID: {target_id}")

    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0, 0.75],
    )

    print_joint_information(robot_id)
    print("\nThe robot and target are visible in the PyBullet GUI.")
    print("The arm will move slightly while the terminal prints distance-to-target updates.")
    print("Press Ctrl+C to stop the simulation.")

    manual_joint_demo(robot_id, target_position)


if __name__ == "__main__":
    main()
