import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np


def simulate_lidar(robot_id, num_rays=36, ray_length=5.0):
    """
    Simulate a basic LIDAR sensor by casting rays in a circle around the rover.
    Debug lines are drawn for each ray (red if an obstacle is hit, green otherwise).
    """
    base_pos, _ = p.getBasePositionAndOrientation(robot_id)
    sensor_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.1]

    ray_from_list = []
    ray_to_list = []
    for i in range(num_rays):
        angle = 2 * math.pi * i / num_rays
        dx = ray_length * math.cos(angle)
        dy = ray_length * math.sin(angle)
        ray_from_list.append(sensor_pos)
        ray_to_list.append([sensor_pos[0] + dx, sensor_pos[1] + dy, sensor_pos[2]])

    results = p.rayTestBatch(ray_from_list, ray_to_list)

    for i, res in enumerate(results):
        hit_fraction = res[2]
        hit_pos = res[3] if hit_fraction < 1.0 else ray_to_list[i]
        color = [1, 0, 0] if hit_fraction < 1.0 else [0, 1, 0]
        p.addUserDebugLine(ray_from_list[i], hit_pos, lineColorRGB=color, lineWidth=1, lifeTime=1 / 240.)


def load_and_control_urdf(urdf_path):
    # Connect to PyBullet in GUI mode.
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load a plane for reference.
    plane_id = p.loadURDF("plane.urdf")

    # Load the rover URDF slightly above the ground.
    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1], useFixedBase=False)
        print(f"Successfully loaded URDF: {urdf_path}")
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        p.disconnect()
        return

    # Build a mapping from joint names to indices.
    joint_dict = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_dict[joint_name] = i
    print("Joint mapping:", joint_dict)

    # Control parameters.
    force = 20           # Maximum force for each wheel.
    forward_speed = 5.0  # Base forward speed.
    Kp = 2.0             # Proportional gain for turning.

    # Toggle variables.
    camera_fixed = True     # If True, the camera is fixed behind the rover.
    lidar_enabled = False   # Toggle for LIDAR simulation.
    drive_to_target_enabled = False  # Toggle for autonomous drive-to-target mode.

    # Set a target position (x, y) on the plane.
    target_position = [2.0, 2.0]

    print("Controls:")
    print("  Arrow keys: Manual driving")
    print("  C: Toggle camera mode (fixed/free)")
    print("  S: Toggle LIDAR simulation")
    print("  D: Toggle drive-to-target mode (autonomous)")
    print("A front camera view will also appear in a separate window.")

    try:
        while True:
            keys = p.getKeyboardEvents()

            # Toggle camera mode with 'c'.
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                camera_fixed = not camera_fixed
                print("Camera mode:", "Fixed behind rover" if camera_fixed else "Free camera")

            # Toggle LIDAR simulation with 's'.
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                lidar_enabled = not lidar_enabled
                print("LIDAR simulation:", "Enabled" if lidar_enabled else "Disabled")

            # Toggle drive-to-target mode with 'd'.
            if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                drive_to_target_enabled = not drive_to_target_enabled
                mode = "Drive-to-target mode enabled" if drive_to_target_enabled else "Manual control enabled"
                print(mode)

            # Initialize wheel velocities.
            left_velocity = 0.0
            right_velocity = 0.0

            # --- Autonomous drive-to-target mode ---
            if drive_to_target_enabled:
                # Get rover's base position and orientation.
                base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
                euler = p.getEulerFromQuaternion(base_orn)
                roll, pitch, rover_yaw = euler  # in radians

                # Compute the difference to the target.
                dx = target_position[0] - base_pos[0]
                dy = target_position[1] - base_pos[1]
                distance = math.hypot(dx, dy)

                # If close enough, stop.
                if distance < 0.2:
                    left_velocity, right_velocity = 0.0, 0.0
                    print("Target reached.")
                else:
                    # Determine desired heading.
                    desired_angle = math.atan2(dy, dx)
                    angle_error = desired_angle - rover_yaw
                    # Normalize angle error to [-pi, pi].
                    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

                    # Scale forward speed by how aligned the rover is with the target.
                    linear_speed = forward_speed * max(math.cos(angle_error), 0)

                    # Compute angular correction.
                    angular_correction = Kp * angle_error

                    # Differential drive: adjust left/right wheel speeds.
                    left_velocity = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction

            # --- Manual control (if drive-to-target is disabled) ---
            else:
                # Inverted control mapping example.
                if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                    left_velocity = 5.0  # example turning values
                    right_velocity = -5.0
                elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                    left_velocity = -5.0
                    right_velocity = 5.0
                elif p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                    left_velocity = forward_speed
                    right_velocity = forward_speed
                elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                    left_velocity = -forward_speed
                    right_velocity = -forward_speed

            # Apply velocity control to each wheel joint.
            p.setJointMotorControl2(robot_id,
                                    joint_dict['wheel_FL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=left_velocity,
                                    force=force)
            p.setJointMotorControl2(robot_id,
                                    joint_dict['wheel_RL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=left_velocity,
                                    force=force)
            p.setJointMotorControl2(robot_id,
                                    joint_dict['wheel_FR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=right_velocity,
                                    force=force)
            p.setJointMotorControl2(robot_id,
                                    joint_dict['wheel_RR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=right_velocity,
                                    force=force)

            p.stepSimulation()
            time.sleep(1 / 120.)

            # Get the rover's base position and orientation for camera and display.
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            euler = p.getEulerFromQuaternion(base_orn)
            roll, pitch, rover_yaw = euler  # in radians

            # Update the fixed camera if enabled.
            if camera_fixed:
                # Adjust camera yaw relative to the rover's yaw (90° offset here).
                camera_yaw = (math.degrees(rover_yaw) + 90) % 360
                p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                             cameraYaw=camera_yaw,
                                             cameraPitch=-10,
                                             cameraTargetPosition=base_pos)

            # Update LIDAR simulation if enabled.
            if lidar_enabled:
                simulate_lidar(robot_id)

            # -------------------------------
            # IMU Display
            # -------------------------------
            imu_text = f"IMU: Roll={math.degrees(roll):.1f} Pitch={math.degrees(pitch):.1f} Yaw={math.degrees(rover_yaw):.1f}"
            p.addUserDebugText(imu_text, [base_pos[0], base_pos[1], base_pos[2] + 0.5],
                               textColorRGB=[1, 1, 1], textSize=1.2, lifeTime=1 / 240.)

    except KeyboardInterrupt:
        print("Simulation terminated by user.")
    finally:
        p.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    load_and_control_urdf(urdf_file)
