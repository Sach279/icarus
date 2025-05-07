import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np


def simulate_lidar(robot_id, num_rays=36, ray_length=5.0):
    """
    Simulate a basic LIDAR sensor by casting rays in a circle around the rover.
    Draw debug lines for each ray (red if an obstacle is hit, green otherwise).
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
    force = 20  # Maximum force for each wheel.
    forward_speed = 5.0  # Speed for moving forward/backward.
    turn_speed = 3.0  # Speed for turning.

    # Toggle variables.
    camera_fixed = True  # If True, the debug visualizer camera is fixed behind the rover.
    lidar_enabled = False  # Toggle LIDAR sensor simulation.

    print("Controls:")
    print("  Left Arrow: Move backward")
    print("  Right Arrow: Move forward")
    print("  Up Arrow: Turn right")
    print("  Down Arrow: Turn left")
    print("  C: Toggle camera mode (fixed/free)")
    print("  S: Toggle LIDAR sensor simulation")
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

            # Initialize wheel velocities.
            left_velocity = 0.0
            right_velocity = 0.0

            # Inverted control mapping:
            # Left arrow: move backward (both wheels reverse).
            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                left_velocity = turn_speed
                right_velocity = -turn_speed
            # Right arrow: move forward (both wheels forward).
            elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                left_velocity = -turn_speed
                right_velocity = turn_speed
            # Up arrow: turn right (left wheels forward, right wheels reverse).
            elif p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                left_velocity = forward_speed
                right_velocity = forward_speed
            # Down arrow: turn left (left wheels reverse, right wheels forward).
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

            # Get the rover's base position and orientation.
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            euler = p.getEulerFromQuaternion(base_orn)
            roll, pitch, rover_yaw = euler  # in radians

            # Update the fixed camera if enabled.
            if camera_fixed:
                # Use a 90° offset instead of 180°.
                camera_yaw = (math.degrees(rover_yaw) + 90) % 360
                p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                             cameraYaw=camera_yaw,
                                             cameraPitch=-10,
                                             cameraTargetPosition=base_pos)

            # Update LIDAR simulation if enabled.
            if lidar_enabled:
                simulate_lidar(robot_id)

            # -------------------------------
            # Front Camera (picture-in-picture)
            # -------------------------------
            # width, height = 320, 240
            # fov = 60
            # aspect = width / height
            # near = 0.1
            # far = 100
            # front_offset = 0.3  # distance from base to camera
            # cam_height_offset = 0.2  # height of the camera above base
            # front_cam_pos = [
            #     base_pos[0] + front_offset * math.cos(rover_yaw),
            #     base_pos[1] + front_offset * math.sin(rover_yaw),
            #     base_pos[2] + cam_height_offset
            # ]
            # front_cam_target = [
            #     base_pos[0] + (front_offset + 0.5) * math.cos(rover_yaw),
            #     base_pos[1] + (front_offset + 0.5) * math.sin(rover_yaw),
            #     base_pos[2] + cam_height_offset
            # ]
            # up_vector = [0, 0, 1]
            # viewMatrix = p.computeViewMatrix(front_cam_pos, front_cam_target, up_vector)
            # projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
            # img_arr = p.getCameraImage(width, height, viewMatrix, projectionMatrix)
            # rgb_array = np.reshape(img_arr[2], (height, width, 4))
            # # Convert the image array to uint8 to avoid type issues.
            # rgb_array = rgb_array.astype(np.uint8)
            # rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
            # cv2.imshow("Front Camera", rgb_array)
            # cv2.waitKey(1)

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
