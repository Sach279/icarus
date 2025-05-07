import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np

def simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140):
    """
    Simulate a basic front-facing LIDAR sensor by casting rays within a limited FOV
    (default 140°) in front of the rover. Debug lines are drawn for each ray (red if an obstacle is hit, green otherwise).
    """
    # Get the rover's current position and orientation.
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    sensor_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.1]
    
    # Extract the yaw (heading) from the orientation.
    _, _, rover_yaw = p.getEulerFromQuaternion(base_orn)
    
    # Define the FOV in radians and calculate the starting angle.
    fov_rad = math.radians(fov_deg)
    start_angle = rover_yaw - fov_rad / 2

    ray_from_list = []
    ray_to_list = []
    # Cast rays evenly within the specified FOV.
    for i in range(num_rays):
        angle = start_angle + (i / (num_rays - 1)) * fov_rad
        dx = ray_length * math.cos(angle)
        dy = ray_length * math.sin(angle)
        ray_from_list.append(sensor_pos)
        ray_to_list.append([sensor_pos[0] + dx, sensor_pos[1] + dy, sensor_pos[2]])

    results = p.rayTestBatch(ray_from_list, ray_to_list)

    for i, res in enumerate(results):
        hit_fraction = res[2]
        hit_pos = res[3] if hit_fraction < 1.0 else ray_to_list[i]
        color = [1, 0, 0] if hit_fraction < 1.0 else [0, 1, 0]
        p.addUserDebugLine(ray_from_list[i], hit_pos, lineColorRGB=color, lineWidth=1, lifeTime=1/240.)

def load_and_control_urdf(urdf_path):
    # Connect to PyBullet in GUI mode.
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    print("[DEBUG] Connected to PyBullet, gravity set to -9.81.")

    # Load a plane for reference.
    plane_id = p.loadURDF("plane.urdf")
    print("[DEBUG] Plane URDF loaded.")

    # Load the rover URDF slightly above the ground.
    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1], useFixedBase=False)
        print(f"[DEBUG] Successfully loaded URDF: {urdf_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load URDF: {e}")
        p.disconnect()
        return

    # Build a mapping from joint names to indices.
    joint_dict = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_dict[joint_name] = i
    print(f"[DEBUG] Joint mapping created with {num_joints} joints: {joint_dict}")

    # Control parameters.
    force = 20           # Maximum force for each wheel.
    forward_speed = 5.0  # Base forward speed.
    # PID Gains for steering:
    Kp = 2.0             # Proportional gain.
    Ki = 0.25            # Integral gain.
    Kd = 0.01            # Derivative gain.
    # Define two alignment thresholds.
    threshold_stage1 = math.radians(90)  # Stage 1: if error > 90 deg -> rotate on the spot.
    threshold_stage2 = math.radians(40)  # Stage 2: between 90 and 40 deg -> transitional drive.

    # Create debug sliders to set the target's x and y coordinates.
    target_x_slider = p.addUserDebugParameter("Target X", -10, 10, 2.0)
    target_y_slider = p.addUserDebugParameter("Target Y", -10, 10, 0.0)
    print("[DEBUG] Debug sliders created for target coordinates.")

    # Create a visual representation of the target as a red transparent box.
    target_visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.2, 0.2, 0.2],
        rgbaColor=[1, 0, 0, 0.5]
    )
    target_body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual_shape_id,
        baseCollisionShapeIndex=-1,
        basePosition=[2.0, 2.0, 0.2]
    )
    print("[DEBUG] Target visual marker created.")

    # LIDAR toggle.
    lidar_enabled = False

    # Camera mode flag: True for auto camera, False for free camera.
    auto_camera = True

    # Initialize the camera in auto mode with an overhead view.
    p.resetDebugVisualizerCamera(
        cameraDistance=5,
        cameraYaw=0,
        cameraPitch=-89,
        cameraTargetPosition=[0, 0, 0]
    )
    print("[DEBUG] Initial camera set to auto overhead view (distance=5, pitch=-89).")

    print("Controls:")
    print("  C: Toggle camera mode (Auto vs Free camera)")
    print("  R: Reset free camera to overhead view (0,0,5, pitch=-89)")
    print("  S: Toggle LIDAR sensor visualization")
    print("  (In Auto mode, the camera resizes to keep both rover and target visible)")

    frame_count = 0  # For periodic debug messages

    # Initialize PID state variables.
    error_integral = 0.0
    previous_error = 0.0
    dt = 1 / 120.0  # Assuming simulation loop runs at 120 Hz.

    try:
        while True:
            frame_count += 1

            # Check keyboard events.
            keys = p.getKeyboardEvents()
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                auto_camera = not auto_camera
                mode = "Auto Camera" if auto_camera else "Free Camera"
                print(f"[DEBUG] Camera mode toggled: {mode}")
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                if not auto_camera:
                    p.resetDebugVisualizerCamera(
                        cameraDistance=5,
                        cameraYaw=0,
                        cameraPitch=-89,
                        cameraTargetPosition=[0, 0, 0]
                    )
                    print("[DEBUG] Free camera reset to overhead view.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                lidar_enabled = not lidar_enabled
                status = "Enabled" if lidar_enabled else "Disabled"
                print(f"[DEBUG] LIDAR simulation toggled: {status}")

            # Read target coordinates from the sliders.
            target_x = p.readUserDebugParameter(target_x_slider)
            target_y = p.readUserDebugParameter(target_y_slider)
            target_position = [target_x, target_y]
            # Update the target box's position.
            p.resetBasePositionAndOrientation(
                target_body_id,
                [target_x, target_y, 0.2],
                [0, 0, 0, 1]
            )

            # Get the rover's base position and orientation.
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            euler = p.getEulerFromQuaternion(base_orn)
            roll, pitch, rover_yaw = euler  # in radians

            # Autonomous drive-to-target control.
            dx = target_position[0] - base_pos[0]
            dy = target_position[1] - base_pos[1]
            distance = math.hypot(dx, dy)

            if distance < 0.2:
                left_velocity, right_velocity = 0.0, 0.0
                p.addUserDebugText(
                    "Target reached",
                    [base_pos[0], base_pos[1], base_pos[2] + 0.5],
                    textColorRGB=[1, 1, 0],
                    lifeTime=1.0
                )
                if frame_count % 120 == 0:
                    print("[DEBUG] Target reached. Stopping rover.")
            else:
                # Calculate desired heading and error.
                desired_angle = math.atan2(dy, dx)
                angle_error = desired_angle - rover_yaw
                # Normalize angle error to [-pi, pi]
                angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

                if frame_count % 120 == 0:
                    print(f"[DEBUG] Desired angle: {desired_angle:.2f}, Rover yaw: {rover_yaw:.2f}, Angle error: {angle_error:.2f}")

                # Compute PID terms.
                error = angle_error
                error_integral += error * dt
                error_derivative = (error - previous_error) / dt
                pid_output = Kp * error + Ki * error_integral + Kd * error_derivative
                previous_error = error

                # Two-stage alignment:
                if abs(angle_error) > threshold_stage1:
                    # Stage 1: If error > 90°, rotate on the spot.
                    linear_speed = 0.0
                    angular_speed = pid_output
                    left_velocity = -angular_speed
                    right_velocity = angular_speed
                    if frame_count % 120 == 0:
                        print("[DEBUG] Stage 1: Rotating on the spot (error > 90°).")
                elif abs(angle_error) > threshold_stage2:
                    # Stage 2: If error is between 90° and 40°, scale forward speed.
                    scale = (threshold_stage1 - abs(angle_error)) / (threshold_stage1 - threshold_stage2)
                    linear_speed = forward_speed * scale
                    angular_correction = pid_output
                    left_velocity = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction
                    if frame_count % 120 == 0:
                        print(f"[DEBUG] Stage 2: Transitional drive (error between 90° and 40°), scale: {scale:.2f}.")
                else:
                    # Stage 3: If error <= 40°, drive forward at full speed.
                    linear_speed = forward_speed
                    angular_correction = pid_output
                    left_velocity = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction
                    if frame_count % 120 == 0:
                        print("[DEBUG] Stage 3: Target aligned. Driving forward at full speed.")

            # Apply computed velocities to wheel joints.
            p.setJointMotorControl2(robot_id, joint_dict['wheel_FL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=-left_velocity,
                                    force=force)
            p.setJointMotorControl2(robot_id, joint_dict['wheel_RL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=-left_velocity,
                                    force=force)
            p.setJointMotorControl2(robot_id, joint_dict['wheel_FR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=right_velocity,
                                    force=force)
            p.setJointMotorControl2(robot_id, joint_dict['wheel_RR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=right_velocity,
                                    force=force)

            # --- Camera Update ---
            if auto_camera:
                # Compute the midpoint between the rover and target.
                mid_x = (base_pos[0] + target_position[0]) / 2.0
                mid_y = (base_pos[1] + target_position[1]) / 2.0
                midpoint = [mid_x, mid_y, 0]
                # Compute horizontal separation.
                separation = math.hypot(base_pos[0] - target_position[0],
                                        base_pos[1] - target_position[1])
                # With an assumed horizontal FOV of 60°, compute the distance needed.
                FOV = 60  # degrees
                required_distance = (separation / 2) / math.tan(math.radians(FOV / 2))
                min_distance = 5
                camera_distance = max(min_distance, required_distance * 1.2)
                p.resetDebugVisualizerCamera(
                    cameraDistance=camera_distance,
                    cameraYaw=0,
                    cameraPitch=-89,
                    cameraTargetPosition=midpoint
                )
                if frame_count % 120 == 0:
                    print(f"[DEBUG] Auto camera updated: midpoint = {midpoint}, separation = {separation:.2f}, distance = {camera_distance:.2f}")
            # In Free Camera mode, the camera is manually controlled by the user.

            # Update LIDAR visualization if enabled.
            if lidar_enabled:
                simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140)

            # Display IMU data.
            imu_text = f"IMU: Roll={math.degrees(roll):.1f} Pitch={math.degrees(pitch):.1f} Yaw={math.degrees(rover_yaw):.1f}"
            p.addUserDebugText(imu_text,
                               [base_pos[0], base_pos[1], base_pos[2] + 0.5],
                               textColorRGB=[1, 1, 1],
                               textSize=1.2,
                               lifeTime=1/120.)
            
            if frame_count % 120 == 0:
                print(f"[DEBUG] Frame {frame_count}:")
                print(f"        Rover position: {base_pos}")
                print(f"        Target position: {target_position}")
                print(f"        Distance to target: {distance:.2f}")
            
            p.stepSimulation()
            time.sleep(1/120.)

    except KeyboardInterrupt:
        print("[DEBUG] Simulation terminated by user.")
    finally:
        p.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    load_and_control_urdf(urdf_file)
