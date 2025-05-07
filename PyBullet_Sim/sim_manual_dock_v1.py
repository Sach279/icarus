import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np

def simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140):
    """
    Simulate a basic front-facing LIDAR sensor by casting rays within a limited FOV
    (default 140°) in front of the rover. Debug lines are drawn for each ray.
    """
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    sensor_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.1]
    _, _, rover_yaw = p.getEulerFromQuaternion(base_orn)
    fov_rad = math.radians(fov_deg)
    start_angle = rover_yaw - fov_rad / 2

    ray_from_list = []
    ray_to_list = []
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
    # Connect and configure simulation.
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    print("[DEBUG] Connected to PyBullet, gravity set to -9.81.")

    plane_id = p.loadURDF("plane.urdf")
    print("[DEBUG] Plane URDF loaded.")

    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1], useFixedBase=False)
        print(f"[DEBUG] Successfully loaded rover URDF: {urdf_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load rover URDF: {e}")
        p.disconnect()
        return

    joint_dict = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_dict[joint_name] = i
    print(f"[DEBUG] Joint mapping created with {num_joints} joints: {joint_dict}")

    # Control and PID parameters.
    force = 20
    forward_speed = 5.0
    Kp = 2.0
    Ki = 0.25
    Kd = 0.01
    threshold_stage1 = math.radians(90)
    threshold_stage2 = math.radians(40)
    # Reduced final approach distance and docking offset.
    final_approach_distance = 0.12
    docking_offset = 0.12
    # New parameter: docking lift to raise the block.
    docking_lift = 0.03
    # Threshold to consider the rover aligned enough to stop.
    stop_alignment_threshold = math.radians(10)

    # === Debug Sliders for Two Phases ===
    # Pick-up phase: block (structure) position.
    block_x_slider = p.addUserDebugParameter("Block X", -10, 10, 2.0)
    block_y_slider = p.addUserDebugParameter("Block Y", -10, 10, 0.0)
    print("[DEBUG] Debug sliders created for block (pick-up) position.")

    # Transport phase: destination target (red transparent cube).
    dest_x_slider = p.addUserDebugParameter("Destination X", -10, 10, 0.0)
    dest_y_slider = p.addUserDebugParameter("Destination Y", -10, 10, 0.0)
    print("[DEBUG] Debug sliders created for destination position.")

    # Load the block from CubeStructure.urdf.
    block_urdf_path = r"C:\Users\akshi\Documents\Building Block\Models\CubeStructure.urdf"
    try:
        # Load with useFixedBase=False so it can be moved/attached.
        block_body_id = p.loadURDF(block_urdf_path, basePosition=[2.0, 2.0, 0.2], useFixedBase=False)
        print("[DEBUG] CubeStructure.urdf loaded as block (pick-up target).")
    except Exception as e:
        print(f"[ERROR] Failed to load CubeStructure.urdf: {e}")
        p.disconnect()
        return

    # Create destination target as a red transparent cube.
    dest_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                               halfExtents=[0.2, 0.2, 0.2],
                                               rgbaColor=[1, 0, 0, 0.5])
    dest_body_id = p.createMultiBody(baseMass=0,
                                     baseVisualShapeIndex=dest_visual_shape_id,
                                     baseCollisionShapeIndex=-1,
                                     basePosition=[0.0, 0.0, 0.2])
    print("[DEBUG] Destination target (red cube) created.")

    lidar_enabled = False
    auto_camera = True

    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0])
    print("[DEBUG] Initial camera set to auto overhead view (distance=5, pitch=-89).")

    print("Controls:")
    print("  C: Toggle camera mode (Auto vs Free)")
    print("  R: Reset free camera to overhead view")
    print("  S: Toggle LIDAR visualization")
    print("  D: Dock (engage coupling and lift block)")
    print("  U: Undock (release block at destination)")

    frame_count = 0
    error_integral = 0.0
    previous_error = 0.0
    dt = 1/120.0

    docked = False
    dock_constraint = None

    try:
        while True:
            frame_count += 1
            keys = p.getKeyboardEvents()
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                auto_camera = not auto_camera
                mode = "Auto Camera" if auto_camera else "Free Camera"
                print(f"[DEBUG] Camera mode toggled: {mode}")
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                if not auto_camera:
                    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])
                    print("[DEBUG] Free camera reset to overhead view.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                lidar_enabled = not lidar_enabled
                status = "Enabled" if lidar_enabled else "Disabled"
                print(f"[DEBUG] LIDAR simulation toggled: {status}")

            # --- Update Targets Based on Docking State ---
            if not docked:
                # Pick-up phase: update block position from sliders.
                block_x = p.readUserDebugParameter(block_x_slider)
                block_y = p.readUserDebugParameter(block_y_slider)
                block_position = [block_x, block_y]
                p.resetBasePositionAndOrientation(block_body_id, [block_x, block_y, 0.2], [0, 0, 0, 1])
                driving_target = block_position
            else:
                # Transport phase: update destination target from sliders.
                dest_x = p.readUserDebugParameter(dest_x_slider)
                dest_y = p.readUserDebugParameter(dest_y_slider)
                dest_position = [dest_x, dest_y]
                p.resetBasePositionAndOrientation(dest_body_id, [dest_x, dest_y, 0.2], [0, 0, 0, 1])
                driving_target = dest_position

            # Get rover state.
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)

            dx = driving_target[0] - base_pos[0]
            dy = driving_target[1] - base_pos[1]
            distance = math.hypot(dx, dy)

            # --- Final Approach and Docking/Undocking Behavior ---
            if distance < final_approach_distance:
                # Override desired angle to approach perpendicular to target.
                if abs(dx) > abs(dy):
                    discrete_desired_angle = 0.0 if dx > 0 else math.pi
                else:
                    discrete_desired_angle = math.pi/2 if dy > 0 else -math.pi/2
                desired_angle = discrete_desired_angle
                if frame_count % 120 == 0:
                    phase = "Docking" if not docked else "Transport"
                    print(f"[DEBUG] Final approach ({phase}): Overriding desired angle to {math.degrees(desired_angle):.0f}° (dx={dx:.2f}, dy={dy:.2f}).")
                angle_error = desired_angle - rover_yaw
                angle_error = (angle_error + math.pi) % (2*math.pi) - math.pi
                error = angle_error
                error_integral += error * dt
                error_derivative = (error - previous_error) / dt
                pid_output = Kp * error + Ki * error_integral + Kd * error_derivative
                previous_error = error

                # If the rover is well aligned, come to a full stop.
                if abs(angle_error) < stop_alignment_threshold:
                    linear_speed = 0.0
                    angular_correction = 0.0
                    left_velocity = 0.0
                    right_velocity = 0.0
                    if frame_count % 120 == 0:
                        print("[DEBUG] Final approach reached and aligned: stopping rover.")
                else:
                    # Otherwise, use a slow approach.
                    linear_speed = forward_speed * 0.2
                    angular_correction = pid_output
                    left_velocity = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction

                # Docking/undocking commands.
                if not docked:
                    if abs(angle_error) < math.radians(10) and ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                        # Compute relative transform and add a Z-offset to lift the block.
                        pos_rover, orn_rover = p.getBasePositionAndOrientation(robot_id)
                        pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
                        invPos, invOrn = p.invertTransform(pos_rover, orn_rover)
                        relPos, _ = p.multiplyTransforms(invPos, invOrn, pos_block, orn_block)
                        relPos = [relPos[0], relPos[1], relPos[2] + docking_lift]
                        dock_constraint = p.createConstraint(robot_id, -1, block_body_id, -1, p.JOINT_FIXED,
                                                             [0, 0, 0], relPos, [0, 0, 0])
                        docked = True
                        print("[DEBUG] Docking engaged: Block attached and lifted.")
                else:
                    if ord('u') in keys and keys[ord('u')] & p.KEY_WAS_TRIGGERED:
                        p.removeConstraint(dock_constraint)
                        dock_constraint = None
                        docked = False
                        print("[DEBUG] Undocking: Block released at destination.")
                        # When undocking, stop the rover.
                        linear_speed = 0.0
                        left_velocity = 0.0
                        right_velocity = 0.0

            else:
                # --- Normal Two-Stage Alignment Control ---
                desired_angle = math.atan2(dy, dx)
                angle_error = desired_angle - rover_yaw
                angle_error = (angle_error + math.pi) % (2*math.pi) - math.pi
                if frame_count % 120 == 0:
                    print(f"[DEBUG] Desired angle: {math.degrees(desired_angle):.1f}°, Rover yaw: {math.degrees(rover_yaw):.1f}°, Angle error: {math.degrees(angle_error):.1f}°")
                error = angle_error
                error_integral += error * dt
                error_derivative = (error - previous_error) / dt
                pid_output = Kp * error + Ki * error_integral + Kd * error_derivative
                previous_error = error

                if abs(angle_error) > threshold_stage1:
                    linear_speed = 0.0
                    angular_speed = pid_output
                    left_velocity = -angular_speed
                    right_velocity = angular_speed
                    if frame_count % 120 == 0:
                        print("[DEBUG] Stage 1: Rotating (error > 90°).")
                elif abs(angle_error) > threshold_stage2:
                    scale = (threshold_stage1 - abs(angle_error)) / (threshold_stage1 - threshold_stage2)
                    linear_speed = forward_speed * scale
                    angular_correction = pid_output
                    left_velocity = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction
                    if frame_count % 120 == 0:
                        print(f"[DEBUG] Stage 2: Transitional drive, scale: {scale:.2f}.")
                else:
                    linear_speed = forward_speed
                    angular_correction = pid_output
                    left_velocity = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction
                    if frame_count % 120 == 0:
                        print("[DEBUG] Stage 3: Aligned. Driving at full speed.")

            # Apply wheel commands.
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
                mid_x = (base_pos[0] + driving_target[0]) / 2.0
                mid_y = (base_pos[1] + driving_target[1]) / 2.0
                midpoint = [mid_x, mid_y, 0]
                separation = math.hypot(base_pos[0] - driving_target[0],
                                        base_pos[1] - driving_target[1])
                FOV = 60
                required_distance = (separation / 2) / math.tan(math.radians(FOV / 2))
                min_distance = 5
                camera_distance = max(min_distance, required_distance * 1.2)
                p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                             cameraYaw=0,
                                             cameraPitch=-89,
                                             cameraTargetPosition=midpoint)
                if frame_count % 120 == 0:
                    print(f"[DEBUG] Auto camera updated: midpoint = {midpoint}, separation = {separation:.2f}, distance = {camera_distance:.2f}")
            # LIDAR update.
            if lidar_enabled:
                simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140)

            imu_text = f"IMU: Roll={math.degrees(roll):.1f} Pitch={math.degrees(pitch):.1f} Yaw={math.degrees(rover_yaw):.1f}"
            p.addUserDebugText(imu_text,
                               [base_pos[0], base_pos[1], base_pos[2] + 0.5],
                               textColorRGB=[1, 1, 1],
                               textSize=1.2,
                               lifeTime=1/120.)
            
            if frame_count % 120 == 0:
                print(f"[DEBUG] Frame {frame_count}:")
                print(f"        Rover pos: {base_pos}")
                print(f"        Driving target: {driving_target}")
                print(f"        Distance: {distance:.2f}")
            
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
