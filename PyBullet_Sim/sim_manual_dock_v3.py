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

def pickStrictSideAngle(dx, dy):
    """
    Always pick one of the four cardinal directions (0, ±90°, 180°)
    based on which absolute coordinate is larger.
    """
    if abs(dx) >= abs(dy):
        return 0.0 if dx >= 0 else math.pi
    else:
        return math.pi/2 if dy >= 0 else -math.pi/2

def load_and_control_urdf(urdf_path):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    print("[DEBUG] Connected to PyBullet, gravity set to -9.81.")

    plane_id = p.loadURDF("plane.urdf")
    print("[DEBUG] Plane URDF loaded.")

    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0.1], useFixedBase=False)
        print(f"[DEBUG] Successfully loaded rover URDF: {urdf_path}")
    except Exception as e:
        print("[ERROR]", e)
        p.disconnect()
        return

    # Build joint mapping.
    joint_dict = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_dict[joint_name] = i
    print(f"[DEBUG] Joint mapping with {num_joints} joints created.")

    # Control and PID parameters.
    force = 20
    forward_speed = 5.0
    Kp = 2.0
    Ki = 0.25
    Kd = 0.01
    threshold_stage1 = math.radians(90)
    threshold_stage2 = math.radians(40)
    final_approach_distance = 0.12
    docking_offset = 0.12
    docking_lift = 0.03
    stop_alignment_threshold = math.radians(10)

    # For the transport phase, destination target is controlled via sliders.
    dest_x_slider = p.addUserDebugParameter("Destination X", -10, 10, 0.0)
    dest_y_slider = p.addUserDebugParameter("Destination Y", -10, 10, 0.0)
    print("[DEBUG] Debug sliders created for destination position.")

    # Load the block (structure) from CubeStructure.urdf.
    block_urdf_path = r"C:\Users\akshi\Documents\Building Block\Models\CubeStructure.urdf"
    try:
        # Note: useFixedBase=False so that the block can be attached/docked.
        block_body_id = p.loadURDF(block_urdf_path, basePosition=[2.0, 2.0, 0.5], useFixedBase=False)
        print("[DEBUG] CubeStructure.urdf loaded as block (pick-up target).")
    except Exception as e:
        print("[ERROR] Failed to load CubeStructure.urdf:", e)
        p.disconnect()
        return

    # Create the destination target (red transparent cube).
    dest_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=[1,0,0,0.5])
    dest_body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=dest_visual_shape_id,
                                     baseCollisionShapeIndex=-1, basePosition=[0,0,0.2])
    print("[DEBUG] Destination target (red cube) created.")

    lidar_enabled = False
    auto_camera = True
    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])
    print("[DEBUG] Camera set to auto overhead view.")

    print("Controls:")
    print("  C: Toggle camera mode (Auto vs Free)")
    print("  R: Reset free camera")
    print("  S: Toggle LIDAR")
    print("  D: Dock (attach and lift block)")
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
                print(f"[DEBUG] Camera mode toggled: {auto_camera}")
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                if not auto_camera:
                    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0,
                                                 cameraPitch=-89, cameraTargetPosition=[0,0,0])
                    print("[DEBUG] Free camera reset.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                lidar_enabled = not lidar_enabled
                print("[DEBUG] LIDAR toggled:", lidar_enabled)

            # --- Determine Driving Target ---
            # In pick-up phase, get block position from simulation.
            if not docked:
                pos_block, _ = p.getBasePositionAndOrientation(block_body_id)
                target = [pos_block[0], pos_block[1]]
            else:
                # In transport phase, destination target is updated via sliders.
                dx_ = p.readUserDebugParameter(dest_x_slider)
                dy_ = p.readUserDebugParameter(dest_y_slider)
                p.resetBasePositionAndOrientation(dest_body_id, [dx_, dy_, 0.2], [0,0,0,1])
                target = [dx_, dy_]

            # Get rover state.
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)

            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]
            distance = math.hypot(dx, dy)

            # Compute a stable discrete approach angle (from block/destination).
            desired_angle = pickStrictSideAngle(dx, dy)
            angle_error = desired_angle - rover_yaw
            angle_error = (angle_error + math.pi) % (2*math.pi) - math.pi

            error = angle_error
            error_integral += error * dt
            error_derivative = (error - previous_error) / dt
            pid_output = Kp * error + Ki * error_integral + Kd * error_derivative
            previous_error = error

            # --- Final Approach and Docking/Undocking Behavior ---
            if distance < final_approach_distance:
                # In final approach, enforce discrete desired angle.
                if abs(angle_error) < stop_alignment_threshold:
                    left_velocity = 0.0
                    right_velocity = 0.0
                    print(f"[DEBUG] Final approach reached and aligned; stopping rover.")
                else:
                    linear_speed = forward_speed * 0.2
                    angular_correction = pid_output
                    left_velocity = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction

                if not docked:
                    if abs(angle_error) < stop_alignment_threshold and ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                        pos_rover, orn_rover = p.getBasePositionAndOrientation(robot_id)
                        pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
                        invPos, invOrn = p.invertTransform(pos_rover, orn_rover)
                        relPos, _ = p.multiplyTransforms(invPos, invOrn, pos_block, orn_block)
                        relPos = [relPos[0], relPos[1], relPos[2] + docking_lift]
                        dock_constraint = p.createConstraint(robot_id, -1, block_body_id, -1,
                                                             p.JOINT_FIXED, [0,0,0], relPos, [0,0,0])
                        docked = True
                        print("[DEBUG] Docked: Block attached and lifted.")
                else:
                    if abs(angle_error) < stop_alignment_threshold and ord('u') in keys and keys[ord('u')] & p.KEY_WAS_TRIGGERED:
                        p.removeConstraint(dock_constraint)
                        dock_constraint = None
                        docked = False
                        left_velocity = 0.0
                        right_velocity = 0.0
                        print("[DEBUG] Undocked: Block released at destination.")
            else:
                # --- Normal Two-Stage Alignment Control ---
                if abs(angle_error) > threshold_stage1:
                    left_velocity = -pid_output
                    right_velocity = pid_output
                elif abs(angle_error) > threshold_stage2:
                    scale = (threshold_stage1 - abs(angle_error)) / (threshold_stage1 - threshold_stage2)
                    linear_speed = forward_speed * scale
                    angular_correction = pid_output
                    left_velocity = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction
                else:
                    linear_speed = forward_speed
                    angular_correction = pid_output
                    left_velocity = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction

            p.setJointMotorControl2(robot_id, joint_dict['wheel_FL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=-left_velocity, force=force)
            p.setJointMotorControl2(robot_id, joint_dict['wheel_RL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=-left_velocity, force=force)
            p.setJointMotorControl2(robot_id, joint_dict['wheel_FR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=right_velocity, force=force)
            p.setJointMotorControl2(robot_id, joint_dict['wheel_RR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=right_velocity, force=force)

            if auto_camera:
                mid_x = (base_pos[0] + target[0]) / 2.0
                mid_y = (base_pos[1] + target[1]) / 2.0
                midpoint = [mid_x, mid_y, 0]
                separation = math.hypot(base_pos[0]-target[0], base_pos[1]-target[1])
                FOV = 60
                required_distance = (separation/2)/math.tan(math.radians(FOV/2))
                min_dist = 1.5
                cam_dist = max(min_dist, required_distance*1.2)
                p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=0,
                                             cameraPitch=-89, cameraTargetPosition=midpoint)
            if lidar_enabled:
                simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140)

            imu_text = f"IMU: Roll={roll:.1f} Pitch={pitch:.1f} Yaw={math.degrees(rover_yaw):.1f}"
            p.addUserDebugText(imu_text, [base_pos[0], base_pos[1], base_pos[2]+0.5],
                               textColorRGB=[1,1,1], textSize=1.2, lifeTime=1/120.)
            
            p.stepSimulation()
            time.sleep(1/120.)
            
            if frame_count % 120 == 0:
                print(f"[DEBUG] Frame {frame_count}: Distance={distance:.2f} Angle error={math.degrees(angle_error):.1f}")
    except KeyboardInterrupt:
        print("[DEBUG] Simulation terminated by user.")
    finally:
        p.disconnect()
        cv2.destroyAllWindows()

if __name__=="__main__":
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    load_and_control_urdf(urdf_file)
