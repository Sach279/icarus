import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random

# --- State Constants ---
STATE_IDLE = 0
STATE_FIND_BLOCK = 1
STATE_APPROACH_BLOCK = 2
STATE_ALIGN_BLOCK = 3
STATE_MANUAL_DOCK = 4
STATE_LOCATE_DEST = 5
STATE_APPROACH_DEST = 6
STATE_ALIGN_DEST = 7
STATE_ROTATE_DEST = 8
STATE_MANUAL_UNDOCK = 9
STATE_DELIVERED = 10

# --- PID Controller Class ---
class PIDController:
    """Simple PID controller for angular adjustments."""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_integral = 0.0
        self.previous_error = 0.0

    def compute(self, error, dt):
        self.error_integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.error_integral + self.Kd * derivative
        self.previous_error = error
        return output

# --- Sensor Simulation ---
def simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140):
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    sensor_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.1]
    _, _, rover_yaw = p.getEulerFromQuaternion(base_orn)
    fov_rad = math.radians(fov_deg)
    start_angle = rover_yaw - fov_rad / 2

    ray_from_list = []
    ray_to_list = []
    for i in range(num_rays):
        angle = start_angle + (i/(num_rays-1))*fov_rad
        dx = ray_length * math.cos(angle)
        dy = ray_length * math.sin(angle)
        ray_from_list.append(sensor_pos)
        ray_to_list.append([sensor_pos[0]+dx, sensor_pos[1]+dy, sensor_pos[2]])
    results = p.rayTestBatch(ray_from_list, ray_to_list)
    for i, res in enumerate(results):
        hit_fraction = res[2]
        hit_pos = res[3] if hit_fraction < 1.0 else ray_to_list[i]
        color = [1,0,0] if hit_fraction < 1.0 else [0,1,0]
        p.addUserDebugLine(ray_from_list[i], hit_pos, lineColorRGB=color, lineWidth=1, lifeTime=1/240.)

# --- Helper Functions ---
def pick_strict_side_angle(dx, dy):
    if abs(dx) >= abs(dy):
        return 0.0 if dx >= 0 else math.pi
    else:
        return math.pi/2 if dy >= 0 else -math.pi/2

def set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity, force):
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

def update_camera_view(auto_camera, base_pos, target, FOV=60):
    if auto_camera and target is not None:
        mid_x = (base_pos[0] + target[0]) / 2.0
        mid_y = (base_pos[1] + target[1]) / 2.0
        midpoint = [mid_x, mid_y, 0]
        separation = math.hypot(base_pos[0]-target[0], base_pos[1]-target[1])
        required_distance = (separation/2)/math.tan(math.radians(FOV/2))
        min_dist = 1.5
        cam_dist = max(min_dist, required_distance*1.2)
        p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=0,
                                     cameraPitch=-89, cameraTargetPosition=midpoint)

def update_block_visualization(block_body_id, block_center_sphere):
    block_center, _ = p.getBasePositionAndOrientation(block_body_id)
    p.resetBasePositionAndOrientation(block_center_sphere, block_center, [0,0,0,1])
    pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
    rot_mat = p.getMatrixFromQuaternion(orn_block)
    axis_x = [rot_mat[0], rot_mat[3], rot_mat[6]]
    axis_y = [rot_mat[1], rot_mat[4], rot_mat[7]]
    axis_z = [rot_mat[2], rot_mat[5], rot_mat[8]]
    scale = 0.15
    p.addUserDebugLine(pos_block,
                       [pos_block[0]+scale*axis_x[0],
                        pos_block[1]+scale*axis_x[1],
                        pos_block[2]+scale*axis_x[2]],
                       [1,0,0], lifeTime=1)
    p.addUserDebugLine(pos_block,
                       [pos_block[0]+scale*axis_y[0],
                        pos_block[1]+scale*axis_y[1],
                        pos_block[2]+scale*axis_y[2]],
                       [0,1,0], lifeTime=1)
    p.addUserDebugLine(pos_block,
                       [pos_block[0]+scale*axis_z[0],
                        pos_block[1]+scale*axis_z[1],
                        pos_block[2]+scale*axis_z[2]],
                       [0,0,1], lifeTime=1)

def load_robot(urdf_path):
    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0.1], useFixedBase=False)
        print("[DEBUG] Robot loaded.")
    except Exception as e:
        print("[ERROR] Failed to load robot URDF:", e)
        p.disconnect()
        return None, None
    joint_dict = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_dict[info[1].decode("utf-8")] = i
    return robot_id, joint_dict

def load_block(block_urdf_path):
    r_val = random.uniform(1,3)
    angle = random.uniform(0,2*math.pi)
    block_x = r_val * math.cos(angle)
    block_y = r_val * math.sin(angle)
    block_z = 0.01
    random_z_rotation = random.uniform(0,2*math.pi)
    orientation = p.getQuaternionFromEuler([0,0,random_z_rotation])
    try:
        block_body_id = p.loadURDF(block_urdf_path, basePosition=[block_x, block_y, block_z],
                                   baseOrientation=orientation, useFixedBase=False)
        print(f"[DEBUG] Block loaded at ({block_x:.2f}, {block_y:.2f}, {block_z}) with rotation {math.degrees(random_z_rotation):.1f}°")
    except Exception as e:
        print("[ERROR] Failed to load block:", e)
        p.disconnect()
        return None, None
    block_center_sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                     radius=0.05,
                                                     rgbaColor=[1,0,0,0.5])
    block_center_sphere = p.createMultiBody(baseMass=0,
                                            baseVisualShapeIndex=block_center_sphere_visual,
                                            baseCollisionShapeIndex=-1,
                                            basePosition=[block_x, block_y, block_z])
    print("[DEBUG] Block center sphere created.")
    return block_body_id, block_center_sphere

def create_destination_visual():
    dest_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2,0.2,0.2],
                                        rgbaColor=[1,0,0,0.5])
    dest_body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=dest_shape_id,
                                     baseCollisionShapeIndex=-1, basePosition=[0,0,0.2])
    print("[DEBUG] Destination target created.")
    return dest_body_id

def create_target_visual():
    """Create a small orange transparent sphere as the target marker."""
    target_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                       radius=0.05,
                                       rgbaColor=[1, 0.5, 0, 0.5])
    target_vis_id = p.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=target_shape,
                                      baseCollisionShapeIndex=-1,
                                      basePosition=[0,0,0.2])
    print("[DEBUG] Target visualization created.")
    return target_vis_id

def initialize_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(1)
    print("[DEBUG] PyBullet connected; gravity set to -9.81.")
    p.loadURDF("plane.urdf")
    print("[DEBUG] Plane URDF loaded.")

# --- NEW: Drive-to-Target Function ---
def drive_to_target(robot_id, joint_dict, target, forward_speed, pid_controller, dt, force, stop_distance=0.12):
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
    dx = target[0] - base_pos[0]
    dy = target[1] - base_pos[1]
    distance = math.hypot(dx, dy)
    desired_angle = math.atan2(dy, dx)
    angle_error = (desired_angle - yaw + math.pi) % (2*math.pi) - math.pi
    pid_output = pid_controller.compute(angle_error, dt)
    if distance > stop_distance:
        left_velocity = forward_speed - pid_output
        right_velocity = forward_speed + pid_output
    else:
        left_velocity = right_velocity = 0.0
    set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity, force)
    return distance, angle_error

# --- Main Simulation Loop ---
def load_and_control_urdf(urdf_path, block_urdf_path):
    initialize_simulation()
    robot_id, joint_dict = load_robot(urdf_path)
    if robot_id is None:
        return
    # --- Control and Motion Parameters ---
    force = 20
    forward_speed = 5.0
    Kp, Ki, Kd = 2.0, 0.15, 0.01
    threshold_stage1 = math.radians(90)
    final_approach_distance = 0.12
    predock_offset = 0.2
    stop_alignment_threshold = math.radians(2)
    pid_rotation_scale = 4
    dt = 1/120.0
    angle_pid = PIDController(Kp, Ki, Kd)
    # --- UI Elements ---
    dest_x_slider = p.addUserDebugParameter("Destination X", -10, 10, 0.0)
    dest_y_slider = p.addUserDebugParameter("Destination Y", -10, 10, 0.0)
    # Add slider for block yaw rotation (in degrees)
    block_yaw_slider = p.addUserDebugParameter("Block Yaw", -180, 180, 0.0)
    print("[DEBUG] Destination sliders and Block Yaw slider created.")
    block_body_id, block_center_sphere = load_block(block_urdf_path)
    if block_body_id is None:
        return
    dest_body_id = create_destination_visual()
    target_vis_id = create_target_visual()  # Visualization for state 2 target
    # --- Camera and LIDAR Settings ---
    lidar_enabled = False
    auto_camera = True
    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])
    print("[DEBUG] Camera set to overhead view.")
    print("Controls:")
    print("  C: Toggle camera (Auto vs Free)")
    print("  R: Reset camera")
    print("  S: Toggle LIDAR")
    print("  D: Dock (attach block)")
    print("  U: Undock (release block)")
    print("\nState Legend:")
    print("  0: IDLE")
    print("  1: FIND_BLOCK")
    print("  2: APPROACH_BLOCK")
    print("  3: ALIGN_BLOCK")
    print("  4: MANUAL_DOCK")
    print("  5: LOCATE_DEST")
    print("  6: APPROACH_DEST")
    print("  7: ALIGN_DEST")
    print("  8: ROTATE_DEST")
    print("  9: MANUAL_UNDOCK")
    print(" 10: DELIVERED")
    current_state = STATE_IDLE
    idle_start_time = None
    dock_constraint = None
    frame_count = 0

    try:
        while True:
            frame_count += 1
            keys = p.getKeyboardEvents()
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                auto_camera = not auto_camera
                print("[DEBUG] Auto camera toggled:", auto_camera)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED and not auto_camera:
                p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89,
                                             cameraTargetPosition=[0,0,0])
                print("[DEBUG] Camera reset.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                lidar_enabled = not lidar_enabled
                print("[DEBUG] LIDAR toggled:", lidar_enabled)

            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)
            target = None

            # --- Update Block Orientation from Slider ---
            block_yaw_deg = p.readUserDebugParameter(block_yaw_slider)
            block_yaw_rad = math.radians(block_yaw_deg)
            block_pos, _ = p.getBasePositionAndOrientation(block_body_id)
            p.resetBasePositionAndOrientation(block_body_id, block_pos, p.getQuaternionFromEuler([0, 0, block_yaw_rad]))

            update_block_visualization(block_body_id, block_center_sphere)

            # --- State Machine ---
            if current_state == STATE_IDLE:
                if idle_start_time is None:
                    idle_start_time = time.time()
                    print("[DEBUG] IDLE: Starting idle period.")
                if time.time() - idle_start_time < 2.0:
                    set_motor_speeds(robot_id, joint_dict, 0, 0, force)
                    print("[DEBUG] IDLE: Waiting...")
                else:
                    current_state = STATE_FIND_BLOCK
                    idle_start_time = None
                    print("[DEBUG] IDLE complete: Transition to FIND_BLOCK")

            elif current_state == STATE_FIND_BLOCK:
                pos_block, _ = p.getBasePositionAndOrientation(block_body_id)
                target = [pos_block[0], pos_block[1]]
                dx = target[0] - base_pos[0]
                dy = target[1] - base_pos[1]
                desired_angle = math.atan2(dy, dx)
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                if abs(angle_error) < math.radians(30):
                    print("[DEBUG] FIND_BLOCK: Block detected; transitioning to APPROACH_BLOCK")
                    current_state = STATE_APPROACH_BLOCK
                else:
                    left_velocity = -angle_error
                    right_velocity = angle_error
                    set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity, force)
                    print(f"[DEBUG] FIND_BLOCK: Rotating (angle error={math.degrees(angle_error):.1f}°)")

            elif current_state == STATE_APPROACH_BLOCK:
                pos_block, _ = p.getBasePositionAndOrientation(block_body_id)
                # Compute docking target: block center shifted 0.2m perpendicular to chosen face.
                rot_mat = p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(block_body_id)[1])
                local_x = np.array([rot_mat[0], rot_mat[3]])
                local_y = np.array([rot_mat[1], rot_mat[4]])
                vec = np.array([base_pos[0]-pos_block[0], base_pos[1]-pos_block[1]])
                norm = np.linalg.norm(vec)
                vec_norm = vec/norm if norm > 1e-3 else np.array([1,0])
                dot_x = np.dot(vec_norm, local_x)
                dot_y = np.dot(vec_norm, local_y)
                if abs(dot_x) >= abs(dot_y):
                    chosen_axis = local_x if dot_x > 0 else -local_x
                else:
                    chosen_axis = local_y if dot_y > 0 else -local_y
                docking_offset = 0.2
                target = [pos_block[0] + docking_offset * chosen_axis[0],
                          pos_block[1] + docking_offset * chosen_axis[1]]
                # Update the target visualization (orange sphere)
                p.resetBasePositionAndOrientation(target_vis_id, [target[0], target[1], 0.2], [0,0,0,1])
                distance, angle_error = drive_to_target(robot_id, joint_dict, target,
                                                        forward_speed, angle_pid, dt, force,
                                                        stop_distance=final_approach_distance)
                print(f"[DEBUG] APPROACH_BLOCK: Dist={distance:.2f}, AngleErr={math.degrees(angle_error):.1f}")
                if distance <= final_approach_distance:
                    current_state = STATE_ALIGN_BLOCK
                    print("[DEBUG] Transition to ALIGN_BLOCK")

            elif current_state == STATE_ALIGN_BLOCK:
                pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
                rot_mat = p.getMatrixFromQuaternion(orn_block)
                local_x = np.array([rot_mat[0], rot_mat[3]])
                local_y = np.array([rot_mat[1], rot_mat[4]])
                vec = np.array([base_pos[0]-pos_block[0], base_pos[1]-pos_block[1]])
                norm = np.linalg.norm(vec)
                vec_norm = vec/norm if norm > 1e-3 else np.array([1,0])
                dot_x = np.dot(vec_norm, local_x)
                dot_y = np.dot(vec_norm, local_y)
                print(f"[DEBUG] ALIGN_BLOCK: vec_norm={vec_norm}, dot_x={dot_x:.2f}, dot_y={dot_y:.2f}")
                if abs(dot_x) >= abs(dot_y):
                    chosen_axis = local_x if dot_x > 0 else -local_x
                else:
                    chosen_axis = local_y if dot_y > 0 else -local_y
                desired_angle = math.atan2(chosen_axis[1], chosen_axis[0])
                desired_angle = (desired_angle + math.pi) % (2*math.pi) - math.pi
                angle_error = (desired_angle - math.pi - rover_yaw + math.pi) % (2*math.pi) - math.pi
                print(f"[DEBUG] ALIGN_BLOCK: desired_angle={math.degrees(desired_angle):.1f}°, rover_yaw={math.degrees(rover_yaw):.1f}°, angle_error={math.degrees(angle_error):.1f}°")
                pid_output = angle_pid.compute(angle_error, dt)
                dx = pos_block[0] - base_pos[0]
                dy = pos_block[1] - base_pos[1]
                distance = math.hypot(dx, dy)
                if abs(angle_error) > stop_alignment_threshold and distance > final_approach_distance:
                    left_velocity = -Kp * angle_error + 1
                    right_velocity = Kp * angle_error + 1
                else:
                    left_velocity = right_velocity = 0.0
                    print("[DEBUG] ALIGN_BLOCK: Perfect alignment achieved. Transition to MANUAL_DOCK")
                    current_state = STATE_MANUAL_DOCK
                set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity, force)

            elif current_state == STATE_MANUAL_DOCK:
                set_motor_speeds(robot_id, joint_dict, 0, 0, force)
                print("[DEBUG] MANUAL_DOCK: Waiting for Dock key (D)...")
                if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                    pos_rover, orn_rover = p.getBasePositionAndOrientation(robot_id)
                    pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
                    invPos, invOrn = p.invertTransform(pos_rover, orn_rover)
                    relPos, _ = p.multiplyTransforms(invPos, invOrn, pos_block, orn_block)
                    relPos = [relPos[0], relPos[1], relPos[2] + 0.03]
                    dock_constraint = p.createConstraint(robot_id, -1, block_body_id, -1,
                                                         p.JOINT_FIXED, [0,0,0],
                                                         relPos, [0,0,0])
                    print("[DEBUG] Docked: Block attached and lifted. Transition to LOCATE_DEST.")
                    current_state = STATE_LOCATE_DEST

            elif current_state == STATE_LOCATE_DEST:
                dx_slider = p.readUserDebugParameter(dest_x_slider)
                dy_slider = p.readUserDebugParameter(dest_y_slider)
                target = [dx_slider, dy_slider]
                print(f"[DEBUG] LOCATE_DEST: Destination set to ({dx_slider:.2f}, {dy_slider:.2f}). Transition to APPROACH_DEST.")
                current_state = STATE_APPROACH_DEST

            elif current_state == STATE_APPROACH_DEST:
                dx_slider = p.readUserDebugParameter(dest_x_slider)
                dy_slider = p.readUserDebugParameter(dest_y_slider)
                target = [dx_slider, dy_slider]
                distance, angle_error = drive_to_target(robot_id, joint_dict, target, forward_speed,
                                                        angle_pid, dt, force, stop_distance=final_approach_distance)
                print(f"[DEBUG] APPROACH_DEST: Dist={distance:.2f}, AngleErr={math.degrees(angle_error):.1f}")
                if distance <= final_approach_distance:
                    current_state = STATE_ALIGN_DEST
                    print("[DEBUG] Transition to ALIGN_DEST")

            elif current_state == STATE_ALIGN_DEST:
                dx_slider = p.readUserDebugParameter(dest_x_slider)
                dy_slider = p.readUserDebugParameter(dest_y_slider)
                target = [dx_slider, dy_slider]
                desired_angle = pick_strict_side_angle(target[0]-base_pos[0], target[1]-base_pos[1])
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_output = angle_pid.compute(angle_error, dt)
                if abs(angle_error) > stop_alignment_threshold:
                    left_velocity = -Kp * angle_error * pid_rotation_scale
                    right_velocity = Kp * angle_error * pid_rotation_scale
                else:
                    left_velocity = right_velocity = 0.0
                    print("[DEBUG] ALIGN_DEST: Coarse alignment achieved. Transition to ROTATE_DEST")
                    current_state = STATE_ROTATE_DEST
                set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity, force)

            elif current_state == STATE_ROTATE_DEST:
                dx_slider = p.readUserDebugParameter(dest_x_slider)
                dy_slider = p.readUserDebugParameter(dest_y_slider)
                target = [dx_slider, dy_slider]
                desired_angle = pick_strict_side_angle(target[0]-base_pos[0], target[1]-base_pos[1])
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_output = angle_pid.compute(angle_error, dt)
                if abs(angle_error) > stop_alignment_threshold:
                    left_velocity = -Kp * angle_error * pid_rotation_scale
                    right_velocity = Kp * angle_error * pid_rotation_scale
                else:
                    left_velocity = right_velocity = 0.0
                    print("[DEBUG] ROTATE_DEST: Fine alignment complete. Transition to MANUAL_UNDOCK")
                    current_state = STATE_MANUAL_UNDOCK
                set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity, force)

            elif current_state == STATE_MANUAL_UNDOCK:
                set_motor_speeds(robot_id, joint_dict, 0, 0, force)
                print("[DEBUG] MANUAL_UNDOCK: Waiting for Undock key (U)...")
                if ord('u') in keys and keys[ord('u')] & p.KEY_WAS_TRIGGERED:
                    if dock_constraint is not None:
                        p.removeConstraint(dock_constraint)
                        dock_constraint = None
                    print("[DEBUG] Undocked: Block released. Transition to DELIVERED")
                    current_state = STATE_DELIVERED

            elif current_state == STATE_DELIVERED:
                set_motor_speeds(robot_id, joint_dict, 0, 0, force)
                print("[DEBUG] DELIVERED: Process complete, rover idle.")

            update_camera_view(auto_camera, base_pos, target)
            if lidar_enabled:
                simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140)
            imu_text = f"IMU: Roll={roll:.1f} Pitch={pitch:.1f} Yaw={math.degrees(rover_yaw):.1f}"
            p.addUserDebugText(imu_text, [base_pos[0], base_pos[1], base_pos[2]+0.5],
                               textColorRGB=[1,1,1], textSize=1.2, lifeTime=1/120.)
            p.stepSimulation()
            time.sleep(dt)
            if frame_count % 120 == 0:
                print(f"[DEBUG] Frame {frame_count}: State={current_state}")
    except KeyboardInterrupt:
        print("[DEBUG] Simulation terminated by user.")
    finally:
        p.disconnect()

if __name__ == "__main__":
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    block_urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeStructure.urdf"
    load_and_control_urdf(urdf_file, block_urdf_file)
