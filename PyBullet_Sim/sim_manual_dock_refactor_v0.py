import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random

# --- Simulation State Constants (using numbers) ---
STATE_IDLE = 0
STATE_APPROACH_BLOCK = 1
STATE_FINAL_ALIGN_BLOCK = 2
STATE_APPROACH_DEST = 3
STATE_FINAL_ALIGN_DEST = 4
STATE_DELIVERED = 5


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
    """
    Simulate a LiDAR sensor by casting rays and drawing debug lines.
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
        p.addUserDebugLine(ray_from_list[i], hit_pos, lineColorRGB=color, lineWidth=1, lifeTime=1 / 240.)


# --- Helper Functions ---
def pick_strict_side_angle(dx, dy):
    """
    Determine docking angle based on the dominant axis direction.
    """
    if abs(dx) >= abs(dy):
        return 0.0 if dx >= 0 else math.pi
    else:
        return math.pi / 2 if dy >= 0 else -math.pi / 2


def set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity, force):
    """
    Set motor speeds for the robot's wheels.
    """
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
    """
    Update the camera view based on the robot and target positions.
    """
    if auto_camera and target is not None:
        mid_x = (base_pos[0] + target[0]) / 2.0
        mid_y = (base_pos[1] + target[1]) / 2.0
        midpoint = [mid_x, mid_y, 0]
        separation = math.hypot(base_pos[0] - target[0], base_pos[1] - target[1])
        required_distance = (separation / 2) / math.tan(math.radians(FOV / 2))
        min_dist = 1.5
        cam_dist = max(min_dist, required_distance * 1.2)
        p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=0,
                                     cameraPitch=-89, cameraTargetPosition=midpoint)


def update_block_visualization(block_body_id, block_center_sphere):
    """
    Update the visualization of the block's center and orientation frame.
    """
    # Update block center sphere.
    block_center, _ = p.getBasePositionAndOrientation(block_body_id)
    p.resetBasePositionAndOrientation(block_center_sphere, block_center, [0, 0, 0, 1])

    # Draw orientation axes.
    pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
    rot_mat = p.getMatrixFromQuaternion(orn_block)
    axis_x = [rot_mat[0], rot_mat[3], rot_mat[6]]
    axis_y = [rot_mat[1], rot_mat[4], rot_mat[7]]
    axis_z = [rot_mat[2], rot_mat[5], rot_mat[8]]
    scale = 0.15
    p.addUserDebugLine(pos_block,
                       [pos_block[0] + scale * axis_x[0],
                        pos_block[1] + scale * axis_x[1],
                        pos_block[2] + scale * axis_x[2]],
                       [1, 0, 0],
                       lifeTime=1)
    p.addUserDebugLine(pos_block,
                       [pos_block[0] + scale * axis_y[0],
                        pos_block[1] + scale * axis_y[1],
                        pos_block[2] + scale * axis_y[2]],
                       [0, 1, 0],
                       lifeTime=1)
    p.addUserDebugLine(pos_block,
                       [pos_block[0] + scale * axis_z[0],
                        pos_block[1] + scale * axis_z[1],
                        pos_block[2] + scale * axis_z[2]],
                       [0, 0, 1],
                       lifeTime=1)


def load_robot(urdf_path):
    """
    Load the robot URDF and return its ID and joint mapping.
    """
    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1], useFixedBase=False)
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
    """
    Load the block URDF at a random position and create its visualization.
    Returns the block's ID and the ID of the center sphere visualization.
    """
    r_val = random.uniform(1, 3)
    angle = random.uniform(0, 2 * math.pi)
    block_x = r_val * math.cos(angle)
    block_y = r_val * math.sin(angle)
    block_z = 0.01  # Slightly above ground

    random_z_rotation = random.uniform(0, 2 * math.pi)
    orientation = p.getQuaternionFromEuler([0, 0, random_z_rotation])

    try:
        block_body_id = p.loadURDF(block_urdf_path, basePosition=[block_x, block_y, block_z],
                                   baseOrientation=orientation, useFixedBase=False)
        print(
            f"[DEBUG] Block loaded at ({block_x:.2f}, {block_y:.2f}, {block_z}) with rotation {math.degrees(random_z_rotation):.1f}°")
    except Exception as e:
        print("[ERROR] Failed to load block:", e)
        p.disconnect()
        return None, None

    block_center_sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                     radius=0.05,
                                                     rgbaColor=[1, 0, 0, 0.5])
    block_center_sphere = p.createMultiBody(baseMass=0,
                                            baseVisualShapeIndex=block_center_sphere_visual,
                                            baseCollisionShapeIndex=-1,
                                            basePosition=[block_x, block_y, block_z])
    print("[DEBUG] Block center sphere created.")
    return block_body_id, block_center_sphere


def create_destination_visual():
    """
    Create a visual destination target.
    """
    dest_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2],
                                        rgbaColor=[1, 0, 0, 0.5])
    dest_body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=dest_shape_id,
                                     baseCollisionShapeIndex=-1, basePosition=[0, 0, 0.2])
    print("[DEBUG] Destination target created.")
    return dest_body_id


def initialize_simulation():
    """
    Initialize the PyBullet simulation environment.
    """
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    print("[DEBUG] PyBullet connected; gravity set to -9.81.")

    p.loadURDF("plane.urdf")
    print("[DEBUG] Plane URDF loaded.")


# --- Main Simulation Loop ---
def load_and_control_urdf(urdf_path, block_urdf_path):
    """
    Load URDFs, initialize control parameters, and execute the main simulation loop.
    """
    initialize_simulation()

    robot_id, joint_dict = load_robot(urdf_path)
    if robot_id is None:
        return

    # --- Control and Motion Parameters ---
    force = 20
    forward_speed = 5.0
    Kp, Ki, Kd = 2.0, 0.25, 0.01
    threshold_stage1 = math.radians(90)
    final_approach_distance = 0.12
    predock_offset = 0.2
    stop_alignment_threshold = math.radians(2)
    pid_rotation_scale = 4  # scaling factor for PID rotation
    dt = 1 / 120.0

    # Initialize PID controller.
    angle_pid = PIDController(Kp, Ki, Kd)

    # --- UI Elements ---
    dest_x_slider = p.addUserDebugParameter("Destination X", -10, 10, 0.0)
    dest_y_slider = p.addUserDebugParameter("Destination Y", -10, 10, 0.0)
    print("[DEBUG] Destination sliders created.")

    block_body_id, block_center_sphere = load_block(block_urdf_path)
    if block_body_id is None:
        return

    dest_body_id = create_destination_visual()

    # --- Camera and LIDAR Settings ---
    lidar_enabled = False
    auto_camera = True
    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0])
    print("[DEBUG] Camera set to overhead view.")

    print("Controls:")
    print("  C: Toggle camera (Auto vs Free)")
    print("  R: Reset camera")
    print("  S: Toggle LIDAR")
    print("  D: Dock (attach block)")
    print("  U: Undock (release block)")
    print("\nState Legend:")
    print("  0: IDLE")
    print("  1: APPROACH_BLOCK")
    print("  2: FINAL_ALIGN_BLOCK")
    print("  3: APPROACH_DEST")
    print("  4: FINAL_ALIGN_DEST")
    print("  5: DELIVERED")

    current_state = STATE_IDLE
    idle_start_time = None
    dock_constraint = None
    frame_count = 0

    try:
        while True:
            frame_count += 1
            keys = p.getKeyboardEvents()
            # --- Process Keyboard Inputs ---
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                auto_camera = not auto_camera
                print("[DEBUG] Auto camera toggled:", auto_camera)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED and not auto_camera:
                p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89,
                                             cameraTargetPosition=[0, 0, 0])
                print("[DEBUG] Camera reset.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                lidar_enabled = not lidar_enabled
                print("[DEBUG] LIDAR toggled:", lidar_enabled)

            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)
            target = None

            # --- Update Visualizations ---
            update_block_visualization(block_body_id, block_center_sphere)

            # --- State Machine ---
            if current_state == STATE_IDLE:
                if idle_start_time is None:
                    idle_start_time = time.time()
                    print("[DEBUG] IDLE: Starting idle period.")
                if time.time() - idle_start_time < 2.0:
                    left_velocity = right_velocity = 0.0
                    print("[DEBUG] IDLE: Waiting...")
                else:
                    current_state = STATE_APPROACH_BLOCK
                    idle_start_time = None
                    print("[DEBUG] IDLE complete: Transition to APPROACH_BLOCK")

            elif current_state == STATE_APPROACH_BLOCK:
                # Approach the block by aligning with its center.
                pos_block, _ = p.getBasePositionAndOrientation(block_body_id)
                target = [pos_block[0], pos_block[1]]
                dx = target[0] - base_pos[0]
                dy = target[1] - base_pos[1]
                distance = math.hypot(dx, dy)
                desired_angle = pick_strict_side_angle(dx, dy)
                angle_error = (desired_angle - rover_yaw + math.pi) % (2 * math.pi) - math.pi

                pid_output = angle_pid.compute(angle_error, dt)
                if abs(angle_error) > threshold_stage1:
                    print("[DEBUG] APPROACH_BLOCK: Rotating in place (angle error too high).")
                    left_velocity = -Kp * angle_error
                    right_velocity = Kp * angle_error
                else:
                    if distance > final_approach_distance + predock_offset:
                        left_velocity = forward_speed - pid_output
                        right_velocity = forward_speed + pid_output
                    else:
                        current_state = STATE_FINAL_ALIGN_BLOCK
                        print("[DEBUG] Transition to FINAL_ALIGN_BLOCK")
                        left_velocity = right_velocity = 0.0

            elif current_state == STATE_FINAL_ALIGN_BLOCK:
                # Align rover's orientation with the block's docking face.
                pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
                dx = pos_block[0] - base_pos[0]
                dy = pos_block[1] - base_pos[1]
                rot_mat = p.getMatrixFromQuaternion(orn_block)
                local_x = np.array([rot_mat[0], rot_mat[3]])
                local_y = np.array([rot_mat[1], rot_mat[4]])
                vec = np.array([base_pos[0] - pos_block[0], base_pos[1] - pos_block[1]])
                norm = np.linalg.norm(vec)
                vec_norm = vec / norm if norm > 1e-3 else np.array([1, 0])

                dot_x = np.dot(vec_norm, local_x)
                dot_y = np.dot(vec_norm, local_y)
                print(f"[DEBUG] FINAL_ALIGN_BLOCK: vec_norm={vec_norm}, dot_x={dot_x:.2f}, dot_y={dot_y:.2f}")

                if abs(dot_x) >= abs(dot_y):
                    chosen_axis = local_x if dot_x > 0 else -local_x
                else:
                    chosen_axis = local_y if dot_y > 0 else -local_y

                desired_angle = math.atan2(chosen_axis[1], chosen_axis[0])
                desired_angle = (desired_angle + math.pi) % (2 * math.pi) - math.pi
                angle_error = (desired_angle - rover_yaw + math.pi) % (2 * math.pi) - math.pi
                print(
                    f"[DEBUG] FINAL_ALIGN_BLOCK: desired_angle={math.degrees(desired_angle):.1f}°, rover_yaw={math.degrees(rover_yaw):.1f}°, angle_error={math.degrees(angle_error):.1f}°")

                pid_output = angle_pid.compute(angle_error, dt)
                if abs(angle_error) > stop_alignment_threshold:
                    left_velocity = -Kp * angle_error * pid_rotation_scale
                    right_velocity = Kp * angle_error * pid_rotation_scale
                else:
                    left_velocity = right_velocity = 0.0
                    print("[DEBUG] FINAL_ALIGN_BLOCK: Alignment complete.")
                    # (Transition to next state as needed.)

            elif current_state == STATE_APPROACH_DEST:
                # Move toward the destination specified via sliders.
                dx_slider = p.readUserDebugParameter(dest_x_slider)
                dy_slider = p.readUserDebugParameter(dest_y_slider)
                target = [dx_slider, dy_slider]
                dx = target[0] - base_pos[0]
                dy = target[1] - base_pos[1]
                distance = math.hypot(dx, dy)
                desired_angle = pick_strict_side_angle(dx, dy)
                angle_error = (desired_angle - rover_yaw + math.pi) % (2 * math.pi) - math.pi

                pid_output = angle_pid.compute(angle_error, dt)
                if distance > final_approach_distance:
                    left_velocity = forward_speed - pid_output
                    right_velocity = forward_speed + pid_output
                else:
                    current_state = STATE_FINAL_ALIGN_DEST
                    left_velocity = right_velocity = 0.0
                    print("[DEBUG] Transition to FINAL_ALIGN_DEST")

            elif current_state == STATE_FINAL_ALIGN_DEST:
                # Fine-tune alignment at the destination.
                dx_slider = p.readUserDebugParameter(dest_x_slider)
                dy_slider = p.readUserDebugParameter(dest_y_slider)
                target = [dx_slider, dy_slider]
                dx_fine = target[0] - base_pos[0]
                dy_fine = target[1] - base_pos[1]
                distance = math.hypot(dx_fine, dy_fine)
                desired_angle = pick_strict_side_angle(dx_fine, dy_fine)
                angle_error = (desired_angle - rover_yaw + math.pi) % (2 * math.pi) - math.pi

                pid_output = angle_pid.compute(angle_error, dt)
                if distance > final_approach_distance and abs(angle_error) > stop_alignment_threshold:
                    linear_speed = forward_speed * 0.2
                    left_velocity = linear_speed - pid_output
                    right_velocity = linear_speed + pid_output
                else:
                    left_velocity = right_velocity = 0.0
                    print("[DEBUG] FINAL_ALIGN_DEST: Perfect alignment achieved.")
                    if ord('u') in keys and keys[ord('u')] & p.KEY_WAS_TRIGGERED:
                        if dock_constraint is not None:
                            p.removeConstraint(dock_constraint)
                            dock_constraint = None
                        current_state = STATE_DELIVERED
                        print("[DEBUG] Undocked: Block released. Transition to DELIVERED.")

            elif current_state == STATE_DELIVERED:
                left_velocity = right_velocity = 0.0
                target = target if target is not None else [0, 0]
                print("[DEBUG] DELIVERED: Process complete, rover idle.")

            # --- Actuation and Simulation Step ---
            set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity, force)
            update_camera_view(auto_camera, base_pos, target)
            if lidar_enabled:
                simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140)

            imu_text = f"IMU: Roll={roll:.1f} Pitch={pitch:.1f} Yaw={math.degrees(rover_yaw):.1f}"
            p.addUserDebugText(imu_text, [base_pos[0], base_pos[1], base_pos[2] + 0.5],
                               textColorRGB=[1, 1, 1], textSize=1.2, lifeTime=1 / 120.)

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
