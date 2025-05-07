import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random
import traceback

# --- State Constants ---
STATE_IDLE = 0
STATE_FIND_BLOCK = 1          # Find closest available block
STATE_APPROACH_BLOCK = 2
STATE_ALIGN_BLOCK = 3
STATE_FINAL_DRIVE_BLOCK = 4
STATE_DOCKING = 5
STATE_LOCATE_DEST = 6         # Calculate specific placement spot for this block
STATE_APPROACH_DEST = 7
STATE_ALIGN_DEST = 8
STATE_FINAL_DRIVE_DEST = 9
STATE_UNDOCKING = 10
STATE_RETREAT = 11
STATE_ALL_DELIVERED = 12      # When no blocks are left

# --- Control Parameters ---
FORCE = 20.0
FORWARD_SPEED = 5.0
ROTATION_SPEED_SCALE = 2.0 # <<< Reduced Rotation Scale
MAX_ROTATION_VELOCITY = 3.0 # <<< Max angular velocity (rad/s) for PID clamping
FINAL_DRIVE_SPEED = 1.0
RETREAT_SPEED = 1.5
RETREAT_DISTANCE = 0.15 # <<< Maybe increase slightly if collisions still occur (e.g., 0.18)
PLACEMENT_OFFSET = 0.11

# --- PID Parameters ---
# NOTE: These might need further tuning after other changes
KP = 1.8 # Slightly reduced Kp
KI = 0.05 # Reduced Ki
KD = 0.02 # Slightly increased Kd

# --- Thresholds and Distances ---
FIND_BLOCK_ANGLE_THRESHOLD = math.radians(30)
APPROACH_BLOCK_STOP_DISTANCE = 0.05
ALIGNMENT_STABILITY_THRESHOLD = math.radians(1.5) # Tightened
ANGULAR_VELOCITY_STABILITY_THRESHOLD = 0.05 # rad/s <<< New threshold for stability check
STABILITY_DURATION = 0.75 # Increased
FINAL_APPROACH_DISTANCE = 0.12
DOCKING_DISTANCE = 0.11
FINAL_DRIVE_TIMEOUT = 10.0

# --- Simulation Parameters ---
SIMULATION_TIMESTEP = 1/240.0
NUM_BLOCKS = 3

# --- PID Controller Class ---
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.MAX_INTEGRAL = 5.0 # Example Integral Clamp Limit

    def compute(self, error, dt):
        dt = max(dt, 1e-6)
        self.error_integral += error * dt
        # Clamp integral term to prevent windup
        self.error_integral = max(min(self.error_integral, self.MAX_INTEGRAL), -self.MAX_INTEGRAL)

        derivative = (error - self.previous_error) / dt
        # Optional: Clamp derivative term to prevent spikes
        # MAX_DERIVATIVE = 100.0
        # derivative = max(min(derivative, MAX_DERIVATIVE), -MAX_DERIVATIVE)

        output = self.Kp * error + self.Ki * self.error_integral + self.Kd * derivative
        self.previous_error = error

        # <<< Clamp final output >>>
        output = max(min(output, MAX_ROTATION_VELOCITY), -MAX_ROTATION_VELOCITY)
        return output

    def reset(self):
        self.error_integral = 0.0
        self.previous_error = 0.0

# --- Helper Functions ---
def is_valid_coordinate(coord):
    return not (math.isnan(coord) or math.isinf(coord))

def are_valid_coordinates(*coords):
    if not coords: return False
    flat_coords = [c for point in coords if point is not None for c in point]
    if not flat_coords: return False
    return all(is_valid_coordinate(c) for c in flat_coords)

def simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140):
    # (No changes from previous correct version)
    try:
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        if not are_valid_coordinates(base_pos): return
        sensor_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.1]
        _, _, rover_yaw = p.getEulerFromQuaternion(base_orn)
        fov_rad = math.radians(fov_deg)
        start_angle = rover_yaw - fov_rad/2
        ray_from_list, ray_to_list = [], []
        for i in range(num_rays):
            angle = start_angle + (i/(num_rays-1))*fov_rad
            dx = ray_length * math.cos(angle); dy = ray_length * math.sin(angle)
            if not (is_valid_coordinate(dx) and is_valid_coordinate(dy)): continue
            ray_from = sensor_pos; ray_to = [sensor_pos[0]+dx, sensor_pos[1]+dy, sensor_pos[2]]
            if not are_valid_coordinates(ray_from, ray_to): continue
            ray_from_list.append(ray_from); ray_to_list.append(ray_to)
        if not ray_from_list: return
        results = p.rayTestBatch(ray_from_list, ray_to_list)
        for i, res in enumerate(results):
            hit_fraction = res[2]; hit_pos = res[3] if hit_fraction < 1.0 else ray_to_list[i]
            color = [1, 0, 0] if hit_fraction < 1.0 else [0, 1, 0]
            if are_valid_coordinates(ray_from_list[i], hit_pos):
                try: p.addUserDebugLine(ray_from_list[i], hit_pos, lineColorRGB=color, lineWidth=1, lifeTime=SIMULATION_TIMESTEP * 2)
                except p.error: pass
    except p.error: pass
    except Exception as e: print(f"[ERROR] Unexpected error in simulate_lidar: {e}"); traceback.print_exc()


def pick_strict_side_angle(dx, dy):
    # (No changes needed)
    if abs(dx) >= abs(dy): return 0.0 if dx >= 0 else math.pi
    else: return math.pi/2 if dy >= 0 else -math.pi/2

def set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity):
    # (No changes needed from previous correct version)
    if not (is_valid_coordinate(left_velocity) and is_valid_coordinate(right_velocity)):
        print(f"[ERROR] Invalid motor velocities requested: L={left_velocity}, R={right_velocity}. Stopping motors.")
        left_velocity = 0.0; right_velocity = 0.0
    try:
        p.setJointMotorControl2(robot_id, joint_dict['wheel_FL_joint'], controlMode=p.VELOCITY_CONTROL, targetVelocity=-left_velocity, force=FORCE)
        p.setJointMotorControl2(robot_id, joint_dict['wheel_RL_joint'], controlMode=p.VELOCITY_CONTROL, targetVelocity=-left_velocity, force=FORCE)
        p.setJointMotorControl2(robot_id, joint_dict['wheel_FR_joint'], controlMode=p.VELOCITY_CONTROL, targetVelocity=right_velocity, force=FORCE)
        p.setJointMotorControl2(robot_id, joint_dict['wheel_RR_joint'], controlMode=p.VELOCITY_CONTROL, targetVelocity=right_velocity, force=FORCE)
    except p.error: pass

def update_camera_view(auto_camera, base_pos, target, FOV=60):
    # (No changes needed from previous correct version)
    if auto_camera and target is not None and are_valid_coordinates(base_pos, target):
        mid_x = (base_pos[0] + target[0]) / 2.0; mid_y = (base_pos[1] + target[1]) / 2.0
        midpoint = [mid_x, mid_y, 0]; separation = math.hypot(base_pos[0]-target[0], base_pos[1]-target[1])
        required_distance = (separation/2)/math.tan(math.radians(FOV/2)) if separation > 0.1 else 1.0
        min_dist = 1.5; cam_dist = max(min_dist, required_distance*1.2)
        try: p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=midpoint)
        except p.error: pass

def update_all_block_visualizations(block_viz_dict):
    # (No changes needed from previous correct version)
    if block_viz_dict is None: return
    for block_id, sphere_id in block_viz_dict.items():
        try:
            pos_block, orn_block = p.getBasePositionAndOrientation(block_id)
            if not are_valid_coordinates(pos_block): continue
            p.resetBasePositionAndOrientation(sphere_id, pos_block, [0,0,0,1])
            rot_mat = p.getMatrixFromQuaternion(orn_block)
            axis_x = [rot_mat[0], rot_mat[3], rot_mat[6]]; axis_y = [rot_mat[1], rot_mat[4], rot_mat[7]]; axis_z = [rot_mat[2], rot_mat[5], rot_mat[8]]
            scale = 0.15
            end_x = [pos_block[0]+scale*axis_x[0], pos_block[1]+scale*axis_x[1], pos_block[2]+scale*axis_x[2]]
            if are_valid_coordinates(pos_block, end_x):
                try: p.addUserDebugLine(pos_block, end_x, [1,0,0], lifeTime=SIMULATION_TIMESTEP * 2)
                except p.error: pass
            end_y = [pos_block[0]+scale*axis_y[0], pos_block[1]+scale*axis_y[1], pos_block[2]+scale*axis_y[2]]
            if are_valid_coordinates(pos_block, end_y):
                 try: p.addUserDebugLine(pos_block, end_y, [0,1,0], lifeTime=SIMULATION_TIMESTEP * 2)
                 except p.error: pass
            end_z = [pos_block[0]+scale*axis_z[0], pos_block[1]+scale*axis_z[1], pos_block[2]+scale*axis_z[2]]
            if are_valid_coordinates(pos_block, end_z):
                 try: p.addUserDebugLine(pos_block, end_z, [0,0,1], lifeTime=SIMULATION_TIMESTEP * 2)
                 except p.error: pass
        except p.error: pass

# --- Loading Functions ---
def load_robot(urdf_path):
    # (No changes needed)
    print("[DEBUG] Attempting to load robot from:", urdf_path)
    try: robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0.1], useFixedBase=False); print(f"[DEBUG] Robot loaded successfully. ID: {robot_id}")
    except Exception as e: print("!!!!!!!!!!!!!!!!!!\n[ERROR] FAILED TO LOAD ROBOT URDF:", e); traceback.print_exc(); print("!!!!!!!!!!!!!!!!!!") ; time.sleep(10); return None, None
    joint_dict = {}; num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints): info = p.getJointInfo(robot_id, i); joint_dict[info[1].decode("utf-8")] = i
    print(f"[DEBUG] Robot joint mapping created: {joint_dict}"); return robot_id, joint_dict

def load_blocks(block_urdf_path, num_to_load):
    # (No changes needed)
    blocks = {}; block_states = {}
    print(f"[DEBUG] Attempting to load {num_to_load} blocks...")
    for i in range(num_to_load):
        r_val = random.uniform(1, 3); angle = random.uniform(0, 2*math.pi)
        block_x = r_val * math.cos(angle); block_y = r_val * math.sin(angle); block_z = 0.01
        random_z_rotation = random.uniform(0, 2*math.pi); orientation = p.getQuaternionFromEuler([0,0,random_z_rotation])
        print(f"[DEBUG]   Spawning block {i+1} at ({block_x:.2f}, {block_y:.2f}) rot {math.degrees(random_z_rotation):.1f}°")
        try:
            block_id = p.loadURDF(block_urdf_path, basePosition=[block_x, block_y, block_z], baseOrientation=orientation, useFixedBase=False)
            print(f"[DEBUG]   Block {i+1} loaded successfully. ID: {block_id}")
            sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1,0,0,0.5])
            sphere_id = p.createMultiBody(0, sphere_visual, -1, [block_x, block_y, block_z])
            blocks[block_id] = sphere_id
            block_states[block_id] = 'AVAILABLE'
        except Exception as e: print(f"!!!!!!!!!!!!!!!!!!\n[ERROR] FAILED TO LOAD BLOCK {i+1} URDF:", e); traceback.print_exc(); print("!!!!!!!!!!!!!!!!!!"); time.sleep(10); return None, None, None # Return failure
    print(f"[DEBUG] Successfully loaded {len(blocks)} blocks.")
    return blocks, block_states

def create_destination_visual():
    # (No changes needed)
    dest_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.055,0.055,0.055], rgbaColor=[0,0,1,0.5])
    dest_body_id = p.createMultiBody(0, dest_shape_id, -1, [0,0,-1])
    print("[DEBUG] Destination placement marker (blue cube) created.")
    return dest_body_id

def create_target_visual():
    # (No changes needed)
    target_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0.5, 0, 0.5])
    target_vis_id = p.createMultiBody(0, target_shape, -1, [0,0,-1])
    print("[DEBUG] Approach target visualization (orange sphere) created.")
    return target_vis_id

def initialize_simulation():
    # (No changes needed)
    if p.isConnected(): print("[WARN] Already connected to PyBullet.")
    try: client_id = p.connect(p.GUI); assert client_id >= 0, "Failed connection"; print(f"[DEBUG] PyBullet connected with client ID: {client_id}")
    except Exception as e: print(f"[FATAL] PyBullet connection error: {e}"); exit()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=SIMULATION_TIMESTEP)
    p.setRealTimeSimulation(0)
    print("[DEBUG] Gravity set, timestep configured, real-time simulation OFF.")
    try: p.loadURDF("plane.urdf"); print("[DEBUG] Plane URDF loaded.")
    except Exception as e: print("[ERROR] Failed to load plane.urdf:", e); time.sleep(5); p.disconnect(); exit()

# --- Builder Bot Class ---
class BuilderBot:
    def __init__(self, robot_id, joint_dict, block_viz_dict, initial_block_states, target_vis_id, dest_vis_id):
        self.robot_id = robot_id
        self.joint_dict = joint_dict
        self.block_viz_dict = block_viz_dict
        self.block_states = initial_block_states
        self.target_block_id = None
        self.target_vis_id = target_vis_id
        self.dest_vis_id = dest_vis_id

        self.current_state = STATE_IDLE
        self.dock_constraint = None
        self.angle_pid = PIDController(KP, KI, KD) # Use updated gains

        self.stability_timer = None
        self.retreat_start_pos = None
        self.base_destination = [0.0, -2.0]
        self.current_placement_target = [0.0, 0.0]
        self.placed_block_count = 0
        self.final_drive_start_time = None
        self.final_drive_target_yaw = 0.0
        self.retreat_yaw = 0.0

        self.base_dest_x_slider = p.addUserDebugParameter("Base Dest X", -5, 5, self.base_destination[0])
        self.base_dest_y_slider = p.addUserDebugParameter("Base Dest Y", -5, 5, self.base_destination[1])

        print("[DEBUG] BuilderBot initialized.")

    def get_state_name(self):
        states = ["IDLE", "FIND_BLOCK", "APPROACH_BLOCK", "ALIGN_BLOCK",
                  "FINAL_DRIVE_BLOCK", "DOCKING", "LOCATE_DEST", "APPROACH_DEST",
                  "ALIGN_DEST", "FINAL_DRIVE_DEST", "UNDOCKING", "RETREAT", "ALL_DELIVERED"]
        return states[self.current_state] if 0 <= self.current_state < len(states) else "UNKNOWN"

    def reset_stability_timer(self): self.stability_timer = None

    # <<< Updated stability check >>>
    def check_stability(self, angle_aligned, robot_id):
        try:
            # Get angular velocity
            lin_vel, ang_vel = p.getBaseVelocity(robot_id)
            angular_velocity_low = abs(ang_vel[2]) < ANGULAR_VELOCITY_STABILITY_THRESHOLD # Check Z-axis angular velocity
        except p.error:
            angular_velocity_low = False # Assume not stable if error getting velocity

        if angle_aligned and angular_velocity_low: # Check BOTH angle AND low velocity
            if self.stability_timer is None: self.stability_timer = time.time()
            elif time.time() - self.stability_timer >= STABILITY_DURATION: self.reset_stability_timer(); return True
        else:
            self.reset_stability_timer() # Reset if angle OR velocity condition fails
        return False

    def run_step(self):
        try: base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id); roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)
        except p.error as e: print(f"[ERROR] Failed to get robot pose: {e}. Resetting state."); self.current_state = STATE_IDLE; return None

        target = None; left_velocity = 0.0; right_velocity = 0.0

        # --- State Machine Logic ---
        if self.current_state == STATE_IDLE:
            # (No change)
            if self.stability_timer is None: self.stability_timer = time.time()
            if time.time() - self.stability_timer >= 1.0: self.current_state = STATE_FIND_BLOCK; self.reset_stability_timer(); self.angle_pid.reset(); print("[INFO] IDLE complete: Transition to FIND_BLOCK")
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_FIND_BLOCK:
            # (No change)
            closest_block_id = -1; min_dist_sq = float('inf'); available_blocks_exist = False
            for block_id, status in self.block_states.items():
                if status == 'AVAILABLE': available_blocks_exist = True;
                try:
                    pos_block, _ = p.getBasePositionAndOrientation(block_id)
                    if not are_valid_coordinates(pos_block) or status != 'AVAILABLE': continue
                    dist_sq = (pos_block[0] - base_pos[0])**2 + (pos_block[1] - base_pos[1])**2
                    if dist_sq < min_dist_sq: min_dist_sq = dist_sq; closest_block_id = block_id
                except p.error: continue
            if closest_block_id != -1:
                self.target_block_id = closest_block_id
                pos_block, _ = p.getBasePositionAndOrientation(self.target_block_id); target = [pos_block[0], pos_block[1]]
                dx = target[0] - base_pos[0]; dy = target[1] - base_pos[1]; desired_angle = math.atan2(dy, dx)
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                if abs(angle_error) < FIND_BLOCK_ANGLE_THRESHOLD: print(f"[INFO] FIND_BLOCK: Aligned with closest block {self.target_block_id}. Transitioning to APPROACH_BLOCK"); self.current_state = STATE_APPROACH_BLOCK; self.angle_pid.reset()
                else: pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP); left_velocity = -pid_output * ROTATION_SPEED_SCALE * 0.5; right_velocity = pid_output * ROTATION_SPEED_SCALE * 0.5
            elif available_blocks_exist: print("[WARN] FIND_BLOCK: Available blocks exist, but failed to find closest/valid one. Rotating slowly."); left_velocity = -0.5; right_velocity = 0.5
            else: print("[INFO] FIND_BLOCK: No available blocks found. Transitioning to ALL_DELIVERED."); self.current_state = STATE_ALL_DELIVERED
            if closest_block_id == -1: left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_APPROACH_BLOCK:
            # (No change)
            target_status = self.block_states.get(self.target_block_id, 'NOT_FOUND')
            if self.target_block_id is None or target_status != 'AVAILABLE':
                 print(f"[WARN] APPROACH_BLOCK: Invalid target block (ID: {self.target_block_id}, Status: '{target_status}'). Returning to FIND_BLOCK.")
                 self.current_state = STATE_FIND_BLOCK; self.target_block_id = None; return target
            try:
                pos_block, orn_block = p.getBasePositionAndOrientation(self.target_block_id)
                if not are_valid_coordinates(pos_block): raise p.error("Invalid block coordinates")
                rot_mat = p.getMatrixFromQuaternion(orn_block); local_x = np.array([rot_mat[0], rot_mat[3]]); local_y = np.array([rot_mat[1], rot_mat[4]])
                vec = np.array([base_pos[0]-pos_block[0], base_pos[1]-pos_block[1]]); norm = np.linalg.norm(vec); vec_norm = vec/norm if norm > 1e-3 else np.array([1,0])
                dot_x = np.dot(vec_norm, local_x); dot_y = np.dot(vec_norm, local_y)
                chosen_axis = local_x if abs(dot_x) >= abs(dot_y) else local_y; chosen_axis = chosen_axis if (dot_x >= 0 if abs(dot_x) >= abs(dot_y) else dot_y >= 0) else -chosen_axis
                offset_distance = FINAL_APPROACH_DISTANCE + 0.1; target = [pos_block[0] + offset_distance * chosen_axis[0], pos_block[1] + offset_distance * chosen_axis[1]]
                if are_valid_coordinates(target): p.resetBasePositionAndOrientation(self.target_vis_id, [target[0], target[1], 0.2], [0,0,0,1])
                dx = target[0] - base_pos[0]; dy = target[1] - base_pos[1]; distance = math.hypot(dx, dy)
                desired_angle = math.atan2(dy, dx); angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                if distance > APPROACH_BLOCK_STOP_DISTANCE:
                    speed_factor = max(0.1, math.cos(angle_error)**2); current_speed = FORWARD_SPEED * speed_factor
                    left_velocity = current_speed - pid_output; right_velocity = current_speed + pid_output
                else:
                    print("[INFO] APPROACH_BLOCK: Reached offset point. Transition to ALIGN_BLOCK")
                    self.current_state = STATE_ALIGN_BLOCK; self.angle_pid.reset(); self.reset_stability_timer()
                    p.resetBasePositionAndOrientation(self.target_vis_id, [0,0,-1], [0,0,0,1]); left_velocity = right_velocity = 0.0
            except p.error as e: print(f"[ERROR] PyBullet error in APPROACH_BLOCK: {e}. Resetting."); self.current_state = STATE_FIND_BLOCK; self.target_block_id = None

        elif self.current_state == STATE_ALIGN_BLOCK:
             if self.target_block_id is None: self.current_state = STATE_FIND_BLOCK; return target
             try:
                pos_block, orn_block = p.getBasePositionAndOrientation(self.target_block_id)
                if not are_valid_coordinates(pos_block): raise p.error("Invalid block coordinates")
                rot_mat = p.getMatrixFromQuaternion(orn_block); local_x = np.array([rot_mat[0], rot_mat[3]]); local_y = np.array([rot_mat[1], rot_mat[4]])
                vec = np.array([base_pos[0]-pos_block[0], base_pos[1]-pos_block[1]]); norm = np.linalg.norm(vec); vec_norm = vec/norm if norm > 1e-3 else np.array([1,0])
                dot_x = np.dot(vec_norm, local_x); dot_y = np.dot(vec_norm, local_y)
                chosen_axis = local_x if abs(dot_x) >= abs(dot_y) else local_y; chosen_axis = chosen_axis if (dot_x >= 0 if abs(dot_x) >= abs(dot_y) else dot_y >= 0) else -chosen_axis
                desired_angle = math.atan2(chosen_axis[1], chosen_axis[0]) + math.pi; desired_angle = (desired_angle + math.pi) % (2*math.pi) - math.pi
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP);
                aligned = abs(angle_error) < ALIGNMENT_STABILITY_THRESHOLD
                if not aligned:
                    # Apply rotation using clamped PID output and scale
                    left_velocity = -pid_output * ROTATION_SPEED_SCALE; right_velocity = pid_output * ROTATION_SPEED_SCALE;
                    self.reset_stability_timer()
                else:
                    left_velocity = right_velocity = 0.0
                    # <<< Use updated stability check >>>
                    if self.check_stability(aligned, self.robot_id):
                        print("[INFO] ALIGN_BLOCK: Alignment stable. Transition to FINAL_DRIVE_BLOCK")
                        self.final_drive_target_yaw = rover_yaw # Store current yaw
                        self.current_state = STATE_FINAL_DRIVE_BLOCK; self.angle_pid.reset()
                        self.final_drive_start_time = None
             except p.error as e: print(f"[ERROR] PyBullet error in ALIGN_BLOCK: {e}. Resetting."); self.current_state = STATE_FIND_BLOCK; self.target_block_id = None

        elif self.current_state == STATE_FINAL_DRIVE_BLOCK:
            # (No changes needed - already includes timeout and yaw correction)
            if self.final_drive_start_time is None: self.final_drive_start_time = time.time(); print(f"[INFO] FINAL_DRIVE_BLOCK: Entered state for block {self.target_block_id}. Starting timer.")
            if time.time() - self.final_drive_start_time > FINAL_DRIVE_TIMEOUT:
                print(f"[WARN] FINAL_DRIVE_BLOCK: Timeout ({FINAL_DRIVE_TIMEOUT}s) reached for block {self.target_block_id}. Aborting.")
                if self.target_block_id is not None: self.block_states[self.target_block_id] = 'AVAILABLE'
                self.target_block_id = None; self.final_drive_start_time = None; self.current_state = STATE_FIND_BLOCK; left_velocity = right_velocity = 0.0
            else:
                if self.target_block_id is None: self.current_state = STATE_FIND_BLOCK; self.final_drive_start_time = None; return target
                try:
                    pos_block, _ = p.getBasePositionAndOrientation(self.target_block_id)
                    if not are_valid_coordinates(pos_block): raise p.error("Invalid block coordinates")
                    target = [pos_block[0], pos_block[1]]
                    dx = pos_block[0] - base_pos[0]; dy = pos_block[1] - base_pos[1]; distance = math.hypot(dx, dy)
                    if distance > DOCKING_DISTANCE:
                        angle_error = (self.final_drive_target_yaw - rover_yaw + math.pi) % (2*math.pi) - math.pi
                        pid_correction = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                        left_velocity = FINAL_DRIVE_SPEED - pid_correction * 0.5; right_velocity = FINAL_DRIVE_SPEED + pid_correction * 0.5
                    else: print("[INFO] FINAL_DRIVE_BLOCK: Reached docking distance. Transition to DOCKING."); self.current_state = STATE_DOCKING; self.final_drive_start_time = None; left_velocity = right_velocity = 0.0
                except p.error as e: print(f"[ERROR] PyBullet error in FINAL_DRIVE_BLOCK: {e}. Resetting."); self.current_state = STATE_FIND_BLOCK; self.target_block_id = None; self.final_drive_start_time = None

        elif self.current_state == STATE_DOCKING:
            # (No changes needed)
            if self.target_block_id is None: self.current_state = STATE_FIND_BLOCK; return target
            if self.dock_constraint is None:
                try:
                    pos_rover, orn_rover = p.getBasePositionAndOrientation(self.robot_id); pos_block, orn_block = p.getBasePositionAndOrientation(self.target_block_id)
                    if not (are_valid_coordinates(pos_rover) and are_valid_coordinates(pos_block)): raise p.error("Invalid coordinates for docking")
                    invPos, invOrn = p.invertTransform(pos_rover, orn_rover); relPos, relOrn = p.multiplyTransforms(invPos, invOrn, pos_block, orn_block)
                    docking_lift = 0.03; relPos = [relPos[0], relPos[1], relPos[2] + docking_lift]
                    self.dock_constraint = p.createConstraint(self.robot_id, -1, self.target_block_id, -1, p.JOINT_FIXED, [0,0,0], parentFramePosition=relPos, childFramePosition=[0,0,0], parentFrameOrientation=relOrn)
                    self.block_states[self.target_block_id] = 'HELD'
                    print(f"[INFO] DOCKING: Constraint created for block {self.target_block_id}. Block attached."); print("[INFO] DOCKING: Transitioning to LOCATE_DEST."); self.current_state = STATE_LOCATE_DEST; self.angle_pid.reset()
                except p.error as e: print(f"[ERROR] Failed to create docking constraint for block {self.target_block_id}: {e}"); self.block_states[self.target_block_id] = 'AVAILABLE'; self.current_state = STATE_FIND_BLOCK; self.target_block_id = None
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_LOCATE_DEST:
            # (No change)
            base_dest_x = p.readUserDebugParameter(self.base_dest_x_slider); base_dest_y = p.readUserDebugParameter(self.base_dest_y_slider)
            self.base_destination = [base_dest_x, base_dest_y]
            target_x = self.base_destination[0] + self.placed_block_count * PLACEMENT_OFFSET; target_y = self.base_destination[1]
            self.current_placement_target = [target_x, target_y]; target = self.current_placement_target
            if are_valid_coordinates(target): p.resetBasePositionAndOrientation(self.dest_vis_id, [target[0], target[1], 0.05], [0,0,0,1])
            print(f"[INFO] LOCATE_DEST: Base Dest ({base_dest_x:.2f}, {base_dest_y:.2f}), Placed Count {self.placed_block_count}, Target ({target_x:.2f}, {target_y:.2f}). Transition to APPROACH_DEST.")
            self.current_state = STATE_APPROACH_DEST; self.angle_pid.reset(); left_velocity = right_velocity = 0.0


        elif self.current_state == STATE_APPROACH_DEST:
            # (No change)
            target = self.current_placement_target; dx = target[0] - base_pos[0]; dy = target[1] - base_pos[1]; distance = math.hypot(dx, dy)
            desired_angle = math.atan2(dy, dx); angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
            pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
            if distance > FINAL_APPROACH_DISTANCE:
                speed_factor = max(0.1, math.cos(angle_error)**2); current_speed = FORWARD_SPEED * speed_factor
                left_velocity = current_speed - pid_output; right_velocity = current_speed + pid_output
            else: print("[INFO] APPROACH_DEST: Reached destination proximity. Transition to ALIGN_DEST"); self.current_state = STATE_ALIGN_DEST; self.angle_pid.reset(); self.reset_stability_timer(); left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_ALIGN_DEST:
            target = self.current_placement_target
            # <<< Consistent Y-axis alignment >>>
            angle_to_pos_y = (math.pi/2 - rover_yaw + math.pi) % (2*math.pi) - math.pi
            angle_to_neg_y = (-math.pi/2 - rover_yaw + math.pi) % (2*math.pi) - math.pi
            desired_angle = math.pi/2 if abs(angle_to_pos_y) <= abs(angle_to_neg_y) else -math.pi/2
            angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
            pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP) # <<< Use clamped PID output >>>
            aligned = abs(angle_error) < ALIGNMENT_STABILITY_THRESHOLD
            if not aligned:
                 # <<< Apply rotation using clamped PID output and scale >>>
                left_velocity = -pid_output * ROTATION_SPEED_SCALE; right_velocity = pid_output * ROTATION_SPEED_SCALE;
                self.reset_stability_timer()
            else:
                left_velocity = right_velocity = 0.0
                # <<< Use updated stability check >>>
                if self.check_stability(aligned, self.robot_id):
                    print("[INFO] ALIGN_DEST: Alignment stable. Transition to FINAL_DRIVE_DEST")
                    self.final_drive_target_yaw = rover_yaw # Store final yaw
                    self.current_state = STATE_FINAL_DRIVE_DEST; self.angle_pid.reset()

        elif self.current_state == STATE_FINAL_DRIVE_DEST:
            target = self.current_placement_target; dx = target[0] - base_pos[0]; dy = target[1] - base_pos[1]; distance = math.hypot(dx, dy)
            if distance > DOCKING_DISTANCE - 0.01:
                angle_error = (self.final_drive_target_yaw - rover_yaw + math.pi) % (2*math.pi) - math.pi # <<< Correct using stored yaw
                pid_correction = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP) # <<< Use clamped PID output >>>
                left_velocity = FINAL_DRIVE_SPEED - pid_correction * 0.5; right_velocity = FINAL_DRIVE_SPEED + pid_correction * 0.5
            else: print("[INFO] FINAL_DRIVE_DEST: Reached release distance. Transition to UNDOCKING."); self.current_state = STATE_UNDOCKING; left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_UNDOCKING:
            if self.dock_constraint is not None:
                try: p.removeConstraint(self.dock_constraint); print(f"[INFO] UNDOCKING: Constraint removed for block {self.target_block_id}.")
                except p.error as e: print(f"[WARN] Error removing constraint: {e}")
                self.dock_constraint = None
                if self.target_block_id is not None:
                    self.block_states[self.target_block_id] = 'PLACED'; self.placed_block_count += 1
                    print(f"[INFO] UNDOCKING: Incremented placed_block_count to {self.placed_block_count}")
                    self.target_block_id = None
            print("[INFO] UNDOCKING: Transitioning to RETREAT."); self.current_state = STATE_RETREAT
            self.retreat_yaw = rover_yaw # <<< Store yaw for retreat
            self.retreat_start_pos = base_pos; left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_RETREAT:
            if self.retreat_start_pos is None: self.retreat_start_pos = base_pos
            distance_retreated = math.hypot(base_pos[0] - self.retreat_start_pos[0], base_pos[1] - self.retreat_start_pos[1])
            if distance_retreated < RETREAT_DISTANCE:
                # <<< Use stored retreat yaw for optional correction >>>
                left_velocity = -RETREAT_SPEED
                right_velocity = -RETREAT_SPEED
                angle_error = (self.retreat_yaw - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_correction = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP) # <<< Use clamped PID output >>>
                left_velocity -= pid_correction * 0.1 # Gentle correction
                right_velocity += pid_correction * 0.1
            else: print("[INFO] RETREAT: Retreat complete. Transitioning back to FIND_BLOCK."); self.current_state = STATE_FIND_BLOCK; self.retreat_start_pos = None; self.angle_pid.reset(); left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_ALL_DELIVERED:
            left_velocity = right_velocity = 0.0

        # --- Apply Motor Speeds ---
        if self.robot_id is not None: set_motor_speeds(self.robot_id, self.joint_dict, left_velocity, right_velocity)
        return target

# --- Main Simulation Setup ---
def run_simulation(urdf_path, block_urdf_path):
    # (Initialize simulation, load assets - unchanged from previous version)
    initialize_simulation()
    robot_id, joint_dict = load_robot(urdf_path)
    if robot_id is None: print("[FATAL] Robot loading failed."); p.disconnect(); return
    block_visualizations, block_states = load_blocks(block_urdf_path, NUM_BLOCKS)
    if block_visualizations is None: print("[FATAL] Block loading failed."); p.disconnect(); return
    dest_body_id = create_destination_visual()
    target_vis_id = create_target_visual()
    lidar_enabled = False; auto_camera = True
    builder_bot = BuilderBot(robot_id, joint_dict, block_visualizations, block_states, target_vis_id, dest_body_id)
    frame_count = 0; last_print_time = time.time(); transport_line_id = None

    try:
        while p.isConnected():
            frame_count += 1; current_time = time.time(); keys = p.getKeyboardEvents()
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED: auto_camera = not auto_camera; print("[INFO] Auto camera toggled:", auto_camera)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED and not auto_camera: p.resetDebugVisualizerCamera(5, 0, -89, [0,0,0]); print("[INFO] Camera reset.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED: lidar_enabled = not lidar_enabled; print("[INFO] LIDAR toggled:", lidar_enabled)

            try: base_pos, base_orn = p.getBasePositionAndOrientation(robot_id); camera_target = builder_bot.run_step()
            except p.error as e: print(f"[ERROR] PyBullet error during bot step (state {builder_bot.current_state}): {e}"); builder_bot.current_state = STATE_IDLE; continue

            try:
                 update_all_block_visualizations(builder_bot.block_viz_dict)
                 if auto_camera: update_camera_view(auto_camera, base_pos, camera_target)
                 if lidar_enabled: simulate_lidar(robot_id)
            except p.error: pass

            # (Transport line drawing - unchanged)
            if builder_bot.dock_constraint is not None:
                try:
                    dest_pos = builder_bot.current_placement_target
                    if are_valid_coordinates(base_pos, dest_pos):
                        if transport_line_id is not None: p.removeUserDebugItem(transport_line_id)
                        transport_line_id = p.addUserDebugLine(base_pos, [dest_pos[0], dest_pos[1], base_pos[2]], [0, 1, 1], 2, SIMULATION_TIMESTEP * 3)
                except p.error: transport_line_id = None
            elif transport_line_id is not None: p.removeUserDebugItem(transport_line_id); transport_line_id = None

            # (IMU/State/Blocks Text - unchanged)
            try:
                 roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)
                 imu_text = f"IMU: R:{math.degrees(roll):.0f} P:{math.degrees(pitch):.0f} Y:{math.degrees(rover_yaw):.0f}"
                 state_text = f"State: {builder_bot.get_state_name()}"
                 blocks_text = f"Placed: {builder_bot.placed_block_count}/{NUM_BLOCKS}"
                 debug_text_pos = [base_pos[0]-0.3, base_pos[1]+0.3, base_pos[2]+0.5]
                 if are_valid_coordinates(debug_text_pos):
                     p.addUserDebugText(imu_text, debug_text_pos, [1,1,1], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=1) # Use replaceItem
                     p.addUserDebugText(state_text, [debug_text_pos[0], debug_text_pos[1]-0.1, debug_text_pos[2]], [1,1,0], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=2)
                     p.addUserDebugText(blocks_text, [debug_text_pos[0], debug_text_pos[1]-0.2, debug_text_pos[2]], [0,1,1], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=3)
            except p.error: pass

            p.stepSimulation()

            if current_time - last_print_time >= 1.0:
                print(f"[INFO] Time: {current_time:.1f} State={builder_bot.get_state_name()} ({builder_bot.current_state}) TargetBlock: {builder_bot.target_block_id}")
                last_print_time = current_time

    except KeyboardInterrupt: print("[INFO] Simulation terminated by user.")
    except Exception as e: print(f"[ERROR] An unhandled exception occurred: {e}"); traceback.print_exc()
    finally:
        if p.isConnected(): print("[INFO] Disconnecting from PyBullet."); p.disconnect()

# --- Run Script ---
if __name__ == "__main__":
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    block_urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeStructure.urdf"
    run_simulation(urdf_file, block_urdf_file)