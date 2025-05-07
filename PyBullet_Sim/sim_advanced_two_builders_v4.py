import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random
import traceback

# --- State Constants ---
STATE_IDLE = 0
STATE_FIND_BLOCK = 1          # Find closest/assigned available block
STATE_APPROACH_BLOCK = 2
STATE_ALIGN_BLOCK = 3
STATE_FINAL_DRIVE_BLOCK = 4
STATE_DOCKING = 5
STATE_LOCATE_DEST = 6         # Calculate specific placement spot for this block
STATE_APPROACH_DEST = 7
STATE_WAITING_FOR_DEST = 13   # <<< New State: Waiting for destination lock
STATE_ALIGN_DEST = 8          # Renumbered from here down
STATE_FINAL_DRIVE_DEST = 9
STATE_UNDOCKING = 10
STATE_RETREAT = 11
STATE_ALL_DELIVERED = 12

# --- Control Parameters ---
FORCE = 20.0
FORWARD_SPEED = 5.0
ROTATION_SPEED_SCALE = 2.0
MAX_ROTATION_VELOCITY = 3.0
FINAL_DRIVE_SPEED = 1.0
RETREAT_SPEED = 1.5
RETREAT_DISTANCE = 0.18
PLACEMENT_OFFSET = 0.11

# --- PID Parameters ---
KP = 1.8; KI = 0.05; KD = 0.02

# --- Thresholds and Distances ---
FIND_BLOCK_ANGLE_THRESHOLD = math.radians(30)
APPROACH_BLOCK_STOP_DISTANCE = 0.05
ALIGNMENT_STABILITY_THRESHOLD = math.radians(2.0)
ANGULAR_VELOCITY_STABILITY_THRESHOLD = 0.05
STABILITY_DURATION = 0.75
FINAL_APPROACH_DISTANCE = 0.12
DOCKING_DISTANCE = 0.11
FINAL_DRIVE_TIMEOUT = 10.0
ROBOT_COLLISION_THRESHOLD = 0.35 # <<< Min distance between robot centers

# --- Simulation Parameters ---
SIMULATION_TIMESTEP = 1/240.0
NUM_BLOCKS = 10
NUM_ROBOTS = 2

# --- Environment Parameters ---
BLOCK_SPAWN_AREA = [-2.0, 2.0, 1.0, 3.0]
DEFAULT_DESTINATION = [0.0, -2.0]

# --- PID Controller Class ---
# (No changes)
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.error_integral = 0.0; self.previous_error = 0.0
        self.MAX_INTEGRAL = 5.0
    def compute(self, error, dt):
        dt = max(dt, 1e-6); self.error_integral += error * dt
        self.error_integral = max(min(self.error_integral, self.MAX_INTEGRAL), -self.MAX_INTEGRAL)
        derivative = (error - self.previous_error) / dt; self.previous_error = error
        output = self.Kp * error + self.Ki * self.error_integral + self.Kd * derivative
        output = max(min(output, MAX_ROTATION_VELOCITY), -MAX_ROTATION_VELOCITY)
        return output
    def reset(self): self.error_integral = 0.0; self.previous_error = 0.0

# --- Helper Functions ---
def is_valid_coordinate(coord): return not (math.isnan(coord) or math.isinf(coord))
def are_valid_coordinates(*coords):
    if not coords: return False
    flat_coords = [c for point in coords if point is not None for c in point]
    if not flat_coords: return False
    return all(is_valid_coordinate(c) for c in flat_coords)

def simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140):
    # (No changes)
    try:
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        if not are_valid_coordinates(base_pos): return
        sensor_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.1]; _, _, rover_yaw = p.getEulerFromQuaternion(base_orn)
        fov_rad = math.radians(fov_deg); start_angle = rover_yaw - fov_rad/2; ray_from_list, ray_to_list = [], []
        for i in range(num_rays):
            angle = start_angle + (i/(num_rays-1))*fov_rad; dx = ray_length * math.cos(angle); dy = ray_length * math.sin(angle)
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
    # (No changes)
    if not (is_valid_coordinate(dx) and is_valid_coordinate(dy)): return 0.0
    if abs(dx) >= abs(dy): return 0.0 if dx >= 0 else math.pi
    else: return math.pi/2 if dy >= 0 else -math.pi/2

def set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity):
    # (No changes)
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
    # (No changes)
    if auto_camera and target is not None and are_valid_coordinates(base_pos, target):
        mid_x = (base_pos[0] + target[0]) / 2.0; mid_y = (base_pos[1] + target[1]) / 2.0
        midpoint = [mid_x, mid_y, 0]; separation = math.hypot(base_pos[0]-target[0], base_pos[1]-target[1])
        required_distance = (separation/2)/math.tan(math.radians(FOV/2)) if separation > 0.1 else 1.0
        min_dist = 1.5; cam_dist = max(min_dist, required_distance*1.2)
        try: p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=midpoint)
        except p.error: pass

def update_all_block_visualizations(block_viz_dict):
    if block_viz_dict is None: return
    for block_id, sphere_id in block_viz_dict.items():
        try:
            pos_block, orn_block = p.getBasePositionAndOrientation(block_id)
            if not are_valid_coordinates(pos_block): continue

            # Update center sphere
            p.resetBasePositionAndOrientation(sphere_id, pos_block, [0,0,0,1])

            # Draw orientation axes
            rot_mat = p.getMatrixFromQuaternion(orn_block)
            axis_x = [rot_mat[0], rot_mat[3], rot_mat[6]]
            axis_y = [rot_mat[1], rot_mat[4], rot_mat[7]]
            axis_z = [rot_mat[2], rot_mat[5], rot_mat[8]]
            scale = 0.15

            # --- Corrected X-axis drawing ---
            end_x = [pos_block[0]+scale*axis_x[0], pos_block[1]+scale*axis_x[1], pos_block[2]+scale*axis_x[2]]
            if are_valid_coordinates(pos_block, end_x):
                # Indent the try block
                try:
                    p.addUserDebugLine(pos_block, end_x, [1,0,0], lifeTime=SIMULATION_TIMESTEP * 2)
                except p.error: pass # Ignore draw errors

            # --- Corrected Y-axis drawing ---
            end_y = [pos_block[0]+scale*axis_y[0], pos_block[1]+scale*axis_y[1], pos_block[2]+scale*axis_y[2]]
            if are_valid_coordinates(pos_block, end_y):
                 # Indent the try block
                 try:
                     p.addUserDebugLine(pos_block, end_y, [0,1,0], lifeTime=SIMULATION_TIMESTEP * 2)
                 except p.error: pass

            # --- Corrected Z-axis drawing ---
            end_z = [pos_block[0]+scale*axis_z[0], pos_block[1]+scale*axis_z[1], pos_block[2]+scale*axis_z[2]]
            if are_valid_coordinates(pos_block, end_z):
                 # Indent the try block
                 try:
                     p.addUserDebugLine(pos_block, end_z, [0,0,1], lifeTime=SIMULATION_TIMESTEP * 2)
                 except p.error: pass

        except p.error: pass # Ignore if block ID is invalid getting pose

# --- Loading Functions ---
def load_robots(urdf_path, num_to_load):
    # (No changes)
    robot_ids = []; joint_dicts = []
    start_separation = 0.5
    print(f"[DEBUG] Attempting to load {num_to_load} robots...")
    for i in range(num_to_load):
        start_x = (i - (num_to_load - 1) / 2.0) * start_separation
        start_pos = [start_x, 0, 0.1]
        print(f"[DEBUG]   Loading robot {i} at {start_pos}")
        try:
            robot_id = p.loadURDF(urdf_path, basePosition=start_pos, useFixedBase=False)
            print(f"[DEBUG]   Robot {i} loaded successfully. ID: {robot_id}")
            joint_dict = {}; num_joints = p.getNumJoints(robot_id)
            for j in range(num_joints): info = p.getJointInfo(robot_id, j); joint_dict[info[1].decode("utf-8")] = j
            robot_ids.append(robot_id); joint_dicts.append(joint_dict)
            print(f"[DEBUG]   Robot {i} joint mapping created: {joint_dict}")
        except Exception as e: print(f"!!!!!!!!!!!!!!!!!!\n[ERROR] FAILED TO LOAD ROBOT {i} URDF:", e); traceback.print_exc(); print("!!!!!!!!!!!!!!!!!!"); time.sleep(10); return None, None
    return robot_ids, joint_dicts

def load_blocks(block_urdf_path, num_to_load, spawn_area):
    # (No changes)
    blocks = {}; block_states = {}
    min_x, max_x, min_y, max_y = spawn_area
    print(f"[DEBUG] Attempting to load {num_to_load} blocks in area X:({min_x:.1f},{max_x:.1f}), Y:({min_y:.1f},{max_y:.1f})...")
    for i in range(num_to_load):
        block_x = random.uniform(min_x, max_x); block_y = random.uniform(min_y, max_y); block_z = 0.01
        random_z_rotation = random.uniform(0, 2*math.pi); orientation = p.getQuaternionFromEuler([0,0,random_z_rotation])
        print(f"[DEBUG]   Spawning block {i+1} at ({block_x:.2f}, {block_y:.2f}) rot {math.degrees(random_z_rotation):.1f}°")
        try:
            block_id = p.loadURDF(block_urdf_path, basePosition=[block_x, block_y, block_z], baseOrientation=orientation, useFixedBase=False)
            print(f"[DEBUG]   Block {i+1} loaded successfully. ID: {block_id}")
            sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1,0,0,0.5])
            sphere_id = p.createMultiBody(0, sphere_visual, -1, [block_x, block_y, block_z])
            blocks[block_id] = sphere_id
            block_states[block_id] = 'AVAILABLE'
        except Exception as e: print(f"!!!!!!!!!!!!!!!!!!\n[ERROR] FAILED TO LOAD BLOCK {i+1} URDF:", e); traceback.print_exc(); print("!!!!!!!!!!!!!!!!!!"); time.sleep(10); return None, None
    print(f"[DEBUG] Successfully loaded {len(blocks)} blocks.")
    return blocks, block_states

def create_destination_visual():
    # (No changes)
    dest_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.055,0.055,0.055], rgbaColor=[0,0,1,0.5])
    dest_body_id = p.createMultiBody(0, dest_shape_id, -1, [0,0,-1])
    print("[DEBUG] Destination placement marker (blue cube) created.")
    return dest_body_id

def create_target_visual():
    # (No changes)
    target_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0.5, 0, 0.5])
    target_vis_id = p.createMultiBody(0, target_shape, -1, [0,0,-1])
    print("[DEBUG] Approach target visualization (orange sphere) created.")
    return target_vis_id

def initialize_simulation():
    # (No changes)
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
    def __init__(self, robot_instance_id, robot_id, joint_dict, block_viz_dict, shared_block_states, shared_data, target_vis_id, dest_vis_id):
        # (Init unchanged)
        self.robot_instance_id = robot_instance_id
        self.robot_id = robot_id; self.joint_dict = joint_dict
        self.block_viz_dict = block_viz_dict; self.block_states = shared_block_states; self.shared_data = shared_data
        self.target_block_id = None; self.target_vis_id = target_vis_id; self.dest_vis_id = dest_vis_id
        self.current_state = STATE_IDLE; self.dock_constraint = None
        self.angle_pid = PIDController(KP, KI, KD)
        self.stability_timer = None; self.retreat_start_pos = None
        self.base_destination = list(DEFAULT_DESTINATION)
        self.current_placement_target = [0.0, 0.0]
        self.final_drive_start_time = None; self.final_drive_target_yaw = 0.0; self.retreat_yaw = 0.0
        self.base_dest_x_slider = p.addUserDebugParameter(f"Base Dest X", -5, 5, self.base_destination[0])
        self.base_dest_y_slider = p.addUserDebugParameter(f"Base Dest Y", -5, 5, self.base_destination[1])
        print(f"[DEBUG] BuilderBot {self.robot_instance_id} (ID: {self.robot_id}) initialized.")

    def get_state_name(self):
        states = ["IDLE","FIND_BLOCK","APPROACH_BLOCK","ALIGN_BLOCK","FINAL_DRIVE_BLOCK","DOCKING",
                  "LOCATE_DEST","APPROACH_DEST", "ALIGN_DEST","FINAL_DRIVE_DEST","UNDOCKING",
                  "RETREAT","ALL_DELIVERED", "WAITING_FOR_DEST"] # <<< Added WAITING state name
        # Adjust state indices if needed based on where WAITING was inserted
        state_map = {
            STATE_IDLE: "IDLE", STATE_FIND_BLOCK: "FIND_BLOCK", STATE_APPROACH_BLOCK: "APPROACH_BLOCK",
            STATE_ALIGN_BLOCK: "ALIGN_BLOCK", STATE_FINAL_DRIVE_BLOCK: "FINAL_DRIVE_BLOCK",
            STATE_DOCKING: "DOCKING", STATE_LOCATE_DEST: "LOCATE_DEST", STATE_APPROACH_DEST: "APPROACH_DEST",
            STATE_WAITING_FOR_DEST: "WAITING_FOR_DEST", STATE_ALIGN_DEST: "ALIGN_DEST",
            STATE_FINAL_DRIVE_DEST: "FINAL_DRIVE_DEST", STATE_UNDOCKING: "UNDOCKING",
            STATE_RETREAT: "RETREAT", STATE_ALL_DELIVERED: "ALL_DELIVERED"
        }
        return state_map.get(self.current_state, "UNKNOWN")


    def reset_stability_timer(self): self.stability_timer = None
    def check_stability(self, angle_aligned, robot_id):
        # (Unchanged)
        try: lin_vel, ang_vel = p.getBaseVelocity(robot_id); angular_velocity_low = abs(ang_vel[2]) < ANGULAR_VELOCITY_STABILITY_THRESHOLD
        except p.error: angular_velocity_low = False
        if angle_aligned and angular_velocity_low:
            if self.stability_timer is None: self.stability_timer = time.time()
            elif time.time() - self.stability_timer >= STABILITY_DURATION: self.reset_stability_timer(); return True
        else: self.reset_stability_timer()
        return False

    # <<< New Collision Check Method >>>
    def check_robot_collision(self, base_pos, all_robots):
        for other_bot in all_robots:
            if other_bot.robot_id == self.robot_id: continue # Don't check against self
            try:
                other_pos, _ = p.getBasePositionAndOrientation(other_bot.robot_id)
                if not are_valid_coordinates(other_pos): continue # Skip if other robot pose invalid
                dist_sq = (base_pos[0] - other_pos[0])**2 + (base_pos[1] - other_pos[1])**2
                # Simple threshold check (could be more complex later)
                if dist_sq < ROBOT_COLLISION_THRESHOLD**2:
                    print(f"[WARN] Robot {self.robot_instance_id}: Potential collision detected with Robot {other_bot.robot_instance_id}! Stopping.")
                    return True # Collision risk
            except p.error:
                continue # Ignore error if other robot ID is invalid
        return False # No collision risk detected

    def run_step(self, all_robots): # <<< Accept list of all robot objects
        try: base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id); roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)
        except p.error as e: print(f"[ERROR] Robot {self.robot_instance_id}: Failed to get pose: {e}. Resetting state."); self.current_state = STATE_IDLE; return None

        target = None; left_velocity = 0.0; right_velocity = 0.0
        collision_detected = False # Flag for collision override

        # --- State Machine Logic ---
        if self.current_state == STATE_IDLE:
            # (No change)
            if self.stability_timer is None: self.stability_timer = time.time()
            if time.time() - self.stability_timer >= 1.0 + random.uniform(0, 0.5*self.robot_instance_id): self.current_state = STATE_FIND_BLOCK; self.reset_stability_timer(); self.angle_pid.reset(); print(f"[INFO] Robot {self.robot_instance_id}: IDLE complete -> FIND_BLOCK")
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_FIND_BLOCK:
            # (Logic unchanged - uses shared data correctly)
            my_target_id = -1; rotate_only = False; perform_calculation = False
            if self.robot_instance_id == 0 and not self.shared_data.get('block_targets_assigned', False):
                perform_calculation = True; available_blocks = []
                r0_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                for block_id, status in self.block_states.items():
                    if status == 'AVAILABLE':
                        try: pos_block, _ = p.getBasePositionAndOrientation(block_id); dist_sq = (pos_block[0] - r0_pos[0])**2 + (pos_block[1] - r0_pos[1])**2; available_blocks.append((dist_sq, block_id))
                        except p.error: continue
                available_blocks.sort(); self.shared_data['target_for_robot_0'] = available_blocks[0][1] if len(available_blocks) > 0 else -1; self.shared_data['target_for_robot_1'] = available_blocks[1][1] if len(available_blocks) > 1 else -1; self.shared_data['block_targets_assigned'] = True; #print(f"[DEBUG] Robot 0: Calc Targets: R0={self.shared_data['target_for_robot_0']}, R1={self.shared_data['target_for_robot_1']}")
            if self.shared_data.get('block_targets_assigned', False):
                my_target_id = self.shared_data.get(f'target_for_robot_{self.robot_instance_id}', -1)
                if my_target_id != -1 and self.block_states.get(my_target_id) == 'AVAILABLE':
                    self.target_block_id = my_target_id
                    try:
                        pos_block, _ = p.getBasePositionAndOrientation(self.target_block_id); target = [pos_block[0], pos_block[1]]
                        dx = target[0] - base_pos[0]; dy = target[1] - base_pos[1]; desired_angle = math.atan2(dy, dx)
                        angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                        if abs(angle_error) < FIND_BLOCK_ANGLE_THRESHOLD: print(f"[INFO] Robot {self.robot_instance_id}: FIND_BLOCK: Aligned with assigned block {self.target_block_id}. -> APPROACH_BLOCK"); self.current_state = STATE_APPROACH_BLOCK; self.angle_pid.reset();
                        else: pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP); left_velocity = -pid_output * ROTATION_SPEED_SCALE * 0.5; right_velocity = pid_output * ROTATION_SPEED_SCALE * 0.5; rotate_only = True
                    except p.error as e: print(f"[WARN] Robot {self.robot_instance_id}: FIND_BLOCK: Error getting pose for target {self.target_block_id}. Error: {e}. Retrying find."); self.target_block_id = None; rotate_only = True; left_velocity = -0.3; right_velocity = 0.3
                else:
                    self.target_block_id = None
                    if not any(status == 'AVAILABLE' for status in self.block_states.values()): print(f"[INFO] Robot {self.robot_instance_id}: FIND_BLOCK: No available blocks remain. -> ALL_DELIVERED."); self.current_state = STATE_ALL_DELIVERED
                    else:
                        if self.shared_data.get('block_targets_assigned', False): left_velocity = -0.3; right_velocity = 0.3; rotate_only = True
            else:
                 if self.robot_instance_id != 0: pass
            if not rotate_only: left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_APPROACH_BLOCK:
            # (Logic unchanged)
            target_status = self.block_states.get(self.target_block_id, 'NOT_FOUND')
            if self.target_block_id is None or target_status != 'AVAILABLE': print(f"[WARN] Robot {self.robot_instance_id}: APPROACH_BLOCK: Target block {self.target_block_id} no longer available (Status: '{target_status}'). -> FIND_BLOCK."); self.current_state = STATE_FIND_BLOCK; self.target_block_id = None; return target
            try:
                pos_block, orn_block = p.getBasePositionAndOrientation(self.target_block_id); # ... rest of approach logic ...
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
                else: print(f"[INFO] Robot {self.robot_instance_id}: APPROACH_BLOCK: Reached offset point. -> ALIGN_BLOCK"); self.current_state = STATE_ALIGN_BLOCK; self.angle_pid.reset(); self.reset_stability_timer(); p.resetBasePositionAndOrientation(self.target_vis_id, [0,0,-1], [0,0,0,1]); left_velocity = right_velocity = 0.0
            except p.error as e: print(f"[ERROR] Robot {self.robot_instance_id}: PyBullet error in APPROACH_BLOCK: {e}. Resetting."); self.current_state = STATE_FIND_BLOCK; self.target_block_id = None


        elif self.current_state == STATE_ALIGN_BLOCK:
             # (Logic unchanged)
             target_status = self.block_states.get(self.target_block_id, 'NOT_FOUND')
             if self.target_block_id is None or target_status != 'AVAILABLE': self.current_state = STATE_FIND_BLOCK; return target
             try:
                pos_block, orn_block = p.getBasePositionAndOrientation(self.target_block_id); # ... rest of align logic ...
                if not are_valid_coordinates(pos_block): raise p.error("Invalid block coordinates")
                rot_mat = p.getMatrixFromQuaternion(orn_block); local_x = np.array([rot_mat[0], rot_mat[3]]); local_y = np.array([rot_mat[1], rot_mat[4]])
                vec = np.array([base_pos[0]-pos_block[0], base_pos[1]-pos_block[1]]); norm = np.linalg.norm(vec); vec_norm = vec/norm if norm > 1e-3 else np.array([1,0])
                dot_x = np.dot(vec_norm, local_x); dot_y = np.dot(vec_norm, local_y)
                chosen_axis = local_x if abs(dot_x) >= abs(dot_y) else local_y; chosen_axis = chosen_axis if (dot_x >= 0 if abs(dot_x) >= abs(dot_y) else dot_y >= 0) else -chosen_axis
                desired_angle = math.atan2(chosen_axis[1], chosen_axis[0]) + math.pi; desired_angle = (desired_angle + math.pi) % (2*math.pi) - math.pi
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP); aligned = abs(angle_error) < ALIGNMENT_STABILITY_THRESHOLD
                if not aligned: left_velocity = -pid_output * ROTATION_SPEED_SCALE; right_velocity = pid_output * ROTATION_SPEED_SCALE; self.reset_stability_timer()
                else:
                    left_velocity = right_velocity = 0.0
                    if self.check_stability(aligned, self.robot_id):
                        print(f"[INFO] Robot {self.robot_instance_id}: ALIGN_BLOCK: Alignment stable. -> FINAL_DRIVE_BLOCK")
                        self.final_drive_target_yaw = rover_yaw
                        self.current_state = STATE_FINAL_DRIVE_BLOCK; self.angle_pid.reset()
                        self.final_drive_start_time = None
             except p.error as e: print(f"[ERROR] Robot {self.robot_instance_id}: PyBullet error in ALIGN_BLOCK: {e}. Resetting."); self.current_state = STATE_FIND_BLOCK; self.target_block_id = None


        elif self.current_state == STATE_FINAL_DRIVE_BLOCK:
            # (Logic unchanged)
            if self.final_drive_start_time is None: self.final_drive_start_time = time.time(); print(f"[INFO] Robot {self.robot_instance_id}: FINAL_DRIVE_BLOCK: Entered state for block {self.target_block_id}. Starting timer.")
            if time.time() - self.final_drive_start_time > FINAL_DRIVE_TIMEOUT:
                print(f"[WARN] Robot {self.robot_instance_id}: FINAL_DRIVE_BLOCK: Timeout ({FINAL_DRIVE_TIMEOUT}s) reached for block {self.target_block_id}. Aborting.")
                if self.target_block_id is not None and self.target_block_id in self.block_states: self.block_states[self.target_block_id] = 'AVAILABLE'
                self.target_block_id = None; self.final_drive_start_time = None; self.current_state = STATE_FIND_BLOCK; left_velocity = right_velocity = 0.0
            else:
                target_status = self.block_states.get(self.target_block_id, 'NOT_FOUND')
                if self.target_block_id is None or target_status != 'AVAILABLE': self.current_state = STATE_FIND_BLOCK; self.final_drive_start_time = None; return target
                try:
                    pos_block, _ = p.getBasePositionAndOrientation(self.target_block_id); # ... rest of drive logic ...
                    if not are_valid_coordinates(pos_block): raise p.error("Invalid block coordinates")
                    target = [pos_block[0], pos_block[1]]
                    dx = pos_block[0] - base_pos[0]; dy = pos_block[1] - base_pos[1]; distance = math.hypot(dx, dy)
                    if distance > DOCKING_DISTANCE:
                        angle_error = (self.final_drive_target_yaw - rover_yaw + math.pi) % (2*math.pi) - math.pi
                        pid_correction = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                        left_velocity = FINAL_DRIVE_SPEED - pid_correction * 0.5; right_velocity = FINAL_DRIVE_SPEED + pid_correction * 0.5
                    else: print(f"[INFO] Robot {self.robot_instance_id}: FINAL_DRIVE_BLOCK: Reached docking distance. -> DOCKING."); self.current_state = STATE_DOCKING; self.final_drive_start_time = None; left_velocity = right_velocity = 0.0
                except p.error as e: print(f"[ERROR] Robot {self.robot_instance_id}: PyBullet error in FINAL_DRIVE_BLOCK: {e}. Resetting."); self.current_state = STATE_FIND_BLOCK; self.target_block_id = None; self.final_drive_start_time = None

        elif self.current_state == STATE_DOCKING:
            # (Logic unchanged - resets shared flag)
            target_status = self.block_states.get(self.target_block_id, 'NOT_FOUND')
            if self.target_block_id is None or target_status != 'AVAILABLE': self.current_state = STATE_FIND_BLOCK; return target
            if self.dock_constraint is None:
                try:
                    pos_rover, orn_rover = p.getBasePositionAndOrientation(self.robot_id); pos_block, orn_block = p.getBasePositionAndOrientation(self.target_block_id)
                    if not (are_valid_coordinates(pos_rover) and are_valid_coordinates(pos_block)): raise p.error("Invalid coordinates for docking")
                    invPos, invOrn = p.invertTransform(pos_rover, orn_rover); relPos, relOrn = p.multiplyTransforms(invPos, invOrn, pos_block, orn_block)
                    docking_lift = 0.03; relPos = [relPos[0], relPos[1], relPos[2] + docking_lift]
                    self.dock_constraint = p.createConstraint(self.robot_id, -1, self.target_block_id, -1, p.JOINT_FIXED, [0,0,0], parentFramePosition=relPos, childFramePosition=[0,0,0], parentFrameOrientation=relOrn)
                    self.block_states[self.target_block_id] = 'HELD'
                    print(f"[INFO] Robot {self.robot_instance_id}: DOCKING: Constraint created for block {self.target_block_id}. Block attached.")
                    self.shared_data['block_targets_assigned'] = False # <<< Reset assignment flag
                    print(f"[DEBUG] Robot {self.robot_instance_id}: Reset block_targets_assigned flag.")
                    print(f"[INFO] Robot {self.robot_instance_id}: DOCKING: -> LOCATE_DEST."); self.current_state = STATE_LOCATE_DEST; self.angle_pid.reset()
                except p.error as e: print(f"[ERROR] Robot {self.robot_instance_id}: Failed to create docking constraint for block {self.target_block_id}: {e}"); self.block_states[self.target_block_id] = 'AVAILABLE'; self.current_state = STATE_FIND_BLOCK; self.target_block_id = None
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_LOCATE_DEST:
             # <<< Updated Logic: Reads and increments shared index >>>
            base_dest_x = p.readUserDebugParameter(self.base_dest_x_slider); base_dest_y = p.readUserDebugParameter(self.base_dest_y_slider)
            self.base_destination = [base_dest_x, base_dest_y]
            # <<< Atomically read and increment the slot index >>>
            # (Note: This is not truly atomic in Python's GIL, but sufficient for this simulation)
            placement_index = self.shared_data['next_placement_slot_index']
            self.shared_data['next_placement_slot_index'] += 1
            # <<< End of atomic update simulation >>>
            target_x = self.base_destination[0] + placement_index * PLACEMENT_OFFSET; target_y = self.base_destination[1]
            self.current_placement_target = [target_x, target_y]; target = self.current_placement_target
            if are_valid_coordinates(target): p.resetBasePositionAndOrientation(self.dest_vis_id, [target[0], target[1], 0.05], [0,0,0,1])
            print(f"[INFO] Robot {self.robot_instance_id}: LOCATE_DEST: Claimed Slot Index {placement_index}. Target ({target_x:.2f}, {target_y:.2f}). -> APPROACH_DEST.")
            self.current_state = STATE_APPROACH_DEST; self.angle_pid.reset(); left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_APPROACH_DEST:
            target = self.current_placement_target; dx = target[0] - base_pos[0]; dy = target[1] - base_pos[1]; distance = math.hypot(dx, dy)
            desired_angle = math.atan2(dy, dx); angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
            pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
            if distance > FINAL_APPROACH_DISTANCE:
                speed_factor = max(0.1, math.cos(angle_error)**2); current_speed = FORWARD_SPEED * speed_factor
                left_velocity = current_speed - pid_output; right_velocity = current_speed + pid_output
            else:
                 # <<< Check Destination Lock Before Aligning >>>
                current_locker = self.shared_data['destination_locked_by']
                if current_locker == -1: # Lock is free
                    self.shared_data['destination_locked_by'] = self.robot_instance_id # Acquire lock
                    print(f"[INFO] Robot {self.robot_instance_id}: APPROACH_DEST: Reached proximity and acquired dest lock. -> ALIGN_DEST")
                    self.current_state = STATE_ALIGN_DEST; self.angle_pid.reset(); self.reset_stability_timer(); left_velocity = right_velocity = 0.0
                elif current_locker == self.robot_instance_id: # Already have lock (shouldn't happen here, but safe)
                    print(f"[INFO] Robot {self.robot_instance_id}: APPROACH_DEST: Reached proximity (already had lock). -> ALIGN_DEST")
                    self.current_state = STATE_ALIGN_DEST; self.angle_pid.reset(); self.reset_stability_timer(); left_velocity = right_velocity = 0.0
                else: # Lock held by other robot
                    print(f"[INFO] Robot {self.robot_instance_id}: APPROACH_DEST: Reached proximity but dest locked by {current_locker}. -> WAITING_FOR_DEST")
                    self.current_state = STATE_WAITING_FOR_DEST; left_velocity = right_velocity = 0.0 # Wait

        # <<< New State: Waiting for Destination Lock >>>
        elif self.current_state == STATE_WAITING_FOR_DEST:
            print(f"[INFO] Robot {self.robot_instance_id}: WAITING_FOR_DEST: Checking lock...")
            current_locker = self.shared_data['destination_locked_by']
            if current_locker == -1: # Lock is free now
                 self.shared_data['destination_locked_by'] = self.robot_instance_id # Acquire lock
                 print(f"[INFO] Robot {self.robot_instance_id}: WAITING_FOR_DEST: Lock acquired. -> ALIGN_DEST")
                 self.current_state = STATE_ALIGN_DEST; self.angle_pid.reset(); self.reset_stability_timer()
            # else: Stay in this state and do nothing (keep checking next frame)
            left_velocity = right_velocity = 0.0


        elif self.current_state == STATE_ALIGN_DEST:
            # (Logic unchanged - uses Y-axis alignment)
            target = self.current_placement_target
            angle_to_pos_y = (math.pi/2 - rover_yaw + math.pi) % (2*math.pi) - math.pi; angle_to_neg_y = (-math.pi/2 - rover_yaw + math.pi) % (2*math.pi) - math.pi
            desired_angle = math.pi/2 if abs(angle_to_pos_y) <= abs(angle_to_neg_y) else -math.pi/2
            angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
            pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP); aligned = abs(angle_error) < ALIGNMENT_STABILITY_THRESHOLD
            if not aligned: left_velocity = -pid_output * ROTATION_SPEED_SCALE; right_velocity = pid_output * ROTATION_SPEED_SCALE; self.reset_stability_timer()
            else:
                left_velocity = right_velocity = 0.0
                if self.check_stability(aligned, self.robot_id):
                    print(f"[INFO] Robot {self.robot_instance_id}: ALIGN_DEST: Alignment stable. -> FINAL_DRIVE_DEST")
                    self.final_drive_target_yaw = rover_yaw
                    self.current_state = STATE_FINAL_DRIVE_DEST; self.angle_pid.reset()

        elif self.current_state == STATE_FINAL_DRIVE_DEST:
             # (Logic unchanged)
            target = self.current_placement_target; dx = target[0] - base_pos[0]; dy = target[1] - base_pos[1]; distance = math.hypot(dx, dy)
            if distance > DOCKING_DISTANCE - 0.01:
                angle_error = (self.final_drive_target_yaw - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_correction = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                left_velocity = FINAL_DRIVE_SPEED - pid_correction * 0.5; right_velocity = FINAL_DRIVE_SPEED + pid_correction * 0.5
            else: print(f"[INFO] Robot {self.robot_instance_id}: FINAL_DRIVE_DEST: Reached release distance. -> UNDOCKING."); self.current_state = STATE_UNDOCKING; left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_UNDOCKING:
            # (Logic unchanged - increments shared count)
            if self.dock_constraint is not None:
                try: p.removeConstraint(self.dock_constraint); print(f"[INFO] Robot {self.robot_instance_id}: UNDOCKING: Constraint removed for block {self.target_block_id}.")
                except p.error as e: print(f"[WARN] Robot {self.robot_instance_id}: Error removing constraint: {e}")
                self.dock_constraint = None
                if self.target_block_id is not None:
                    self.block_states[self.target_block_id] = 'PLACED'
                    self.shared_data['placed_count'] += 1
                    print(f"[INFO] Robot {self.robot_instance_id}: UNDOCKING: Incremented shared placed_count to {self.shared_data['placed_count']}")
                    self.target_block_id = None
            print(f"[INFO] Robot {self.robot_instance_id}: UNDOCKING: -> RETREAT."); self.current_state = STATE_RETREAT
            self.retreat_yaw = rover_yaw; self.retreat_start_pos = base_pos; left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_RETREAT:
            # <<< Release Destination Lock >>>
            if self.shared_data['destination_locked_by'] == self.robot_instance_id:
                print(f"[INFO] Robot {self.robot_instance_id}: RETREAT: Releasing destination lock.")
                self.shared_data['destination_locked_by'] = -1 # Release lock
            # <<< End Lock Release >>>

            if self.retreat_start_pos is None: self.retreat_start_pos = base_pos
            distance_retreated = math.hypot(base_pos[0] - self.retreat_start_pos[0], base_pos[1] - self.retreat_start_pos[1])
            if distance_retreated < RETREAT_DISTANCE:
                left_velocity = -RETREAT_SPEED; right_velocity = -RETREAT_SPEED
                angle_error = (self.retreat_yaw - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_correction = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                left_velocity -= pid_correction * 0.1; right_velocity += pid_correction * 0.1
            else: print(f"[INFO] Robot {self.robot_instance_id}: RETREAT: Retreat complete. -> FIND_BLOCK."); self.current_state = STATE_FIND_BLOCK; self.retreat_start_pos = None; self.angle_pid.reset(); left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_ALL_DELIVERED:
            left_velocity = right_velocity = 0.0

        # --- Collision Avoidance Override ---
        if self.current_state in [STATE_APPROACH_BLOCK, STATE_APPROACH_DEST]: # Only check during approach phases
             if self.check_robot_collision(base_pos, all_robots):
                 left_velocity = 0.0
                 right_velocity = 0.0
                 collision_detected = True # Set flag

        # --- Apply Motor Speeds ---
        if self.robot_id is not None:
             set_motor_speeds(self.robot_id, self.joint_dict, left_velocity, right_velocity)

        return target


# --- Main Simulation Setup ---
def run_simulation(urdf_path, block_urdf_path):
    initialize_simulation()
    robot_ids, joint_dicts = load_robots(urdf_path, NUM_ROBOTS)
    if robot_ids is None: print("[FATAL] Robot loading failed."); p.disconnect(); return

    block_visualizations, block_states = load_blocks(block_urdf_path, NUM_BLOCKS, BLOCK_SPAWN_AREA)
    if block_visualizations is None: print("[FATAL] Block loading failed."); p.disconnect(); return

    # <<< Initialize Shared Data including lock >>>
    shared_data = {
        'placed_count': 0,
        'next_placement_slot_index': 0, # Used by LOCATE_DEST to claim slot
        'target_for_robot_0': -1,
        'target_for_robot_1': -1,
        'block_targets_assigned': False,
        'destination_locked_by': -1  # -1: unlocked, 0: robot 0, 1: robot 1
    }

    dest_body_id = create_destination_visual()
    target_vis_id = create_target_visual()
    lidar_enabled = False; auto_camera = True

    robots = []
    for i in range(NUM_ROBOTS):
        bot = BuilderBot(
            robot_instance_id=i, robot_id=robot_ids[i], joint_dict=joint_dicts[i],
            block_viz_dict=block_visualizations, shared_block_states=block_states,
            shared_data=shared_data, target_vis_id=target_vis_id, dest_vis_id=dest_body_id
        )
        robots.append(bot)

    # <<< Populate other_robots list (removed, pass full list instead) >>>

    frame_count = 0; last_print_time = time.time(); transport_line_ids = [None] * NUM_ROBOTS

    # Main Loop
    try:
        while p.isConnected():
            frame_count += 1; current_time = time.time(); keys = p.getKeyboardEvents()
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED: auto_camera = not auto_camera; print("[INFO] Auto camera toggled:", auto_camera)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED and not auto_camera: p.resetDebugVisualizerCamera(5, 0, -89, [0,0,0]); print("[INFO] Camera reset.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED: lidar_enabled = not lidar_enabled; print("[INFO] LIDAR toggled:", lidar_enabled)

            camera_target_overall = None
            active_bot_index = 0

            # --- Run Step for Each Bot ---
            for i, bot in enumerate(robots):
                try:
                    # <<< Pass the full list of robots for collision check >>>
                    current_camera_target = bot.run_step(robots)
                    # (Camera focus logic unchanged)
                    if bot.current_state != STATE_IDLE and bot.current_state != STATE_ALL_DELIVERED:
                        active_bot_index = i
                        camera_target_overall = current_camera_target
                    elif i == 0 and camera_target_overall is None: camera_target_overall = current_camera_target
                except p.error as e: print(f"[ERROR] PyBullet error during bot {i} step (state {bot.current_state}): {e}"); bot.current_state = STATE_IDLE; continue
                except Exception as e_main: print(f"[FATAL] Unhandled error in bot {i} run_step: {e_main}"); traceback.print_exc(); break # Exit loop on major error


            # --- Update Visualizations ---
            try:
                 update_all_block_visualizations(robots[0].block_viz_dict)
                 if auto_camera:
                     valid_active_bot_index = active_bot_index if active_bot_index < len(robots) else 0
                     if valid_active_bot_index < len(robots):
                         active_bot_pos, _ = p.getBasePositionAndOrientation(robots[valid_active_bot_index].robot_id)
                         update_camera_view(auto_camera, active_bot_pos, camera_target_overall)
                 if lidar_enabled:
                     for bot in robots: simulate_lidar(bot.robot_id)
            except p.error: pass
            except IndexError: pass

            # --- Draw Transport Lines ---
            # (No change)
            for i, bot in enumerate(robots):
                if bot.dock_constraint is not None:
                    try:
                        bot_base_pos, _ = p.getBasePositionAndOrientation(bot.robot_id)
                        dest_pos = bot.current_placement_target
                        if are_valid_coordinates(bot_base_pos, dest_pos):
                            if transport_line_ids[i] is not None: p.removeUserDebugItem(transport_line_ids[i])
                            transport_line_ids[i] = p.addUserDebugLine(bot_base_pos, [dest_pos[0], dest_pos[1], bot_base_pos[2]], [0, 1, 1], 2, SIMULATION_TIMESTEP * 3)
                    except p.error: transport_line_ids[i] = None
                elif transport_line_ids[i] is not None:
                    try: p.removeUserDebugItem(transport_line_ids[i])
                    except p.error: pass
                    transport_line_ids[i] = None

            # --- IMU/State/Blocks Text ---
            # (No change)
            try:
                 bot0 = robots[0]; base_pos0, base_orn0 = p.getBasePositionAndOrientation(bot0.robot_id); roll0, pitch0, yaw0 = p.getEulerFromQuaternion(base_orn0)
                 imu0_text = f"R0 IMU: Y:{math.degrees(yaw0):.0f}"; state0_text = f"R0 St: {bot0.get_state_name()}"; target0_text= f"R0 Tgt: {bot0.target_block_id}"
                 debug_pos0 = [base_pos0[0]-0.3, base_pos0[1]+0.4, base_pos0[2]+0.5]
                 if are_valid_coordinates(debug_pos0):
                     p.addUserDebugText(imu0_text, debug_pos0, [1,1,1], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=1)
                     p.addUserDebugText(state0_text, [debug_pos0[0], debug_pos0[1]-0.1, debug_pos0[2]], [1,1,0], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=2)
                     p.addUserDebugText(target0_text, [debug_pos0[0], debug_pos0[1]-0.2, debug_pos0[2]], [1,0.7,0], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=4)
                 if len(robots) > 1:
                     bot1 = robots[1]; base_pos1, base_orn1 = p.getBasePositionAndOrientation(bot1.robot_id); roll1, pitch1, yaw1 = p.getEulerFromQuaternion(base_orn1)
                     imu1_text = f"R1 IMU: Y:{math.degrees(yaw1):.0f}"; state1_text = f"R1 St: {bot1.get_state_name()}"; target1_text= f"R1 Tgt: {bot1.target_block_id}"
                     debug_pos1 = [base_pos1[0]-0.3, base_pos1[1]+0.4, base_pos1[2]+0.5]
                     if are_valid_coordinates(debug_pos1):
                        p.addUserDebugText(imu1_text, debug_pos1, [1,1,1], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=5)
                        p.addUserDebugText(state1_text, [debug_pos1[0], debug_pos1[1]-0.1, debug_pos1[2]], [1,1,0], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=6)
                        p.addUserDebugText(target1_text, [debug_pos1[0], debug_pos1[1]-0.2, debug_pos1[2]], [1,0.7,0], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=7)
                 blocks_text = f"Placed: {shared_data['placed_count']}/{NUM_BLOCKS}" # Use shared count
                 p.addUserDebugText(blocks_text, [debug_pos0[0], debug_pos0[1]-0.3, debug_pos0[2]], [0,1,1], 1.0, SIMULATION_TIMESTEP*2, replaceItemUniqueId=3)
            except p.error: pass
            except IndexError: pass

            p.stepSimulation()

            if current_time - last_print_time >= 1.0:
                 print("-" * 20)
                 for i, bot in enumerate(robots): print(f"[INFO] Time: {current_time:.1f} Robot {i}: State={bot.get_state_name()} ({bot.current_state}) TargetBlock: {bot.target_block_id}")
                 print(f"Shared Data: {shared_data}")
                 print("-" * 20)
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