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
STATE_LOCATE_DEST = 6
STATE_APPROACH_DEST = 7
STATE_ALIGN_DEST = 8
STATE_FINAL_DRIVE_DEST = 9
STATE_UNDOCKING = 10
STATE_RETREAT = 11
STATE_ALL_DELIVERED = 12      # New state when no blocks are left

# --- Control Parameters ---
FORCE = 20.0
FORWARD_SPEED = 5.0
ROTATION_SPEED_SCALE = 4.0
FINAL_DRIVE_SPEED = 1.0
RETREAT_SPEED = 1.5
RETREAT_DISTANCE = 0.15

# --- PID Parameters ---
KP = 2.0
KI = 0.1
KD = 0.01

# --- Thresholds and Distances ---
FIND_BLOCK_ANGLE_THRESHOLD = math.radians(30)
APPROACH_BLOCK_STOP_DISTANCE = 0.05
ALIGNMENT_STABILITY_THRESHOLD = math.radians(2)
STABILITY_DURATION = 0.5
FINAL_APPROACH_DISTANCE = 0.12
DOCKING_DISTANCE = 0.11

# --- Simulation Parameters ---
SIMULATION_TIMESTEP = 1/240.0
NUM_BLOCKS = 3 # <<<--- Set the number of blocks to spawn

# --- PID Controller Class ---
class PIDController:
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

    def reset(self):
        self.error_integral = 0.0
        self.previous_error = 0.0

# --- Helper Functions ---
def simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140):
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    sensor_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.1]
    _, _, rover_yaw = p.getEulerFromQuaternion(base_orn)
    fov_rad = math.radians(fov_deg)
    start_angle = rover_yaw - fov_rad/2
    ray_from_list, ray_to_list = [], []
    for i in range(num_rays):
        angle = start_angle + (i/(num_rays-1))*fov_rad
        dx = ray_length * math.cos(angle)
        dy = ray_length * math.sin(angle)
        ray_from_list.append(sensor_pos)
        ray_to_list.append([sensor_pos[0]+dx, sensor_pos[1]+dy, sensor_pos[2]])
    try:
        results = p.rayTestBatch(ray_from_list, ray_to_list)
        for i, res in enumerate(results):
            hit_fraction = res[2]
            hit_pos = res[3] if hit_fraction < 1.0 else ray_to_list[i]
            color = [1, 0, 0] if hit_fraction < 1.0 else [0, 1, 0]
            p.addUserDebugLine(ray_from_list[i], hit_pos, lineColorRGB=color, lineWidth=1, lifeTime=SIMULATION_TIMESTEP * 2)
    except p.error as e:
        # print(f"[WARN] Lidar raycast failed: {e}") # Ignore error if objects are removed
        pass


def pick_strict_side_angle(dx, dy):
    if abs(dx) >= abs(dy):
        return 0.0 if dx >= 0 else math.pi
    else:
        return math.pi/2 if dy >= 0 else -math.pi/2

def set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity):
    try:
        p.setJointMotorControl2(robot_id, joint_dict['wheel_FL_joint'], controlMode=p.VELOCITY_CONTROL, targetVelocity=-left_velocity, force=FORCE)
        p.setJointMotorControl2(robot_id, joint_dict['wheel_RL_joint'], controlMode=p.VELOCITY_CONTROL, targetVelocity=-left_velocity, force=FORCE)
        p.setJointMotorControl2(robot_id, joint_dict['wheel_FR_joint'], controlMode=p.VELOCITY_CONTROL, targetVelocity=right_velocity, force=FORCE)
        p.setJointMotorControl2(robot_id, joint_dict['wheel_RR_joint'], controlMode=p.VELOCITY_CONTROL, targetVelocity=right_velocity, force=FORCE)
    except p.error as e:
        # print(f"[WARN] Failed to set motor speed: {e}")
        pass # Ignore if robot removed etc.

def update_camera_view(auto_camera, base_pos, target, FOV=60):
    if auto_camera and target is not None:
        mid_x = (base_pos[0] + target[0]) / 2.0
        mid_y = (base_pos[1] + target[1]) / 2.0
        midpoint = [mid_x, mid_y, 0]
        separation = math.hypot(base_pos[0]-target[0], base_pos[1]-target[1])
        required_distance = (separation/2)/math.tan(math.radians(FOV/2)) if separation > 0.1 else 1.0
        min_dist = 1.5
        cam_dist = max(min_dist, required_distance*1.2)
        try:
            p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=0,
                                        cameraPitch=-89, cameraTargetPosition=midpoint)
        except p.error as e:
            pass # Ignore camera errors


def update_all_block_visualizations(block_viz_dict):
    """Iterates through the dictionary to update all block visualizations."""
    if block_viz_dict is None: return
    for block_id, sphere_id in block_viz_dict.items():
        try:
            block_center, _ = p.getBasePositionAndOrientation(block_id)
            p.resetBasePositionAndOrientation(sphere_id, block_center, [0,0,0,1])
            pos_block, orn_block = p.getBasePositionAndOrientation(block_id)
            rot_mat = p.getMatrixFromQuaternion(orn_block)
            axis_x = [rot_mat[0], rot_mat[3], rot_mat[6]]
            axis_y = [rot_mat[1], rot_mat[4], rot_mat[7]]
            axis_z = [rot_mat[2], rot_mat[5], rot_mat[8]]
            scale = 0.15
            p.addUserDebugLine(pos_block, [pos_block[0]+scale*axis_x[0], pos_block[1]+scale*axis_x[1], pos_block[2]+scale*axis_x[2]], [1,0,0], lifeTime=SIMULATION_TIMESTEP * 2)
            p.addUserDebugLine(pos_block, [pos_block[0]+scale*axis_y[0], pos_block[1]+scale*axis_y[1], pos_block[2]+scale*axis_y[2]], [0,1,0], lifeTime=SIMULATION_TIMESTEP * 2)
            p.addUserDebugLine(pos_block, [pos_block[0]+scale*axis_z[0], pos_block[1]+scale*axis_z[1], pos_block[2]+scale*axis_z[2]], [0,0,1], lifeTime=SIMULATION_TIMESTEP * 2)
        except p.error as e:
            # print(f"[WARN] Error updating viz for block {block_id}: {e}")
            pass # Ignore if block doesn't exist

# --- Loading Functions with Enhanced Debugging ---
def load_robot(urdf_path):
    print("[DEBUG] Attempting to load robot from:", urdf_path)
    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0.1], useFixedBase=False)
        print(f"[DEBUG] Robot loaded successfully. ID: {robot_id}")
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("[ERROR] FAILED TO LOAD ROBOT URDF:", e)
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        time.sleep(10)
        return None, None

    joint_dict = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_dict[info[1].decode("utf-8")] = i
    print(f"[DEBUG] Robot joint mapping created: {joint_dict}")
    return robot_id, joint_dict

def load_blocks(block_urdf_path, num_to_load):
    """Loads multiple blocks at random positions/orientations."""
    blocks = {} # Dictionary: {block_id: sphere_id}
    block_states = {} # Dictionary: {block_id: state_string}
    print(f"[DEBUG] Attempting to load {num_to_load} blocks...")
    for i in range(num_to_load):
        r_val = random.uniform(1, 3) # Avoid spawning too close to origin (0,0)
        angle = random.uniform(0, 2*math.pi)
        block_x = r_val * math.cos(angle)
        block_y = r_val * math.sin(angle)
        block_z = 0.01
        random_z_rotation = random.uniform(0, 2*math.pi)
        orientation = p.getQuaternionFromEuler([0,0,random_z_rotation])

        print(f"[DEBUG]   Spawning block {i+1} at ({block_x:.2f}, {block_y:.2f}) rot {math.degrees(random_z_rotation):.1f}°")
        try:
            block_id = p.loadURDF(block_urdf_path, basePosition=[block_x, block_y, block_z],
                                       baseOrientation=orientation, useFixedBase=False)
            print(f"[DEBUG]   Block {i+1} loaded successfully. ID: {block_id}")

            # Create visualization sphere for this block
            sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1,0,0,0.5])
            sphere_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_visual,
                                          baseCollisionShapeIndex=-1, basePosition=[block_x, block_y, block_z])
            blocks[block_id] = sphere_id
            block_states[block_id] = 'AVAILABLE'

        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"[ERROR] FAILED TO LOAD BLOCK {i+1} URDF:", e)
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            time.sleep(10)
            # Continue trying to load others, but return None if any fail for simplicity?
            # Or return the partially loaded dicts? For now, let's return failure.
            return None, None, None

    print(f"[DEBUG] Successfully loaded {len(blocks)} blocks.")
    return blocks, block_states


def create_destination_visual():
    dest_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2,0.2,0.2],
                                        rgbaColor=[0,0,1,0.5]) # Changed to blue
    dest_body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=dest_shape_id,
                                     baseCollisionShapeIndex=-1, basePosition=[0,0,-1]) # Hide initially
    print("[DEBUG] Destination target (blue cube) created.")
    return dest_body_id

def create_target_visual():
    target_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                       radius=0.05,
                                       rgbaColor=[1, 0.5, 0, 0.5]) # Orange sphere
    target_vis_id = p.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=target_shape,
                                      baseCollisionShapeIndex=-1,
                                      basePosition=[0,0,-1]) # Initially hidden
    print("[DEBUG] Target visualization (orange sphere) created.")
    return target_vis_id

def initialize_simulation():
    if p.isConnected():
        print("[WARN] Already connected to PyBullet.")
    try:
        client_id = p.connect(p.GUI)
        if client_id < 0:
           print("[FATAL] Failed to connect to PyBullet GUI.")
           exit()
        print(f"[DEBUG] PyBullet connected with client ID: {client_id}")
    except p.error as e:
        print(f"[FATAL] PyBullet connection error: {e}")
        exit()

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=SIMULATION_TIMESTEP)
    p.setRealTimeSimulation(0)
    print("[DEBUG] Gravity set, timestep configured, real-time simulation OFF.")
    try:
        p.loadURDF("plane.urdf")
        print("[DEBUG] Plane URDF loaded.")
    except Exception as e:
        print("[ERROR] Failed to load plane.urdf:", e)
        time.sleep(5)
        p.disconnect()
        exit()

# --- Builder Bot Class ---
class BuilderBot:
    def __init__(self, robot_id, joint_dict, block_viz_dict, initial_block_states, target_vis_id, dest_vis_id):
        self.robot_id = robot_id
        self.joint_dict = joint_dict
        self.block_viz_dict = block_viz_dict # Dict: {block_id: sphere_id}
        self.block_states = initial_block_states # Dict: {block_id: 'AVAILABLE'/'HELD'/'PLACED'}
        self.target_block_id = None # ID of the block currently being pursued/held
        self.target_vis_id = target_vis_id
        self.dest_vis_id = dest_vis_id # ID for the blue destination cube

        self.current_state = STATE_IDLE
        self.dock_constraint = None
        self.angle_pid = PIDController(KP, KI, KD)

        self.stability_timer = None
        self.retreat_start_pos = None
        self.current_destination = [0, 0] # Store current destination target

        # Destination reading sliders
        self.dest_x_slider = p.addUserDebugParameter("Destination X", -10, 10, 0.0)
        self.dest_y_slider = p.addUserDebugParameter("Destination Y", -10, 10, -2.0) # Start destination elsewhere

        print("[DEBUG] BuilderBot initialized.")
        print("State Legend:")
        # (Add new state name)
        print("  0: IDLE, 1: FIND_BLOCK, 2: APPROACH_BLOCK, 3: ALIGN_BLOCK,")
        print("  4: FINAL_DRIVE_BLOCK, 5: DOCKING, 6: LOCATE_DEST, 7: APPROACH_DEST,")
        print("  8: ALIGN_DEST, 9: FINAL_DRIVE_DEST, 10: UNDOCKING, 11: RETREAT, 12: ALL_DELIVERED")


    def get_state_name(self):
        states = ["IDLE", "FIND_BLOCK", "APPROACH_BLOCK", "ALIGN_BLOCK",
                  "FINAL_DRIVE_BLOCK", "DOCKING", "LOCATE_DEST", "APPROACH_DEST",
                  "ALIGN_DEST", "FINAL_DRIVE_DEST", "UNDOCKING", "RETREAT", "ALL_DELIVERED"]
        return states[self.current_state] if 0 <= self.current_state < len(states) else "UNKNOWN"

    def reset_stability_timer(self):
        self.stability_timer = None

    def check_stability(self, condition_met):
        if condition_met:
            if self.stability_timer is None:
                self.stability_timer = time.time()
            elif time.time() - self.stability_timer >= STABILITY_DURATION:
                self.reset_stability_timer()
                return True
        else:
            self.reset_stability_timer()
        return False

    def run_step(self):
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)
        except p.error as e:
             print(f"[ERROR] Failed to get robot pose: {e}. Resetting state.")
             self.current_state = STATE_IDLE # Try to recover
             return None # Skip rest of step

        target = None # For camera update
        left_velocity = 0.0
        right_velocity = 0.0

        # --- State Machine Logic (using if/elif/else) ---
        if self.current_state == STATE_IDLE:
            if self.stability_timer is None: self.stability_timer = time.time()
            if time.time() - self.stability_timer >= 1.0: # Shorter idle
                self.current_state = STATE_FIND_BLOCK
                self.reset_stability_timer()
                self.angle_pid.reset()
                print("[INFO] IDLE complete: Transition to FIND_BLOCK")
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_FIND_BLOCK:
            closest_block_id = -1
            min_dist_sq = float('inf')
            available_blocks_exist = False

            # Iterate through all known blocks
            for block_id, status in self.block_states.items():
                if status == 'AVAILABLE':
                    available_blocks_exist = True
                    try:
                        pos_block, _ = p.getBasePositionAndOrientation(block_id)
                        dist_sq = (pos_block[0] - base_pos[0])**2 + (pos_block[1] - base_pos[1])**2
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            closest_block_id = block_id
                    except p.error:
                        print(f"[WARN] Block ID {block_id} invalid in FIND_BLOCK, skipping.")
                        continue # Skip this block if it doesn't exist

            if closest_block_id != -1:
                self.target_block_id = closest_block_id
                print(f"[INFO] FIND_BLOCK: Closest available block is {self.target_block_id} at distance {math.sqrt(min_dist_sq):.2f}")

                # Now, check if we need to rotate towards it
                pos_block, _ = p.getBasePositionAndOrientation(self.target_block_id)
                target = [pos_block[0], pos_block[1]] # Target for rotation check
                dx = target[0] - base_pos[0]
                dy = target[1] - base_pos[1]
                desired_angle = math.atan2(dy, dx)
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi

                if abs(angle_error) < FIND_BLOCK_ANGLE_THRESHOLD:
                    print("[INFO] FIND_BLOCK: Aligned with closest block. Transitioning to APPROACH_BLOCK")
                    self.current_state = STATE_APPROACH_BLOCK
                    self.angle_pid.reset()
                    left_velocity = right_velocity = 0.0
                else:
                    # Rotate towards the closest block
                    pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                    left_velocity = -pid_output * ROTATION_SPEED_SCALE * 0.5
                    right_velocity = pid_output * ROTATION_SPEED_SCALE * 0.5
            elif available_blocks_exist:
                 # Available blocks exist, but maybe we couldn't get pose? Rotate slowly.
                 print("[WARN] FIND_BLOCK: Available blocks exist, but couldn't find closest. Rotating slowly.")
                 left_velocity = -0.5
                 right_velocity = 0.5
            else:
                print("[INFO] FIND_BLOCK: No available blocks found. Transitioning to ALL_DELIVERED.")
                self.current_state = STATE_ALL_DELIVERED
                left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_APPROACH_BLOCK:
            if self.target_block_id is None or self.block_states.get(self.target_block_id) != 'AVAILABLE':
                 print("[WARN] APPROACH_BLOCK: Target block ID invalid or not available. Returning to FIND_BLOCK.")
                 self.current_state = STATE_FIND_BLOCK
                 self.target_block_id = None
            else:
                try:
                    pos_block, orn_block = p.getBasePositionAndOrientation(self.target_block_id)
                    rot_mat = p.getMatrixFromQuaternion(orn_block)
                    local_x = np.array([rot_mat[0], rot_mat[3]])
                    local_y = np.array([rot_mat[1], rot_mat[4]])
                    vec = np.array([base_pos[0]-pos_block[0], base_pos[1]-pos_block[1]])
                    norm = np.linalg.norm(vec)
                    vec_norm = vec/norm if norm > 1e-3 else np.array([1,0])
                    dot_x = np.dot(vec_norm, local_x)
                    dot_y = np.dot(vec_norm, local_y)

                    if abs(dot_x) >= abs(dot_y): chosen_axis = local_x if dot_x > 0 else -local_x
                    else: chosen_axis = local_y if dot_y > 0 else -local_y

                    offset_distance = FINAL_APPROACH_DISTANCE + 0.1
                    target = [pos_block[0] + offset_distance * chosen_axis[0],
                              pos_block[1] + offset_distance * chosen_axis[1]]

                    p.resetBasePositionAndOrientation(self.target_vis_id, [target[0], target[1], 0.2], [0,0,0,1])

                    dx = target[0] - base_pos[0]
                    dy = target[1] - base_pos[1]
                    distance = math.hypot(dx, dy)
                    desired_angle = math.atan2(dy, dx)
                    angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                    pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)

                    if distance > APPROACH_BLOCK_STOP_DISTANCE:
                        speed_factor = max(0.1, math.cos(angle_error)**2)
                        current_speed = FORWARD_SPEED * speed_factor
                        left_velocity = current_speed - pid_output
                        right_velocity = current_speed + pid_output
                    else:
                        print("[INFO] APPROACH_BLOCK: Reached offset point. Transition to ALIGN_BLOCK")
                        self.current_state = STATE_ALIGN_BLOCK
                        self.angle_pid.reset()
                        self.reset_stability_timer()
                        p.resetBasePositionAndOrientation(self.target_vis_id, [0,0,-1], [0,0,0,1])
                        left_velocity = right_velocity = 0.0
                except p.error as e:
                    print(f"[ERROR] PyBullet error in APPROACH_BLOCK: {e}. Resetting.")
                    self.current_state = STATE_FIND_BLOCK
                    self.target_block_id = None

        elif self.current_state == STATE_ALIGN_BLOCK:
            if self.target_block_id is None: self.current_state = STATE_FIND_BLOCK; return target # Safety check
            try:
                pos_block, orn_block = p.getBasePositionAndOrientation(self.target_block_id)
                rot_mat = p.getMatrixFromQuaternion(orn_block)
                local_x = np.array([rot_mat[0], rot_mat[3]])
                local_y = np.array([rot_mat[1], rot_mat[4]])
                vec = np.array([base_pos[0]-pos_block[0], base_pos[1]-pos_block[1]])
                norm = np.linalg.norm(vec)
                vec_norm = vec/norm if norm > 1e-3 else np.array([1,0])
                dot_x = np.dot(vec_norm, local_x)
                dot_y = np.dot(vec_norm, local_y)

                if abs(dot_x) >= abs(dot_y): chosen_axis = local_x if dot_x > 0 else -local_x
                else: chosen_axis = local_y if dot_y > 0 else -local_y

                desired_angle = math.atan2(chosen_axis[1], chosen_axis[0]) + math.pi
                desired_angle = (desired_angle + math.pi) % (2*math.pi) - math.pi
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                aligned = abs(angle_error) < ALIGNMENT_STABILITY_THRESHOLD

                if not aligned:
                    left_velocity = -pid_output * ROTATION_SPEED_SCALE
                    right_velocity = pid_output * ROTATION_SPEED_SCALE
                    self.reset_stability_timer()
                else:
                    left_velocity = right_velocity = 0.0
                    if self.check_stability(aligned):
                        print("[INFO] ALIGN_BLOCK: Alignment stable. Transition to FINAL_DRIVE_BLOCK")
                        self.current_state = STATE_FINAL_DRIVE_BLOCK
                        self.angle_pid.reset()
            except p.error as e:
                print(f"[ERROR] PyBullet error in ALIGN_BLOCK: {e}. Resetting.")
                self.current_state = STATE_FIND_BLOCK
                self.target_block_id = None


        elif self.current_state == STATE_FINAL_DRIVE_BLOCK:
            if self.target_block_id is None: self.current_state = STATE_FIND_BLOCK; return target # Safety check
            try:
                pos_block, _ = p.getBasePositionAndOrientation(self.target_block_id)
                target = [pos_block[0], pos_block[1]] # Target for camera
                dx = pos_block[0] - base_pos[0]
                dy = pos_block[1] - base_pos[1]
                distance = math.hypot(dx, dy)

                if distance > DOCKING_DISTANCE:
                    desired_angle = rover_yaw
                    angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                    pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                    left_velocity = FINAL_DRIVE_SPEED - pid_output
                    right_velocity = FINAL_DRIVE_SPEED + pid_output
                else:
                    print("[INFO] FINAL_DRIVE_BLOCK: Reached docking distance. Transition to DOCKING.")
                    self.current_state = STATE_DOCKING
                    left_velocity = right_velocity = 0.0
            except p.error as e:
                print(f"[ERROR] PyBullet error in FINAL_DRIVE_BLOCK: {e}. Resetting.")
                self.current_state = STATE_FIND_BLOCK
                self.target_block_id = None


        elif self.current_state == STATE_DOCKING:
            if self.target_block_id is None: self.current_state = STATE_FIND_BLOCK; return target # Safety check
            if self.dock_constraint is None:
                try:
                    pos_rover, orn_rover = p.getBasePositionAndOrientation(self.robot_id)
                    pos_block, orn_block = p.getBasePositionAndOrientation(self.target_block_id)
                    invPos, invOrn = p.invertTransform(pos_rover, orn_rover)
                    relPos, relOrn = p.multiplyTransforms(invPos, invOrn, pos_block, orn_block)
                    docking_lift = 0.03
                    relPos = [relPos[0], relPos[1], relPos[2] + docking_lift]
                    self.dock_constraint = p.createConstraint(self.robot_id, -1, self.target_block_id, -1,
                                                             p.JOINT_FIXED, [0,0,0],
                                                             parentFramePosition=relPos,
                                                             childFramePosition=[0,0,0],
                                                             parentFrameOrientation=relOrn)
                    self.block_states[self.target_block_id] = 'HELD' # Update state
                    print(f"[INFO] DOCKING: Constraint created for block {self.target_block_id}. Block attached.")
                    print("[INFO] DOCKING: Transitioning to LOCATE_DEST.")
                    self.current_state = STATE_LOCATE_DEST
                    self.angle_pid.reset()
                except p.error as e:
                    print(f"[ERROR] Failed to create docking constraint for block {self.target_block_id}: {e}")
                    self.block_states[self.target_block_id] = 'AVAILABLE' # Mark as available again
                    self.current_state = STATE_FIND_BLOCK # Go find another block
                    self.target_block_id = None
            left_velocity = right_velocity = 0.0


        elif self.current_state == STATE_LOCATE_DEST:
            dx_slider = p.readUserDebugParameter(self.dest_x_slider)
            dy_slider = p.readUserDebugParameter(self.dest_y_slider)
            self.current_destination = [dx_slider, dy_slider] # Store destination
            target = self.current_destination # For camera
            # Update blue destination cube visualization
            p.resetBasePositionAndOrientation(self.dest_vis_id, [target[0], target[1], 0.2], [0,0,0,1])
            print(f"[INFO] LOCATE_DEST: Destination set to ({dx_slider:.2f}, {dy_slider:.2f}). Transition to APPROACH_DEST.")
            self.current_state = STATE_APPROACH_DEST
            self.angle_pid.reset()
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_APPROACH_DEST:
            target = self.current_destination
            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]
            distance = math.hypot(dx, dy)
            desired_angle = math.atan2(dy, dx)
            angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
            pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)

            if distance > FINAL_APPROACH_DISTANCE:
                speed_factor = max(0.1, math.cos(angle_error)**2)
                current_speed = FORWARD_SPEED * speed_factor
                left_velocity = current_speed - pid_output
                right_velocity = current_speed + pid_output
            else:
                print("[INFO] APPROACH_DEST: Reached destination proximity. Transition to ALIGN_DEST")
                self.current_state = STATE_ALIGN_DEST
                self.angle_pid.reset()
                self.reset_stability_timer()
                left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_ALIGN_DEST:
            target = self.current_destination
            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]
            desired_angle = pick_strict_side_angle(dx, dy)
            angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
            pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
            aligned = abs(angle_error) < ALIGNMENT_STABILITY_THRESHOLD

            if not aligned:
                left_velocity = -pid_output * ROTATION_SPEED_SCALE
                right_velocity = pid_output * ROTATION_SPEED_SCALE
                self.reset_stability_timer()
            else:
                left_velocity = right_velocity = 0.0
                if self.check_stability(aligned):
                    print("[INFO] ALIGN_DEST: Alignment stable. Transition to FINAL_DRIVE_DEST")
                    self.current_state = STATE_FINAL_DRIVE_DEST
                    self.angle_pid.reset()

        elif self.current_state == STATE_FINAL_DRIVE_DEST:
            target = self.current_destination
            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]
            distance = math.hypot(dx, dy)

            if distance > DOCKING_DISTANCE - 0.01:
                desired_angle = rover_yaw
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                left_velocity = FINAL_DRIVE_SPEED - pid_output
                right_velocity = FINAL_DRIVE_SPEED + pid_output
            else:
                print("[INFO] FINAL_DRIVE_DEST: Reached release distance. Transition to UNDOCKING.")
                self.current_state = STATE_UNDOCKING
                left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_UNDOCKING:
            if self.dock_constraint is not None:
                try:
                    p.removeConstraint(self.dock_constraint)
                    print(f"[INFO] UNDOCKING: Constraint removed for block {self.target_block_id}.")
                except p.error as e:
                     print(f"[WARN] Error removing constraint (might already be removed): {e}")
                self.dock_constraint = None
                if self.target_block_id is not None:
                    self.block_states[self.target_block_id] = 'PLACED' # Mark as placed
                    self.target_block_id = None # No longer holding a target
            print("[INFO] UNDOCKING: Transitioning to RETREAT.")
            self.current_state = STATE_RETREAT
            self.retreat_start_pos = base_pos
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_RETREAT:
            if self.retreat_start_pos is None: self.retreat_start_pos = base_pos
            distance_retreated = math.hypot(base_pos[0] - self.retreat_start_pos[0],
                                           base_pos[1] - self.retreat_start_pos[1])
            if distance_retreated < RETREAT_DISTANCE:
                left_velocity = -RETREAT_SPEED
                right_velocity = -RETREAT_SPEED
            else:
                print("[INFO] RETREAT: Retreat complete. Transitioning back to FIND_BLOCK.")
                self.current_state = STATE_FIND_BLOCK # Go find the next block
                self.retreat_start_pos = None
                self.angle_pid.reset()
                left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_ALL_DELIVERED:
            print("[INFO] ALL_DELIVERED: All blocks placed. Task complete.")
            left_velocity = right_velocity = 0.0
            # Stay in this state

        # --- Apply Motor Speeds ---
        if self.robot_id is not None:
             set_motor_speeds(self.robot_id, self.joint_dict, left_velocity, right_velocity)

        # --- Return target for camera ---
        if target is None and self.target_block_id is not None and self.block_states.get(self.target_block_id) == 'HELD':
            # If holding block but no other target, keep camera focused near destination
            target = self.current_destination

        return target


# --- Main Simulation Setup ---
def run_simulation(urdf_path, block_urdf_path):
    initialize_simulation()
    robot_id, joint_dict = load_robot(urdf_path)
    if robot_id is None: return # Exit if loading failed

    # Load multiple blocks
    block_visualizations, block_states = load_blocks(block_urdf_path, NUM_BLOCKS)
    if block_visualizations is None: return # Exit if loading failed

    dest_body_id = create_destination_visual()
    target_vis_id = create_target_visual()

    lidar_enabled = False
    auto_camera = True

    builder_bot = BuilderBot(robot_id, joint_dict, block_visualizations, block_states, target_vis_id, dest_body_id)

    frame_count = 0
    last_print_time = time.time()
    transport_line_id = None # To store the ID of the transport debug line

    try:
        while p.isConnected():
            frame_count += 1
            current_time = time.time()
            keys = p.getKeyboardEvents()

            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED: auto_camera = not auto_camera; print("[INFO] Auto camera toggled:", auto_camera)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED and not auto_camera: p.resetDebugVisualizerCamera(5, 0, -89, [0,0,0]); print("[INFO] Camera reset.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED: lidar_enabled = not lidar_enabled; print("[INFO] LIDAR toggled:", lidar_enabled)

            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            camera_target = builder_bot.run_step() # Execute bot logic

            # --- Update Visualizations ---
            update_all_block_visualizations(builder_bot.block_viz_dict) # Update all blocks
            if auto_camera: update_camera_view(auto_camera, base_pos, camera_target)
            if lidar_enabled: simulate_lidar(robot_id)

            # --- Draw Transport Line ---
            if builder_bot.dock_constraint is not None:
                dest_pos, _ = p.getBasePositionAndOrientation(builder_bot.dest_vis_id)
                if transport_line_id is not None: # Remove old line
                    p.removeUserDebugItem(transport_line_id)
                # Draw new line from robot to destination
                transport_line_id = p.addUserDebugLine(base_pos, [dest_pos[0], dest_pos[1], base_pos[2]], # Keep line horizontal
                                                       lineColorRGB=[0, 1, 1], # Cyan line
                                                       lineWidth=2, lifeTime=SIMULATION_TIMESTEP * 3)
            elif transport_line_id is not None: # Remove line if not docked
                 p.removeUserDebugItem(transport_line_id)
                 transport_line_id = None


            # --- IMU Display ---
            try:
                 roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)
                 imu_text = f"IMU: R:{math.degrees(roll):.0f} P:{math.degrees(pitch):.0f} Y:{math.degrees(rover_yaw):.0f}"
                 state_text = f"State: {builder_bot.get_state_name()}"
                 p.addUserDebugText(imu_text, [base_pos[0]-0.3, base_pos[1]+0.3, base_pos[2]+0.5], textColorRGB=[1,1,1], textSize=1.0, lifeTime=SIMULATION_TIMESTEP*2)
                 p.addUserDebugText(state_text, [base_pos[0]-0.3, base_pos[1]+0.2, base_pos[2]+0.5], textColorRGB=[1,1,0], textSize=1.0, lifeTime=SIMULATION_TIMESTEP*2) # Yellow state text
            except p.error as e: pass

            p.stepSimulation() # Step physics

            # Optional: Slight delay if running too fast, though stepSimulation should handle timing
            # time.sleep(max(0, SIMULATION_TIMESTEP - (time.time() - current_time)))

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