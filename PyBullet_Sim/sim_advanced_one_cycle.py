import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random
import traceback # Import traceback for detailed error printing

# --- State Constants ---
STATE_IDLE = 0
STATE_FIND_BLOCK = 1
STATE_APPROACH_BLOCK = 2  # Drive towards offset point near block
STATE_ALIGN_BLOCK = 3     # Rotate to align docking face
STATE_FINAL_DRIVE_BLOCK = 4 # Drive straight forward to dock
STATE_DOCKING = 5         # Create constraint
STATE_LOCATE_DEST = 6     # Read destination sliders
STATE_APPROACH_DEST = 7   # Drive towards destination
STATE_ALIGN_DEST = 8      # Align perpendicular to destination approach
STATE_FINAL_DRIVE_DEST = 9 # Drive straight forward to place
STATE_UNDOCKING = 10        # Remove constraint
STATE_RETREAT = 11          # Move back slightly
STATE_DELIVERED = 12        # Cycle complete

# --- Control Parameters ---
FORCE = 20.0
FORWARD_SPEED = 5.0
ROTATION_SPEED_SCALE = 4.0 # Factor to amplify PID output for rotation
FINAL_DRIVE_SPEED = 1.0 # Slow speed for final approach/placement drive
RETREAT_SPEED = 1.5
RETREAT_DISTANCE = 0.15

# --- PID Parameters ---
KP = 2.0
KI = 0.1 # Reduced Ki slightly
KD = 0.01

# --- Thresholds and Distances ---
FIND_BLOCK_ANGLE_THRESHOLD = math.radians(30) # Angle tolerance for finding block
APPROACH_BLOCK_STOP_DISTANCE = 0.05 # How close to get to the offset point
ALIGNMENT_STABILITY_THRESHOLD = math.radians(2) # Angle tolerance for alignment
STABILITY_DURATION = 0.5 # Seconds the alignment must hold
FINAL_APPROACH_DISTANCE = 0.12 # Distance threshold for final alignment/drive states
DOCKING_DISTANCE = 0.11 # Distance threshold to trigger docking/undocking after final drive

# --- Simulation Parameters ---
SIMULATION_TIMESTEP = 1/240.0 # Use a fixed timestep

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
    results = p.rayTestBatch(ray_from_list, ray_to_list)
    for i, res in enumerate(results):
        hit_fraction = res[2]
        hit_pos = res[3] if hit_fraction < 1.0 else ray_to_list[i]
        color = [1, 0, 0] if hit_fraction < 1.0 else [0, 1, 0]
        p.addUserDebugLine(ray_from_list[i], hit_pos, lineColorRGB=color, lineWidth=1, lifeTime=SIMULATION_TIMESTEP * 2)

def pick_strict_side_angle(dx, dy):
    if abs(dx) >= abs(dy):
        return 0.0 if dx >= 0 else math.pi
    else:
        return math.pi/2 if dy >= 0 else -math.pi/2

def set_motor_speeds(robot_id, joint_dict, left_velocity, right_velocity):
    p.setJointMotorControl2(robot_id, joint_dict['wheel_FL_joint'],
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=-left_velocity, force=FORCE)
    p.setJointMotorControl2(robot_id, joint_dict['wheel_RL_joint'],
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=-left_velocity, force=FORCE)
    p.setJointMotorControl2(robot_id, joint_dict['wheel_FR_joint'],
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=right_velocity, force=FORCE)
    p.setJointMotorControl2(robot_id, joint_dict['wheel_RR_joint'],
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=right_velocity, force=FORCE)

def update_camera_view(auto_camera, base_pos, target, FOV=60):
    if auto_camera and target is not None:
        mid_x = (base_pos[0] + target[0]) / 2.0
        mid_y = (base_pos[1] + target[1]) / 2.0
        midpoint = [mid_x, mid_y, 0]
        separation = math.hypot(base_pos[0]-target[0], base_pos[1]-target[1])
        required_distance = (separation/2)/math.tan(math.radians(FOV/2)) if separation > 0.1 else 1.0
        min_dist = 1.5
        cam_dist = max(min_dist, required_distance*1.2)
        p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=0,
                                     cameraPitch=-89, cameraTargetPosition=midpoint)

def update_block_visualization(block_body_id, block_center_sphere):
    if block_body_id is None or block_center_sphere is None: return # Safety check
    try:
        block_center, _ = p.getBasePositionAndOrientation(block_body_id)
        p.resetBasePositionAndOrientation(block_center_sphere, block_center, [0,0,0,1])
        pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
        rot_mat = p.getMatrixFromQuaternion(orn_block)
        axis_x = [rot_mat[0], rot_mat[3], rot_mat[6]]
        axis_y = [rot_mat[1], rot_mat[4], rot_mat[7]]
        axis_z = [rot_mat[2], rot_mat[5], rot_mat[8]]
        scale = 0.15
        p.addUserDebugLine(pos_block, [pos_block[0]+scale*axis_x[0], pos_block[1]+scale*axis_x[1], pos_block[2]+scale*axis_x[2]], [1,0,0], lifeTime=SIMULATION_TIMESTEP * 2)
        p.addUserDebugLine(pos_block, [pos_block[0]+scale*axis_y[0], pos_block[1]+scale*axis_y[1], pos_block[2]+scale*axis_y[2]], [0,1,0], lifeTime=SIMULATION_TIMESTEP * 2)
        p.addUserDebugLine(pos_block, [pos_block[0]+scale*axis_z[0], pos_block[1]+scale*axis_z[1], pos_block[2]+scale*axis_z[2]], [0,0,1], lifeTime=SIMULATION_TIMESTEP * 2)
    except p.error as e:
        # Handle cases where the body might have been removed unexpectedly
        # print(f"[WARN] PyBullet error updating block visualization: {e}")
        pass


# --- Loading Functions with Enhanced Debugging ---
def load_robot(urdf_path):
    print("[DEBUG] Attempting to load robot from:", urdf_path) # ADDED
    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0.1], useFixedBase=False)
        print(f"[DEBUG] Robot loaded successfully. ID: {robot_id}") # MODIFIED
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") # ADDED
        print("[ERROR] FAILED TO LOAD ROBOT URDF:", e)       # MODIFIED
        traceback.print_exc()                                 # ADDED
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") # ADDED
        time.sleep(10) # Keep window potentially open      # ADDED
        # p.disconnect() # Temporarily disable disconnect on error for debugging
        return None, None

    joint_dict = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_dict[info[1].decode("utf-8")] = i
    print(f"[DEBUG] Robot joint mapping created: {joint_dict}") # ADDED
    return robot_id, joint_dict

def load_block(block_urdf_path):
    r_val = random.uniform(1,3)
    angle = random.uniform(0,2*math.pi)
    block_x = r_val * math.cos(angle)
    block_y = r_val * math.sin(angle)
    block_z = 0.01
    random_z_rotation = random.uniform(0,2*math.pi)
    orientation = p.getQuaternionFromEuler([0,0,random_z_rotation])

    print("[DEBUG] Attempting to load block from:", block_urdf_path) # ADDED
    print(f"[DEBUG]   at position: ({block_x:.2f}, {block_y:.2f}, {block_z}) orientation: {orientation}") # ADDED
    try:
        block_body_id = p.loadURDF(block_urdf_path, basePosition=[block_x, block_y, block_z],
                                   baseOrientation=orientation, useFixedBase=False)
        print(f"[DEBUG] Block loaded successfully. ID: {block_body_id} at ({block_x:.2f}, {block_y:.2f}, {block_z}) with rotation {math.degrees(random_z_rotation):.1f}°") # MODIFIED
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") # ADDED
        print("[ERROR] FAILED TO LOAD BLOCK URDF:", e)      # MODIFIED
        traceback.print_exc()                                # ADDED
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") # ADDED
        time.sleep(10) # Keep window potentially open     # ADDED
        # p.disconnect() # Temporarily disable disconnect on error for debugging
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
    target_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                       radius=0.05,
                                       rgbaColor=[1, 0.5, 0, 0.5]) # Orange sphere
    target_vis_id = p.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=target_shape,
                                      baseCollisionShapeIndex=-1,
                                      basePosition=[0,0,-1]) # Initially hidden
    print("[DEBUG] Target visualization created.")
    return target_vis_id

def initialize_simulation():
    # Check if already connected
    if p.isConnected():
        print("[WARN] Already connected to PyBullet.")
        # Optionally disconnect and reconnect, or just return
        # p.disconnect()
    try:
        client_id = p.connect(p.GUI)
        if client_id < 0:
           print("[FATAL] Failed to connect to PyBullet GUI.")
           exit() # Exit if connection failed
        print(f"[DEBUG] PyBullet connected with client ID: {client_id}")
    except p.error as e:
        print(f"[FATAL] PyBullet connection error: {e}")
        exit()

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    # Use setPhysicsEngineParameter to set timestep
    p.setPhysicsEngineParameter(fixedTimeStep=SIMULATION_TIMESTEP)
    # Disable real-time simulation, step manually
    p.setRealTimeSimulation(1)
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
    def __init__(self, robot_id, joint_dict, block_id, target_vis_id):
        self.robot_id = robot_id
        self.joint_dict = joint_dict
        self.block_id = block_id
        self.target_vis_id = target_vis_id

        self.current_state = STATE_IDLE
        self.dock_constraint = None
        self.angle_pid = PIDController(KP, KI, KD)

        self.stability_timer = None
        self.retreat_start_pos = None

        # Destination reading (replace sliders later)
        self.dest_x_slider = p.addUserDebugParameter("Destination X", -10, 10, 0.0)
        self.dest_y_slider = p.addUserDebugParameter("Destination Y", -10, 10, 0.0)
        self.block_yaw_slider = p.addUserDebugParameter("Block Yaw", -180, 180, 0.0)

        print("[DEBUG] BuilderBot initialized.")
        print("State Legend:")
        print("  0: IDLE, 1: FIND_BLOCK, 2: APPROACH_BLOCK, 3: ALIGN_BLOCK,")
        print("  4: FINAL_DRIVE_BLOCK, 5: DOCKING, 6: LOCATE_DEST, 7: APPROACH_DEST,")
        print("  8: ALIGN_DEST, 9: FINAL_DRIVE_DEST, 10: UNDOCKING, 11: RETREAT, 12: DELIVERED")


    def get_state_name(self):
        states = ["IDLE", "FIND_BLOCK", "APPROACH_BLOCK", "ALIGN_BLOCK",
                  "FINAL_DRIVE_BLOCK", "DOCKING", "LOCATE_DEST", "APPROACH_DEST",
                  "ALIGN_DEST", "FINAL_DRIVE_DEST", "UNDOCKING", "RETREAT", "DELIVERED"]
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
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)

        target = None # For camera update
        left_velocity = 0.0
        right_velocity = 0.0

        # --- Update Block Orientation from Slider (for testing) ---
        block_yaw_deg = p.readUserDebugParameter(self.block_yaw_slider)
        block_yaw_rad = math.radians(block_yaw_deg)
        try:
            block_pos_curr, _ = p.getBasePositionAndOrientation(self.block_id)
            # Only apply if not docked
            if self.dock_constraint is None:
                 p.resetBasePositionAndOrientation(self.block_id, block_pos_curr, p.getQuaternionFromEuler([0, 0, block_yaw_rad]))
        except p.error as e:
            # Handle case where block might have been removed
            # print(f"[WARN] Pybullet error getting/setting block pose: {e}")
            pass

        # --- State Machine Logic (using if/elif/else) ---
        if self.current_state == STATE_IDLE:
            if self.stability_timer is None:
                self.stability_timer = time.time()
                # print("[DEBUG] IDLE: Starting idle period.") # Reduce print frequency
            if time.time() - self.stability_timer >= 2.0:
                self.current_state = STATE_FIND_BLOCK
                self.reset_stability_timer()
                self.angle_pid.reset()
                print("[INFO] IDLE complete: Transition to FIND_BLOCK")
            # else:
                # print("[DEBUG] IDLE: Waiting...")
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_FIND_BLOCK:
            try:
                pos_block, _ = p.getBasePositionAndOrientation(self.block_id)
                target = [pos_block[0], pos_block[1]]
                dx = target[0] - base_pos[0]
                dy = target[1] - base_pos[1]
                desired_angle = math.atan2(dy, dx)
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi

                if abs(angle_error) < FIND_BLOCK_ANGLE_THRESHOLD:
                    print("[INFO] FIND_BLOCK: Block detected; transitioning to APPROACH_BLOCK")
                    self.current_state = STATE_APPROACH_BLOCK
                    self.angle_pid.reset()
                    left_velocity = right_velocity = 0.0
                else:
                    pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                    left_velocity = -pid_output * ROTATION_SPEED_SCALE * 0.5
                    right_velocity = pid_output * ROTATION_SPEED_SCALE * 0.5
                    # print(f"[DEBUG] FIND_BLOCK: Rotating (angle error={math.degrees(angle_error):.1f}°)") # Reduce print
            except p.error as e:
                print(f"[ERROR] PyBullet error in FIND_BLOCK (likely block removed): {e}")
                self.current_state = STATE_IDLE # Go back to idle if block disappears


        elif self.current_state == STATE_APPROACH_BLOCK:
            try:
                pos_block, orn_block = p.getBasePositionAndOrientation(self.block_id)
                rot_mat = p.getMatrixFromQuaternion(orn_block)
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
                    # print(f"[DEBUG] APPROACH_BLOCK: Driving to offset. Dist={distance:.2f}, AngleErr={math.degrees(angle_error):.1f}")
                else:
                    print("[INFO] APPROACH_BLOCK: Reached offset point. Transition to ALIGN_BLOCK")
                    self.current_state = STATE_ALIGN_BLOCK
                    self.angle_pid.reset()
                    self.reset_stability_timer()
                    p.resetBasePositionAndOrientation(self.target_vis_id, [0,0,-1], [0,0,0,1])
                    left_velocity = right_velocity = 0.0
            except p.error as e:
                print(f"[ERROR] PyBullet error in APPROACH_BLOCK: {e}")
                self.current_state = STATE_IDLE


        elif self.current_state == STATE_ALIGN_BLOCK:
            try:
                pos_block, orn_block = p.getBasePositionAndOrientation(self.block_id)
                rot_mat = p.getMatrixFromQuaternion(orn_block)
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

                desired_angle = math.atan2(chosen_axis[1], chosen_axis[0]) + math.pi
                desired_angle = (desired_angle + math.pi) % (2*math.pi) - math.pi
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                aligned = abs(angle_error) < ALIGNMENT_STABILITY_THRESHOLD

                if not aligned:
                    left_velocity = -pid_output * ROTATION_SPEED_SCALE
                    right_velocity = pid_output * ROTATION_SPEED_SCALE
                    self.reset_stability_timer()
                    # print(f"[DEBUG] ALIGN_BLOCK: Rotating. AngleErr={math.degrees(angle_error):.1f}°")
                else:
                    left_velocity = right_velocity = 0.0
                    # print(f"[DEBUG] ALIGN_BLOCK: Angle error within threshold ({math.degrees(angle_error):.1f}°). Checking stability...")
                    if self.check_stability(aligned):
                        print("[INFO] ALIGN_BLOCK: Alignment stable. Transition to FINAL_DRIVE_BLOCK")
                        self.current_state = STATE_FINAL_DRIVE_BLOCK
                        self.angle_pid.reset()
                    # else:
                        # print(f"[DEBUG] ALIGN_BLOCK: Waiting for stability ({time.time() - self.stability_timer:.2f}s / {STABILITY_DURATION:.2f}s)")
            except p.error as e:
                print(f"[ERROR] PyBullet error in ALIGN_BLOCK: {e}")
                self.current_state = STATE_IDLE

        elif self.current_state == STATE_FINAL_DRIVE_BLOCK:
            try:
                pos_block, _ = p.getBasePositionAndOrientation(self.block_id)
                dx = pos_block[0] - base_pos[0]
                dy = pos_block[1] - base_pos[1]
                distance = math.hypot(dx, dy)

                if distance > DOCKING_DISTANCE:
                    desired_angle = rover_yaw
                    angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                    pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                    left_velocity = FINAL_DRIVE_SPEED - pid_output
                    right_velocity = FINAL_DRIVE_SPEED + pid_output
                    # print(f"[DEBUG] FINAL_DRIVE_BLOCK: Driving forward. Dist={distance:.3f}")
                else:
                    print("[INFO] FINAL_DRIVE_BLOCK: Reached docking distance. Transition to DOCKING.")
                    self.current_state = STATE_DOCKING
                    left_velocity = right_velocity = 0.0
            except p.error as e:
                print(f"[ERROR] PyBullet error in FINAL_DRIVE_BLOCK: {e}")
                self.current_state = STATE_IDLE

        elif self.current_state == STATE_DOCKING:
            if self.dock_constraint is None:
                try:
                    pos_rover, orn_rover = p.getBasePositionAndOrientation(self.robot_id)
                    pos_block, orn_block = p.getBasePositionAndOrientation(self.block_id)
                    invPos, invOrn = p.invertTransform(pos_rover, orn_rover)
                    relPos, relOrn = p.multiplyTransforms(invPos, invOrn, pos_block, orn_block)
                    docking_lift = 0.03
                    relPos = [relPos[0], relPos[1], relPos[2] + docking_lift]
                    self.dock_constraint = p.createConstraint(self.robot_id, -1, self.block_id, -1,
                                                             p.JOINT_FIXED, [0,0,0],
                                                             parentFramePosition=relPos,
                                                             childFramePosition=[0,0,0],
                                                             parentFrameOrientation=relOrn)
                    print("[INFO] DOCKING: Constraint created. Block attached.")
                    # Transition immediately after creating constraint
                    print("[INFO] DOCKING: Transitioning to LOCATE_DEST.")
                    self.current_state = STATE_LOCATE_DEST
                    self.angle_pid.reset()
                except p.error as e:
                    print(f"[ERROR] Failed to create docking constraint: {e}")
                    self.current_state = STATE_IDLE # Go back if docking fails
            left_velocity = right_velocity = 0.0


        elif self.current_state == STATE_LOCATE_DEST:
            dx_slider = p.readUserDebugParameter(self.dest_x_slider)
            dy_slider = p.readUserDebugParameter(self.dest_y_slider)
            target = [dx_slider, dy_slider] # Keep target for camera
            print(f"[INFO] LOCATE_DEST: Destination set to ({dx_slider:.2f}, {dy_slider:.2f}). Transition to APPROACH_DEST.")
            self.current_state = STATE_APPROACH_DEST
            self.angle_pid.reset()
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_APPROACH_DEST:
            dx_slider = p.readUserDebugParameter(self.dest_x_slider)
            dy_slider = p.readUserDebugParameter(self.dest_y_slider)
            target = [dx_slider, dy_slider]
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
                # print(f"[DEBUG] APPROACH_DEST: Dist={distance:.2f}, AngleErr={math.degrees(angle_error):.1f}")
            else:
                print("[INFO] APPROACH_DEST: Reached destination proximity. Transition to ALIGN_DEST")
                self.current_state = STATE_ALIGN_DEST
                self.angle_pid.reset()
                self.reset_stability_timer()
                left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_ALIGN_DEST:
            dx_slider = p.readUserDebugParameter(self.dest_x_slider)
            dy_slider = p.readUserDebugParameter(self.dest_y_slider)
            target = [dx_slider, dy_slider]
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
                # print(f"[DEBUG] ALIGN_DEST: Rotating. AngleErr={math.degrees(angle_error):.1f}°")
            else:
                left_velocity = right_velocity = 0.0
                # print(f"[DEBUG] ALIGN_DEST: Angle error within threshold ({math.degrees(angle_error):.1f}°). Checking stability...")
                if self.check_stability(aligned):
                    print("[INFO] ALIGN_DEST: Alignment stable. Transition to FINAL_DRIVE_DEST")
                    self.current_state = STATE_FINAL_DRIVE_DEST
                    self.angle_pid.reset()
                # else:
                    # print(f"[DEBUG] ALIGN_DEST: Waiting for stability ({time.time() - self.stability_timer:.2f}s / {STABILITY_DURATION:.2f}s)")

        elif self.current_state == STATE_FINAL_DRIVE_DEST:
            dx_slider = p.readUserDebugParameter(self.dest_x_slider)
            dy_slider = p.readUserDebugParameter(self.dest_y_slider)
            target = [dx_slider, dy_slider]
            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]
            distance = math.hypot(dx, dy)

            if distance > DOCKING_DISTANCE - 0.01:
                desired_angle = rover_yaw
                angle_error = (desired_angle - rover_yaw + math.pi) % (2*math.pi) - math.pi
                pid_output = self.angle_pid.compute(angle_error, SIMULATION_TIMESTEP)
                left_velocity = FINAL_DRIVE_SPEED - pid_output
                right_velocity = FINAL_DRIVE_SPEED + pid_output
                # print(f"[DEBUG] FINAL_DRIVE_DEST: Driving forward. Dist={distance:.3f}")
            else:
                print("[INFO] FINAL_DRIVE_DEST: Reached release distance. Transition to UNDOCKING.")
                self.current_state = STATE_UNDOCKING
                left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_UNDOCKING:
            if self.dock_constraint is not None:
                try:
                    p.removeConstraint(self.dock_constraint)
                    self.dock_constraint = None
                    print("[INFO] UNDOCKING: Constraint removed. Block released.")
                except p.error as e:
                     print(f"[WARN] Error removing constraint (might already be removed): {e}")
                     self.dock_constraint = None # Ensure it's None even if remove failed

            print("[INFO] UNDOCKING: Transitioning to RETREAT.")
            self.current_state = STATE_RETREAT
            self.retreat_start_pos = base_pos
            left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_RETREAT:
            if self.retreat_start_pos is None:
                self.retreat_start_pos = base_pos
            distance_retreated = math.hypot(base_pos[0] - self.retreat_start_pos[0],
                                           base_pos[1] - self.retreat_start_pos[1])
            if distance_retreated < RETREAT_DISTANCE:
                left_velocity = -RETREAT_SPEED
                right_velocity = -RETREAT_SPEED
                # print(f"[DEBUG] RETREAT: Moving backward ({distance_retreated:.2f} / {RETREAT_DISTANCE:.2f})")
            else:
                print("[INFO] RETREAT: Retreat complete. Transition to DELIVERED.")
                self.current_state = STATE_DELIVERED
                self.retreat_start_pos = None
                left_velocity = right_velocity = 0.0

        elif self.current_state == STATE_DELIVERED:
            left_velocity = right_velocity = 0.0
            if target is None: # Keep camera target if needed
                 dx_slider = p.readUserDebugParameter(self.dest_x_slider)
                 dy_slider = p.readUserDebugParameter(self.dest_y_slider)
                 target = [dx_slider, dy_slider]
            # print("[DEBUG] DELIVERED: Process complete, rover idle.") # Reduce printing
            # Optionally transition back to IDLE automatically after a pause
            if self.stability_timer is None: self.stability_timer = time.time()
            if time.time() - self.stability_timer > 5.0: # Wait 5 seconds
                self.current_state = STATE_IDLE
                self.reset_stability_timer()
                print("[INFO] DELIVERED: Restarting cycle -> IDLE")


        # --- Apply Motor Speeds ---
        # Only apply if robot ID is valid
        if self.robot_id is not None:
             set_motor_speeds(self.robot_id, self.joint_dict, left_velocity, right_velocity)

        # --- Return target for camera ---
        return target


# --- Main Simulation Setup ---
def run_simulation(urdf_path, block_urdf_path):
    # Make sure simulation is initialized correctly
    initialize_simulation()

    # Load robot and block, exit gracefully if loading fails
    robot_id, joint_dict = load_robot(urdf_path)
    if robot_id is None:
        print("[FATAL] Robot loading failed. Exiting.")
        if p.isConnected(): p.disconnect()
        return

    block_body_id, block_center_sphere = load_block(block_urdf_path)
    if block_body_id is None:
        print("[FATAL] Block loading failed. Exiting.")
        if p.isConnected(): p.disconnect()
        return

    # Create visualization markers
    dest_body_id = create_destination_visual()
    target_vis_id = create_target_visual()

    # --- Camera and LIDAR Settings ---
    lidar_enabled = False
    auto_camera = True

    # Create the bot instance
    builder_bot = BuilderBot(robot_id, joint_dict, block_body_id, target_vis_id)

    frame_count = 0
    last_print_time = time.time()

    # Main simulation loop
    try:
        while p.isConnected(): # Loop while connected
            frame_count += 1
            current_time = time.time()
            keys = p.getKeyboardEvents()

            # --- Handle User Input (Toggles) ---
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                auto_camera = not auto_camera
                print("[INFO] Auto camera toggled:", auto_camera)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED and not auto_camera:
                p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89,
                                             cameraTargetPosition=[0,0,0])
                print("[INFO] Camera reset.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                lidar_enabled = not lidar_enabled
                print("[INFO] LIDAR toggled:", lidar_enabled)

            # --- Run Bot Step ---
            try:
                base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
                camera_target = builder_bot.run_step()
            except p.error as e:
                 print(f"[ERROR] PyBullet error during bot step (state {builder_bot.current_state}): {e}")
                 # Attempt to recover or exit gracefully
                 builder_bot.current_state = STATE_IDLE # Go back to idle as recovery
                 # Or break the loop:
                 # break

            # --- Update Visualizations ---
            try:
                 update_block_visualization(block_body_id, block_center_sphere)
                 if auto_camera: update_camera_view(auto_camera, base_pos, camera_target)
                 if lidar_enabled: simulate_lidar(robot_id)
            except p.error as e:
                 # Ignore visualization errors if bodies might be invalid temporarily
                 # print(f"[WARN] PyBullet error during visualization update: {e}")
                 pass


            # --- IMU Display ---
            try:
                 roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)
                 imu_text = f"IMU: Roll={math.degrees(roll):.1f} Pitch={math.degrees(pitch):.1f} Yaw={math.degrees(rover_yaw):.1f}"
                 p.addUserDebugText(imu_text, [base_pos[0], base_pos[1], base_pos[2]+0.5],
                               textColorRGB=[1,1,1], textSize=1.2, lifeTime=SIMULATION_TIMESTEP*2)
            except p.error as e:
                 pass # Ignore text errors


            # --- Simulation Step ---
            p.stepSimulation()
            # time.sleep(SIMULATION_TIMESTEP) # Let stepSimulation handle timing with RealTimeSim=0

            # --- Periodic Debug Print ---
            if current_time - last_print_time >= 1.0: # Print every 1 second
                print(f"[INFO] Time: {current_time:.1f} State={builder_bot.get_state_name()} ({builder_bot.current_state})")
                last_print_time = current_time

    except KeyboardInterrupt:
        print("[INFO] Simulation terminated by user.")
    except Exception as e:
        print(f"[ERROR] An unhandled exception occurred in the main loop: {e}")
        traceback.print_exc() # Print detailed exception info
    finally:
        if p.isConnected():
            print("[INFO] Disconnecting from PyBullet.")
            p.disconnect()

# --- Run Script ---
if __name__ == "__main__":
    # Use raw strings for Windows paths
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    block_urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeStructure.urdf"
    run_simulation(urdf_file, block_urdf_file)