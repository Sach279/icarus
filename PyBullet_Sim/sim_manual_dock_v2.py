import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np

def simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140):
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
    If |dx| == |dy|, pick X as a tiebreak (or pick Y, your choice).
    """
    if abs(dx) >= abs(dy):
        # Approach along X
        return 0.0 if dx >= 0 else math.pi
    else:
        # Approach along Y
        return math.pi/2 if dy >= 0 else -math.pi/2

def load_and_control_urdf(urdf_path):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)

    plane_id = p.loadURDF("plane.urdf")

    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0.1], useFixedBase=False)
    except Exception as e:
        print("[ERROR]", e)
        p.disconnect()
        return

    # Build joint dict
    joint_dict = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_dict[joint_name] = i

    force = 20
    forward_speed = 5.0
    Kp = 2.0
    Ki = 0.25
    Kd = 0.01

    threshold_stage1 = math.radians(90)
    threshold_stage2 = math.radians(40)
    final_approach_distance = 0.13
    docking_lift = 0.03
    stop_alignment_threshold = math.radians(10)

    # Debug Sliders
    block_x_slider = p.addUserDebugParameter("Block X", -10, 10, 2.0)
    block_y_slider = p.addUserDebugParameter("Block Y", -10, 10, 0.0)
    dest_x_slider = p.addUserDebugParameter("Destination X", -10, 10, 0.0)
    dest_y_slider = p.addUserDebugParameter("Destination Y", -10, 10, 0.0)

    # Load block (useFixedBase=False)
    block_urdf_path = r"C:\Users\akshi\Documents\Building Block\Models\CubeStructure.urdf"
    try:
        block_body_id = p.loadURDF(block_urdf_path, basePosition=[2.0, 2.0, 0.2], useFixedBase=False)
    except:
        p.disconnect()
        return

    # Red cube for destination
    dest_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2,0.2,0.2], rgbaColor=[1,0,0,0.5])
    dest_body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=dest_shape_id,
                                     baseCollisionShapeIndex=-1, basePosition=[0,0,0.2])

    # States
    docked = False
    dock_constraint = None
    approach_angle_block = 0.0  # stable approach angles
    approach_angle_dest  = 0.0

    # We store separate approach angles for block vs. destination
    # and only recalc them when user changes from undocked -> approach block
    # or from docked -> approach destination.

    # PID states
    error_integral = 0.0
    previous_error = 0.0
    dt = 1/120.0

    # camera
    auto_camera = True
    lidar_enabled = False

    print("Controls:")
    print("  C: Toggle camera (auto/free)")
    print("  R: Reset camera (when free)")
    print("  S: Toggle LIDAR")
    print("  D: Dock (attach block)")
    print("  U: Undock (release block)")

    frame_count = 0

    try:
        while True:
            frame_count += 1
            keys = p.getKeyboardEvents()

            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                auto_camera = not auto_camera
                print("[DEBUG] auto_camera =", auto_camera)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                if not auto_camera:
                    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0,
                                                 cameraPitch=-89, cameraTargetPosition=[0,0,0])
                    print("[DEBUG] camera reset")

            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                lidar_enabled = not lidar_enabled
                print("[DEBUG] LIDAR =", lidar_enabled)

            # If not docked => approach the block
            if not docked:
                bx = p.readUserDebugParameter(block_x_slider)
                by = p.readUserDebugParameter(block_y_slider)
                p.resetBasePositionAndOrientation(block_body_id, [bx, by, 0.2], [0,0,0,1])
                target = [bx, by]

                # Recalc approach angle if needed
                # We'll do it once at the beginning or if the user changes sliders drastically
                # but for simplicity, recalc each frame is fine => stable side approach anyway.
                base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
                dx = target[0] - base_pos[0]
                dy = target[1] - base_pos[1]
                approach_angle_block = pickStrictSideAngle(dx, dy)
                desired_angle = approach_angle_block
            else:
                # If docked => approach the destination
                dx_ = p.readUserDebugParameter(dest_x_slider)
                dy_ = p.readUserDebugParameter(dest_y_slider)
                p.resetBasePositionAndOrientation(dest_body_id, [dx_, dy_, 0.2], [0,0,0,1])
                target = [dx_, dy_]

                # Recalc approach angle for destination
                base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
                dx = target[0] - base_pos[0]
                dy = target[1] - base_pos[1]
                approach_angle_dest = pickStrictSideAngle(dx, dy)
                desired_angle = approach_angle_dest

            # Get rover state again
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)

            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]
            distance = math.hypot(dx, dy)

            # Always use the stable approach angle
            angle_error = desired_angle - rover_yaw
            angle_error = (angle_error + math.pi) % (2*math.pi) - math.pi

            # PID
            error = angle_error
            error_integral += error * dt
            error_derivative = (error - previous_error) / dt
            pid_output = Kp*error + Ki*error_integral + Kd*error_derivative
            previous_error = error

            # final approach
            if distance < final_approach_distance:
                if abs(angle_error) < stop_alignment_threshold:
                    # Stop
                    left_velocity  = 0.0
                    right_velocity = 0.0
                else:
                    # Slow approach
                    linear_speed = forward_speed*0.2
                    angular_correction = pid_output
                    left_velocity  = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction

                # Docking/undocking
                if not docked:
                    # Press D to dock
                    if abs(angle_error) < stop_alignment_threshold and ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                        # create constraint
                        pos_rover, orn_rover = p.getBasePositionAndOrientation(robot_id)
                        pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
                        invPos, invOrn = p.invertTransform(pos_rover, orn_rover)
                        relPos, _ = p.multiplyTransforms(invPos, invOrn, pos_block, orn_block)
                        # add lift
                        relPos = [relPos[0], relPos[1], relPos[2] + docking_lift]
                        dock_constraint = p.createConstraint(robot_id, -1, block_body_id, -1,
                                                             p.JOINT_FIXED, [0,0,0],
                                                             relPos, [0,0,0])
                        docked = True
                        print("[DEBUG] Docked & lifted block.")
                else:
                    # Press U to undock
                    if abs(angle_error) < stop_alignment_threshold and ord('u') in keys and keys[ord('u')] & p.KEY_WAS_TRIGGERED:
                        p.removeConstraint(dock_constraint)
                        dock_constraint = None
                        docked = False
                        print("[DEBUG] Undocked block at destination.")
                        left_velocity  = 0.0
                        right_velocity = 0.0
            else:
                # normal approach
                if abs(angle_error) > threshold_stage1:
                    # rotate in place
                    left_velocity  = -pid_output
                    right_velocity =  pid_output
                elif abs(angle_error) > threshold_stage2:
                    scale = (threshold_stage1 - abs(angle_error)) / (threshold_stage1 - threshold_stage2)
                    linear_speed = forward_speed*scale
                    angular_correction = pid_output
                    left_velocity  = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction
                else:
                    # fully aligned
                    linear_speed = forward_speed
                    angular_correction = pid_output
                    left_velocity  = linear_speed - angular_correction
                    right_velocity = linear_speed + angular_correction

            # Apply velocities
            p.setJointMotorControl2(robot_id, joint_dict['wheel_FL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=-left_velocity, force=force)
            p.setJointMotorControl2(robot_id, joint_dict['wheel_RL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=-left_velocity, force=force)
            p.setJointMotorControl2(robot_id, joint_dict['wheel_FR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity= right_velocity, force=force)
            p.setJointMotorControl2(robot_id, joint_dict['wheel_RR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity= right_velocity, force=force)

            # auto camera
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
            # LIDAR
            if lidar_enabled:
                simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140)

            # IMU
            imu_text = f"IMU: Roll={roll:.1f} Pitch={pitch:.1f} Yaw={math.degrees(rover_yaw):.1f}"
            p.addUserDebugText(imu_text, [base_pos[0], base_pos[1], base_pos[2]+0.5],
                               textColorRGB=[1,1,1], textSize=1.2, lifeTime=1/120.)

            p.stepSimulation()
            time.sleep(1/120.)
            if frame_count % 120 == 0:
                print(f"[DEBUG] Frame {frame_count}: dist={distance:.2f} angle_err={math.degrees(angle_error):.1f}")

    except KeyboardInterrupt:
        print("[DEBUG] Simulation terminated by user.")
    finally:
        p.disconnect()
        cv2.destroyAllWindows()

if __name__=="__main__":
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    load_and_control_urdf(urdf_file)
