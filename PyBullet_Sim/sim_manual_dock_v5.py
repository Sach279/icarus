import pybullet as p
import pybullet_data
import time
import math
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
        p.addUserDebugLine(ray_from_list[i], hit_pos, lineColorRGB=color, lineWidth=1, lifeTime=1 / 240.)


def pickStrictSideAngle(dx, dy):
    if abs(dx) >= abs(dy):
        return 0.0 if dx >= 0 else math.pi
    else:
        return math.pi / 2 if dy >= 0 else -math.pi / 2


def pickClosestSide(dx, dy):
    if abs(dx) >= abs(dy):
        if dx >= 0:
            return 0.0
        else:
            return math.pi
    else:
        if dy >= 0:
            return math.pi / 2
        else:
            return -math.pi / 2


def load_and_control_urdf(urdf_path):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    print("[DEBUG] Connected to PyBullet, gravity set to -9.81.")

    p.loadURDF("plane.urdf")
    print("[DEBUG] Plane URDF loaded.")

    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1], useFixedBase=False)
        print("[DEBUG] Rover loaded.")
    except Exception as e:
        print("[ERROR] Failed to load rover URDF:", e)
        p.disconnect()
        return

    joint_dict = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_dict[info[1].decode("utf-8")] = i

    # Control parameters.
    force = 20
    forward_speed = 5.0
    Kp, Ki, Kd = 2.0, 0.25, 0.01
    threshold_stage1 = math.radians(90)  # ±90° must be maintained.
    threshold_stage2 = math.radians(40)  # (Not used now)
    final_approach_distance = 0.12
    predock_offset = 0.2
    docking_lift = 0.03
    stop_alignment_threshold = math.radians(2)  # Tighter: 2°

    # Destination target sliders.
    dest_x_slider = p.addUserDebugParameter("Destination X", -10, 10, 0.0)
    dest_y_slider = p.addUserDebugParameter("Destination Y", -10, 10, 0.0)
    print("[DEBUG] Destination sliders created.")

    # Load block (structure) from its URDF.
    block_urdf_path = r"C:\Users\akshi\Documents\Building Block\Models\CubeStructure.urdf"
    try:
        block_body_id = p.loadURDF(block_urdf_path, basePosition=[1.0, -0.5, 0.01], useFixedBase=False)
        print("[DEBUG] Block loaded.")
    except Exception as e:
        print("[ERROR] Failed to load block:", e)
        p.disconnect()
        return

    # Create a red transparent sphere to visualize the center of the block.
    # Adjust the sphere's radius and color as needed.
    block_center_sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                     radius=0.05,
                                                     rgbaColor=[1, 0, 0, 0.5])
    block_center_sphere = p.createMultiBody(baseMass=0,
                                            baseVisualShapeIndex=block_center_sphere_visual,
                                            baseCollisionShapeIndex=-1,
                                            basePosition=[2.0, 0.0, 0.01])
    print("[DEBUG] Block center sphere created.")

    # Create a red transparent cube as destination target.
    dest_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.5])
    dest_body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=dest_shape_id,
                                     baseCollisionShapeIndex=-1, basePosition=[0, 0, 0.2])
    print("[DEBUG] Destination target created.")

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
    print("")
    print("State Legend:")
    print("  0: APPROACH_BLOCK")
    print("  1: FINAL_ALIGN_BLOCK")
    print("  2: APPROACH_DEST")
    print("  3: FINAL_ALIGN_DEST")
    print("  4: DELIVERED")

    # Numeric state variable:
    # 0: APPROACH_BLOCK, 1: FINAL_ALIGN_BLOCK, 2: APPROACH_DEST, 3: FINAL_ALIGN_DEST, 4: DELIVERED.
    state = 0

    error_integral = 0.0
    previous_error = 0.0
    dt = 1 / 120.0
    dock_constraint = None
    pre_dock_target_block = None
    pre_dock_target_dest = None

    frame_count = 0

    try:
        while True:
            frame_count += 1
            keys = p.getKeyboardEvents()
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                auto_camera = not auto_camera
                print("[DEBUG] Auto camera toggled:", auto_camera)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                if not auto_camera:
                    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0,
                                                 cameraPitch=-89, cameraTargetPosition=[0, 0, 0])
                    print("[DEBUG] Camera reset.")
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                lidar_enabled = not lidar_enabled
                print("[DEBUG] LIDAR toggled:", lidar_enabled)

            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            roll, pitch, rover_yaw = p.getEulerFromQuaternion(base_orn)
            target = None

            # --- Update Block Center Visualization ---
            block_center, _ = p.getBasePositionAndOrientation(block_body_id)
            p.resetBasePositionAndOrientation(block_center_sphere, block_center, [0, 0, 0, 1])

            # --- Draw Orientation Frame for the Block ---
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

            match state:
                case 0:  # APPROACH_BLOCK
                    pos_block, _ = p.getBasePositionAndOrientation(block_body_id)
                    target = [pos_block[0], pos_block[1]]
                    dx = target[0] - base_pos[0]
                    dy = target[1] - base_pos[1]
                    distance = math.hypot(dx, dy)
                    desired_angle = pickStrictSideAngle(dx, dy)
                    angle_error = desired_angle - rover_yaw
                    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

                    error = angle_error
                    error_integral += error * dt
                    error_derivative = (error - previous_error) / dt
                    pid_output = Kp * error + Ki * error_integral + Kd * error_derivative
                    previous_error = error

                    if abs(angle_error) > threshold_stage1:
                        print("[DEBUG] Lost block from sight => rotate in place.")
                        left_velocity = -Kp * angle_error
                        right_velocity = Kp * angle_error
                    else:
                        if distance > final_approach_distance + predock_offset:
                            left_velocity = forward_speed - pid_output
                            right_velocity = forward_speed + pid_output
                        else:
                            pre_dock_target_block = [
                                target[0] - predock_offset * math.cos(desired_angle),
                                target[1] - predock_offset * math.sin(desired_angle)
                            ]
                            first_time = 1
                            stage = 1
                            state = 1
                            print("[DEBUG] Transition to FINAL_ALIGN_BLOCK")
                            left_velocity = 0.0
                            right_velocity = 0.0

                case 1:  # FINAL_ALIGN_BLOCK
                    pos_block, _ = p.getBasePositionAndOrientation(block_body_id)
                    dx = pos_block[0] - base_pos[0]
                    dy = pos_block[1] - base_pos[1]
                    distance_array = [dx, dy]
                    distance = math.hypot(dx, dy)

                    if first_time == 1:
                        desired_angle_for_check = pickStrictSideAngle(dx, dy)
                        first_time = 0

                    if abs(dx) > abs(dy):
                        close_axis = 1
                    else:
                        close_axis = 0

                    angle_error_check = desired_angle_for_check - rover_yaw
                    error = (angle_error_check + math.pi) % (2 * math.pi) - math.pi

                    # if distance < final_approach_distance or abs(angle_error) < stop_alignment_threshold:

                    # print(f"[DEBUG] desired:{desired_angle_for_check:.2f}, angle:{error:.2f}, rover_yaw:{rover_yaw:.2f}, lv:{left_velocity:.2f}, rv:{right_velocity:.2f}, kp:{Kp}")

                    if stage == 1:
                        if abs(error) > 0.01:
                            k = 8
                            left_velocity = -(Kp * error * k)
                            right_velocity = (Kp * error * k)
                        else:
                            # print(f"[DEBUG] Stage 1 of aligning Done.")
                            left_velocity = 0.0
                            right_velocity = 0.0
                            stage = 2

                    if stage == 2:
                        print(f"[DEBUG] axis:{close_axis}, dist:{abs(distance_array[close_axis]):.2f}")
                        if abs(distance_array[close_axis]) > 0.01:
                            linear_speed = forward_speed * 0.5
                            left_velocity = linear_speed
                            right_velocity = linear_speed
                        else:
                            # print(f"[DEBUG] Stage 2 of aligning Done.")
                            left_velocity = 0.0
                            right_velocity = 0.0
                            stage = 3

                    if stage == 3:
                        # At this stage the rover needs to turn towards the block
                        print(f"[DEBUG] Stage 3 of aligning.")
                        pos_block, _ = p.getBasePositionAndOrientation(block_body_id)
                        dx_b = pos_block[0] - base_pos[0]
                        dy_b = pos_block[1] - base_pos[1]
                        block_angle = math.atan2(dy_b, dx_b)
                        desired_angle = block_angle
                        desired_angle = (desired_angle + math.pi) % (2 * math.pi) - math.pi

                        angle_error = desired_angle - rover_yaw
                        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

                        # Here, if we're not yet within our stop threshold, rotate in place.
                        if abs(angle_error) > stop_alignment_threshold:
                            left_velocity = -(Kp * angle_error * k)
                            right_velocity = (Kp * angle_error * k)
                        else:
                            # print(f"[DEBUG] Stage 2 of aligning Done.")
                            left_velocity = 0.0
                            right_velocity = 0.0
                            stage = 4

                    if stage == 4:
                        print(f"[DEBUG] Stage 3 of aligning.")
                        if abs(distance) > 0.12:
                            linear_speed = forward_speed * 0.2
                            left_velocity = linear_speed
                            right_velocity = linear_speed
                        else:
                            # print(f"[DEBUG] Stage 2 of aligning Done.")
                            left_velocity = 0.0
                            right_velocity = 0.0
                            stage = 5

                    if stage == 5:
                        print(f"[DEBUG] FINAL_ALIGN_BLOCK: Perfectly aligned and stopped")
                        if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                            pos_rover, orn_rover = p.getBasePositionAndOrientation(robot_id)
                            pos_block, orn_block = p.getBasePositionAndOrientation(block_body_id)
                            invPos, invOrn = p.invertTransform(pos_rover, orn_rover)
                            relPos, _ = p.multiplyTransforms(invPos, invOrn, pos_block, orn_block)
                            relPos = [relPos[0], relPos[1], relPos[2] + docking_lift]
                            dock_constraint = p.createConstraint(robot_id, -1, block_body_id, -1,
                                                                 p.JOINT_FIXED, [0, 0, 0],
                                                                 relPos, [0, 0, 0])
                            docked = True
                            state = 2
                            print("[DEBUG] Docked: Block attached and lifted. Transition to APPROACH_DEST.")

                case 2:  # APPROACH_DEST
                    dx_slider = p.readUserDebugParameter(dest_x_slider)
                    dy_slider = p.readUserDebugParameter(dest_y_slider)
                    target = [dx_slider, dy_slider]
                    dx = target[0] - base_pos[0]
                    dy = target[1] - base_pos[1]
                    distance = math.hypot(dx, dy)
                    desired_angle = pickStrictSideAngle(dx, dy)
                    angle_error = desired_angle - rover_yaw
                    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

                    error = angle_error
                    error_integral += error * dt
                    error_derivative = (error - previous_error) / dt
                    pid_output = Kp * error + Ki * error_integral + Kd * error_derivative
                    previous_error = error

                    if distance > final_approach_distance:
                        left_velocity = forward_speed - pid_output
                        right_velocity = forward_speed + pid_output
                    else:
                        state = 3
                        left_velocity = 0.0
                        right_velocity = 0.0
                        print("[DEBUG] Transition to FINAL_ALIGN_DEST")

                case 3:  # FINAL_ALIGN_DEST
                    dx_slider = p.readUserDebugParameter(dest_x_slider)
                    dy_slider = p.readUserDebugParameter(dest_y_slider)
                    target = [dx_slider, dy_slider]
                    dx_fine = target[0] - base_pos[0]
                    dy_fine = target[1] - base_pos[1]
                    distance = math.hypot(dx_fine, dy_fine)
                    desired_angle = pickStrictSideAngle(dx_fine, dy_fine)
                    angle_error = desired_angle - rover_yaw
                    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

                    error = angle_error
                    error_integral += error * dt
                    error_derivative = (error - previous_error) / dt
                    pid_output = Kp * error + Ki * error_integral + Kd * error_derivative
                    previous_error = error

                    if distance > final_approach_distance or abs(angle_error) > stop_alignment_threshold:
                        linear_speed = forward_speed * 0.2
                        left_velocity = linear_speed - pid_output
                        right_velocity = linear_speed + pid_output
                    else:
                        left_velocity = 0.0
                        right_velocity = 0.0
                        print("[DEBUG] FINAL_ALIGN_DEST: Perfectly aligned and stopped.")
                        if abs(angle_error) < stop_alignment_threshold and ord('u') in keys and keys[
                            ord('u')] & p.KEY_WAS_TRIGGERED:
                            p.removeConstraint(dock_constraint)
                            dock_constraint = None
                            docked = False
                            state = 4
                            print("[DEBUG] Undocked: Block released. Transition to DELIVERED.")
                case 4:  # DELIVERED
                    left_velocity = 0.0
                    right_velocity = 0.0
                    target = target if target is not None else [0, 0]
                    angle_error = 0.0
                    print("[DEBUG] DELIVERED: Process complete, rover idle.")

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

            if auto_camera and target is not None:
                mid_x = (base_pos[0] + target[0]) / 2.0
                mid_y = (base_pos[1] + target[1]) / 2.0
                midpoint = [mid_x, mid_y, 0]
                separation = math.hypot(base_pos[0] - target[0], base_pos[1] - target[1])
                FOV = 60
                required_distance = (separation / 2) / math.tan(math.radians(FOV / 2))
                min_dist = 1.5
                cam_dist = max(min_dist, required_distance * 1.2)
                p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=0,
                                             cameraPitch=-89, cameraTargetPosition=midpoint)
            if lidar_enabled:
                simulate_lidar(robot_id, num_rays=36, ray_length=5.0, fov_deg=140)

            imu_text = f"IMU: Roll={roll:.1f} Pitch={pitch:.1f} Yaw={math.degrees(rover_yaw):.1f}"
            p.addUserDebugText(imu_text, [base_pos[0], base_pos[1], base_pos[2] + 0.5],
                               textColorRGB=[1, 1, 1], textSize=1.2, lifeTime=1 / 120.)

            p.stepSimulation()
            time.sleep(1 / 120.)
            if frame_count % 120 == 0:
                if 'angle_error' not in locals():
                    angle_error = 0.0
                if 'distance' not in locals():
                    distance = 0.0
                print(
                    f"[DEBUG] Frame {frame_count}: Dist={distance:.2f}, AngleErr={math.degrees(angle_error):.1f}, State={state}")
    except KeyboardInterrupt:
        print("[DEBUG] Simulation terminated by user.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    load_and_control_urdf(urdf_file)
