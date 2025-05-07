import pybullet as p
import pybullet_data
import time
import math

def load_and_control_urdf(urdf_path):
    # Connect to PyBullet in GUI mode.
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load a plane for visual reference.
    plane_id = p.loadURDF("plane.urdf")

    # Load the rover URDF slightly above ground.
    try:
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1], useFixedBase=False)
        print(f"Successfully loaded URDF: {urdf_path}")
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        p.disconnect()
        return

    # Build a mapping from joint names to joint indices.
    joint_dict = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_dict[joint_name] = i
    print("Joint mapping:", joint_dict)

    # Control parameters.
    force = 10          # Maximum force to apply to each wheel.
    forward_speed = 5.0 # Speed for moving forward/backward.
    turn_speed = 3.0    # Speed for turning.
    
    # Camera mode toggle: True means the camera is fixed behind the rover.
    camera_fixed = True

    print("Controls:")
    print("  Left Arrow: Move forward")
    print("  Right Arrow: Move backward")
    print("  Up Arrow: Turn right")
    print("  Down Arrow: Turn left")
    print("  C: Toggle between free camera and fixed camera behind the rover")

    try:
        while True:
            keys = p.getKeyboardEvents()

            # Toggle camera mode if the 'c' key is pressed.
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                camera_fixed = not camera_fixed
                if camera_fixed:
                    print("Camera set to fixed behind rover.")
                else:
                    print("Free camera mode activated.")

            # Initialize default wheel velocities.
            left_velocity = 0.0
            right_velocity = 0.0

            # Inverted control mapping:
            # Left arrow: move forward (both wheels forward)
            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                left_velocity = forward_speed
                right_velocity = forward_speed
            # Right arrow: move backward (both wheels backward)
            elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                left_velocity = -forward_speed
                right_velocity = -forward_speed
            # Up arrow: turn right (left wheels forward, right wheels reverse)
            elif p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                left_velocity = turn_speed
                right_velocity = -turn_speed
            # Down arrow: turn left (left wheels reverse, right wheels forward)
            elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                left_velocity = -turn_speed
                right_velocity = turn_speed

            # Apply velocity control to each wheel joint.
            p.setJointMotorControl2(robot_id,
                                    joint_dict['wheel_FL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=left_velocity,
                                    force=force)
            p.setJointMotorControl2(robot_id,
                                    joint_dict['wheel_RL_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=left_velocity,
                                    force=force)
            p.setJointMotorControl2(robot_id,
                                    joint_dict['wheel_FR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=right_velocity,
                                    force=force)
            p.setJointMotorControl2(robot_id,
                                    joint_dict['wheel_RR_joint'],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=right_velocity,
                                    force=force)

            p.stepSimulation()
            time.sleep(1./240.)  # Maintain simulation rate.

            # If fixed camera mode is active, update the camera dynamically.
            if camera_fixed:
                # Get the rover's current base position and orientation.
                base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
                # Convert the quaternion to Euler angles and extract the yaw.
                euler = p.getEulerFromQuaternion(base_orn)
                rover_yaw = euler[2]  # Yaw angle in radians.
                # Calculate the camera yaw: position the camera behind the rover.
                camera_yaw = (math.degrees(rover_yaw) + 90) % 360
                # Update the camera view.
                p.resetDebugVisualizerCamera(cameraDistance=1,
                                             cameraYaw=camera_yaw,
                                             cameraPitch=-10,
                                             cameraTargetPosition=base_pos)
    except KeyboardInterrupt:
        print("Simulation terminated by user.")
    finally:
        p.disconnect()

if __name__ == "__main__":
    # Use a raw string to handle Windows backslashes.
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    load_and_control_urdf(urdf_file)
