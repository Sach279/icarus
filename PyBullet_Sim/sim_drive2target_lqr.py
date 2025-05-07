import pybullet as p
import pybullet_data
import time
import math
import numpy as np
from scipy.linalg import solve_continuous_are

# Constants
GRAVITY = -9.81
FRAME_RATE = 120
FORCE = 20
FORWARD_SPEED = 10.0

# LQR Controller parameters
Q = np.diag([10, 10, 1])
R = np.diag([1, 1])


def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, GRAVITY)
    p.setRealTimeSimulation(1)
    p.loadURDF("plane.urdf")


def load_rover(urdf_path):
    rover_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1])
    joint_mapping = {}
    for i in range(p.getNumJoints(rover_id)):
        info = p.getJointInfo(rover_id, i)
        joint_mapping[info[1].decode()] = i
    return rover_id, joint_mapping


def create_target_visual():
    shape_id = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.1], rgbaColor=[1, 0, 0, 0.5])
    return p.createMultiBody(0, shape_id, -1, [2.0, 2.0, 0.1])


def set_motor_speeds(rover_id, joints, left_speed, right_speed):
    wheels = [('wheel_FL_joint', -left_speed), ('wheel_RL_joint', -left_speed),
              ('wheel_FR_joint', right_speed), ('wheel_RR_joint', right_speed)]
    for wheel_name, speed in wheels:
        p.setJointMotorControl2(rover_id, joints[wheel_name], p.VELOCITY_CONTROL,
                                targetVelocity=speed, force=FORCE)


def update_camera_auto(rover_pos, target_pos):
    midpoint = [(rover_pos[0] + target_pos[0])/2, (rover_pos[1] + target_pos[1])/2, 0]
    separation = np.linalg.norm(np.array(rover_pos[:2]) - np.array(target_pos[:2]))
    camera_dist = max(2, (separation / 2) / math.tan(math.radians(30)) * 1.2)
    p.resetDebugVisualizerCamera(camera_dist, 0, -89, midpoint)


def compute_lqr_gain():
    v0 = FORWARD_SPEED
    A = np.array([[0, 0, 0],
                  [0, 0, v0],
                  [0, 0, 0]])

    B = np.array([[1, 0],
                  [0, 0],
                  [0, 1]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    return K


def main_loop(rover_id, joints, target_visual_id):
    auto_camera = True
    target_x_slider = p.addUserDebugParameter("Target X", -10, 10, 2.0)
    target_y_slider = p.addUserDebugParameter("Target Y", -10, 10, 0.0)
    target_yaw_slider = p.addUserDebugParameter("Target Yaw", -math.pi, math.pi, 0.0)

    K = compute_lqr_gain()

    while True:
        target_pos = [p.readUserDebugParameter(target_x_slider),
                      p.readUserDebugParameter(target_y_slider), 0.1]
        target_yaw = p.readUserDebugParameter(target_yaw_slider)

        target_orientation = p.getQuaternionFromEuler([0, 0, target_yaw])
        p.resetBasePositionAndOrientation(target_visual_id, target_pos, target_orientation)

        rover_pos, rover_orn = p.getBasePositionAndOrientation(rover_id)
        _, _, rover_yaw = p.getEulerFromQuaternion(rover_orn)

        dx, dy = target_pos[0] - rover_pos[0], target_pos[1] - rover_pos[1]
        distance, angle_to_target = math.hypot(dx, dy), math.atan2(dy, dx)

        angle_diff = (angle_to_target - target_yaw + math.pi) % (2 * math.pi) - math.pi
        desired_angle = target_yaw + round(angle_diff / (math.pi/2)) * (math.pi/2)
        angle_error = (desired_angle - rover_yaw + math.pi) % (2 * math.pi) - math.pi

        state_error = np.array([dx, dy, angle_error])
        control = -K.dot(state_error)

        left_speed = np.clip(FORWARD_SPEED + control[0], -FORWARD_SPEED, FORWARD_SPEED)
        right_speed = np.clip(FORWARD_SPEED + control[1], -FORWARD_SPEED, FORWARD_SPEED)

        set_motor_speeds(rover_id, joints, left_speed, right_speed)

        if auto_camera:
            update_camera_auto(rover_pos, target_pos)

        p.stepSimulation()
        time.sleep(1 / FRAME_RATE)


if __name__ == "__main__":
    setup_simulation()
    rover_id, joints = load_rover(r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf")
    target_visual_id = create_target_visual()
    try:
        main_loop(rover_id, joints, target_visual_id)
    except KeyboardInterrupt:
        p.disconnect()
