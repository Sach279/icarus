import pybullet as p
import pybullet_data
import time

def load_and_simulate_urdf(urdf_path):
    # Connect to PyBullet in GUI mode for visualization
    physicsClient = p.connect(p.GUI)

    # Set additional search path for assets (e.g., plane.urdf)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity
    p.setGravity(0, 0, -9.81)

    # Load a plane for reference
    plane_id = p.loadURDF("plane.urdf")

    # Spawn the rover a bit above the ground so it's clearly visible
    try:
        # Adjust the basePosition: change the Z value to raise or lower the robot.
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1], useFixedBase=False)
        print(f"Successfully loaded URDF: {urdf_path}")
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        p.disconnect()
        return

    # Reset the camera so that it’s closer to the rover.
    # cameraDistance: how far the camera is from the target
    # cameraYaw: rotation around the target (in degrees)
    # cameraPitch: angle above/below the horizontal plane (in degrees)
    # cameraTargetPosition: the point the camera is looking at (here, near the rover)
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.1])

    # Run the simulation indefinitely until you stop it (Ctrl+C)
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)  # maintain simulation rate
    except KeyboardInterrupt:
        print("Simulation terminated by user.")
    finally:
        p.disconnect()

if __name__ == "__main__":
    # Use a raw string to avoid backslash escape issues in the file path.
    urdf_file = r"C:\Users\akshi\Documents\Building Block\Models\CubeBuilder.urdf"
    load_and_simulate_urdf(urdf_file)
