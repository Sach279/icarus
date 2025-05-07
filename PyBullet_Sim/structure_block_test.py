import pybullet as p
import pybullet_data
import time

# Connect to PyBullet simulation with a graphical interface
physicsClient = p.connect(p.GUI)

# Add the default search path for URDF files from pybullet_data
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity for the simulation (m/s^2)
p.setGravity(0, 0, -9.81)

# Load a ground plane to serve as the environment
planeId = p.loadURDF("plane.urdf")

# Specify the URDF file to load; change 'r2d2.urdf' to your own file if needed
urdf_path = r"C:\Users\akshi\Documents\Building Block\Models\CubeStructure.urdf"
startPos = [0, 0, 0.01]  # Spawn position (x, y, z)
startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # No initial rotation

# Load the URDF into the simulation
urdfId = p.loadURDF(urdf_path, startPos, startOrientation)

# Run the simulation loop
while p.isConnected():
    p.stepSimulation()
    time.sleep(1.0 / 240.0)  # Simulation step time
