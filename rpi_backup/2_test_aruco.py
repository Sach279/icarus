import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os
from picamera.array import PiRGBArray
from picamera import PiCamera

# Create a directory to save captured images
save_dir = "captured_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"[DEBUG] Created directory: {save_dir}")

# Initialize the PiCamera and set resolution and framerate
print("[DEBUG] Initializing camera...")
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(0.1)
print("[DEBUG] Camera warmed up.")

# Define the ArUco dictionary and detector parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
print("[DEBUG] ArUco dictionary and parameters set.")

# Counter for saved images
frame_counter = 0
processed_frames = 0

print("[DEBUG] Starting frame capture...")
# Start capturing frames from the camera continuously
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    processed_frames += 1
    image = frame.array
    print(f"[DEBUG] Processing frame #{processed_frames}")

    # Convert the captured frame to grayscale for marker detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the grayscale image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        print(f"[DEBUG] Detected marker IDs: {ids.flatten()}")
        # Draw detected markers on the image
        image = aruco.drawDetectedMarkers(image, corners, ids)

        # Generate a filename using the current timestamp and counter
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(save_dir, f"frame_{timestamp}_{frame_counter}.jpg")
        cv2.imwrite(filename, image)
        print(f"[DEBUG] Saved image: {filename}")
        frame_counter += 1
    else:
        print("[DEBUG] No markers detected in this frame.")

    # Clear the stream for the next frame
    rawCapture.truncate(0)
    
    # For debugging, optionally add a break condition after a certain number of frames
    # if processed_frames >= 100:
    #     print("[DEBUG] Processed 100 frames, exiting loop for debugging.")
    #     break

print("[DEBUG] Exiting frame capture loop.")
