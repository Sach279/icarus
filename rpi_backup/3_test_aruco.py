import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os
from picamera2 import Picamera2

# Create a directory to save captured images
save_dir = "captured_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"[DEBUG] Created directory: {save_dir}")

# Initialize Picamera2 and configure it for a preview mode
print("[DEBUG] Initializing Picamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration({"format": "XRGB8888", "size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow camera to warm up
print("[DEBUG] Camera started and warmed up.")

# Define the ArUco dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
print("[DEBUG] ArUco dictionary and parameters set.")

frame_counter = 0
processed_frames = 0

print("[DEBUG] Starting frame capture loop. Press Ctrl+C to exit.")

try:
    while True:
        processed_frames += 1
        # Capture a frame from Picamera2
        image = picam2.capture_array()
        print(f"[DEBUG] Processing frame #{processed_frames}")

        # If the captured image has an alpha channel (XRGB -> BGRA), convert it to BGR
        if image.shape[2] == 4:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            image_bgr = image

        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None:
            print(f"[DEBUG] Detected marker IDs: {ids.flatten()}")
            # Draw detected markers on the image
            image_bgr = aruco.drawDetectedMarkers(image_bgr, corners, ids)

            # Generate a filename using a timestamp and counter
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(save_dir, f"frame_{timestamp}_{frame_counter}.jpg")
            cv2.imwrite(filename, image_bgr)
            print(f"[DEBUG] Saved image: {filename}")
            frame_counter += 1
        else:
            print("[DEBUG] No markers detected in this frame.")

        # Pause briefly to reduce CPU usage (adjust as needed)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[DEBUG] Exiting frame capture loop.")

finally:
    picam2.stop()
    print("[DEBUG] Camera stopped.")
