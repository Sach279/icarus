import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket
import json
import os
from picamera2 import Picamera2

# Set the UDP target IP address and port (update UDP_IP with your laptop's IP)
UDP_IP = "192.168.230.44"  # Replace with your laptop's IP address
UDP_PORT = 5005

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Create a directory to save captured images
save_dir = "captured_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("[DEBUG] Created directory:", save_dir)

# Initialize Picamera2 and configure it
print("[DEBUG] Initializing Picamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration({"format": "XRGB8888", "size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow the camera to warm up
print("[DEBUG] Camera started and warmed up.")

# Setup the ArUco dictionary and detector parameters using the 4x4_100 dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()
print("[DEBUG] ArUco dictionary (DICT_4X4_100) and parameters set.")

frame_counter = 0

print("[DEBUG] Starting UDP sender loop. Press Ctrl+C to exit.")

try:
    while True:
        # Capture a frame from Picamera2
        frame = picam2.capture_array()
        print("[DEBUG] Captured a frame from Picamera2")

        # Adjust contrast (and optionally brightness)
        alpha = 1.5  # Increase contrast by 50%
        beta = 0     # No change in brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        print("[DEBUG] Adjusted contrast of the frame.")

        # Convert from XRGB8888 (with alpha channel) to BGR if necessary
        if frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            print("[DEBUG] Converted frame from BGRA to BGR")
        else:
            frame_bgr = frame
            print("[DEBUG] Frame already in BGR format")

        # Convert frame to grayscale for marker detection
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        print("[DEBUG] Converted frame to grayscale")

        # Detect ArUco markers in the grayscale frame
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None:
            print(f"[DEBUG] Detected {len(ids)} marker(s)")
            markers = []
            for marker_id, marker_corners in zip(ids.flatten(), corners):
                print(f"[DEBUG] Marker ID {marker_id} detected with corners: {marker_corners}")
                markers.append({
                    "id": int(marker_id),
                    "corners": marker_corners.reshape((4, 2)).tolist()
                })

            # Save the image locally with drawn markers
            #image_with_markers = aruco.drawDetectedMarkers(frame_bgr.copy(), corners, ids)
            #timestamp = time.strftime("%Y%m%d-%H%M%S")
            #filename = os.path.join(save_dir, f"frame_{timestamp}_{frame_counter}.jpg")
            #cv2.imwrite(filename, image_with_markers)
            #print(f"[DEBUG] Saved image locally as: {filename}")
            frame_counter += 1

            # Prepare UDP data with marker info
            data = {
                "markers": markers
            }
            json_data = json.dumps(data)
            # Send the JSON data over UDP
            sock.sendto(json_data.encode('utf-8'), (UDP_IP, UDP_PORT))
            print("[DEBUG] Sent UDP packet:", json_data)
        else:
            print("[DEBUG] No markers detected in this frame")

        # Short delay to reduce CPU usage
        # time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[DEBUG] Exiting UDP sender loop.")

finally:
    picam2.stop()
    print("[DEBUG] Camera stopped.")
