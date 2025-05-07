#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os
import socket
import json
import RPi.GPIO as GPIO
from picamera2 import Picamera2

#------------------------------
# UDP Socket Setup
#------------------------------
UDP_IP = "192.168.230.44"  # Replace with your laptop's IP address
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#------------------------------
# Motor & Drive Unit Definitions
#------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor:
    def __init__(self, pin_forward, pin_reverse, use_pwm=False, pwm_freq=1000, inverted=False):
        self.pin_forward = pin_forward
        self.pin_reverse = pin_reverse
        self.use_pwm = use_pwm
        self.inverted = inverted
        GPIO.setup(self.pin_forward, GPIO.OUT)
        GPIO.setup(self.pin_reverse, GPIO.OUT)
        if self.use_pwm:
            self.pwm_forward = GPIO.PWM(self.pin_forward, pwm_freq)
            self.pwm_reverse = GPIO.PWM(self.pin_reverse, pwm_freq)
            self.pwm_forward.start(0)
            self.pwm_reverse.start(0)

    def forward(self, speed=100):
        if self.inverted:
            self._reverse_logic(speed)
        else:
            self._forward_logic(speed)

    def reverse(self, speed=100):
        if self.inverted:
            self._forward_logic(speed)
        else:
            self._reverse_logic(speed)

    def _forward_logic(self, speed):
        if self.use_pwm:
            self.pwm_forward.ChangeDutyCycle(speed)
            self.pwm_reverse.ChangeDutyCycle(0)
        else:
            GPIO.output(self.pin_forward, GPIO.HIGH)
            GPIO.output(self.pin_reverse, GPIO.LOW)

    def _reverse_logic(self, speed):
        if self.use_pwm:
            self.pwm_forward.ChangeDutyCycle(0)
            self.pwm_reverse.ChangeDutyCycle(speed)
        else:
            GPIO.output(self.pin_forward, GPIO.LOW)
            GPIO.output(self.pin_reverse, GPIO.HIGH)

    def stop(self):
        if self.use_pwm:
            self.pwm_forward.ChangeDutyCycle(0)
            self.pwm_reverse.ChangeDutyCycle(0)
        else:
            GPIO.output(self.pin_forward, GPIO.LOW)
            GPIO.output(self.pin_reverse, GPIO.LOW)

class DifferentialDrive:
    def __init__(self, left_motors, right_motors):
        self.left_motors = left_motors
        self.right_motors = right_motors

    def forward(self, speed=100):
        for motor in self.left_motors:
            motor.forward(speed)
        for motor in self.right_motors:
            motor.forward(speed)

    def reverse(self, speed=100):
        for motor in self.left_motors:
            motor.reverse(speed)
        for motor in self.right_motors:
            motor.reverse(speed)

    def pivot_left(self, speed=50):
        for motor in self.left_motors:
            motor.reverse(speed)
        for motor in self.right_motors:
            motor.forward(speed)

    def pivot_right(self, speed=50):
        for motor in self.left_motors:
            motor.forward(speed)
        for motor in self.right_motors:
            motor.reverse(speed)

    def stop(self):
        for motor in self.left_motors:
            motor.stop()
        for motor in self.right_motors:
            motor.stop()

#-------------------------------------------------------------------------
# Motor assignments (adjust GPIO pins as needed)
# Motor 1: forward = 13, reverse = 6
# Motor 2: forward = 26, reverse = 19
# Motor 3: forward = 16, reverse = 12
# Motor 4: forward = 20, reverse = 21 (or inverted if required)
#-------------------------------------------------------------------------
motor1 = Motor(pin_forward=13,  pin_reverse=6)
motor2 = Motor(pin_forward=26,  pin_reverse=19)
motor3 = Motor(pin_forward=16,  pin_reverse=12)
motor4 = Motor(pin_forward=20,  pin_reverse=21)
drive = DifferentialDrive(left_motors=[motor1, motor2],
                          right_motors=[motor3, motor4])

#------------------------------
# Camera and ArUco Setup
#------------------------------
print("[DEBUG] Initializing Picamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration({"format": "XRGB8888", "size": (640, 480)})
# config = picam2.create_preview_configuration({"format": "XRGB8888", "size": (1280, 720)})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow the camera to warm up

picam2.set_controls({
    "AeEnable": True,       # Enable auto-exposure
    "AwbEnable": True,      # Enable auto white-balance
    # "ExposureTime": 20000,  # Manually set exposure time (microseconds)
    # "AnalogueGain": 2.0,    # Manually set gain/ISO
    "Brightness": 0.5,      # Range often -1.0 to 1.0 (implementation-dependent)
    "Contrast": 1.2,        # Increase contrast slightly
    "Saturation": 1.2,      # Increase saturation slightly
    "Sharpness": 1.2        # Increase sharpness slightly
})

print("[DEBUG] Picamera2 started.")


# Use the 4x4_100 ArUco dictionary.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Frame dimensions and center.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
center_x = FRAME_WIDTH / 2

# Control parameters.
steering_threshold = 120  # Pixel threshold for pivoting.
drive_speed = 10         # Base forward speed.
# (Pivot and search speeds will be determined dynamically or set as fallback.)
search_speed = 10
explore = 3        

# Directory to save debug images.
save_dir = "captured_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("[DEBUG] Created directory:", save_dir)

print("[DEBUG] Starting main control loop. Press Ctrl+C to exit.")

#------------------------------
# Main Control Loop
#------------------------------
try:
    while True:
        # Capture a frame.
        frame = picam2.capture_array()
        if frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            print("[DEBUG] Converted frame from BGRA to BGR")
        else:
            frame_bgr = frame
            print("[DEBUG] Frame is in BGR format")
        
        # Convert frame to grayscale.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        print("[DEBUG] Converted frame to grayscale")
        
        # Detect ArUco markers.
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        udp_data = {}  # Prepare UDP message payload.
        
        if ids is not None:
            print(f"[DEBUG] Detected {len(ids)} marker(s)")
            # For simplicity, use the first detected marker.
            marker_corners = corners[0].reshape((4, 2))
            marker_id = int(ids[0])
            marker_center = np.mean(marker_corners, axis=0)
            print(f"[DEBUG] Marker ID {marker_id} center: {marker_center}")
            
            # Calculate horizontal error relative to the image center.
            error = marker_center[0] - center_x
            print(f"[DEBUG] Horizontal error: {error}")
            
            # Dynamic speed regulation:
            if abs(error) > steering_threshold:
                # Dynamically adjust pivot speed based on error magnitude.
                min_pivot_speed = 20
                max_pivot_speed = 50
                pivot_gain = 0.5
                dynamic_pivot_speed = int(max(min_pivot_speed, min(max_pivot_speed, pivot_gain * abs(error))))
                print(f"[DEBUG] Dynamic pivot speed: {dynamic_pivot_speed}")
                if error > 0:
                    print("[DEBUG] Marker is to the right. Pivoting right.")
                    drive.pivot_right(speed=dynamic_pivot_speed)
                else:
                    print("[DEBUG] Marker is to the left. Pivoting left.")
                    drive.pivot_left(speed=dynamic_pivot_speed)
            else:
                # If marker is centered, drive forward at a preset speed.
                dynamic_forward_speed = drive_speed
                print(f"[DEBUG] Marker is centered. Driving forward at speed: {dynamic_forward_speed}")
                drive.forward(speed=dynamic_forward_speed)
            
            # Save image with marker outline.
            image_with_marker = aruco.drawDetectedMarkers(frame_bgr.copy(), corners, ids)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(save_dir, f"frame_{timestamp}_{marker_id}.jpg")
            cv2.imwrite(filename, image_with_marker)
            print(f"[DEBUG] Saved image with marker as: {filename}")
            
            # Build UDP data payload.
            udp_data = {
                "markers": [
                    {
                        "id": marker_id,
                        "corners": marker_corners.tolist(),
                        "center": marker_center.tolist(),
                        "error": float(error)
                    }
                ]
            }
        else:
            print("[DEBUG] No marker detected. Searching...")
            # Optionally, you could set a search maneuver here.
            
            if explore < 6:
                drive.pivot_right(speed=search_speed)
                explore += 1
            # elif explore > 6 & explore < 12:
            elif explore < 12:
                drive.pivot_left(speed=search_speed)
                explore += 1
            else:
                explore = 0

            print(F"[DEBUG] Exploration step: {explore}")

            time.sleep(0.05)
                
            udp_data = {"markers": []}  # No markers detected.
        
        # Send UDP message.
        json_data = json.dumps(udp_data)
        sock.sendto(json_data.encode('utf-8'), (UDP_IP, UDP_PORT))
        print("[DEBUG] Sent UDP packet:", json_data)
        
        # Allow drive command to run briefly.
        time.sleep(0.02)
        drive.stop()
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n[DEBUG] Exiting control loop.")

finally:
    picam2.stop()
    drive.stop()
    GPIO.cleanup()
    print("[DEBUG] Shutdown complete.")
