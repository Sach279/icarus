#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os
import RPi.GPIO as GPIO
from picamera2 import Picamera2

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

# Assign motors to GPIO pins.
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
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow camera to warm up
print("[DEBUG] Picamera2 started.")

# Use the 4x4_100 ArUco dictionary.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Frame dimensions and center.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
center_x = FRAME_WIDTH / 2

# Control parameters.
steering_threshold = 30   # Pixel threshold for pivoting.
drive_speed = 70/2          # Speed for forward movement.
pivot_speed = 50/2          # Speed for pivoting.
search_speed = 30/2         # Speed for searching (pivoting) when no marker is seen.

# Create directory to save images for debugging.
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

        # Adjust contrast (and optionally brightness)
        alpha = 1.5  # Increase contrast by 50%
        beta = 0     # No change in brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
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
        if ids is not None:
            print(f"[DEBUG] Detected {len(ids)} marker(s)")
            # Use the first detected marker.
            marker_corners = corners[0].reshape((4, 2))
            marker_id = int(ids[0])
            marker_center = np.mean(marker_corners, axis=0)
            print(f"[DEBUG] Marker ID {marker_id} center: {marker_center}")

            # Determine horizontal error from center.
            error = marker_center[0] - center_x
            print(f"[DEBUG] Horizontal error: {error}")

            # Control logic:
            if abs(error) > steering_threshold:
                if error > 0:
                    print("[DEBUG] Marker is to the right. Pivoting right.")
                    drive.pivot_right(speed=pivot_speed)
                else:
                    print("[DEBUG] Marker is to the left. Pivoting left.")
                    drive.pivot_left(speed=pivot_speed)
            else:
                print("[DEBUG] Marker is centered. Driving forward.")
                drive.forward(speed=drive_speed)

            # Save frame with marker outline for debugging.
            #image_with_marker = aruco.drawDetectedMarkers(frame_bgr.copy(), corners, ids)
            #timestamp = time.strftime("%Y%m%d-%H%M%S")
            #filename = os.path.join(save_dir, f"frame_{timestamp}_{marker_id}.jpg")
            #cv2.imwrite(filename, image_with_marker)
            #print(f"[DEBUG] Saved image with marker as: {filename}")
        else:
            print("[DEBUG] No marker detected. Searching...")
            # If no marker, pivot slowly to search.
            # drive.pivot_right(speed=search_speed)

        # Allow the drive command to run for a short duration.
        time.sleep(0.1)
        drive.stop()
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[DEBUG] Exiting control loop.")

finally:
    picam2.stop()
    drive.stop()
    GPIO.cleanup()
    print("[DEBUG] Shutdown complete.")
