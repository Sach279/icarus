--- START OF FILE 4_follower_advanced_pid.py ---

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
import math # For calculations if needed

#------------------------------
# UDP Socket Setup
#------------------------------
UDP_IP = "192.168.230.44"  # Replace with your laptop's IP address
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#------------------------------
# Camera Calibration Parameters
#------------------------------
# !!! IMPORTANT: Replace these with your actual camera calibration results !!!
# You need to perform a separate camera calibration process (e.g., using a chessboard)
# with your specific Picamera2 module to get these values.
# Save them typically as a .npz file and load them here.
CALIB_DATA_PATH = "camera_calibration.npz" # e.g., from a previous calibration script

camera_matrix = None
dist_coeffs = None

try:
    calib_data = np.load(CALIB_DATA_PATH)
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]
    print(f"[INFO] Loaded camera calibration data from {CALIB_DATA_PATH}")
except FileNotFoundError:
    print(f"[ERROR] Camera calibration file not found at {CALIB_DATA_PATH}")
    print("[ERROR] Pose estimation will NOT work without calibration data.")
    print("[ERROR] Please run a calibration script and save the results to this location.")
    # Decide if you want to exit or try to run without pose (not recommended for this script)
    # import sys
    # sys.exit(1)
    # Continue running, but state will remain 'calibration_missing' or 'searching' without pose

#------------------------------
# Marker Parameters
#------------------------------
MARKER_SIZE = 0.05 # Size of the ArUco marker side in meters (e.g., 5 cm)
# !!! IMPORTANT: Replace this with the actual size of your physical markers !!!

TARGET_MARKER_ID = 42 # The specific ID of the marker you want to follow/dock to - CHANGE THIS

#------------------------------
# PID Controller Class
#------------------------------
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(-100, 100), integral_limits=None, derivative_filter_alpha=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral_limits = integral_limits or (-float('inf'), float('inf')) # Default no limits
        self.derivative_filter_alpha = derivative_filter_alpha # EMA filter coefficient (0 to 1)

        self._integral = 0
        self._previous_value = None # Used for derivative calculation from measurement
        self._filtered_derivative = 0 # Used for EMA filter

        self._last_time = None
        self.last_output = 0.0

    def update(self, current_value, current_time=None):
        if current_time is None:
            current_time = time.time()

        # If this is the first update or time has reset, initialize
        if self._last_time is None or current_time <= self._last_time:
            self._last_time = current_time
            self._previous_value = current_value
            # self._filtered_derivative remains 0
            return 0.0 # No output on first valid update

        dt = current_time - self._last_time

        error = self.setpoint - current_value

        # Proportional term
        P_term = self.Kp * error

        # Integral term (with anti-windup via clamping)
        self._integral += error * dt
        if self.integral_limits is not None:
             self._integral = np.clip(self._integral, self.integral_limits[0], self.integral_limits[1])

        I_term = self.Ki * self._integral

        # Derivative term (calculate from measurement, filter)
        derivative = (current_value - self._previous_value) / dt
        if self.derivative_filter_alpha is not None and 0 < self.derivative_filter_alpha < 1:
            self._filtered_derivative = (self.derivative_filter_alpha * derivative) + ((1 - self.derivative_filter_alpha) * self._filtered_derivative)
            derivative_to_use = self._filtered_derivative
        else:
            # No filter or filter coefficient is 0 or 1
            if self.derivative_filter_alpha == 0: # Alpha 0 means immediate filter = current derivative
                 derivative_to_use = derivative
            elif self.derivative_filter_alpha is None or self.derivative_filter_alpha >= 1: # Alpha >= 1 or None means no filtering
                 derivative_to_use = derivative
            else: # Should not happen with checks above, but as fallback
                 derivative_to_use = derivative

        D_term = self.Kd * derivative_to_use

        # Combine terms
        output = P_term + I_term + D_term

        # Clamp output
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        self.last_output = output

        # Update state for next iteration
        # self._previous_error = error # Not used for derivative calc anymore
        self._previous_value = current_value
        self._last_time = current_time

        return output

    def reset(self):
        """Resets the integral and previous error/value terms."""
        self._integral = 0
        # self._previous_error = 0
        self._previous_value = None # Resetting this prevents derivative kick on re-acquisition
        self._filtered_derivative = 0
        self._last_time = None # Reset time to avoid large dt on first update after reset
        self.last_output = 0.0

#------------------------------
# Motor & Drive Unit Definitions
#------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor:
    def __init__(self, pin_forward, pin_reverse, use_pwm=True, pwm_freq=1000, inverted=False):
        self.pin_forward = pin_forward
        self.pin_reverse = pin_reverse
        self.use_pwm = use_pwm
        self.inverted = inverted
        GPIO.setup(self.pin_forward, GPIO.OUT)
        GPIO.setup(self.pin_reverse, GPIO.OUT)
        if self.use_pwm:
            # Ensure pins support PWM if use_pwm is True
            try:
                self.pwm_forward = GPIO.PWM(self.pin_forward, pwm_freq)
                self.pwm_reverse = GPIO.PWM(self.pin_reverse, pwm_freq)
                self.pwm_forward.start(0)
                self.pwm_reverse.start(0)
            except ValueError as e:
                 print(f"[ERROR] Pin {e} likely does not support hardware PWM. Check your pins or set use_pwm=False.")
                 # Fallback to non-PWM if PWM setup fails
                 self.use_pwm = False
                 print("[INFO] Falling back to non-PWM motor control.")


    def forward(self, speed=100):
        speed = max(0, min(100, speed)) # Clamp speed [0, 100]
        if self.inverted:
            self._reverse_logic(speed)
        else:
            self._forward_logic(speed)

    def reverse(self, speed=100):
        speed = max(0, min(100, speed)) # Clamp speed [0, 100]
        if self.inverted:
            self._forward_logic(speed)
        else:
            self._reverse_logic(speed)

    def _forward_logic(self, speed):
        if self.use_pwm:
            self.pwm_forward.ChangeDutyCycle(speed)
            self.pwm_reverse.ChangeDutyCycle(0)
        else:
             # If not using PWM, speed > 0 means HIGH, speed == 0 means LOW
            GPIO.output(self.pin_forward, GPIO.HIGH if speed > 0 else GPIO.LOW)
            GPIO.output(self.pin_reverse, GPIO.LOW)

    def _reverse_logic(self, speed):
        if self.use_pwm:
            self.pwm_forward.ChangeDutyCycle(0)
            self.pwm_reverse.ChangeDutyCycle(speed)
        else:
            # If not using PWM, speed > 0 means HIGH, speed == 0 means LOW
            GPIO.output(self.pin_forward, GPIO.LOW)
            GPIO.output(self.pin_reverse, GPIO.HIGH if speed > 0 else GPIO.LOW)

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

    def set_speeds(self, left_speed, right_speed):
        # Ensure speeds are within [-100, 100] range and apply them
        # Note: Individual motor forward/reverse methods handle [0, 100] clamping
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))

        # Apply speeds to left motors
        if left_speed > 0:
            for motor in self.left_motors:
                motor.forward(left_speed)
        elif left_speed < 0:
            for motor in self.left_motors:
                motor.reverse(abs(left_speed))
        else:
            for motor in self.left_motors:
                motor.stop()

        # Apply speeds to right motors
        if right_speed > 0:
            for motor in self.right_motors:
                motor.forward(right_speed)
        elif right_speed < 0:
            for motor in self.right_motors:
                motor.reverse(abs(right_speed))
        else:
            for motor in self.right_motors:
                motor.stop()

    def stop(self):
        self.set_speeds(0, 0)


#-------------------------------------------------------------------------
# Motor assignments (adjust GPIO pins as needed)
# Motor 1: forward = 13, reverse = 6
# Motor 2: forward = 26, reverse = 19
# Motor 3: forward = 16, reverse = 12
# Motor 4: forward = 20, reverse = 21 (or inverted if required)
#
# PWM is HIGHLY RECOMMENDED for smoother PID control.
# Ensure your chosen pins support hardware PWM or use software PWM (less precise).
# The default use_pwm=True in Motor class constructor enables PWM.
#-------------------------------------------------------------------------
motor1 = Motor(pin_forward=13,  pin_reverse=6, use_pwm=True)
motor2 = Motor(pin_forward=26,  pin_reverse=19, use_pwm=True)
motor3 = Motor(pin_forward=16,  pin_reverse=12, use_pwm=True)
motor4 = Motor(pin_forward=20,  pin_reverse=21, use_pwm=True) # Adjust inversion if needed

drive = DifferentialDrive(left_motors=[motor1, motor2],
                          right_motors=[motor3, motor4])

#------------------------------
# Camera and ArUco Setup
#------------------------------
print("[INFO] Initializing Picamera2...")
picam2 = Picamera2()
# Use a resolution that balances performance and accuracy. 640x480 is often best for Pi Zero 2W.
config = picam2.create_preview_configuration({"format": "XRGB8888", "size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)  # Allow the camera to warm up and auto-exposure to settle

# Adjust camera controls - these affect image quality and can influence detection
# Auto-exposure and auto-white balance are usually best.
picam2.set_controls({
    "AeEnable": True,       # Enable auto-exposure
    "AwbEnable": True,      # Enable auto white-balance
    "Brightness": 0.5,
    "Contrast": 1.2,
    "Saturation": 1.2,
    "Sharpness": 1.2
})

print("[INFO] Picamera2 started.")

# Use the 4x4_100 ArUco dictionary.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4x4_100)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX # Refine corners for better accuracy (moderate cost)
# parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR # Another option, slightly faster

#------------------------------
# Control & State Machine Parameters
#------------------------------
TARGET_DISTANCE = 0.6  # Target distance to the marker in meters (e.g., 0.6m for docking) - TUNE THIS

# PID Gains (THESE NEED CAREFUL TUNING FOR YOUR ROBOT)
# Tune Kp first (Ki=0, Kd=0), then add Kd, then add Ki.
# Gains for Distance (Z axis error)
KP_DISTANCE = 25.0   # Proportional gain
KI_DISTANCE = 0.2    # Integral gain
KD_DISTANCE = 1.5    # Derivative gain

# Gains for Yaw (Rotation around vertical axis - rvec[1] is used)
KP_YAW = 45.0        # Proportional gain
KI_YAW = 0.8         # Integral gain
KD_YAW = 2.5         # Derivative gain

# PID Output Limits
MAX_FORWARD_SPEED_COMPONENT = 50 # Max speed output from distance PID [-50, 50] - TUNE THIS
MAX_TURN_RATE_COMPONENT = 40     # Max speed output from yaw PID [-40, 40] - TUNE THIS
# Final motor speeds (L/R) will be combination, potentially exceeding 100 before clamping

# Integral Windup Limits
INTEGRAL_LIMIT_DISTANCE = 8.0 # Limits integral term for distance PID
INTEGRAL_LIMIT_YAW = 4.0      # Limits integral term for yaw PID

# Derivative Filter (Exponential Moving Average - EMA)
# Alpha closer to 1 means less filtering (more responsive but sensitive to noise).
# Alpha closer to 0 means more filtering (smoother but adds latency).
# Try values like 0.5 to 0.9. None or <=0 means no filtering.
DERIVATIVE_FILTER_ALPHA = 0.7 # TUNE THIS or set to None

# Deadbands (errors within this range are considered zero for PID input & AtTarget check)
DISTANCE_DEADBAND = 0.08     # Meters +/- from TARGET_DISTANCE - TUNE THIS
YAW_DEADBAND_RVECCMP = 0.05  # rvec[1] value +/- from 0.0 - TUNE THIS

# Minimum Speed Thresholds (to overcome static friction)
# If the calculated PID output component is non-zero but below this, use this minimum speed.
MIN_MOVE_SPEED_LINEAR = 10 # Minimum speed for the forward/backward component - TUNE THIS
MIN_MOVE_SPEED_ANGULAR = 10 # Minimum speed for the turning component - TUNE THIS


# State Machine Parameters
LOST_TIMEOUT = 0.5 # Seconds marker must be lost before transitioning from COASTING to SEARCHING - TUNE THIS
SEARCH_PIVOT_SPEED = 20 # Speed for the search pattern - TUNE THIS
SEARCH_DURATION = 0.7   # Duration of each pivot step before reversing direction - TUNE THIS
COASTING_SPEED_PERCENT = 0 # Percent of last forward speed to maintain when coasting (0 = stop)

# Criteria for entering AT_TARGET state
# Robot must be within both deadbands AND PID outputs must be very low
AT_TARGET_PID_THRESHOLD = 5 # Max absolute value of PID output components to be considered 'at target'

# Directory to save debug images.
save_dir = "captured_frames_pid"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("[INFO] Created directory:", save_dir)

# Initialize PID controllers
# Distance PID: setpoint is TARGET_DISTANCE (meters)
distance_pid = PIDController(
    Kp=KP_DISTANCE, Ki=KI_DISTANCE, Kd=KD_DISTANCE,
    setpoint=TARGET_DISTANCE,
    output_limits=(-MAX_FORWARD_SPEED_COMPONENT, MAX_FORWARD_SPEED_COMPONENT), # Output is forward speed component
    integral_limits=(-INTEGRAL_LIMIT_DISTANCE, INTEGRAL_LIMIT_DISTANCE),
    derivative_filter_alpha=DERIVATIVE_FILTER_ALPHA
)

# Yaw PID: setpoint is 0.0 (want rvec[1] component to be zero)
yaw_pid = PIDController(
    Kp=KP_YAW, Ki=KI_YAW, Kd=KD_YAW,
    setpoint=0.0,
    output_limits=(-MAX_TURN_RATE_COMPONENT, MAX_TURN_RATE_COMPONENT), # Output is turn rate component
    integral_limits=(-INTEGRAL_LIMIT_YAW, INTEGRAL_LIMIT_YAW),
    derivative_filter_alpha=DERIVATIVE_FILTER_ALPHA
)

print("[INFO] PID controllers initialized.")

# State Machine Variables
STATE_SEARCHING = "SEARCHING"
STATE_LOST_COASTING = "LOST_COASTING"
STATE_TRACKING = "TRACKING"
STATE_AT_TARGET = "AT_TARGET"
STATE_CALIBRATION_MISSING = "CALIBRATION_MISSING" # Added state

current_state = STATE_SEARCHING if camera_matrix is not None and dist_coeffs is not None else STATE_CALIBRATION_MISSING
last_state_change_time = time.time()

time_marker_lost = None # Timestamp when marker was first lost
last_known_tvec = None
last_known_rvec = None
last_tracking_forward_speed = 0 # For coasting
last_tracking_turn_rate = 0 # For coasting

search_direction = 1 # 1 for right, -1 for left
last_search_pivot_time = time.time()

prev_loop_time = time.time() # Initialize time for dt calculation

print(f"[INFO] Starting main control loop in state: {current_state}. Press Ctrl+C to exit.")

#------------------------------
# Main Control Loop
#------------------------------
try:
    while True:
        current_loop_time = time.time()
        dt = current_loop_time - prev_loop_time
        prev_loop_time = current_loop_time # Update for next iteration

        # Only capture/process if not in calibration missing state (if we chose not to exit)
        if current_state != STATE_CALIBRATION_MISSING:
            # Capture a frame.
            frame = picam2.capture_array()
            if frame.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = frame

            # Convert frame to grayscale for detection.
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers.
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            image_display = frame_bgr.copy() # Image to draw on

            target_marker_found_this_frame = False
            target_tvec = None
            target_rvec = None
            target_marker_corners = None
            target_marker_center_pixel = None

            # --- Marker Detection and Pose Estimation ---
            if ids is not None:
                 # Look for the specific target marker ID
                 target_index = -1
                 for i, marker_id in enumerate(ids):
                     if marker_id[0] == TARGET_MARKER_ID:
                         target_index = i
                         break # Found the target marker

                 if target_index != -1:
                     target_marker_found_this_frame = True
                     # We have calibration data and found the target, perform pose estimation
                     # Estimate pose *only* for the target marker's corners
                     target_corners_arr = np.array([corners[target_index]])
                     rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(target_corners_arr, MARKER_SIZE, camera_matrix, dist_coeffs)

                     if tvecs is not None and rvecs is not None:
                         target_tvec = tvecs[0][0] # Get the translation vector
                         target_rvec = rvecs[0][0] # Get the rotation vector
                         target_marker_corners = target_corners_arr[0].reshape((4, 2)) # Get corners for UDP payload
                         target_marker_center_pixel = np.mean(target_marker_corners, axis=0) # Get pixel center for UDP payload

                         # Store last known valid pose
                         last_known_tvec = target_tvec
                         last_known_rvec = target_rvec

                         # --- Visualization ---
                         # Draw ALL detected markers (for debugging other markers)
                         image_display = aruco.drawDetectedMarkers(image_display, corners, ids)
                         # Draw axis ONLY for the tracked marker
                         image_display = cv2.drawFrameAxes(image_display, camera_matrix, dist_coeffs, target_rvec, target_tvec, length=MARKER_SIZE * 2)

                         # Reset lost timer
                         time_marker_lost = None
                     else:
                         # Pose estimation failed even if ID was found? (Shouldn't happen often)
                         print("[WARNING] Pose estimation failed for target marker.")


            # --- State Machine Transitions ---
            next_state = current_state

            if current_state == STATE_SEARCHING:
                if target_marker_found_this_frame:
                    print(f"[INFO] Transition: SEARCHING -> TRACKING. Target {TARGET_MARKER_ID} found.")
                    next_state = STATE_TRACKING
                    distance_pid.reset() # Reset PIDs on re-acquisition
                    yaw_pid.reset()
                # No state change if still searching

            elif current_state == STATE_LOST_COASTING:
                 if target_marker_found_this_frame:
                    print(f"[INFO] Transition: LOST_COASTING -> TRACKING. Target {TARGET_MARKER_ID} re-acquired.")
                    next_state = STATE_TRACKING
                    distance_pid.reset() # Reset PIDs on re-acquisition
                    yaw_pid.reset()
                 elif time_marker_lost is not None and (current_loop_time - time_marker_lost) > LOST_TIMEOUT:
                    print(f"[INFO] Transition: LOST_COASTING -> SEARCHING. Timeout ({LOST_TIMEOUT}s) reached.")
                    next_state = STATE_SEARCHING
                    time_marker_lost = None # Reset timer

            elif current_state == STATE_TRACKING:
                 if not target_marker_found_this_frame:
                    print(f"[INFO] Transition: TRACKING -> LOST_COASTING. Target {TARGET_MARKER_ID} lost.")
                    next_state = STATE_LOST_COASTING
                    time_marker_lost = current_loop_time # Start lost timer
                    # Store last command for coasting
                    # last_tracking_forward_speed # Need to calculate this based on last PID outputs
                    # last_tracking_turn_rate
                    # Or simply stop/slow down slightly in the COASTING state logic

                 # Check for AT_TARGET condition (only transition from TRACKING)
                 # Needs last_known_tvec and last_known_rvec to be valid
                 elif last_known_tvec is not None and last_known_rvec is not None:
                     dist_error = last_known_tvec[2] - TARGET_DISTANCE
                     yaw_error = last_known_rvec[1] # Using rvec[1] as yaw error signal

                     is_at_distance = abs(dist_error) < DISTANCE_DEADBAND
                     is_at_yaw = abs(yaw_error) < YAW_DEADBAND_RVECCMP

                     # Check if PID outputs are also near zero, indicating convergence
                     pid_output_linear_low = abs(distance_pid.last_output) < AT_TARGET_PID_THRESHOLD
                     pid_output_angular_low = abs(yaw_pid.last_output) < AT_TARGET_PID_THRESHOLD


                     if is_at_distance and is_at_yaw and pid_output_linear_low and pid_output_angular_low:
                          print(f"[INFO] Transition: TRACKING -> AT_TARGET.")
                          next_state = STATE_AT_TARGET
                          drive.stop() # Ensure stop immediately on entering AT_TARGET
                          # Reset PIDs? Maybe not, keep current state for exiting AT_TARGET.
                          # distance_pid.reset()
                          # yaw_pid.reset()


            elif current_state == STATE_AT_TARGET:
                 # If marker is lost while at target, transition to COASTING
                 if not target_marker_found_this_frame:
                     print(f"[INFO] Transition: AT_TARGET -> LOST_COASTING. Target {TARGET_MARKER_ID} lost.")
                     next_state = STATE_LOST_COASTING
                     time_marker_lost = current_loop_time # Start lost timer
                 # If marker is found, but conditions are no longer met, go back to TRACKING
                 elif last_known_tvec is not None and last_known_rvec is not None:
                     dist_error = last_known_tvec[2] - TARGET_DISTANCE
                     yaw_error = last_known_rvec[1]

                     is_at_distance = abs(dist_error) < DISTANCE_DEADBAND
                     is_at_yaw = abs(yaw_error) < YAW_DEADBAND_RVECCMP

                     # Re-calculate PID outputs based on current pose to see if they are low
                     # Note: PID update state only happens in TRACKING state, so we need a temp calc or check output history
                     # Checking last_output is simpler:
                     pid_output_linear_low = abs(distance_pid.last_output) < AT_TARGET_PID_THRESHOLD
                     pid_output_angular_low = abs(yaw_pid.last_output) < AT_TARGET_PID_THRESHOLD

                     if not (is_at_distance and is_at_yaw and pid_output_linear_low and pid_output_angular_low):
                         print(f"[INFO] Transition: AT_TARGET -> TRACKING. Conditions no longer met.")
                         next_state = STATE_TRACKING
                         # Don't reset PIDs, continue tracking smoothly


            # Update current state
            if next_state != current_state:
                 current_state = next_state
                 last_state_change_time = current_loop_time # Record time of transition
                 # Ensure motors are stopped on state change unless new state commands them
                 # (Except for TRACKING which immediately calculates command)
                 if current_state in [STATE_SEARCHING, STATE_LOST_COASTING, STATE_AT_TARGET]:
                      drive.stop()


            # --- State Actions ---
            left_motor_speed = 0.0
            right_motor_speed = 0.0
            control_status = "stopped" # For UDP payload


            if current_state == STATE_CALIBRATION_MISSING:
                # Robot stays stopped, prints error messages
                control_status = "calibration_missing"
                drive.stop()


            elif current_state == STATE_SEARCHING:
                 control_status = "searching"
                 # Execute search pattern: pivot left/right
                 if (current_loop_time - last_search_pivot_time) > SEARCH_DURATION:
                     search_direction *= -1 # Reverse direction
                     last_search_pivot_time = current_loop_time
                     print(f"[DEBUG] Search direction: {'Right' if search_direction > 0 else 'Left'}")

                 if search_direction > 0:
                     drive.pivot_right(speed=SEARCH_PIVOT_SPEED)
                     left_motor_speed = SEARCH_PIVOT_SPEED
                     right_motor_speed = -SEARCH_PIVOT_SPEED
                 else:
                     drive.pivot_left(speed=SEARCH_PIVOT_SPEED)
                     left_motor_speed = -SEARCH_PIVOT_SPEED
                     right_motor_speed = SEARCH_PIVOT_SPEED


            elif current_state == STATE_LOST_COASTING:
                 control_status = "lost_coasting"
                 # Option 1: Just stop (safest)
                 drive.stop()
                 left_motor_speed = 0
                 right_motor_speed = 0

                 # Option 2: Apply a small coasting speed based on last tracking command
                 # if abs(last_tracking_forward_speed) > 0 or abs(last_tracking_turn_rate) > 0:
                 #     coast_forward = last_tracking_forward_speed * (COASTING_SPEED_PERCENT / 100.0)
                 #     coast_turn = last_tracking_turn_rate * (COASTING_SPEED_PERCENT / 100.0)
                 #     left_motor_speed = coast_forward - coast_turn
                 #     right_motor_speed = coast_forward + coast_turn
                 #     drive.set_speeds(left_motor_speed, right_motor_speed)
                 # else:
                 #     drive.stop() # Stop if last command was stop


            elif current_state == STATE_TRACKING:
                 control_status = "tracking"
                 # PID control based on last known valid pose data
                 if last_known_tvec is not None and last_known_rvec is not None:

                     current_distance = last_known_tvec[2]
                     current_yaw_rvec = last_known_rvec[1] # Use rvec[1] for yaw error

                     # Apply deadbands to inputs for PID calculation
                     distance_input_for_pid = current_distance
                     if abs(current_distance - TARGET_DISTANCE) < DISTANCE_DEADBAND:
                         distance_input_for_pid = TARGET_DISTANCE # Treat as target distance within deadband
                         # print("[DEBUG] Distance within deadband.")


                     yaw_input_for_pid = current_yaw_rvec
                     if abs(current_yaw_rvec) < YAW_DEADBAND_RVECCMP:
                         yaw_input_for_pid = 0.0 # Treat as zero yaw error within deadband
                         # print("[DEBUG] Yaw within deadband.")


                     # Update PID controllers with current values
                     forward_speed_component = distance_pid.update(distance_input_for_pid, current_loop_time)
                     turn_rate_component = yaw_pid.update(yaw_input_for_pid, current_loop_time)

                     # Apply minimum speeds *if* outside the respective deadbands
                     final_forward_speed = forward_speed_component
                     if abs(current_distance - TARGET_DISTANCE) >= DISTANCE_DEADBAND: # If we *need* to move forward/backward
                         if final_forward_speed > 0 and final_forward_speed < MIN_MOVE_SPEED_LINEAR:
                             final_forward_speed = MIN_MOVE_SPEED_LINEAR
                         elif final_forward_speed < 0 and final_forward_speed > -MIN_MOVE_SPEED_LINEAR:
                             final_forward_speed = -MIN_MOVE_SPEED_LINEAR
                     else: # If inside deadband, ensure speed is truly zero unless PID output non-zero
                          # If within deadband, PID output should be zero *unless* I term is non-zero.
                          # We want it to stop if error is low *and* PID is settled.
                          # The AT_TARGET state handles the converged stop.
                          # Here, if inside deadband, let PID output drive motors, min speed only applies outside deadband
                          pass # Let the PID output handle it based on distance_input_for_pid

                     final_turn_rate = turn_rate_component
                     if abs(current_yaw_rvec) >= YAW_DEADBAND_RVECCMP: # If we *need* to turn
                          if final_turn_rate > 0 and final_turn_rate < MIN_MOVE_SPEED_ANGULAR:
                               final_turn_rate = MIN_MOVE_SPEED_ANGULAR
                          elif final_turn_rate < 0 and final_turn_rate > -MIN_MOVE_SPEED_ANGULAR:
                              final_turn_rate = -MIN_MOVE_SPEED_ANGULAR
                     else: # If inside deadband, let PID output handle it based on yaw_input_for_pid
                          pass # Let PID output handle it

                     # Combine components for differential drive
                     left_motor_speed = final_forward_speed - final_turn_rate
                     right_motor_speed = final_forward_speed + final_turn_rate

                     # Clamp final motor speeds between -100 and 100
                     left_motor_speed = np.clip(left_motor_speed, -100, 100)
                     right_motor_speed = np.clip(right_motor_speed, -100, 100)

                     drive.set_speeds(left_motor_speed, right_motor_speed)

                     # Store speeds for potential coasting if lost
                     last_tracking_forward_speed = final_forward_speed # Store calculated components, not final L/R
                     last_tracking_turn_rate = final_turn_rate

                     # Debug prints
                     print(f"[DEBUG] State:{current_state}. Dist: {current_distance:.2f}m (Err: {current_distance-TARGET_DISTANCE:.2f}), Yaw_rvec: {current_yaw_rvec:.2f}. PID(L:{forward_speed_component:.1f}, A:{turn_rate_component:.1f}). Motors(L, R): ({left_motor_speed:.1f}, {right_motor_speed:.1f})")

                 else:
                     # Should ideally transition to LOST_COASTING here if last_known is None
                     # but the transition logic above should handle this.
                     # As a fallback, stop motors if somehow in TRACKING without pose data.
                     drive.stop()
                     left_motor_speed = 0
                     right_motor_speed = 0
                     control_status = "tracking_no_pose"
                     print("[WARNING] In TRACKING state but last_known_pose is None. Stopping.")


            elif current_state == STATE_AT_TARGET:
                 control_status = "at_target"
                 # Robot is intentionally stopped, waiting
                 drive.stop()
                 left_motor_speed = 0
                 right_motor_speed = 0
                 # PID controllers are NOT updated, but we might check their last_output for exiting


            # --- UDP Data ---
            udp_data = {
                "state": current_state,
                "target_id": TARGET_MARKER_ID,
                "control_status": control_status,
                "timestamp": current_loop_time,
                "pose": None, # Will fill if target detected
                "control_output": { # Reflects the command sent to motors
                    "left_motor_speed": float(left_motor_speed),
                    "right_motor_speed": float(right_motor_speed)
                },
                "pid_status": { # Detailed PID info (only meaningful in TRACKING state)
                    "distance": {
                         "Kp": KP_DISTANCE, "Ki": KI_DISTANCE, "Kd": KD_DISTANCE,
                         "setpoint": TARGET_DISTANCE,
                         "current_value": float(last_known_tvec[2]) if last_known_tvec is not None else None,
                         "error": float(last_known_tvec[2] - TARGET_DISTANCE) if last_known_tvec is not None else None,
                         "P": float(distance_pid.Kp * (distance_pid.setpoint - (last_known_tvec[2] if last_known_tvec is not None else distance_pid.setpoint))) if last_known_tvec is not None else 0.0, # Approximate P term
                         "I": float(distance_pid._integral),
                         "D": float(distance_pid._filtered_derivative * distance_pid.Kd), # Using filtered derivative
                         "output": float(distance_pid.last_output)
                    },
                    "yaw": {
                         "Kp": KP_YAW, "Ki": KI_YAW, "Kd": KD_YAW,
                         "setpoint": 0.0,
                         "current_value": float(last_known_rvec[1]) if last_known_rvec is not None else None,
                         "error": float(last_known_rvec[1]) if last_known_rvec is not None else None,
                         "P": float(yaw_pid.Kp * (yaw_pid.setpoint - (last_known_rvec[1] if last_known_rvec is not None else yaw_pid.setpoint))) if last_known_rvec is not None else 0.0, # Approximate P term
                         "I": float(yaw_pid._integral),
                         "D": float(yaw_pid._filtered_derivative * yaw_pid.Kd),
                         "output": float(yaw_pid.last_output)
                    }
                }
            }

            if target_marker_found_this_frame and target_tvec is not None and target_rvec is not None:
                 udp_data["pose"] = {
                     "id": int(TARGET_MARKER_ID),
                     "corners": target_marker_corners.tolist() if target_marker_corners is not None else None,
                     "center_pixel": target_marker_center_pixel.tolist() if target_marker_center_pixel is not None else None,
                     "rvec": target_rvec.tolist(),
                     "tvec": target_tvec.tolist() # Pose data in meters
                 }

            # --- Save Debug Image (optional) ---
            # Save image less often to save CPU/SD card writes
            # Example: save every 20 frames or adjust timing
            # if int(current_loop_time * 10) % 2 == 0: # Save roughly 5 times per second if loop is fast
            if np.random.rand() < 0.1: # Save approximately 10% of frames
                 timestamp = time.strftime("%Y%m%d-%H%M%S")
                 filename_state = current_state.lower()
                 filename = os.path.join(save_dir, f"frame_{timestamp}_{filename_state}_id{TARGET_MARKER_ID}.jpg")

                 try:
                     if image_display is not None:
                         cv2.imwrite(filename, image_display)
                         # print(f"[DEBUG] Saved image: {filename}")
                 except Exception as e:
                     print(f"[ERROR] Could not save image {filename}: {e}")


            # --- Send UDP Message ---
            json_data = json.dumps(udp_data)
            try:
                sock.sendto(json_data.encode('utf-8'), (UDP_IP, UDP_PORT))
                # print("[DEBUG] Sent UDP packet")
            except socket.error as e:
                # Ignore 'Resource temporarily unavailable' (errno 11) which happens when nobody is listening
                if e.errno != 11:
                    print(f"[ERROR] UDP socket error: {e}")

        else: # STATE_CALIBRATION_MISSING
            # Just send a status update periodically
            if (current_loop_time - last_state_change_time) > 1.0: # Send every 1 second
                 udp_data = {
                     "state": current_state,
                     "target_id": TARGET_MARKER_ID,
                     "control_status": "calibration_missing",
                     "timestamp": current_loop_time,
                     "pose": None,
                     "control_output": {"left_motor_speed": 0.0, "right_motor_speed": 0.0},
                     "pid_status": {}
                 }
                 json_data = json.dumps(udp_data)
                 try:
                     sock.sendto(json_data.encode('utf-8'), (UDP_IP, UDP_PORT))
                     # print("[DEBUG] Sent calibration missing UDP packet")
                 except socket.error as e:
                     if e.errno != 11:
                         print(f"[ERROR] UDP socket error: {e}")

                 last_state_change_time = current_loop_time # Update timer


        # Control loop frequency: Limit the loop speed to manage CPU usage.
        # Target a frequency like 10-20 Hz.
        # If frame capture/processing takes longer than desired_dt, the sleep will be skipped.
        desired_dt = 1.0 / 15.0 # Aim for approx 15 Hz
        time_spent_this_loop = time.time() - current_loop_time
        sleep_duration = desired_dt - time_spent_this_loop
        if sleep_duration > 0:
            time.sleep(sleep_duration)
            # print(f"[DEBUG] Loop time: {time_spent_this_loop:.3f}s, Sleeping for {sleep_duration:.3f}s")
        else:
            # print(f"[DEBUG] Loop time exceeded desired_dt: {time_spent_this_loop:.3f}s")
            pass # No sleep needed


except KeyboardInterrupt:
    print("\n[INFO] Exiting control loop due to KeyboardInterrupt.")

except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc() # Print detailed error information

finally:
    print("[INFO] Cleaning up...")
    # Ensure motors and camera are stopped
    try:
        picam2.stop()
        drive.stop()
    except Exception as e:
        print(f"[ERROR] Error during cleanup: {e}")

    # Cleanup GPIO
    try:
        GPIO.cleanup()
    except Exception as e:
         print(f"[ERROR] Error during GPIO cleanup: {e}")

    print("[INFO] Shutdown complete.")

--- END OF FILE 4_follower_advanced_pid.py ---