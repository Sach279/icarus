#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

# Set up GPIO mode and disable warnings.
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor:
    """
    Motor control class with optional inversion.
    Each motor is controlled via two GPIO pins (one for forward, one for reverse).
    If inverted=True, forward() and reverse() are swapped.
    """
    def __init__(self, pin_forward, pin_reverse, use_pwm=False, pwm_freq=1000, inverted=False):
        self.pin_forward = pin_forward
        self.pin_reverse = pin_reverse
        self.use_pwm = use_pwm
        self.inverted = inverted

        # Initialize the GPIO pins as outputs.
        GPIO.setup(self.pin_forward, GPIO.OUT)
        GPIO.setup(self.pin_reverse, GPIO.OUT)

        if self.use_pwm:
            self.pwm_forward = GPIO.PWM(self.pin_forward, pwm_freq)
            self.pwm_reverse = GPIO.PWM(self.pin_reverse, pwm_freq)
            self.pwm_forward.start(0)
            self.pwm_reverse.start(0)

    def forward(self, speed=100):
        """Run motor in its 'forward' direction (accounting for inversion)."""
        if self.inverted:
            self._reverse_logic(speed)
        else:
            self._forward_logic(speed)

    def reverse(self, speed=100):
        """Run motor in its 'reverse' direction (accounting for inversion)."""
        if self.inverted:
            self._forward_logic(speed)
        else:
            self._reverse_logic(speed)

    def _forward_logic(self, speed):
        """The non-inverted forward logic."""
        if self.use_pwm:
            self.pwm_forward.ChangeDutyCycle(speed)
            self.pwm_reverse.ChangeDutyCycle(0)
        else:
            GPIO.output(self.pin_forward, GPIO.HIGH)
            GPIO.output(self.pin_reverse, GPIO.LOW)

    def _reverse_logic(self, speed):
        """The non-inverted reverse logic."""
        if self.use_pwm:
            self.pwm_forward.ChangeDutyCycle(0)
            self.pwm_reverse.ChangeDutyCycle(speed)
        else:
            GPIO.output(self.pin_forward, GPIO.LOW)
            GPIO.output(self.pin_reverse, GPIO.HIGH)

    def stop(self):
        """Stop the motor."""
        if self.use_pwm:
            self.pwm_forward.ChangeDutyCycle(0)
            self.pwm_reverse.ChangeDutyCycle(0)
        else:
            GPIO.output(self.pin_forward, GPIO.LOW)
            GPIO.output(self.pin_reverse, GPIO.LOW)

class DifferentialDrive:
    """
    Differential drive for a robot with two tracks.
    left_motors: List of motors on the left track.
    right_motors: List of motors on the right track.
    """
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

    def pivot_left(self, speed=100):
        """
        Pivot in place to the left:
          - Left track runs in reverse.
          - Right track runs forward.
        """
        for motor in self.left_motors:
            motor.reverse(speed)
        for motor in self.right_motors:
            motor.forward(speed)

    def pivot_right(self, speed=100):
        """
        Pivot in place to the right:
          - Left track runs forward.
          - Right track runs in reverse.
        """
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
# Motor assignments with your specific GPIO pins:
# Motor 1: forward = 6,  reverse = 13
# Motor 2: forward = 19, reverse = 26
# Motor 3: forward = 12, reverse = 16
# Motor 4: forward = 20, reverse = 21 (inverted logic)
#-------------------------------------------------------------------------
motor1 = Motor(pin_forward=13,  pin_reverse=6)
motor2 = Motor(pin_forward=26, pin_reverse=19)
motor3 = Motor(pin_forward=12, pin_reverse=16)
motor4 = Motor(pin_forward=20, pin_reverse=21, inverted=True)

# Group the motors into tracks:
# Left track: Motor 1 and Motor 2.
# Right track: Motor 3 and Motor 4.
drive = DifferentialDrive(left_motors=[motor1, motor2],
                          right_motors=[motor3, motor4])

def print_instructions():
    print("\nEnter a command followed by a duration in seconds.")
    print("Commands:")
    print("  f <seconds>  : Move forward")
    print("  b <seconds>  : Move backward (reverse)")
    print("  l <seconds>  : Pivot left")
    print("  r <seconds>  : Pivot right")
    print("  s            : Stop immediately")
    print("  q            : Quit\n")

def main():
    print_instructions()
    while True:
        try:
            cmd = input("Command: ").strip().lower()
            if not cmd:
                continue
            tokens = cmd.split()
            action = tokens[0]

            if action == 'q':
                print("Quitting...")
                drive.stop()
                break
            elif action == 's':
                print("Stopping motors.")
                drive.stop()
                continue

            # For commands requiring a duration, ensure one was provided.
            if len(tokens) < 2:
                print("Please provide a duration in seconds.")
                continue

            try:
                duration = float(tokens[1])
            except ValueError:
                print("Invalid duration. Please enter a number.")
                continue

            # Execute the command.
            if action == 'f':
                print(f"Moving forward for {duration} seconds...")
                drive.forward()
            elif action == 'b':
                print(f"Moving backward for {duration} seconds...")
                drive.reverse()
            elif action == 'l':
                print(f"Pivoting left for {duration} seconds...")
                drive.pivot_left()
            elif action == 'r':
                print(f"Pivoting right for {duration} seconds...")
                drive.pivot_right()
            else:
                print("Unknown command. Try again.")
                continue

            # Run the command for the specified duration.
            time.sleep(duration)
            drive.stop()

        except KeyboardInterrupt:
            print("\nOperation interrupted by user.")
            break

if __name__ == "__main__":
    try:
        main()
    finally:
        GPIO.cleanup()