#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

# Use BCM pin numbering
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor:
    """
    Simple motor control class.
    Each motor is driven by two GPIO pins: one for forward and one for reverse.
    Optionally, PWM can be used for speed control.
    """
    def __init__(self, pin_forward, pin_reverse, use_pwm=False, pwm_freq=1000):
        self.pin_forward = pin_forward
        self.pin_reverse = pin_reverse
        self.use_pwm = use_pwm

        # Setup pins as outputs
        GPIO.setup(self.pin_forward, GPIO.OUT)
        GPIO.setup(self.pin_reverse, GPIO.OUT)

        if self.use_pwm:
            self.pwm_forward = GPIO.PWM(self.pin_forward, pwm_freq)
            self.pwm_reverse = GPIO.PWM(self.pin_reverse, pwm_freq)
            self.pwm_forward.start(0)
            self.pwm_reverse.start(0)

    def forward(self, speed=100):
        """Run the motor forward. For PWM, 'speed' is the duty cycle (0-100)."""
        if self.use_pwm:
            self.pwm_forward.ChangeDutyCycle(speed)
            self.pwm_reverse.ChangeDutyCycle(0)
        else:
            GPIO.output(self.pin_forward, GPIO.HIGH)
            GPIO.output(self.pin_reverse, GPIO.LOW)

    def reverse(self, speed=100):
        """Run the motor in reverse. For PWM, 'speed' is the duty cycle (0-100)."""
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

# Create motor instances with the specified GPIO pins.
motor1 = Motor(pin_forward=6,  pin_reverse=13)
motor2 = Motor(pin_forward=19, pin_reverse=26)
motor3 = Motor(pin_forward=12, pin_reverse=16)
motor4 = Motor(pin_forward=20, pin_reverse=21)

try:
    # Run Motor 1 and Motor 2 concurrently in the forward direction.
    print("Running Motor 1 and Motor 2 forward for 3 seconds...")
    motor1.forward()
    motor2.forward()
    time.sleep(3)
    motor1.stop()
    motor2.stop()
    
    # Pause briefly between operations.
    time.sleep(1)
    
    # Run Motor 3 and Motor 4 concurrently in the reverse direction.
    print("Running Motor 3 and Motor 4 in reverse for 3 seconds...")
    motor3.reverse()
    motor4.reverse()
    time.sleep(3)
    motor3.stop()
    motor4.stop()
    
except KeyboardInterrupt:
    print("Operation interrupted by user.")
    
finally:
    # Cleanup the GPIO settings.
    GPIO.cleanup()