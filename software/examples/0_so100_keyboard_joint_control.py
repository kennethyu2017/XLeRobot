#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
"""

import time
import logging
import traceback
from dataclasses import dataclass
import draccus

from lerobot.robots.so_follower import (SO100Follower,
                                        SO100FollowerConfig)

from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

from software.src.robots.xlerobot_2wheels import XLerobot2WheelsConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Joint calibration coefficients - manually edit
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
JOINT_CALIBRATION = {
    'shoulder_pan': (6.0, 1.0),      # Joint1: zero position offset, scale factor
    'shoulder_lift': (2.0, 0.97),     # Joint2: zero position offset, scale factor
    'elbow_flex': ( 0.0, 1.05),        # Joint3: zero position offset, scale factor
    'wrist_flex': (0.0, 0.94),        # Joint4: zero position offset, scale factor
    'wrist_roll': (0.0, 0.5),        # Joint5: zero position offset, scale factor
    'gripper': (0.0, 1.0),           # Joint6: zero position offset, scale factor
}

def apply_joint_calibration(joint_name, raw_position):
    """
    Apply joint calibration coefficients
    
    Args:
        joint_name: Joint name
        raw_position: Raw position value
    
    Returns:
        calibrated_position: Calibrated position value
    """
    # for joint_cal in JOINT_CALIBRATION:
    #     if joint_cal[0] == joint_name:
    #         offset = joint_cal[1]  # Zero position offset
    #         scale = joint_cal[2]   # Scale factor
    #         calibrated_position = (raw_position - offset) * scale
    #         return calibrated_position

    if joint_name in JOINT_CALIBRATION:
        offset, scale = JOINT_CALIBRATION[joint_name]
        calibrated_position = (raw_position - offset) * scale
        return calibrated_position

    return raw_position  # If calibration coefficients not found, return raw value

# P control calculation
def _p_control_helper(*, robot: SO100Follower,
                      target_pos:dict[str, float],
                      cur_pos:dict[str,float],
                      kp:float,
                      req_total_error:bool = False)->float | None:
    assert target_pos.keys() == cur_pos.keys()
    robot_action = {}
    total_error = 0.0
    for joint_name in target_pos:
        # if joint_name in cur_pos:
        error = target_pos[joint_name] - cur_pos[joint_name]
        total_error += abs(error)

        # P control: output = Kp * error
        control_output = kp * error

        # Convert control output to position command
        new_position = cur_pos[joint_name] + control_output
        robot_action[f"{joint_name}.pos"] = new_position

    # Send action to robot
    if robot_action:
        robot.send_action(robot_action)
    else:
        raise ValueError(f'robot action if empty.')

    if req_total_error:
        return total_error
    return None


def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to zero position
    
    Args:
        robot: Robot instance
        duration: Time required to move to zero position (seconds)
        kp: Proportional gain
    """
    print("Using P control to slowly move robot to zero position...")
    
    # Get current robot state
    # current_obs = robot.get_observation()
    
    # Extract current joint positions
    # current_positions = {}
    # for key, value in current_obs.items():
    #     if key.endswith('.pos'):
    #         motor_name = key.removesuffix('.pos')
    #         current_positions[motor_name] = value
    
    # Zero position targets
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"Will move to zero position in {duration} seconds using P control,\n"
          f" control frequency: {control_freq}Hz, \n"
          f" proportional gain: {kp}")
    
    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value

        # # P control calculation
        # robot_action = {}
        # for joint_name, target_pos in zero_positions.items():
        #     if joint_name in current_positions:
        #         current_pos = current_positions[joint_name]
        #         error = target_pos - current_pos
        #
        #         # P control: output = Kp * error
        #         control_output = kp * error
        #
        #         # Convert control output to position command
        #         new_position = current_pos + control_output
        #         robot_action[f"{joint_name}.pos"] = new_position
        #
        # # Send action to robot
        # if robot_action:
        #     robot.send_action(robot_action)

        _p_control_helper(robot=robot, target_pos=zero_positions, cur_pos=current_positions, kp=kp)

        # Display progress
        if step % (control_freq // 2) == 0:  # Display progress every 0.5 seconds
            progress = (step / total_steps) * 100
            print(f"Moving to zero position progress: {progress:.1f}%")

        # TODO: not include the bus delay time. inaccurate for control freq. kenn.
        time.sleep(step_time)
    
    print("Robot has moved to zero position")

def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    Use P control to return to start position
    
    Args:
        robot: Robot instance
        start_positions: Start joint positions dictionary
        kp: Proportional gain
        control_freq: Control frequency (Hz)
    """
    print("Returning to start position...")
    
    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds
    
    for step in range(max_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value  # Don't apply calibration coefficients
        
        # P control calculation
        # robot_action = {}
        # total_error = 0
        # for joint_name, target_pos in start_positions.items():
        #     if joint_name in current_positions:
        #         current_pos = current_positions[joint_name]
        #         error = target_pos - current_pos
        #         total_error += abs(error)
        #
        #         # P control: output = Kp * error
        #         control_output = kp * error
        #
        #         # Convert control output to position command
        #         new_position = current_pos + control_output
        #         robot_action[f"{joint_name}.pos"] = new_position
        #
        # # Send action to robot
        # if robot_action:
        #     robot.send_action(robot_action)
        total_error = _p_control_helper(robot=robot,target_pos=start_positions,cur_pos=current_positions,
                                        kp=kp,req_total_error=True)
        
        # Check if start position is reached
        if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
            print("Returned to start position")
            break
        
        time.sleep(control_period)
    
    print("Return to start position completed")

def p_control_loop(robot, keyboard, target_positions, start_positions, kp=0.5, control_freq=50):
    """
    P control loop
    
    Args:
        robot: Robot instance
        keyboard: Keyboard instance
        target_positions: Target joint positions dictionary
        start_positions: Start joint positions dictionary
        kp: Proportional gain
        control_freq: Control frequency (Hz)
    """
    control_period = 1.0 / control_freq
    # Joint control mapping
    joint_controls = {
        'q': ('shoulder_pan', -1),  # Joint1 decrease
        'a': ('shoulder_pan', 1),  # Joint1 increase
        'w': ('shoulder_lift', -1),  # Joint2 decrease
        's': ('shoulder_lift', 1),  # Joint2 increase
        'e': ('elbow_flex', -1),  # Joint3 decrease
        'd': ('elbow_flex', 1),  # Joint3 increase
        'r': ('wrist_flex', -1),  # Joint4 decrease
        'f': ('wrist_flex', 1),  # Joint4 increase
        't': ('wrist_roll', -1),  # Joint5 decrease
        'g': ('wrist_roll', 1),  # Joint5 increase
        'y': ('gripper', -1),  # Joint6 decrease
        'h': ('gripper', 1),  # Joint6 increase
    }

    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")

    while True:
        try:
            # Get keyboard input
            keyboard_action = keyboard.get_action()
            
            if keyboard_action:
                # Process keyboard input, update target positions
                # for key, value in keyboard_action.items():
                for key in keyboard_action:
                    if key == 'x':
                        # Exit program, first return to start position
                        print("Exit command detected, returning to start position...")
                        return_to_start_position(robot, start_positions, 0.2, control_freq)
                        time.sleep(1.)
                        return

                    elif key in joint_controls:
                        joint_name, delta = joint_controls[key]
                        assert joint_name in target_positions
                        # if joint_name in target_positions:
                        current_target = target_positions[joint_name]

                        if joint_name == 'gripper':
                            step = delta * 5.0
                            new_target = current_target + step
                            # clamp into [0,100]
                            new_target = max(0.0, min(100.0, new_target))

                            print(f"Updated Gripper: {current_target:.1f}% -> {new_target:.1f}%")

                        else:
                            new_target = int(current_target + delta)
                            print(f"Updated joint {joint_name}: {current_target} -> {new_target}°")

                        target_positions[joint_name] = new_target
                        print(f"Updated target position {joint_name}: {current_target} -> {new_target}")
            
            # Get current robot state
            current_obs = robot.get_observation()
            
            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # Apply calibration coefficients
                    calibrated_value = apply_joint_calibration(motor_name, value)
                    current_positions[motor_name] = calibrated_value
            
            # P control calculation
            # robot_action = {}
            # for joint_name, target_pos in target_positions.items():
            #     if joint_name in current_positions:
            #         current_pos = current_positions[joint_name]
            #         error = target_pos - current_pos
            #
            #         # P control: output = Kp * error
            #         control_output = kp * error
            #
            #         # Convert control output to position command
            #         new_position = current_pos + control_output
            #         robot_action[f"{joint_name}.pos"] = new_position
            #
            # # Send action to robot
            # if robot_action:
            #     robot.send_action(robot_action)

            _p_control_helper(robot=robot, target_pos=target_positions,cur_pos=current_positions, kp=kp)
            
            time.sleep(control_period)
            
        except KeyboardInterrupt:
            print("User interrupted program")
            raise
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            raise

@dataclass
class XConfig:
    # teleop: TeleoperatorConfig | None = None
    teleop: None = None
    robot: XLerobot2WheelsConfig | None = None

    def __post_init__(self)-> XLerobot2WheelsConfig | None:
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device_cfg = self.robot if self.robot else self.teleop


@draccus.wrap()
def main(cfg:XConfig):
    logger.info(f'load config for x_lerbot_2_wheels --->\n{cfg}')

    """Main function"""
    print("XLeRobot Simplified Keyboard Control Example (P Control)")
    print("="*50)
    
    # try:
    # from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
    # from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

    # from lerobot.robots.so_follower import (SO100Follower,
    #                                         SO100FollowerConfig )
    #
    # from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
    # from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

    # Get port
    # port = input("Please enter SO101 robot USB port (e.g.: /dev/ttyACM0): ").strip()
    # arm_choice:str = ''
    # while not arm_choice in ('l', 'r'):
    #     try:
    #         arm_choice = input("Please choose left or right SO101 arm: [l/r]").strip().lower()
    #
    #     except ValueError as err:
    #         arm_choice = ''
    #         logger.warning(f'key-in value error: {err}, {type(err)}')
    #         continue
    #
    #     else:
    #         if arm_choice not in ('l', 'r'):
    #             logger.warning(f'got illegal input: {arm_choice}, should be "l" or "r":  ')
    #             arm_choice = ''

    port:str=''
    while len(port) == 0:
        match input("Please choose left or right SO101 arm: [l/r]").strip().lower():
            case 'l':
                port=cfg.robot.port_left
                logger.info(f'choose left arm.')
            case 'r':
                port=cfg.robot.port_right
                logger.info(f'choose right arm.')
            case _:
                logger.warning(f'got illegal arm choice, only accept "l" or "r" ')
                port=''

    logger.info(f'will connect to SO101 arm on port {port}.')

    # If directly press enter, use default port
    # if not port:
    #     port = "/dev/ttyACM0"
    #     print(f"Using default port: {port}")
    # else:
    #     print(f"Connecting to port: {port}")

    robot_config = SO100FollowerConfig(port=port)
    robot = SO100Follower(robot_config)
    if not robot.calibration:
        raise ValueError(f'can not load robot calibration file in path:{robot.calibration_dir}.'
                         f' should calibrate robot first.')

    # Configure keyboard
    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)

    try:
        # Connect devices
        robot.connect(calibrate=False)
        keyboard.connect()
        
        print("Devices connected successfully!")
        
        # Ask whether to recalibrate
        # while True:
        #     calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
        #     if calibrate_choice in ['y', 'yes']:
        #         print("Starting recalibration...")
        #         robot.calibrate()
        #         print("Calibration completed!")
        #         break
        #     elif calibrate_choice in ['n', 'no']:
        #         print("Using previous calibration file")
        #         break
        #     else:
        #         print("Please enter y or n")
        
        # Read starting joint angles
        print("Reading starting joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                start_positions[motor_name] = int(value)  # Don't apply calibration coefficients
        
        print("Starting joint angles:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")
        
        # Move to zero position
        move_to_zero_position(robot, duration=3.0, kp=0.5)
        
        # Initialize target positions to current positions (integers)
        target_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
          }
        
        
        print("Keyboard control instructions:")
        print("- Q/A: Joint1 (shoulder_pan) decrease/increase")
        print("- W/S: Joint2 (shoulder_lift) decrease/increase")
        print("- E/D: Joint3 (elbow_flex) decrease/increase")
        print("- R/F: Joint4 (wrist_flex) decrease/increase")
        print("- T/G: Joint5 (wrist_roll) decrease/increase")
        print("- Y/H: Joint6 (gripper) decrease/increase")
        print("- X: Exit program (first return to start position)")
        print("- ESC: Exit program")
        print("="*50)
        print("Note: Robot will continuously move to target position")
        
        # Start P control loop
        # TODO: put into Thread. kenn.
        p_control_loop(robot, keyboard, target_positions, start_positions, kp=0.5, control_freq=50)
        
        # Disconnect
        # robot.disconnect()
        # keyboard.disconnect()
        print("Program ended.")

    except KeyboardInterrupt:
        print("Exiting due to user interrupted program. ")

    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Is the robot correctly connected")
        print("2. Is the USB port correct")
        print("3. Do you have sufficient permissions to access USB device")
        print("4. Is the robot correctly configured")

    finally:
        # Disconnect
        if robot.is_connected:
            print(f'disconnecting from {port}...')
            robot.disconnect()

        if keyboard.is_connected:
            print(f'disconnected from keyboard...')
            keyboard.disconnect()

if __name__ == "__main__":
    main() 
