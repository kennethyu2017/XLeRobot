#!/usr/bin/env python3
"""
Dual-arm keyboard control for SO100/SO101 robots
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
Supports simultaneous control of two robot arms: /dev/ttyACM0 and /dev/ttyACM1
Keyboard mapping: First arm (7y8u9i0o-p=[), Second arm (hbjnkml,;.'/)
"""

import time
import logging
import traceback
from typing import Dict
from functools import partial

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
)

from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.robots.so_follower import (SO101Follower)
from software.examples.example_utils import (SO101_arm_return_to_start_position, inverse_kinematics,
                                             keyboard_teleop_task)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# # Joint calibration coefficients - manually edit
# # Format: [joint_name, zero_position_offset(degrees), scaling_factor]
# JOINT_CALIBRATION = [
#     ['shoulder_pan', 6.0, 1.0],      # Joint1: zero position offset, scaling factor
#     ['shoulder_lift', 2.0, 0.97],     # Joint2: zero position offset, scaling factor
#     ['elbow_flex', 0.0, 1.05],        # Joint3: zero position offset, scaling factor
#     ['wrist_flex', 0.0, 0.94],        # Joint4: zero position offset, scaling factor
#     ['wrist_roll', 0.0, 0.5],        # Joint5: zero position offset, scaling factor
#     ['gripper', 0.0, 1.0],           # Joint6: zero position offset, scaling factor
# ]
#
# def apply_joint_calibration(joint_name, raw_position):
#     """
#     Apply joint calibration coefficients
#
#     Args:
#         joint_name: Joint name
#         raw_position: Raw position value
#
#     Returns:
#         calibrated_position: Calibrated position value
#     """
#     for joint_cal in JOINT_CALIBRATION:
#         if joint_cal[0] == joint_name:
#             offset = joint_cal[1]  # Zero position offset
#             scale = joint_cal[2]   # Scaling factor
#             calibrated_position = (raw_position - offset) * scale
#             return calibrated_position
#     return raw_position  # If no calibration coefficient found, return original value

# def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
#     """
#     Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets
#
#     Parameters:
#         x: End effector x coordinate
#         y: End effector y coordinate
#         l1: Upper arm length (default 0.1159 m)
#         l2: Lower arm length (default 0.1350 m)
#
#     Returns:
#         joint2, joint3: Joint angles in radians as defined in the URDF file
#     """
#     # Calculate joint2 and joint3 offsets in theta1 and theta2
#     theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
#     theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0
#
#     # Calculate distance from origin to target point
#     r = math.sqrt(x**2 + y**2)
#     r_max = l1 + l2  # Maximum reachable distance
#
#     # If target point is beyond maximum workspace, scale it to the boundary
#     if r > r_max:
#         scale_factor = r_max / r
#         x *= scale_factor
#         y *= scale_factor
#         r = r_max
#
#     # If target point is less than minimum workspace (|l1-l2|), scale it
#     r_min = abs(l1 - l2)
#     if r < r_min and r > 0:
#         scale_factor = r_min / r
#         x *= scale_factor
#         y *= scale_factor
#         r = r_min
#
#     # Use law of cosines to calculate theta2
#     cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
#
#     # Calculate theta2 (elbow angle)
#     theta2 = math.pi - math.acos(cos_theta2)
#
#     # Calculate theta1 (shoulder angle)
#     beta = math.atan2(y, x)
#     gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
#     theta1 = beta + gamma
#
#     # Convert theta1 and theta2 to joint2 and joint3 angles
#     joint2 = theta1 + theta1_offset
#     joint3 = theta2 + theta2_offset
#
#     # Ensure angles are within URDF limits
#     joint2 = max(-0.1, min(3.45, joint2))
#     joint3 = max(-0.2, min(math.pi, joint3))
#
#     # Convert from radians to degrees
#     joint2_deg = math.degrees(joint2)
#     joint3_deg = math.degrees(joint3)
#
#     joint2_deg = 90-joint2_deg
#     joint3_deg = joint3_deg-90
#
#     return joint2_deg, joint3_deg

# def move_to_zero_position(robots, duration=3.0, kp=0.5):
#     """
#     Use P control to slowly move all robots to zero position
#
#     Args:
#         robots: Robot instance dictionary {'left_arm': robot1, 'right_arm': robot2}
#         duration: Time required to move to zero position (seconds)
#         kp: Proportional gain
#     """
#     print("Using P control to slowly move all robots to zero position...")
#
#     # Get current states of all robots
#     current_obs = {}
#     for arm_name, robot in robots.items():
#         current_obs[arm_name] = robot.get_observation()
#
#     # Extract current joint positions
#     current_positions = {}
#     for arm_name, obs in current_obs.items():
#         current_positions[arm_name] = {}
#         for key, value in obs.items():
#             if key.endswith('.pos'):
#                 motor_name = key.removesuffix('.pos')
#                 current_positions[arm_name][motor_name] = value
#
#     # Zero position target
#     zero_positions = {
#         'shoulder_pan': 0.0,
#         'shoulder_lift': 0.0,
#         'elbow_flex': 0.0,
#         'wrist_flex': 0.0,
#         'wrist_roll': 0.0,
#         'gripper': 0.0
#     }
#
#     # Calculate control steps
#     control_freq = 50  # 50Hz control frequency
#     total_steps = int(duration * control_freq)
#     step_time = 1.0 / control_freq
#
#     print(f"Will move to zero position in {duration} seconds using P control, control frequency: {control_freq}Hz, proportional gain: {kp}")
#
#     for step in range(total_steps):
#         # Get current states of all robots
#         current_obs = {}
#         current_positions = {}
#         for arm_name, robot in robots.items():
#             current_obs[arm_name] = robot.get_observation()
#             current_positions[arm_name] = {}
#             for key, value in current_obs[arm_name].items():
#                 if key.endswith('.pos'):
#                     motor_name = key.removesuffix('.pos')
#                     # Apply calibration coefficients
#                     calibrated_value = apply_joint_calibration(motor_name, value)
#                     current_positions[arm_name][motor_name] = calibrated_value
#
#         # Perform P control calculation for each robot arm
#         for arm_name, robot in robots.items():
#             robot_action = {}
#             for joint_name, target_pos in zero_positions.items():
#                 if joint_name in current_positions[arm_name]:
#                     current_pos = current_positions[arm_name][joint_name]
#                     error = target_pos - current_pos
#
#                     # P control: output = Kp * error
#                     control_output = kp * error
#
#                     # Convert control output to position command
#                     new_position = current_pos + control_output
#                     robot_action[f"{joint_name}.pos"] = new_position
#
#             # Send action to robot
#             if robot_action:
#                 robot.send_action(robot_action)
#
#         # Display progress
#         if step % (control_freq // 2) == 0:  # Display progress every 0.5 seconds
#             progress = (step / total_steps) * 100
#             print(f"Move to zero position progress: {progress:.1f}%")
#
#         time.sleep(step_time)
#
#     print("All robots have moved to zero position")

# def return_to_start_position(robots:Dict[str, Dict[str,float]],
#                              arm_start_positions:Dict[str,Dict[str,float]],
#                              kp:float,
#                              control_freq:int):
#     """
#     Use P control to return to start position
#
#     Args:
#         robots: Robot instance dictionary {'left_arm': robot1, 'right_arm': robot2}
#         arm_start_positions: Start joint position dictionary {'left_arm': {}, 'right_arm': {}}
#         kp: Proportional gain
#         control_freq: Control frequency (Hz)
#     """
#     print("Returning to start position...")
#
#     control_period = 1.0 / control_freq
#     max_steps = int(5.0 * control_freq)  # Maximum 5 seconds
#
#     for step in range(max_steps):
#         # Get current states of all robots
#         current_obs = {}
#         current_positions = {}
#         for arm_name, robot in robots.items():
#             current_obs[arm_name] = robot.get_observation()
#             current_positions[arm_name] = {}
#             for key, value in current_obs[arm_name].items():
#                 if key.endswith('.pos'):
#                     motor_name = key.removesuffix('.pos')
#                     current_positions[arm_name][motor_name] = value  # Don't apply calibration coefficients
#
#         # Perform P control calculation for each robot arm
#         total_error = 0
#         for arm_name, robot in robots.items():
#             robot_action = {}
#             for joint_name, target_pos in start_positions[arm_name].items():
#                 if joint_name in current_positions[arm_name]:
#                     current_pos = current_positions[arm_name][joint_name]
#                     error = target_pos - current_pos
#                     total_error += abs(error)
#
#                     # P control: output = Kp * error
#                     control_output = kp * error
#
#                     # Convert control output to position command
#                     new_position = current_pos + control_output
#                     robot_action[f"{joint_name}.pos"] = new_position
#
#             # Send action to robot
#             if robot_action:
#                 robot.send_action(robot_action)
#
#         # Check if start position is reached
#         if total_error < 4.0:  # If total error is less than 4 degrees (dual arms), consider reached
#             print("Returned to start position")
#             break
#
#         time.sleep(control_period)
#
#     print("Return to start position completed")


def p_control_loop(*,
                   robots: Dict[str, SO101Follower],
                   keyboard:KeyboardTeleop,
                   arm_target_positions: Dict[str, Dict[str,float]],
                   arm_start_positions: Dict[str, Dict[str,float]],
                   current_ee_xy,
                   kp=0.5,
                   control_freq=50):
    """
    P control loop
    
    Args:
        robots: Robot instance dictionary {'left_arm': robot1, 'right_arm': robot2}
        keyboard: Keyboard instance
        arm_target_positions: Target joint position dictionary
        arm_start_positions: Start joint position dictionary
        current_ee_xy: Current joint position dictionary
        kp: Proportional gain
        control_freq: Control frequency (Hz)
    """
    # bi-arms.
    assert len(robots) == 2
    left_arm = robots['left_arm']
    right_arm = robots['right_arm']

    assert len(arm_start_positions) == 2
    # start_positions = arm_start_positions[arm_name]
    assert len(arm_target_positions) == 2
    # target_positions = arm_target_positions[arm_name]

    control_period = 1.0 / control_freq

    # Initialize dual-arm pitch control variables
    pitch = {'left_arm': 0.0, 'right_arm': 0.0}  # Initial pitch adjustment
    pitch_step = 1  # Pitch adjustment step size
    # First arm control mapping: 7y8u9i0o-p=[
    left_arm_joint_controls = {
        '7': ('shoulder_pan', -1),  # Joint1 decrease
        'y': ('shoulder_pan', 1),  # Joint1 increase
        '0': ('wrist_roll', -1),  # Joint5 decrease
        'o': ('wrist_roll', 1),  # Joint5 increase
        '-': ('gripper', -1),  # Joint6 decrease
        'p': ('gripper', 1),  # Joint6 increase
    }

    # First arm x,y coordinate control
    left_arm_xy_controls = {
        '8': ('x', -0.004),  # x decrease
        'u': ('x', 0.004),  # x increase
        '9': ('y', -0.004),  # y decrease
        'i': ('y', 0.004),  # y increase
    }

    # Second arm control mapping: hbjnkml,;.'/
    right_arm_joint_controls = {
        'h': ('shoulder_pan', -1),  # Joint1 decrease
        'b': ('shoulder_pan', 1),  # Joint1 increase
        ';': ('wrist_roll', -1),  # Joint5 decrease
        'l': ('wrist_roll', 1),  # Joint5 increase
        "'": ('gripper', -1),  # Joint6 decrease
        '/': ('gripper', 1),  # Joint6 increase
    }

    # Second arm x,y coordinate control
    right_arm_xy_controls = {
        'j': ('x', -0.004),  # x decrease
        'n': ('x', 0.004),  # x increase
        'k': ('y', -0.004),  # y decrease
        'm': ('y', 0.004),  # y increase
    }
    
    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")

    print("Dual arm keyboard control instructions:")
    print("--- Left arm control (7y8u9i0o-p=[):")
    print("- 7/y: Joint1 (shoulder_pan) decrease/increase")
    print("- 8/u: Control end effector x coordinate (joint2+3)")
    print("- 9/i: Control end effector y coordinate (joint2+3)")
    print("- =/[: Pitch adjustment increase/decrease (affects wrist_flex)")
    print("- 0/o: Joint5 (wrist_roll) decrease/increase")
    print("- -/p: Joint6 (gripper) decrease/increase")
    print("")
    print("--- Right arm control (hbjnkml,;.'/):")
    print("- h/b: Joint1 (shoulder_pan) decrease/increase")
    print("- j/n: Control end effector x coordinate (joint2+3)")
    print("- k/m: Control end effector y coordinate (joint2+3)")
    print("- ,/.: Pitch adjustment increase/decrease (affects wrist_flex)")
    print("- ;/l: Joint5 (wrist_roll) decrease/increase")
    print("- '/: Joint6 (gripper) decrease/increase")
    print("")
    print("- X: Exit program (return to start position first)")
    print("- ESC: Exit program")
    print("=" * 50)
    print("Note: Dual arm robots will continuously move to target positions")

    while True:
        try:
            # Process keyboard input, update target positions
            for key in keyboard.get_action():
                match key:
                    case 'x':
                        # Exit program, return to start position first
                        print("Exit command detected, returning to start position...")
                        SO101_arm_return_to_start_position(robots=robots,
                                                           arm_start_positions=arm_start_positions,
                                                           kp=0.2,
                                                           control_freq=control_freq)
                        return

                    # First arm pitch control
                    case '=':
                        pitch['left_arm'] += pitch_step
                        print(f"First arm increase pitch adjustment: {pitch['left_arm']:.3f}")
                    case '[':
                        pitch['left_arm'] -= pitch_step
                        print(f"First arm decrease pitch adjustment: {pitch['left_arm']:.3f}")

                    # Second arm pitch control
                    case ',':
                        pitch['right_arm'] += pitch_step
                        print(f"Second arm increase pitch adjustment: {pitch['right_arm']:.3f}")
                    case '.':
                        pitch['right_arm'] -= pitch_step
                        print(f"Second arm decrease pitch adjustment: {pitch['right_arm']:.3f}")

                    # First arm joint control
                    case _k if _k in left_arm_joint_controls:
                        joint_name, delta = left_arm_joint_controls[_k]
                        if joint_name in arm_target_positions['left_arm']:
                            current_target = arm_target_positions['left_arm'][joint_name]
                            new_target = int(current_target + delta)
                            arm_target_positions['left_arm'][joint_name] = new_target
                            print(f" Left arm update target position {joint_name}: {current_target} -> {new_target}")

                    # Second arm joint control
                    case _k if _k in right_arm_joint_controls:
                        joint_name, delta = right_arm_joint_controls[_k]
                        if joint_name in arm_target_positions['right_arm']:
                            current_target = arm_target_positions['right_arm'][joint_name]
                            new_target = int(current_target + delta)
                            arm_target_positions['right_arm'][joint_name] = new_target
                            print(f" Right arm update target position {joint_name}: {current_target} -> {new_target}")

                    # First arm xy control
                    case _k if _k in left_arm_xy_controls:
                        coord, delta = left_arm_xy_controls[_k]
                        if coord == 'x':
                            current_ee_xy['left_arm']['x'] += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_ee_xy['left_arm']['x'], current_ee_xy['left_arm']['y'])
                            arm_target_positions['left_arm']['shoulder_lift'] = joint2_target
                            arm_target_positions['left_arm']['elbow_flex'] = joint3_target
                            print(f"First arm update x coordinate: {current_ee_xy['left_arm']['x']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_ee_xy['left_arm']['y'] += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_ee_xy['left_arm']['x'], current_ee_xy['left_arm']['y'])
                            arm_target_positions['left_arm']['shoulder_lift'] = joint2_target
                            arm_target_positions['left_arm']['elbow_flex'] = joint3_target
                            print(f"First arm update y coordinate: {current_ee_xy['left_arm']['y']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")

                    # Second arm xy control
                    case _k if _k in right_arm_xy_controls:
                        coord, delta = right_arm_xy_controls[_k]
                        if coord == 'x':
                            current_ee_xy['right_arm']['x'] += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_ee_xy['right_arm']['x'], current_ee_xy['right_arm']['y'])
                            arm_target_positions['right_arm']['shoulder_lift'] = joint2_target
                            arm_target_positions['right_arm']['elbow_flex'] = joint3_target
                            print(f"Second arm update x coordinate: {current_ee_xy['right_arm']['x']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_ee_xy['right_arm']['y'] += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_ee_xy['right_arm']['x'], current_ee_xy['right_arm']['y'])
                            arm_target_positions['right_arm']['shoulder_lift'] = joint2_target
                            arm_target_positions['right_arm']['elbow_flex'] = joint3_target
                            print(f"Second arm update y coordinate: {current_ee_xy['right_arm']['y']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
            
            # Apply pitch adjustment to wrist_flex for each robot arm
            for arm_name in ['left_arm', 'right_arm']:
                if 'shoulder_lift' in arm_target_positions[arm_name] and 'elbow_flex' in arm_target_positions[arm_name]:
                    arm_target_positions[arm_name]['wrist_flex'] = (- arm_target_positions[arm_name]['shoulder_lift']
                                                                    - arm_target_positions[arm_name]['elbow_flex']
                                                                    + pitch[arm_name])
            
            # Display current pitch values (every 100 steps to avoid spam)
            if hasattr(p_control_loop, 'step_counter'):
                p_control_loop.step_counter += 1
            else:
                p_control_loop.step_counter = 0
            
            if p_control_loop.step_counter % 100 == 0:
                print(f"Current pitch adjustment: left_arm={pitch['left_arm']:.3f}, right_arm={pitch['right_arm']:.3f}")

            # Perform P control calculation for each robot arm
            for _arm_name, _arm in robots.items():
                robot_action = {}
                # Get current states of all robots
                for key, value in _arm.get_observation().items():
                    if key.endswith('.pos'):
                        motor_name = key.removesuffix('.pos')
                        # TODO: comment calib temply by kenn.
                        # Apply calibration coefficients
                        # calibrated_value = apply_joint_calibration(motor_name, value)
                        # current_joint_positions[arm_name][motor_name] = calibrated_value
                        assert motor_name in arm_target_positions[_arm_name]
                        error = (arm_target_positions[_arm_name][motor_name] -
                                 value)
                        # P control: output = Kp * error
                        control_output = kp * error
                        # Convert control output to position command
                        new_position = value + control_output
                        robot_action[key] = new_position

                # Send action to robot
                assert len(robot_action) == len(arm_target_positions[_arm_name])
                _arm.send_action(robot_action)
            
            time.sleep(control_period)
            
        except KeyboardInterrupt:
            print("User interrupted program")
            raise

        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            raise

def main():
    """Main function"""
    print("LeRobot Dual-arm Keyboard Control Example (P Control)")
    print("="*50)
    # Initialize dual arm x,y coordinate control
    x0, y0 = 0.1629, 0.1131
    current_ee_xy = {
        'left_arm': {'x': x0, 'y': y0},
        'right_arm': {'x': x0, 'y': y0}
    }

    print(f"Initialize dual arm end effector positions: left_arm=({x0:.4f}, {y0:.4f}), right_arm=({x0:.4f}, {y0:.4f})")
    keyboard_teleop_task(p_control_loop=partial(p_control_loop,
                                                current_ee_xy=current_ee_xy,
                                                kp=0.3,
                                                control_freq=20))

if __name__ == "__main__":
    main() 