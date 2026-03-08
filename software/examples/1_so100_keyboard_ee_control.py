#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
"""

import time
import logging
import traceback
from typing import Dict
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
)

from lerobot.robots.so_follower import (SO101Follower)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from software.examples.example_utils import (SO101_arm_return_to_start_position, inverse_kinematics,
											 keyboard_teleop_SO101arm_task)

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
#     if 0 < r < r_min:
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


def p_control_loop(*,
                   robots: Dict[str,SO101Follower],
                   keyboard:KeyboardTeleop,
                   arm_target_positions:Dict[str, Dict[str,float]],
                   arm_start_positions:Dict[str, Dict[str,float]],
                   current_x, current_y, kp, control_freq):
    """
    P control loop
    
    Args:
        robots: robot instance
        keyboard: keyboard instance
        arm_target_positions: target joint position dictionary
        arm_start_positions: start joint position dictionary
        current_x: current x coordinate
        current_y: current y coordinate
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    # single arm.
    assert len(robots) == 1
    arm_name, arm = next(iter(robots.items()))
    assert len(arm_start_positions) == 1
    # start_positions = arm_start_positions[arm_name]
    assert len(arm_target_positions) == 1
    target_positions = arm_target_positions[arm_name]

    control_period = 1.0 / control_freq
    
    # Initialize pitch control variables
    pitch = 0.0  # Initial pitch adjustment
    pitch_step = 1  # Pitch adjustment step size
    
    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")
    print("Keyboard control instructions:")
    print("- Q/A: Joint 1 (shoulder_pan) decrease/increase")
    print("- W/S: Control end effector x coordinate (joint2+3)")
    print("- E/D: Control end effector y coordinate (joint2+3)")
    print("- R/F: Pitch adjustment increase/decrease (affects wrist_flex)")
    print("- T/G: Joint 5 (wrist_roll) decrease/increase")
    print("- Y/H: Joint 6 (gripper) decrease/increase")
    print("- X: Exit program (return to start position first)")
    print("- ESC: Exit program")
    print("=" * 50)
    print("Note: Robot will continuously move to target positions")

    # Joint control mapping
    joint_controls = {
        'q': ('shoulder_pan', -1),  # Joint 1 decrease
        'a': ('shoulder_pan', 1),  # Joint 1 increase
        't': ('wrist_roll', -1),  # Joint 5 decrease
        'g': ('wrist_roll', 1),  # Joint 5 increase
        'y': ('gripper', -1),  # Joint 6 decrease
        'h': ('gripper', 1),  # Joint 6 increase
    }
    # x,y coordinate control
    xy_controls = {
        'w': ('x', -0.004),  # x decrease
        's': ('x', 0.004),  # x increase
        'e': ('y', -0.004),  # y decrease
        'd': ('y', 0.004),  # y increase
    }

    while True:
        try:
            # Process keyboard input, update target positions
            for key in keyboard.get_action():
                match key:
                    case 'x':
                        # Exit program, first return to start position
                        print("Exit command detected, returning to start position...")
                        SO101_arm_return_to_start_position(robots=robots,
                                                           arm_start_positions=arm_start_positions,
                                                           kp=0.2,
                                                           control_freq=control_freq)
                        return

                    # Pitch control
                    case 'r':
                        pitch += pitch_step
                        print(f"Increase pitch adjustment: {pitch:.3f}")
                    case 'f':
                        pitch -= pitch_step
                        print(f"Decrease pitch adjustment: {pitch:.3f}")

                    case k_jnt if k_jnt in joint_controls:
                        joint_name, delta = joint_controls[k_jnt]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"Update target position {joint_name}: {current_target} -> {new_target}")

                    case k_xy if k_xy in xy_controls:
                        coord, delta = xy_controls[k_xy]
                        if coord == 'x':
                            current_x += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions['shoulder_lift'] = joint2_target
                            target_positions['elbow_flex'] = joint3_target
                            print(f"Update x coordinate: {current_x:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_y += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions['shoulder_lift'] = joint2_target
                            target_positions['elbow_flex'] = joint3_target
                            print(f"Update y coordinate: {current_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")

                        # debug info only.
                        _wrist_update = - target_positions['shoulder_lift'] - target_positions['elbow_flex'] + pitch
                        print(f"will also adjust wrist_flex for xy: joint4={_wrist_update:.3f}")
            
            # Apply pitch adjustment to wrist_flex
            # Calculate wrist_flex target position based on shoulder_lift and elbow_flex
            if 'shoulder_lift' in target_positions and 'elbow_flex' in target_positions:
                target_positions['wrist_flex'] = - target_positions['shoulder_lift'] - target_positions['elbow_flex'] + pitch
                # Show current pitch value (display every 100 steps to avoid screen flooding)
                if hasattr(p_control_loop, 'step_counter'):
                    p_control_loop.step_counter += 1
                else:
                    p_control_loop.step_counter = 0
                
                if p_control_loop.step_counter % 100 == 0:
                    print(f"Current pitch adjustment: {pitch:.3f}, wrist_flex target: {target_positions['wrist_flex']:.3f}")
            
            # Get current robot state
            current_obs = arm.get_observation()
            
            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # Apply calibration coefficients
                    # TODO: commented by kenn.
                    # calibrated_value = apply_joint_calibration(motor_name, value)
                    # current_positions[motor_name] = calibrated_value
                    current_positions[motor_name] = value
            
            # P control calculation
            robot_action = {}
            for joint_name, target_pos in target_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos
                    
                    # P control: output = Kp * error
                    control_output = kp * error
                    
                    # Convert control output to position command
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # Send action to robot
            if robot_action:
                arm.send_action(robot_action)
            
            time.sleep(control_period)
            
        except KeyboardInterrupt:
            print("User interrupted program")
            break
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            break

def main():
    # Initialize x,y coordinate control
    x0, y0 = 0.1629, 0.1131
    current_x, current_y = x0, y0
    print(f"Initialize end effector position: x={current_x:.4f}, y={current_y:.4f}")

    keyboard_teleop_SO101arm_task(p_control_loop=partial(p_control_loop,
														 current_x=current_x,
														 current_y=current_y,
														 kp=0.3,
														 control_freq=20)
								  )

if __name__ == "__main__":
    main() 
