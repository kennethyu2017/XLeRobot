#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
"""

import logging
import time
import traceback
from typing import Dict
from functools import partial

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
)

from software.src.robots.xlerobot_2wheels import (
    XLerobot2WheelsConfig,
    XLerobot2Wheels)

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from software.joyconrobotics import JoyconRobotics
from software.examples.example_utils import (SO101_arm_return_to_start_position,
                                             inverse_kinematics,
                                             keyboard_teleop_SO101arm_task)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Joint calibration coefficients - manually edited
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
# JOINT_CALIBRATION = [
#     ["shoulder_pan", 6.0, 1.0],  # Joint 1: zero position offset, scale factor
#     ["shoulder_lift", 2.0, 0.97],  # Joint 2: zero position offset, scale factor
#     ["elbow_flex", 0.0, 1.05],  # Joint 3: zero position offset, scale factor
#     ["wrist_flex", 0.0, 0.94],  # Joint 4: zero position offset, scale factor
#     ["wrist_roll", 0.0, 0.5],  # Joint 5: zero position offset, scale factor
#     ["gripper", 0.0, 1.0],  # Joint 6: zero position offset, scale factor
# ]

class FixedAxesJoyconRobotics(JoyconRobotics):
    def common_update(self):
        # 修改后的更新逻辑：摇杆只控制固定轴向
        speed_scale = 0.0008
        # pitch = -self.position[4] * 60 + 20
        # print(f"pitch_ctrl: {pitch}")
        # 垂直摇杆：只控制X轴（前后）
        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        joycon_stick_v_0 = 1800
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if joycon_stick_v > joycon_stick_v_threshold + joycon_stick_v_0:
            self.position[0] += speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * self.direction_vector[0]
            self.position[2] += speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * self.direction_vector[2]
        elif joycon_stick_v < joycon_stick_v_0 - joycon_stick_v_threshold:
            self.position[0] += speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * self.direction_vector[0]
            self.position[2] += speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * self.direction_vector[2]
        
        # 水平摇杆：只控制Y轴（左右）  
        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        joycon_stick_h_0 = 2000
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if joycon_stick_h > joycon_stick_h_threshold + joycon_stick_h_0:
            self.position[1] += speed_scale * (joycon_stick_h - joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        elif joycon_stick_h < joycon_stick_h_0 - joycon_stick_h_threshold:
            self.position[1] += speed_scale * (joycon_stick_h - joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        
        # Z轴只通过按钮控制
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        # 其他按钮控制（复制原来的逻辑）
        joycon_button_xup = self.joycon.get_button_x() if self.joycon.is_right() else self.joycon.get_button_up()
        joycon_button_xback = self.joycon.get_button_b() if self.joycon.is_right() else self.joycon.get_button_down()
        if joycon_button_xup == 1:
            self.position[0] += 0.001 * self.dof_speed[0]
        elif joycon_button_xback == 1:
            self.position[0] -= 0.001 * self.dof_speed[0]
        
        # Home按钮重置逻辑（简化版）
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
        
        # 夹爪控制逻辑（复制原来的）
        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == 'plus' and status == 1) or (self.joycon.is_left() and event_type == 'minus' and status == 1):
                self.reset_button = 1
                # will trigger Joycon re-calibration.
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == 'a':
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == 'y':
                self.restart_episode_button = status
            elif ((self.joycon.is_right() and event_type == 'zr') or (self.joycon.is_left() and event_type == 'zl')) and not self.change_down_to_gripper:
                self.gripper_toggle_button = status
            elif ((self.joycon.is_right() and event_type == 'stick_r_btn') or (self.joycon.is_left() and event_type == 'stick_l_btn')) and self.change_down_to_gripper:
                self.gripper_toggle_button = status
            else: 
                self.reset_button = 0
            
        if self.gripper_toggle_button == 1 :
            if self.gripper_state == self.gripper_open:
                self.gripper_state = self.gripper_close
            else:
                self.gripper_state = self.gripper_open
            self.gripper_toggle_button = 0

        # 按钮控制状态
        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0
        
        return self.position, self.gripper_state, self.button_control



# def apply_joint_calibration(joint_name, raw_position):
#     """
#     Apply joint calibration coefficients
#
#     Args:
#         joint_name: joint name
#         raw_position: raw position value
#
#     Returns:
#         calibrated_position: calibrated position value
#     """
#     for joint_cal in JOINT_CALIBRATION:
#         if joint_cal[0] == joint_name:
#             offset = joint_cal[1]  # zero position offset
#             scale = joint_cal[2]  # scale factor
#             calibrated_position = (raw_position - offset) * scale
#             return calibrated_position
#     return raw_position  # if no calibration coefficient found, return original value


def p_control_loop(*,
                   robots: Dict[str, SO101Follower],
                   arm_target_positions:Dict[str, Dict[str,float]],
                   arm_start_positions:Dict[str, Dict[str,float]],
                   start_x,
                   start_y,
                   joyconrobotics_right,
                   kp,
                   control_freq,
                   **kwargs):
    """
    P control loop

    Args:
        robots: robot instance

        arm_start_positions: target joint position dictionary
        arm_target_positions:
        start_x: current x coordinate
        start_y: current y coordinate
        joyconrobotics_right: joycon robotics instance
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
    current_x, current_y = start_x, start_y

    step = 0
    while True:
        step+=1
        try:
            pose, gripper, control_button = joyconrobotics_right.get_control()

            # TODO: use 'restart_episode' temply. kenn.
            if control_button == -1:
                # Exit program, first return to start position
                print("Exit command detected, returning to start position...")
                SO101_arm_return_to_start_position(robots=robots,
                                                   arm_start_positions=arm_start_positions,
                                                   kp=0.2,
                                                   control_freq=control_freq)
                return

            x, y, z, roll_, pitch_, yaw = pose
            if step % 50 == 0:
                print(f'x={x}, y={y}, z={z},{roll_=:},{pitch_=:},{yaw=:}')
            pitch = -pitch_ * 60 + 20
            roll = roll_ * 50

            if step % 50 == 0:
                print(f'calculated pitch={pitch} roll={roll}')

            # x, y, z is relative to ergo-origin.
            current_x = 0.1629 + x
            current_y = 0.1131 + z

            # 添加y值控制shoulder_pan关节
            # y值直接映射到shoulder_pan的目标位置，可以调整缩放因子
            y_scale = 300.0  # 缩放因子，可以根据需要调整
            target_positions["shoulder_pan"] = y * y_scale
            
            # Calculate target angles for joint2 and joint3
            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
            target_positions["shoulder_lift"] = joint2_target
            target_positions["elbow_flex"] = joint3_target 
            # target_positions["shoulder_lift"] = joint2_target + pitch
            # target_positions["elbow_flex"] = joint3_target + pitch
            target_positions["wrist_flex"] = (-target_positions["shoulder_lift"] - target_positions["elbow_flex"]
                                              + pitch)
            target_positions["wrist_roll"] = roll

            if gripper == 1:
                target_positions["gripper"] = 60
            else:
                target_positions["gripper"] = 0

            if step % 50 == 0:
                print(f'target_positions={target_positions}')

            # Get current robot state
            current_obs = arm.get_observation()

            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
                    # Apply calibration coefficients
                    #TODO: comment temply by kenn.
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
            raise

        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            raise

def main():
    """Main function"""
    print("LeRobot SingleArm JoyCon(Right) Control Example (P Control)")
    print(f'==== use your RIGHT JoyCon ====')
    print(f'==== use your RIGHT JoyCon ====')
    print(f'==== use your RIGHT JoyCon ====')

    print("=" * 50)

    # 使用修改后的控制类
    joyconrobotics_right = FixedAxesJoyconRobotics(
        "right",
        dof_speed=[2, 2, 2, 1, 1, 1]
    )
    # Initialize x,y coordinate control
    x0, y0 = 0.1629, 0.1131
    print(f"Initialize end effector position: x={x0:.4f}, y={y0:.4f}")

    print("固定轴向控制测试:")
    print("垂直摇杆: 只控制X轴（前后）")
    print("水平摇杆: 只控制Y轴（左右）")
    print("R按钮: Z轴上升")
    print("摇杆按钮: Z轴下降")
    print("Home按钮: 重置位置")
    print("ZR按钮: 切换夹爪")
    print("按Ctrl+C停止")
    try:
        keyboard_teleop_SO101arm_task(p_control_loop=partial(p_control_loop,
                                                             joyconrobotics_right=joyconrobotics_right,
                                                             start_x=x0,
                                                             start_y=y0,
                                                             kp=0.3,
                                                             control_freq=20)
                                      )
    except Exception as e:
        print(e)

    finally:
        joyconrobotics_right.disconnect()

    print("Program ended")


if __name__ == "__main__":
    main()
