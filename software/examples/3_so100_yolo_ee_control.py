#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot with independent YOLO streaming display
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
Keyboard control is identical to 5_so100_keyboard_ee_control.py

YOLO stream displays object detection but does NOT control the robot
Video stream and robot control are completely independent
"""

import time
import logging
# import traceback
import cv2
from threading import Thread,Event
from ultralytics import YOLOE
from ultralytics.engine.model import Model
from functools import partial
from typing import Dict

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
)

from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.robots.so_follower import (SO101Follower)
from software.examples.example_utils import (SO101_arm_return_to_start_position,
                                             inverse_kinematics,
                                             keyboard_teleop_task)

# Set the ultralytics logger level to WARNING or higher
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Joint calibration coefficients - manually edit
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
# JOINT_CALIBRATION = [
#     ['shoulder_pan', 6.0, 1.0],      # Joint1: zero position offset, scale factor
#     ['shoulder_lift', 2.0, 0.97],     # Joint2: zero position offset, scale factor
#     ['elbow_flex', 0.0, 1.05],        # Joint3: zero position offset, scale factor
#     ['wrist_flex', 0.0, 0.94],        # Joint4: zero position offset, scale factor
#     ['wrist_roll', 0.0, 0.5],        # Joint5: zero position offset, scale factor
#     ['gripper', 0.0, 1.0],           # Joint6: zero position offset, scale factor
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
#             scale = joint_cal[2]   # Scale factor
#             calibrated_position = (raw_position - offset) * scale
#             return calibrated_position
#     return raw_position  # If no calibration coefficient found, return raw value


def video_stream_loop_v2(model, camera_index:int, stop_event, target_objects=None):
    """
    Optimized video streaming loop using stream=True for better memory management.
    """
    print("Starting YOLO video stream...")
    # stream=True returns a generator which yields results one by one
    results_generator = model.predict(
        source=camera_index,
        stream=True,
        device=0,  # Ensure it uses your 3090Ti
        verbose=False,  # Keep the terminal clean,
        conf=0.55,
        # TODO: just try.
        # imgsz=1024,
    )

    try:
        for result in results_generator:
            # Check if the stop signal was sent from another thread
            if stop_event.is_set():
                break

            # # result.orig_img is the raw frame from cv2.VideoCapture
            # # result.plot() creates the frame with boxes/masks
            # if len(result.boxes) > 0:
            #     annotated_frame = result.plot()
            # else:
            #     annotated_frame = result.orig_img

            annotated_frame = result.plot()
            # Display the window
            cv2.imshow("YOLO Live Detection", annotated_frame)

            # Standard OpenCV break logic
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Video stream error: {e}")
        raise e
    finally:
        print("Video stream ended")
        cv2.destroyAllWindows()
        # Note: Ultralytics closes the camera automatically when the generator ends


# Independent video streaming function (no robot control)
# def video_stream_loop(model:Model, cap:cv2.VideoCapture,
#                       stop_event:Event,
#                       target_objects=None):
#     """
#     Independent video streaming loop that only displays object detection
#     Does not control the robot - purely for visual feedback
#     """
#     print("Starting YOLO video stream...")
#
#     try:
#     # while True:
#         while not stop_event.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 print("Camera frame not available")
#                 continue
#
#             results = model(frame) #,verbose=False)
#             if not results or not hasattr(results[0], 'boxes') or not results[0].boxes:
#                 # No objects detected - show original frame
#                 annotated_frame = frame
#             else:
#                 # Show detection results
#                 annotated_frame = results[0].plot()
#
#             # Show detection results in a window
#             cv2.imshow("YOLO Live Detection", annotated_frame)
#
#             # Allow quitting vision mode with 'q' or ESC
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord("q") or key == 27:  # 'q' or ESC
#                 break
#
#     except Exception as e:
#         print(f"Video stream error: {e}")
#         raise
#     finally:
#         print("Video stream ended")
#         cv2.destroyAllWindows()


def p_control_loop(*,
                   robots: Dict[str, SO101Follower],
                   keyboard: KeyboardTeleop,
                   arm_target_positions: Dict[str, Dict[str, float]],
                   arm_start_positions: Dict[str, Dict[str, float]],
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
    assert len(arm_target_positions) == 1
    target_positions = arm_target_positions[arm_name]

    control_period = 1.0 / control_freq

    # Initialize pitch control variables
    pitch = 0.0  # Initial pitch adjustment
    pitch_step = 1  # Pitch adjustment step size

    print("Control instructions:")
    print("Keyboard control (independent of video stream):")
    print("- Q/A: Joint 1 (shoulder_pan) decrease/increase")
    print("- W/S: Control end effector x coordinate (joint2+3)")
    print("- E/D: Control end effector y coordinate (joint2+3)")
    print("- R/F: Pitch adjustment increase/decrease (affects wrist_flex)")
    print("- T/G: Joint 5 (wrist_roll) decrease/increase")
    print("- Y/H: Joint 6 (gripper) close/open")
    print("- X: Exit program (return to start position first)")
    print("- ESC: Exit program")
    print("")
    print("Video stream:")
    print("- Independent YOLO detection display (no robot control)")
    print("- Q (in YOLO window): Exit video stream")
    print("=" * 60)
    print("Note: Video stream and keyboard control are completely independent")

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
                            print(
                                f"Update x coordinate: {current_x:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_y += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions['shoulder_lift'] = joint2_target
                            target_positions['elbow_flex'] = joint3_target
                            print(
                                f"Update y coordinate: {current_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")

                        # debug info only.
                        _wrist_update = - target_positions['shoulder_lift'] - target_positions['elbow_flex'] + pitch
                        print(f"will also adjust wrist_flex for xy: joint4={_wrist_update:.3f}")

            # Apply pitch adjustment to wrist_flex
            # Calculate wrist_flex target position based on shoulder_lift and elbow_flex
            if 'shoulder_lift' in target_positions and 'elbow_flex' in target_positions:
                target_positions['wrist_flex'] = - target_positions['shoulder_lift'] - target_positions[
                    'elbow_flex'] + pitch
                # Show current pitch value (display every 100 steps to avoid screen flooding)
                if hasattr(p_control_loop, 'step_counter'):
                    p_control_loop.step_counter += 1
                else:
                    p_control_loop.step_counter = 0

                if p_control_loop.step_counter % 100 == 0:
                    print(
                        f"Current pitch adjustment: {pitch:.3f}, wrist_flex target: {target_positions['wrist_flex']:.3f}")

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
            raise

        except Exception as e:
            print(f"P control loop error: {e}")
            # traceback.print_exc()
            raise


def main():

    """Main function"""
    print("LeRobot Keyboard Control + Independent YOLO Display")
    print("="*60)

    # Initialize x,y coordinate control
    x0, y0 = 0.1629, 0.1131
    current_x, current_y = x0, y0
    print(f"Initialize end effector position: x={current_x:.4f}, y={current_y:.4f}")

    # Initialize YOLO and camera
    model = YOLOE("yoloe-11l-seg.pt").to('cuda')  # or select yoloe-11s/m-seg.pt for different sizes
    print(f" ==== YOLOE Model is running on: {model.device} =====")
    print(f" ==== YOLOE Model is running on: {model.device} =====")
    print(f" ==== YOLOE Model is running on: {model.device} =====")

    # Get detection targets from user input
    print("\n" + "=" * 60)
    print("YOLO Detection Target Setup")
    print("=" * 60)
    target_input = input(
        "Enter objects to detect (separate multiple objects with commas, e.g., bottle,cup,mouse): ").strip()

    # If Enter is pressed directly, use default targets
    if not target_input:
        target_objects = ["bottle"]
        print(f"Using default targets: {target_objects}")
    else:
        # Parse multiple objects separated by commas
        target_objects = [obj.strip() for obj in target_input.split(',') if obj.strip()]
        print(f"Detection targets: {target_objects}")

    # Set text prompt to detect the specified objects
    model.set_classes(target_objects, model.get_text_pe(target_objects))

    # List available cameras and prompt user
    def list_cameras(max_index=5):
        available = []
        for idx in range(max_index):
            cap_test = cv2.VideoCapture(idx)
            if cap_test.isOpened():
                available.append(idx)
                cap_test.release()
        return available

    cameras = list_cameras()
    if not cameras:
        print("No cameras found!")
        return
    print(f"Available cameras: {cameras}")
    camera_index = int(input(f"Select camera index from {cameras}: "))
    # cap = cv2.VideoCapture(selected, cv2.CAP_V4L2)
    # if not cap.isOpened():
    #     print("Camera not found!")
    #     return
    # else:
    #     # Optional: Force a specific resolution to match your YOLO inference shape
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stop_signal = Event()
    # Start video stream in a separate thread
    video_thread = Thread(target=video_stream_loop_v2,
                          args=(model, camera_index, stop_signal, target_objects),
                          daemon=True)
    try:

        video_thread.start()
        keyboard_teleop_task(p_control_loop=partial(p_control_loop,
                                                    current_x=current_x,
                                                    current_y=current_y,
                                                    kp=0.3,
                                                    control_freq=20)
                             )
    except Exception as e:
        print(e)
        # traceback.print_exc()
    finally:
        # maybe signal video_thread to finish using event.
        stop_signal.set()
        video_thread.join()

        # cap.release()
        cv2.destroyAllWindows()
        print("Program ended")


if __name__ == "__main__":
    main()
