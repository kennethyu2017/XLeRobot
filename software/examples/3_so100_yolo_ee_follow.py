#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
"""

import time
import logging
import cv2
from ultralytics import YOLO
from threading import Thread, Event
from typing import Tuple, Dict
from queue import Queue
from functools import partial

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
)

from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.robots.so_follower import (SO101Follower)
from software.examples.example_utils import (SO101_arm_return_to_start_position,
											 inverse_kinematics,
											 keyboard_teleop_SO101arm_task)

# Set the ultralytics logger level to WARNING or higher
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping coefficients for vision control
K_pan = -0.006  # radians per pixel (tune as needed)
K_y = 0.00004   # meters per pixel (tune as needed)

def vision_control_update_v2(*,
                             obj_dxy_queue:Queue[Tuple[float,float]], # in image coordinate.
                             model,
                             camera_index:int,
                             stop_event:Event,
                             object_cls_id:int )->None:
    # stream=True returns a generator which yields results one by one
    results_generator = model.predict(
        source=camera_index,
        stream=True,
        device=0,  # Ensure it uses your 3090 GPU.
        verbose=False,  # Keep the terminal clean,
        conf=0.55,
        # TODO: only support single obj till now. kenn.
        classes=[object_cls_id],
        # TODO: just try.
        # imgsz=1024,
    )

    try:
        for result in results_generator:
            # fps = 30, and inference will take ~12ms. actually we do not need to sleep.
            time.sleep(0.01)

            # Check if the stop signal was sent from another thread
            if stop_event.is_set():
                break

            annotated_frame = result.orig_img
            if len(result.boxes) > 0:
                annotated_frame = result.plot()
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = result.names[cls]
                    # if label == target_object:
                    if cls == object_cls_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        h, w = annotated_frame.shape[:2]
                        dx = cx - w // 2
                        dy = cy - h // 2
                        # Map dx, dy to robot control
                        obj_dx_in_img = -K_pan * dx
                        if abs(obj_dx_in_img) < 0.1:
                            obj_dx_in_img = 0

                        obj_dy_in_img = -K_y * dy
                        # Clamp d_current_y to [-0.004, 0.004] and zero if within ±0.0005
                        obj_dy_in_img = max(min(obj_dy_in_img, 0.005), -0.005)
                        if abs(obj_dy_in_img) < 0.001:
                            obj_dy_in_img = 0

                        obj_dxy_queue.put((obj_dx_in_img, obj_dy_in_img))

                        # TODO: Only follow first target object found.
                        break

            # Show annotated frame in a window
            if annotated_frame is not None:
                cv2.imshow("YOLO11 Live", annotated_frame)
            # Allow quitting vision mode with 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    except Exception as e:
        print(f"Video stream error: {e}")
        raise e
    finally:
        print("Video stream ended")
        cv2.destroyAllWindows()
        # Note: Ultralytics closes the camera automatically when the generator ends


# Vision control update function
# def vision_control_update(target_positions, current_x, current_y, model, cap, K_pan, K_y, target_objects=["mouse"]):
#     ret, frame = cap.read()
#     if not ret:
#         print("Camera frame not available")
#         return current_x, current_y  # No update
#
#     results = model(frame)
#     if not results or not hasattr(results[0], 'boxes') or not results[0].boxes:
#         print("No objects detected")
#         annotated_frame = frame
#     else:
#         # Find target objects in detections
#         annotated_frame = results[0].plot()
#         for box in results[0].boxes:
#             cls = int(box.cls[0])
#             label = results[0].names[cls]
#             if label in target_objects:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cx = (x1 + x2) // 2
#                 cy = (y1 + y2) // 2
#                 h, w = frame.shape[:2]
#                 dx = cx - w // 2
#                 dy = cy - h // 2
#                 # Map dx, dy to robot control
#                 d_current_x = -K_pan * dx
#                 if abs(d_current_x) < 0.1:
#                     d_current_x = 0
#
#                 target_positions['shoulder_pan'] += d_current_x
#                 d_current_y = -K_y * dy
#                 # Clamp d_current_y to [-0.004, 0.004] and zero if within ±0.0005
#                 d_current_y = max(min(d_current_y, 0.005), -0.005)
#                 if abs(d_current_y) < 0.001:
#                     d_current_y = 0
#                 current_y += d_current_y   # Negative sign: up in image = increase y
#                 # Update joint targets using inverse kinematics
#                 joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
#                 target_positions['shoulder_lift'] = joint2_target
#                 target_positions['elbow_flex'] = joint3_target
#                 print(f"{label.capitalize()} center offset: dx={dx}, dy={dy} -> dc_y={d_current_y}, pan: {target_positions['shoulder_pan']:.2f}, y: {current_y:.3f}, joint2: {joint2_target:.2f}, joint3: {joint3_target:.2f}")
#                 break  # Only use first target object found
#     # Show annotated frame in a window
#     cv2.imshow("YOLO11 Live", annotated_frame)
#     # Allow quitting vision mode with 'q'
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         raise KeyboardInterrupt
#     return current_x, current_y

def p_control_loop(*,
                   robots: Dict[str, SO101Follower],
                   keyboard: KeyboardTeleop,
                   arm_target_positions: Dict[str, Dict[str, float]],
                   arm_start_positions: Dict[str, Dict[str, float]],
                   init_x:float,
                   init_y:float,
                   obj_dxy_queue: Queue[Tuple[float,float]],
                   kp:float=0.5,
                   control_freq:int=50
                   ):
    """
    P control loop
    
    Args:
        robots: Robot instance
        keyboard: Keyboard instance
        arm_target_positions: Target joint positions dictionary
        arm_start_positions: Start joint positions dictionary
        kp: Proportional gain
        obj_dxy_queue:
        init_x:
        init_y:
        control_freq: Control frequency (Hz)
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

    # Joint control mapping
    joint_controls = {
        'q': ('shoulder_pan', -1),  # Joint1 decrease
        'a': ('shoulder_pan', 1),  # Joint1 increase
        't': ('wrist_roll', -1),  # Joint5 decrease
        'g': ('wrist_roll', 1),  # Joint5 increase
        'y': ('gripper', -1),  # Joint6 decrease
        'h': ('gripper', 1),  # Joint6 increase
    }

    # x,y coordinate control
    xy_controls = {
        'w': ('x', -0.004),  # x decrease
        's': ('x', 0.004),  # x increase
        'e': ('y', -0.004),  # y decrease
        'd': ('y', 0.004),  # y increase
    }
    
    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")

    print("Control instructions:")
    print("Keyboard control (can be used simultaneously with vision):")
    print("- Q/A: Joint1 (shoulder_pan) decrease/increase")
    print("- W/S: Control end effector x coordinate (joint2+3)")
    print("- E/D: Control end effector y coordinate (joint2+3)")
    print("- R/F: Pitch adjustment increase/decrease (affects wrist_flex)")
    print("- T/G: Joint5 (wrist_roll) decrease/increase")
    print("- Y/H: Joint6 (gripper) close/open")
    print("- X: Exit program (return to start position first)")
    print("- ESC: Exit program")
    print("- Q (in camera window): Exit vision mode")
    print("=" * 50)
    print("Note: Vision and keyboard control work together")

    obj_dx_in_img, obj_dy_in_img = 0., 0.
    target_x, target_y = init_x, init_y
    while True:
        try:
            # non-block read all the queued xy, and use the latest one.
            # NOTE: if obj_dxy_queue is empty, we use the previous x,y, i.e., not update current obj pos.
            while not obj_dxy_queue.empty():
                obj_dx_in_img, obj_dy_in_img = obj_dxy_queue.get_nowait()

            # NOTE: must transform obj xy in img coordinate to robot xy coordinate.
            # --- current_x += d_current_x
            target_positions['shoulder_pan'] += obj_dx_in_img
            target_y += obj_dy_in_img   # Negative sign: up in image = increase y

            # Update joint targets using inverse kinematics
            joint2_target, joint3_target = inverse_kinematics(target_x, target_y)
            target_positions['shoulder_lift'] = joint2_target
            target_positions['elbow_flex'] = joint3_target
            # print(f"{label.capitalize()} center offset: dx={dx}, dy={dy} -> dc_y={d_current_y}, pan: {target_positions['shoulder_pan']:.2f}, y: {current_y:.3f}, joint2: {joint2_target:.2f}, joint3: {joint3_target:.2f}")

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

                    # pitch control
                    case 'r':
                        pitch += pitch_step
                        print(f"Increase pitch adjustment: {pitch:.3f}")
                    case 'f':
                        pitch -= pitch_step
                        print(f"Decrease pitch adjustment: {pitch:.3f}")

                    case k_jnt if k_jnt in joint_controls:
                        joint_name, delta = joint_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"Update target position {joint_name}: {current_target} -> {new_target}")

                    case k_xy if k_xy in xy_controls:
                        coord, delta = xy_controls[key]
                        if coord == 'x':
                            target_x += delta
                            # Calculate joint2 and joint3 target angles
                            joint2_target, joint3_target = inverse_kinematics(target_x, target_y)
                            target_positions['shoulder_lift'] = joint2_target
                            target_positions['elbow_flex'] = joint3_target
                            print(f"Update x coordinate: {target_x:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            target_y += delta
                            # Calculate joint2 and joint3 target angles
                            joint2_target, joint3_target = inverse_kinematics(target_x, target_y)
                            target_positions['shoulder_lift'] = joint2_target
                            target_positions['elbow_flex'] = joint3_target
                            print(f"Update y coordinate: {target_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")

            # Apply pitch adjustment to wrist_flex
            # Calculate wrist_flex target position based on shoulder_lift and elbow_flex
            if 'shoulder_lift' in target_positions and 'elbow_flex' in target_positions:
                target_positions['wrist_flex'] = - target_positions['shoulder_lift'] - target_positions['elbow_flex'] + pitch
                # Display current pitch value (every 100 steps to avoid spam)
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
                    # TODO: commented by kenn. temply.
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
    print("LeRobot Simplified Keyboard Control Example (P Control)")
    print("="*50)

    # Initialize x,y coordinate control
    x0, y0 = 0.1629, 0.1131
    print(f"Initialize end effector position: x={x0:.4f}, y={y0:.4f}")

    # Initialize YOLO and camera
    model = YOLO("yolo11x.pt").to('cuda')  # or select yoloe-11s/m-seg.pt for different sizes
    print(f" ==== YOLOE Model is running on: {model.device} =====")
    print(f" ==== YOLOE Model is running on: {model.device} =====")
    print(f" ==== YOLOE Model is running on: {model.device} =====")

    # Get detection targets from user input
    print("\n" + "=" * 60)
    print("YOLO Detection Target Setup")
    print("=" * 60)
    print(f'loaded yolo model supported detection objects: {model.names}')

    object_cls_id :int = 39
    try:
        object_cls_id = int(input(
            "Enter one object class ID to detect: default is ID of 'bottle' if press enter -> "))
    except ValueError:
        print("Enter ID error, we use default ID 39 of 'bottle'")

    print(f'--- Using detection target ID: {object_cls_id} -----')

    # If Enter is pressed directly, use default targets
    # we only follow one kind object.
    # if not target_input:
    #     target_object = "bottle"
    #     print(f"Using default targets: {target_object}")
    # else:
    #     # Parse multiple objects separated by commas
    #     # target_objects = [obj.strip() for obj in target_input.split(',') if obj.strip()]
    #     target_object = target_input
    #     print(f"Detection targets: {target_object}")

    # Set text prompt to detect the specified objects
    # model.set_classes([target_object], model.get_text_pe([target_object]))

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
    obj_dxy_queue = Queue(maxsize=100)
    # Start video stream in a separate thread
    video_thread = Thread(target=vision_control_update_v2,
                          kwargs=dict(
                              obj_dxy_queue = obj_dxy_queue,
                              model = model,
                              camera_index = camera_index,
                              stop_event = stop_signal,
                              object_cls_id = object_cls_id),
                          daemon=True)
    try:

        video_thread.start()
        keyboard_teleop_SO101arm_task(p_control_loop=partial(p_control_loop,
															 init_x=x0,
															 init_y=y0,
															 obj_dxy_queue = obj_dxy_queue,
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
        # cv2.destroyAllWindows()
        print("Program ended")


if __name__ == "__main__":
    main() 