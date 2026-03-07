from typing import Dict,cast, Callable
import time
import math
import draccus
from dataclasses import dataclass
import traceback
import logging
from functools import partial

from lerobot.motors import MotorCalibration
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
)
from lerobot.robots.so_follower import (SO101Follower,
                                        SO101FollowerConfig)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig


from software.src.robots.xlerobot_2wheels import XLerobot2WheelsConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Keymaps (semantic action: key)
LEFT_KEYMAP = {
	'shoulder_pan+': 'q', 'shoulder_pan-': 'e',
	'wrist_roll+': 'r', 'wrist_roll-': 'f',
	'gripper+': 't', 'gripper-': 'g',
	'x+': 'w', 'x-': 's', 'y+': 'a', 'y-': 'd',
	'pitch+': 'z', 'pitch-': 'x',
	'reset': 'c',
	# For head motors
	"head_motor_1+": "<", "head_motor_1-": ">",
	"head_motor_2+": ",", "head_motor_2-": ".",

	'triangle': 'y',  # Rectangle trajectory key
}
RIGHT_KEYMAP = {
	'shoulder_pan+': '7', 'shoulder_pan-': '9',
	'wrist_roll+': '/', 'wrist_roll-': '*',
	'gripper+': '+', 'gripper-': '-',
	'x+': '8', 'x-': '2', 'y+': '4', 'y-': '6',
	'pitch+': '1', 'pitch-': '3',
	'reset': '0',

	'triangle': 'Y',  # Rectangle trajectory key
}

LEFT_JOINT_MAP = {
	"shoulder_pan": "left_arm_shoulder_pan",
	"shoulder_lift": "left_arm_shoulder_lift",
	"elbow_flex": "left_arm_elbow_flex",
	"wrist_flex": "left_arm_wrist_flex",
	"wrist_roll": "left_arm_wrist_roll",
	"gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
	"shoulder_pan": "right_arm_shoulder_pan",
	"shoulder_lift": "right_arm_shoulder_lift",
	"elbow_flex": "right_arm_elbow_flex",
	"wrist_flex": "right_arm_wrist_flex",
	"wrist_roll": "right_arm_wrist_roll",
	"gripper": "right_arm_gripper",
}

# Head motor mapping
HEAD_MOTOR_MAP = {
	"head_motor_1": "head_motor_1",
	"head_motor_2": "head_motor_2",
}


class RectangularTrajectory:
	"""
	Generates a rectangular trajectory on the x-y plane with sinusoidal velocity profiles.
	The rectangle is divided into 4 line segments, each with smooth acceleration/deceleration.
	"""

	def __init__(self, width=0.06, height=0.06, segment_duration=0.91):
		"""
		Initialize rectangular trajectory parameters.

		Args:
			width: Rectangle width in meters
			height: Rectangle height in meters
			segment_duration: Time for each line segment in seconds
		"""
		self.width = width
		self.height = height
		self.segment_duration = segment_duration
		self.total_duration = 4 * segment_duration

	def get_trajectory_point(self, current_x, current_y, t):
		"""
		Get the target x, y position at time t for the rectangular trajectory.

		Args:
			current_x: Starting x position
			current_y: Starting y position
			t: Time since trajectory start (0 to total_duration)

		Returns:
			tuple: (target_x, target_y)
		"""
		# Determine which segment we're in
		segment = int(t / self.segment_duration)
		segment_t = t % self.segment_duration

		# Normalize segment time (0 to 1)
		normalized_t = segment_t / self.segment_duration

		# Sinusoidal velocity profile: smooth acceleration and deceleration
		# s(t) = 0.5 * (1 - cos(π * t)) gives smooth 0 to 1 transition
		smooth_t = 0.5 * (1 - math.cos(math.pi * normalized_t))

		# Define rectangle corners relative to starting position
		corners = [
			(current_x, current_y),  # Start (bottom-left)
			(current_x + self.width, current_y),  # Bottom-right
			(current_x + self.width, current_y + self.height),  # Top-right
			(current_x, current_y + self.height),  # Top-left
			(current_x, current_y)  # Back to start
		]

		# Clamp segment to valid range
		segment = max(0, min(3, segment))

		# Interpolate between current corner and next corner
		start_corner = corners[segment]
		end_corner = corners[segment + 1]

		target_x = start_corner[0] + smooth_t * (end_corner[0] - start_corner[0])
		target_y = start_corner[1] + smooth_t * (end_corner[1] - start_corner[1])

		return target_x, target_y


class SimpleHeadControl:
	def __init__(self, initial_obs, kp=0.81):
		self.kp = kp
		self.degree_step = 1
		# Initialize head motor positions
		self.target_positions = {
			"head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
			"head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
		}
		self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

	def move_to_zero_position(self, robot):
		self.target_positions = self.zero_pos.copy()
		action = self.p_control_action(robot)
		robot.send_action(action)

	def handle_keys(self, key_state):
		if key_state.get('head_motor_1+'):
			self.target_positions["head_motor_1"] += self.degree_step
			print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
		if key_state.get('head_motor_1-'):
			self.target_positions["head_motor_1"] -= self.degree_step
			print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
		if key_state.get('head_motor_2+'):
			self.target_positions["head_motor_2"] += self.degree_step
			print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
		if key_state.get('head_motor_2-'):
			self.target_positions["head_motor_2"] -= self.degree_step
			print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")

	def p_control_action(self, robot):
		obs = robot.get_observation()
		action = {}
		for motor in self.target_positions:
			current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
			error = self.target_positions[motor] - current
			control = self.kp * error
			action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
		return action


class SimpleTeleopArm:
	def __init__(self, kinematics, joint_map, initial_obs, prefix="left", kp=0.81):
		self.kinematics = kinematics
		self.joint_map = joint_map
		self.prefix = prefix  # To distinguish left and right arm
		self.kp = kp
		# Initial joint positions
		self.joint_positions = {
			"shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
			"shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
			"elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
			"wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
			"wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
			"gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
		}
		# Set initial x/y to fixed values
		self.current_x = 0.1629
		self.current_y = 0.1131
		self.pitch = 0.0
		# Set the degree step and xy step
		self.degree_step = 3
		self.xy_step = 0.0081
		# Set target positions to zero for P control
		self.target_positions = {
			"shoulder_pan": 0.0,
			"shoulder_lift": 0.0,
			"elbow_flex": 0.0,
			"wrist_flex": 0.0,
			"wrist_roll": 0.0,
			"gripper": 0.0,
		}
		self.zero_pos = {
			'shoulder_pan': 0.0,
			'shoulder_lift': 0.0,
			'elbow_flex': 0.0,
			'wrist_flex': 0.0,
			'wrist_roll': 0.0,
			'gripper': 0.0
		}

		# Rectangular trajectory instance
		self.rectangular_trajectory = RectangularTrajectory(
			width=0.06,  # 6cm wide rectangle
			height=0.06,  # 4cm tall rectangle
			segment_duration=1.01  # 3 seconds per line segment
		)

	def move_to_zero_position(self, robot):
		print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
		self.target_positions = self.zero_pos.copy()  # Use copy to avoid reference issues

		# Reset kinematic variables to their initial state
		self.current_x = 0.1629
		self.current_y = 0.1131
		self.pitch = 0.0

		# Don't let handle_keys recalculate wrist_flex - set it explicitly
		self.target_positions["wrist_flex"] = 0.0

		action = self.p_control_action(robot)
		robot.send_action(action)

	def execute_rectangular_trajectory(self, robot, fps=30):
		"""
		Execute a blocking rectangular trajectory on the x-y plane.

		Args:
			robot: Robot instance to send actions to
			fps: Control loop frequency
		"""
		print(f"[{self.prefix}] Starting rectangular trajectory...")
		print(
			f"[{self.prefix}] Rectangle: {self.rectangular_trajectory.width:.3f}m x {self.rectangular_trajectory.height:.3f}m")
		print(f"[{self.prefix}] Duration: {self.rectangular_trajectory.total_duration:.3f}s total")

		# Store starting position
		start_x = self.current_x
		start_y = self.current_y

		# Execute trajectory
		start_time = time.time()
		dt = 1.0 / fps

		while True:
			current_time = time.time()
			elapsed_time = current_time - start_time

			# Check if trajectory is complete
			if elapsed_time >= self.rectangular_trajectory.total_duration:
				print(f"[{self.prefix}] Rectangular trajectory completed!")
				break

			# Get target position from trajectory
			target_x, target_y = self.rectangular_trajectory.get_trajectory_point(
				start_x, start_y, elapsed_time
			)

			# Update current position
			self.current_x = target_x
			self.current_y = target_y

			# Calculate inverse kinematics
			try:
				joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
				self.target_positions["shoulder_lift"] = joint2
				self.target_positions["elbow_flex"] = joint3

				# Update wrist_flex coupling
				self.target_positions["wrist_flex"] = (
						-self.target_positions["shoulder_lift"]
						- self.target_positions["elbow_flex"]
						+ self.pitch
				)

				# Get action
				action = self.p_control_action(robot)

				robot_action = {}
				# Determine which arm is executing and send appropriate action structure
				if self.prefix == "left":
					# Send left arm action with empty actions for other components
					robot_action = {**action, **{}, **{}, **{}}
				elif self.prefix == "right":
					# Send right arm action with empty actions for other components
					robot_action = {**{}, **action, **{}, **{}}

				# Send action to robot
				robot.send_action(robot_action)

				# Get observation and log data
				# obs = robot.get_observation()
				# log_rerun_data(obs, robot_action)

			except Exception as e:
				print(f"[{self.prefix}] IK failed at x={self.current_x:.4f}, y={self.current_y:.4f}: {e}")
				break

		# Maintain control frequency
		# busy_wait(dt)
		time.sleep(dt)

		print(f"[{self.prefix}] Trajectory execution finished.")

	def handle_keys(self, key_state):
		# Joint increments
		if key_state.get('shoulder_pan+'):
			self.target_positions["shoulder_pan"] += self.degree_step
			print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
		if key_state.get('shoulder_pan-'):
			self.target_positions["shoulder_pan"] -= self.degree_step
			print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
		if key_state.get('wrist_roll+'):
			self.target_positions["wrist_roll"] += self.degree_step
			print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
		if key_state.get('wrist_roll-'):
			self.target_positions["wrist_roll"] -= self.degree_step
			print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
		if key_state.get('gripper+'):
			self.target_positions["gripper"] += self.degree_step
			print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
		if key_state.get('gripper-'):
			self.target_positions["gripper"] -= self.degree_step
			print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
		if key_state.get('pitch+'):
			self.pitch += self.degree_step
			print(f"[{self.prefix}] pitch: {self.pitch}")
		if key_state.get('pitch-'):
			self.pitch -= self.degree_step
			print(f"[{self.prefix}] pitch: {self.pitch}")

		# XY plane (IK)
		moved = False
		if key_state.get('x+'):
			self.current_x += self.xy_step
			moved = True
			print(f"[{self.prefix}] x+: {self.current_x:.4f}, y: {self.current_y:.4f}")
		if key_state.get('x-'):
			self.current_x -= self.xy_step
			moved = True
			print(f"[{self.prefix}] x-: {self.current_x:.4f}, y: {self.current_y:.4f}")
		if key_state.get('y+'):
			self.current_y += self.xy_step
			moved = True
			print(f"[{self.prefix}] x: {self.current_x:.4f}, y+: {self.current_y:.4f}")
		if key_state.get('y-'):
			self.current_y -= self.xy_step
			moved = True
			print(f"[{self.prefix}] x: {self.current_x:.4f}, y-: {self.current_y:.4f}")
		if moved:
			joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
			self.target_positions["shoulder_lift"] = joint2
			self.target_positions["elbow_flex"] = joint3
			print(f"[{self.prefix}] shoulder_lift: {joint2}, elbow_flex: {joint3}")

		# Wrist flex is always coupled to pitch and the other two
		self.target_positions["wrist_flex"] = (
				-self.target_positions["shoulder_lift"]
				- self.target_positions["elbow_flex"]
				+ self.pitch
		)

	# print(f"[{self.prefix}] wrist_flex: {self.target_positions['wrist_flex']}")

	def p_control_action(self, robot):
		obs = robot.get_observation()
		current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
		action = {}
		for j in self.target_positions:
			error = self.target_positions[j] - current[j]
			control = self.kp * error
			action[f"{self.joint_map[j]}.pos"] = current[j] + control
		return action


# P control calculation
def p_control_helper(*, robot: SO101Follower,
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

		# TODO: in case of tiny error, can not drive back to zero position.kenn.
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


# using linear interpolation.
def move_by_linear_interpolation(robots: Dict[str,SO101Follower],
								 target_pos: Dict[str, Dict[str,float]],
								 duration:float=3.0,
								 kp:float=0.5):
	assert robots.keys() == target_pos.keys()
	print("Using P control to slowly move robot to zero position...")

	# Calculate control steps
	control_freq = 20 # 50  # 50Hz control frequency
	total_steps = int(duration * control_freq)
	step_time = 1.0 / control_freq

	#
	# print(f"Will move from cur pos:{current_obs} to target position:{target_pos} \n"
	# 	  f"in {duration} seconds using P control,\n"
	# 	  f" control frequency: {control_freq}Hz, \n"
	# 	  f" proportional gain: {kp}")

	arm_ds: Dict[str, Dict[str,float]] = {}
	arm_obs: Dict[str, Dict[str,float]] = {}
	for _arm_name, _arm_target in target_pos.items():
		arm_ds[_arm_name] = {}
		arm_obs[_arm_name] = robots[_arm_name].get_observation()
		for _jnt_name, _jnt_target in _arm_target.items():
			error = float(_jnt_target - arm_obs[_arm_name][_jnt_name+'.pos'])
			arm_ds[_arm_name][_jnt_name] = error / total_steps

	def _single_arm_action(step_cnt:int, ds:Dict[str, float], obs:Dict[str, float])->Dict[str, float]:
		arm_action = {}
		for _jnt, _jnt_ds in ds.items():
			arm_action[f'{_jnt}.pos'] = obs[f'{_jnt}.pos'] + _jnt_ds * step_cnt
		return arm_action

	for step in range(total_steps):
		for _arm_name, _arm in robots.items():
			action = _single_arm_action(step, arm_ds[_arm_name], arm_obs[_arm_name])
			_arm.send_action(action)

		# Display progress
		if step % control_freq == 0:  # Display progress every 1 seconds
			progress = (step / total_steps) * 100
			print(f"Moving arm to zero position progress: {progress:.1f}%")

		# TODO: not include the bus delay time. inaccurate for control freq. kenn.
		time.sleep(step_time)
		# time.sleep(0.1)

	print("Robots moved to zero position")


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
	"""
	Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets

	Parameters:
		x: End effector x coordinate
		y: End effector y coordinate
		l1: Upper arm length (default 0.1159 m)
		l2: Lower arm length (default 0.1350 m)

	Returns:
		joint2, joint3: Joint angles in radians as defined in the URDF file
	"""
	# Calculate joint2 and joint3 offsets in theta1 and theta2
	theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
	theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0

	# Calculate distance from origin to target point
	r = math.sqrt(x ** 2 + y ** 2)
	r_max = l1 + l2  # Maximum reachable distance

	# If target point is beyond maximum workspace, scale it to the boundary
	if r > r_max:
		scale_factor = r_max / r
		x *= scale_factor
		y *= scale_factor
		r = r_max

	# If target point is less than minimum workspace (|l1-l2|), scale it
	r_min = abs(l1 - l2)
	if 0 < r < r_min:
		scale_factor = r_min / r
		x *= scale_factor
		y *= scale_factor
		r = r_min

	# Use law of cosines to calculate theta2
	cos_theta2 = -(r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)

	# Calculate theta2 (elbow angle)
	theta2 = math.pi - math.acos(cos_theta2)

	# Calculate theta1 (shoulder angle)
	beta = math.atan2(y, x)
	gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
	theta1 = beta + gamma

	# Convert theta1 and theta2 to joint2 and joint3 angles
	joint2 = theta1 + theta1_offset
	joint3 = theta2 + theta2_offset

	# Ensure angles are within URDF limits
	joint2 = max(-0.1, min(3.45, joint2))
	joint3 = max(-0.2, min(math.pi, joint3))

	# Convert from radians to degrees
	joint2_deg = math.degrees(joint2)
	joint3_deg = math.degrees(joint3)

	joint2_deg = 90 - joint2_deg
	joint3_deg = joint3_deg - 90

	return joint2_deg, joint3_deg


def SO101_arm_return_to_start_position(*,
									   robots: Dict[str, SO101Follower],
									   arm_start_positions: Dict[str, Dict[str, float]],
									   kp=0.2, control_freq=20):
	"""
	Use P control to return to start position

	Args:
		robots: Robot instance
		arm_start_positions: Start joint positions dictionary
		kp: Proportional gain
		control_freq: Control frequency (Hz)
	"""
	print(f"Returning to start position :{arm_start_positions}")

	control_period = 1.0 / control_freq
	max_steps = int(5.0 * control_freq)  # Maximum 5 seconds

	reached = {k:False for k in robots}
	for step in range(max_steps):
		for _arm_name, _arm in robots.items():
			if reached[_arm_name]:
				continue
			# Get current robot state
			cur_obs = _arm.get_observation()
			renamed = {}
			for key, value in cur_obs.items():
				if key.endswith('.pos'):
					motor_name = key.removesuffix('.pos')
					renamed[motor_name] = value  # Don't apply calibration coefficients

			total_error = p_control_helper(robot=_arm,
										   target_pos=arm_start_positions[_arm_name],
										   cur_pos=renamed,
										   kp=kp, req_total_error=True)
			# Check if start position is reached
			if True:
			# if step % control_freq == 0:
				print(f'{_arm_name} back to start pos total_error: {total_error}')

			if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
				print(f"{_arm_name} Returned to start position.")
				reached[_arm_name] = True

		if all(reached.values()):
			break

		time.sleep(control_period)

	print("Return to start position completed")


@dataclass
class XConfig:
	# teleop: TeleoperatorConfig | None = None
	teleop: None = None
	robot: RobotConfig | None = None

# def __post_init__(self)-> XLerobot2WheelsConfig | None:
#     if bool(self.teleop) == bool(self.robot):
#         raise ValueError("Choose either a teleop or a robot.")
#
#     self.device_cfg = self.robot if self.robot else self.teleop


@draccus.wrap()
def keyboard_teleop_task(cfg: XConfig,
						 p_control_loop:Callable )->None:
	logger.info(f'load config for x_lerbot_2_wheels --->\n{cfg}')

	"""Main function"""
	print("XLeRobot Simplified Keyboard Control Example (P Control)")
	print("=" * 50)

	def _build_single_arm(port, calibration_item_prefix:str) -> SO101Follower:
		arm_cfg = SO101FollowerConfig(port=port, id=cfg.robot.id, calibration_dir=cfg.robot.calibration_dir)
		arm = SO101Follower(arm_cfg)
		if not arm.calibration:
			raise ValueError(f'can not load arm calibration file in path:{arm.calibration_dir}.'
							 f' should calibrate arm first, on port {port}')
		else:
			print(f'load arm calibration from file: {arm.calibration_dir}')

		# we make some name conversion.
		renamed: Dict[str, MotorCalibration] = {}
		for _k, _v in arm.calibration.items():
			if _k.startswith(calibration_item_prefix):
				renamed[_k.removeprefix(calibration_item_prefix)] = _v

		arm.calibration = renamed
		arm.bus.calibration = renamed
		print(f'{calibration_item_prefix.split("_")[0]} SO101 arm calibration: {arm.calibration} ')
		return arm

	_build_left_arm = partial(_build_single_arm,
							  port=cast(XLerobot2WheelsConfig, cfg.robot).port_left,
							  calibration_item_prefix='left_arm_')
	_build_right_arm = partial(_build_single_arm,
							   port=cast(XLerobot2WheelsConfig, cfg.robot).port_right,
							   calibration_item_prefix='right_arm_')

	robots :Dict[str, SO101Follower] = {}
	# left_arm : SO101Follower | None = None
	# right_arm : SO101Follower | None = None
	while not robots:
	# while not left_arm and not right_arm:
		match input("Please choose left or right SO101 arm: [l/r/d]").strip().lower():
			case 'l':
				robots['left_arm'] = _build_left_arm()
				logger.info(f'choose left arm.')
			case 'r':
				robots['right_arm'] = _build_right_arm()
				logger.info(f'choose right arm.')
			case 'd':
				robots['left_arm'] = _build_left_arm()
				robots['right_arm'] = _build_right_arm()
				logger.info(f'choose dual arms.')
			case _:
				logger.warning(f'got illegal arm choice, only accept "l" or "r" or "d" ')

	# Configure keyboard
	keyboard_config = KeyboardTeleopConfig()
	keyboard = KeyboardTeleop(keyboard_config)

	try:
		# Connect devices
		arm_start_positions: Dict[str, Dict[str,float]] = {}
		for _arm_name, _arm in robots.items():
			logger.info(f'connect arm: {_arm_name}')
			_arm.connect(calibrate=False)

			print(f"Reading starting joint angles of arm: {_arm_name}")
			obs = _arm.get_observation()
			renamed_obs:Dict[str, int] = {}
			for k, v in obs.items():
				if k.endswith('.pos'):
					renamed_obs[k.removesuffix('.pos')] = int(v)  # Don't apply calibration coefficients

			print(f"{_arm_name} arm Starting joint angles:")
			for joint_name, position in renamed_obs.items():
				print(f"  {joint_name}: {position}°")

			arm_start_positions[_arm_name] = renamed_obs

		keyboard.connect()
		logger.info('keyboard connected.')

		# use zero position as origin.
		# Initialize target positions to zero positions (integers)
		zero_positions = {
			'left_arm': {
			'shoulder_pan': 0.0,
			'shoulder_lift': 0.0,
			'elbow_flex': 0.0,
			'wrist_flex': 0.0,
			'wrist_roll': 0.0,
			'gripper': 0.0},

			'right_arm': {
				'shoulder_pan': 0.0,
				'shoulder_lift': 0.0,
				'elbow_flex': 0.0,
				'wrist_flex': 0.0,
				'wrist_roll': 0.0,
				'gripper': 0.0},
		}


		move_by_linear_interpolation(robots=robots,
									 target_pos={_k:zero_positions[_k] for _k in robots},
									 duration=5.0,
									 kp=0.3)

		# Start P control loop
		p_control_loop(robots=robots,
					   keyboard=keyboard,
					   arm_target_positions={_k:zero_positions[_k] for _k in robots},
					   arm_start_positions=arm_start_positions)
		# p_control_loop(robot, keyboard, target_positions, start_positions, kp=0.5, control_freq=50)

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
		for _arm_name, _arm in robots.items():
			if _arm.is_connected:
				print(f'disconnecting from {_arm_name}...')
				_arm.disconnect()

		if keyboard.is_connected:
			print(f'disconnected from keyboard...')
			keyboard.disconnect()

