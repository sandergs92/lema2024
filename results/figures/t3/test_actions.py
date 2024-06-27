import cv2 as cv
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

# COLORS = {'blue': [np.array([95, 255, 85]), np.array([120, 255, 255])],
#           'red': [np.array([161, 165, 127]), np.array([178, 255, 255])],
#           'yellow': [np.array([16, 0, 99]), np.array([39, 255, 255])],
#           'green': [np.array([33, 19, 105]), np.array([77, 255, 255])]}

COLORS = {'red': ([np.array([0, 120, 70]), np.array([10, 255, 255])], (0, 0, 255)),
          'green': ([np.array([33, 19, 105]), np.array([77, 255, 255])], (0, 255, 0))}


class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.robot = rob
        # 3 discrete actions: steer left, steer right, gas
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            # IR sensor readings
            "sensor_readings": spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
            # Image of size 64x64 with 3 channels (RGB)
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })
        self.explored_positions = set()
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        if isinstance(self.robot, SimulationRobobo):
            try:
                self.robot.stop_simulation()
            except Exception as e:
                print(f"Error during reset: {e}")
            self.robot.play_simulation()
            time.sleep(0.5)
            state = [0.] * 8
        self.robot.set_phone_tilt(109, 100)
        time.sleep(0.5)
        processed_image, biggest_contours = process_image(
            self.robot.get_image_front())
        resized_image = cv.resize(processed_image, (64, 64), cv.INTER_AREA)
        self.explored_positions = set()
        return {"sensor_readings": np.array(state, dtype=np.float32), "image": resized_image}, {}

    def step(self, action):
        # Execute one time step within the environment

        if action == 0:
            steer_left(self.robot)
        elif action == 1:
            steer_right(self.robot)
        elif action == 2:
            gas(self.robot)

        # Wait for the action to complete
        time.sleep(0.5)

        # Get new observation
        sensor_readings = self.robot.read_irs()
        sensor_camera = self.robot.get_image_front()
        processed_image, biggest_contours = process_image(
            sensor_camera)
        # cv.imwrite(f"{FIGRURES_DIR}/{time.time()}.jpg", processed_image)
        resized_image = cv.resize(processed_image, (64, 64), cv.INTER_AREA)
        next_state = {"sensor_readings": np.array(
            sensor_readings, dtype=np.float32), "image": resized_image}

        # Compute reward
        reward, done = self.compute_reward(
            next_state, action, processed_image, biggest_contours, (round(self.robot.get_position().x, 2), round(self.robot.get_position().y, 2)))

        info = {}

        return next_state, reward, done, False, info

    def compute_reward(self, next_state, action_taken, processed_image, biggest_contours, current_position):
        # If the robot detects food at the base, give a high reward and signal the end of the episode
        if self.robot.base_detects_food():
            return 1000, True

        # Base reward for every action to encourage movement
        base_reward = -1

        # Maximum sensor reading distance
        max_distance = max(v for v in next_state["sensor_readings"] if v is not None)

        # Penalize if too close to an obstacle
        if max_distance >= 150 and "red" not in biggest_contours:
            return -100, True

        # Shape of the processed image
        height, width, _ = processed_image.shape
        img_center_x = width // 2

        # Initialize additional rewards
        center_reward = 0
        explore_reward = 0

        if "red" in biggest_contours:
            _, cx, _ = biggest_contours["red"]
            distance_from_center = abs(cx - img_center_x)

            # Define threshold for being considered in the center (10% of image width)
            center_threshold = width * 0.1

            if distance_from_center < center_threshold:
                # Reward for keeping the red block in the center
                center_reward = 50
                if action_taken == 2:  # Additional reward for moving forward
                    center_reward += 50
                    center_reward += 50 * self.robot._base_food_distance()
                if current_position not in self.explored_positions:
                    self.explored_positions.add(current_position)
                    explore_reward = 50  # Higher reward for new exploration
                else:
                    explore_reward = -10  # Smaller penalty if the area is already explored
            else:
                # Penalize if the block is far from the center
                center_reward = -distance_from_center / img_center_x * 10

        total_reward = base_reward + center_reward + explore_reward
        return total_reward, False


    def render(self, mode='human', close=False):
        pass  # No rendering required for this example


def process_image(image, colors=COLORS, min_area=5000):
    def find_colors(frame, points):
        # Create mask with boundaries
        mask = cv.inRange(frame, points[0], points[1])
        # Find contours from mask
        cnts, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_info = []
        for c in cnts:
            M = cv.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])  # Calculate X position
                cy = int(M['m01'] / M['m00'])  # Calculate Y position
                contours_info.append((c, cx, cy))
        return contours_info

    frame = image
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Conversion to HSV
    mask = np.zeros(frame.shape[:2], dtype="uint8")  # Create a black mask
    biggest_contours = {}

    for name, (clr, bgr) in colors.items():  # For each color in colors
        contours_info = find_colors(hsv, clr)
        biggest_contour = None
        max_area = 0
        for c, cx, cy in contours_info:
            area = cv.contourArea(c)  # Calculate contour area
            if area > max_area:
                max_area = area
                biggest_contour = (c, cx, cy)
            if area > min_area:  # Draw filled contour only if contour is big enough
                # Draw filled contour on the mask
                cv.drawContours(mask, [c], -1, 255, -1)
                cv.circle(frame, (cx, cy), 7, bgr, -1)  # draw circle
        if biggest_contour:
            biggest_contours[name] = biggest_contour

    # Bitwise AND operation to keep only the region inside the contour
    result = cv.bitwise_and(frame, frame, mask=mask)
    return result, biggest_contours


def steer_left(rob: IRobobo):
    rob.move_blocking(-25, 25, 500)


def steer_right(rob: IRobobo):
    rob.move_blocking(25, -25, 500)


def gas(rob: IRobobo):
    rob.move_blocking(100, 100, 500)


def run_all_actions(rob: IRobobo, dataset):

    if dataset == 'train':
        env = RoboboEnv(rob)
        env = Monitor(env, str(FIGRURES_DIR))
        # Create the DQN model
        model = DQN('MultiInputPolicy', env, verbose=1)

        # Train the model
        TIMESTEPS = 1000
        for i in range(1, 30):
            # print('RUN: ', str(i))
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            model.save(f"{FIGRURES_DIR}/{TIMESTEPS * i}")
    elif dataset == 'validation':
        env = RoboboEnv(rob)
        env = Monitor(env, str(FIGRURES_DIR))
        model = DQN.load(f"{FIGRURES_DIR}/slow_model/2000.zip", env=env)
        obs = env.reset()[0]
        total_reward = 0
        rewards = []
        for i in range(1000):
            action, _states = model.predict(obs)
            obs, reward, done, test, info = env.step(action)
            total_reward += reward
            if done:
                print(i)
                obs = env.reset()[0]
                rewards.append(total_reward)
                total_reward = 0
        print("Max reward:", max(rewards))
        print("Min reward:", min(rewards))
        print("Avg reward:", sum(rewards) / len(rewards))
        print(rewards)
    elif dataset == 'testing':
        model = DQN.load(f"{FIGRURES_DIR}/15000.zip")
        while True:
            # Get new observation
            sensor_readings = rob.read_irs()
            processed_image, biggest_contours = process_image(
                rob.get_image_front())
            # cv.imwrite(f"{FIGRURES_DIR}/{time.time()}.jpg", processed_image)
            resized_image = cv.resize(processed_image, (64, 64), cv.INTER_AREA)
            next_state = {"sensor_readings": np.array(
                sensor_readings, dtype=np.float32), "image": resized_image}
            action = model.predict(next_state)[0]
            if action == 0:
                steer_left(rob)
            elif action == 1:
                steer_right(rob)
            elif action == 2:
                gas(rob)
            time.sleep(0.5)
