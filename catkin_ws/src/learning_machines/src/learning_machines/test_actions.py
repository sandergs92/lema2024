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

class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.robot = rob
        self.action_space = spaces.Discrete(4)  # 5s discrete actions: steer left, steer right, gas, brake, reverse
        self.observation_space = spaces.Dict({
            "sensor_readings": spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),  # IR sensor readings
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)  # Image of size 64x64 with 3 channels (RGB)
        })
        self.current_num_food = 0
        self.start_time = time.time()
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
        self.current_num_food = 0
        self.start_time = time.time()
        self.robot.set_phone_tilt(100, 100)
        processed_image, black_percentage, original_percentage = process_image(self.robot.get_image_front())
        resized_image = cv.resize(processed_image, (64,64), cv.INTER_AREA)
        return {"sensor_readings": np.array(state, dtype=np.float32), "image": resized_image}, {}

    def step(self, action):
        # Execute one time step within the environment

        if action == 0:
            steer_left(self.robot)
        elif action == 1:
            steer_right(self.robot)
        elif action == 2:
            gas(self.robot)
        elif action == 3:
            reverse(self.robot)

        # Wait for the action to complete
        time.sleep(0.5)

        # Get new observation
        sensor_readings = self.robot.read_irs()
        processed_image, black_percentage, original_percentage = process_image(self.robot.get_image_front())
        # cv.imwrite(f"{FIGRURES_DIR}/{time.time()}.jpg", processed_image) 
        resized_image = cv.resize(processed_image, (64,64), cv.INTER_AREA)
        next_state = {"sensor_readings": np.array(sensor_readings, dtype=np.float32), "image": resized_image}

        # Compute reward
        reward, done = self.compute_reward(next_state, action, black_percentage, original_percentage)

        info = {}

        return next_state, reward, done, False, info

    def compute_reward(self, next_state, action_taken, black_percentage, original_percentage):
        # Give time
        time_diff = time.time() - self.start_time
        if time_diff >= 180:
            return 0, True

        # Max distance from obstacle
        max_distance = max(v for v in next_state["sensor_readings"] if v is not None)
        if max_distance >= 300 and black_percentage >= 99.:  # Threshold distance for being too close to an obstacle
            # print("Bumped in front of wall!")
            return -100, True
        
        # Spotting reward, penalize if no food in image
        if original_percentage > 0.:
            spot_reward = 50 * (original_percentage / 100)
        else:
            spot_reward = -5
        
        # Food reward
        if self.robot.nr_food_collected() > self.current_num_food:
            difference = self.robot.nr_food_collected() - self.current_num_food
            food_reward = difference * 50
            self.current_num_food = self.robot.nr_food_collected()
        else:
            food_reward = 0

        # Base rewards for actions
        base_reward = 2 if action_taken == 2 else -1

        # print(base_reward, food_reward, spot_reward)
        total_reward = base_reward + food_reward + spot_reward

        # End after 3 mins
        if self.robot.get_sim_time() >= 180:
            # print("TIMES UP")
            return total_reward, True

        return total_reward, False

    def render(self, mode='human', close=False):
        pass  # No rendering required for this example

def process_image(image, colors={'green': [np.array([33, 19, 105]), np.array([77, 255, 255])]}, min_area=5000):
    def find_colors(frame, points):
        mask = cv.inRange(frame, points[0], points[1])  # Create mask with boundaries
        cnts, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # Find contours from mask
        contours_info = []
        for c in cnts:
            M = cv.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])  # Calculate X position
                cy = int(M['m01'] / M['m00'])  # Calculate Y position
                contours_info.append(c)
        return contours_info

    frame = image
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Conversion to HSV
    mask = np.zeros(frame.shape[:2], dtype="uint8")  # Create a black mask

    for name, clr in colors.items():  # For each color in colors
        contours_info = find_colors(hsv, clr)
        for c in contours_info:
            area = cv.contourArea(c)  # Calculate contour area
            if area > min_area:  # Draw filled contour only if contour is big enough
                cv.drawContours(mask, [c], -1, 255, -1)  # Draw filled contour on the mask

    # Bitwise AND operation to keep only the region inside the contour
    result = cv.bitwise_and(frame, frame, mask=mask)

    # Calculate the percentages of black and original content
    total_pixels = frame.shape[0] * frame.shape[1]
    black_pixels = total_pixels - cv.countNonZero(mask)
    original_pixels = cv.countNonZero(mask)

    black_percentage = (black_pixels / total_pixels) * 100
    original_percentage = (original_pixels / total_pixels) * 100

    return result, black_percentage, original_percentage

def steer_left(rob: IRobobo):
    rob.move_blocking(-25, 25, 500)

def steer_right(rob: IRobobo):
    rob.move_blocking(25, -25, 500)

def gas(rob: IRobobo):
    rob.move_blocking(100, 100, 500)

def reverse(rob: IRobobo):
    rob.move_blocking(-50, -50, 500)


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
        print("Avg reward:", sum(rewards) / len(rewards) )
        print(rewards)
    elif dataset == 'testing':
        model = DQN.load(f"{FIGRURES_DIR}/15000.zip")
        while True:
            # Get new observation
            sensor_readings = rob.read_irs()
            processed_image, black_percentage, original_percentage = process_image(rob.get_image_front())
            # cv.imwrite(f"{FIGRURES_DIR}/{time.time()}.jpg", processed_image) 
            resized_image = cv.resize(processed_image, (64,64), cv.INTER_AREA)
            next_state = {"sensor_readings": np.array(sensor_readings, dtype=np.float32), "image": resized_image}
            action = model.predict(next_state)[0]
            if action == 0:
                steer_left(rob)
            elif action == 1:
                steer_right(rob)
            elif action == 2:
                gas(rob)
            elif action == 3:
                reverse(rob)
            time.sleep(0.5)

