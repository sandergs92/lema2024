import cv2
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

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

collided_objects = 0

class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.robot = rob
        self.action_space = spaces.Discrete(4)  # 5s discrete actions: steer left, steer right, gas, brake, reverse
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)  # IR sensor readings
        self.reset()
        self.previous_actions = []
        self.explored_positions = set()

    def detect_color(self, image):
        # Capture the image from the front camera
        imageFrame = image

        # Convert the imageFrame in BGR(RGB color space) to HSV(hue-saturation-value) color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Set range for red color and define mask
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        # Set range for green color and define mask
        green_lower = np.array([25, 52, 72], np.uint8)
        green_upper = np.array([102, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        # Set range for blue color and define mask
        blue_lower = np.array([94, 80, 2], np.uint8)
        blue_upper = np.array([120, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

        # Morphological Transform, Dilation for each color and bitwise_and operator
        kernel = np.ones((5, 5), "uint8")

        # For red color
        red_mask = cv2.dilate(red_mask, kernel)
        res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

        # For green color
        green_mask = cv2.dilate(green_mask, kernel)
        res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)

        # For blue color
        blue_mask = cv2.dilate(blue_mask, kernel)
        res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

        green_detected = False
        green_area = 0

        # Creating contour to track red color
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                # imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        # Creating contour to track green color
        contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                # imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                green_detected = True
                green_area = w * h

        # Creating contour to track blue color
        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

        print("Green detected: ", green_detected)
        print("Green area (width X height): ", green_area)

        return green_detected, green_area
    
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
        self.previous_actions = []
        self.explored_positions = set()
        return np.array(state, dtype=np.float32), {}

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
        time.sleep(1)

        next_state = self.robot.read_irs()
        next_state = np.array(next_state, dtype=np.float32)

        # Compute reward
        # print((round(self.robot.get_position().x, 2), round(self.robot.get_position().y, 2)))
        reward, done = self.compute_reward(next_state, action, (round(self.robot.get_position().x, 2), round(self.robot.get_position().y, 2)))

        info = {}

        return next_state, reward, done, False, info

    def compute_reward(self, irs_values, action_taken, current_position):
        # Count the frequency of the current action in the previous actions
        # action_frequency = self.previous_actions.count(action_taken) if action_taken != 2 else 0
        # penalty = -2 * action_frequency if action_frequency > 0 else 0

        # self.previous_actions.append(action_taken)
        # if len(self.previous_actions) > 4:
        #     self.previous_actions.pop(0)
        
        # Max distance from obstacle
        max_distance = max(v for v in irs_values if v is not None)
        if max_distance >= 300:  # Threshold distance for being too close to an obstacle
            return -100, True

        # Reward for moving towards new areas
        if current_position not in self.explored_positions:
            self.explored_positions.add(current_position)
            exploration_reward = 5  # Higher reward for new exploration
        else:
            exploration_reward = -5  # Penalty if the area is already explored

        # Base rewards for actions
        base_reward = 3 if action_taken == 2 else 1

        total_reward = base_reward + exploration_reward

        return total_reward, False

    def render(self, mode='human', close=False):
        pass  # No rendering required for this example

def steer_left(rob: IRobobo):
    rob.move_blocking(-25, 25, 500)

def steer_right(rob: IRobobo):
    rob.move_blocking(25, -25, 500)

def gas(rob: IRobobo):
    rob.move_blocking(50, 50, 500)

def reverse(rob: IRobobo):
    rob.move_blocking(-50, -50, 500)

# def run_all_actions(rob: IRobobo, dataset):
def run_all_actions(rob: IRobobo):
    env = RoboboEnv(rob)
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    print("Phone tilt angle: ", rob.read_phone_tilt())

    rob.set_phone_tilt(100, 100)
    img = rob.get_image_front()
    env.detect_color(img)

    while rob.nr_food_collected() == 0:
        img = rob.get_image_front()
        rob.move_blocking(-50, 50, 500)
        rob.move_blocking(50, 50, 1000)
        rob.move_blocking(50, -50, 500)
        rob.move_blocking(50, 50, 1000)
        print("Food collected: ", rob.nr_food_collected())

    env.detect_color(img)

    rob.move_blocking(50, 50, 500)
    rob.move_blocking(100, 100, 500)
    rob.move_blocking(-50, 50, 500)
    rob.move_blocking(100, 100, 500)
    rob.move_blocking(-50, 50, 500)
    rob.move_blocking(100, 100, 500)
    rob.move_blocking(-50, 50, 500)

    rob.sleep(5)

    rob.stop_simulation()


    # if dataset == 'train':
    #     env = RoboboEnv(rob)
    #     env = Monitor(env, str(FIGRURES_DIR))
        
    #     # Configure logging
    #     new_logger = configure(str(FIGRURES_DIR), ["stdout", "csv", "tensorboard"])

    #     # Create the DQN model
    #     model = DQN('MlpPolicy', env, verbose=1)
    #     model.set_logger(new_logger)

    #     # Train the model
    #     TIMESTEPS = 1000
    #     for i in range(1, 30):
    #         print('RUN: ', str(i))
    #         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    #         model.save(f"{FIGRURES_DIR}/{TIMESTEPS * i}")
    # elif dataset == 'validation':
    #     env = RoboboEnv(rob)
    #     env = Monitor(env, str(FIGRURES_DIR))
    #     timestep = 1000
    #     for i in range(27):
    #         model = DQN.load(f"{FIGRURES_DIR}/{timestep}.zip", env=env)
    #         obs = env.reset()[0]
    #         for _ in range(1000):
    #             action, _states = model.predict(obs)
    #             obs, reward, done, test, info = env.step(action)
    # elif dataset == 'testing':
    #     model = DQN.load(f"{FIGRURES_DIR}/15000.zip")
    #     while True:
    #         next_state = rob.read_irs()
    #         next_state = np.array(next_state, dtype=np.float32)
    #         action = model.predict(next_state)[0]
    #         if action == 0:
    #             steer_left(rob)
    #         elif action == 1:
    #             steer_right(rob)
    #         elif action == 2:
    #             gas(rob)
    #         elif action == 3:
    #             reverse(rob)
    #         time.sleep(0.5)



