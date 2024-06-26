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
        self.previous_distance_to_base = None
        self.has_red_object = False
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

        # Initialize the previous distance to the base
        robot_pos = self.robot.get_position()
        base_pos = self.robot.base_position()
        self.previous_distance_to_base = self.calculate_distance(robot_pos, base_pos)

        self.has_red_object = False
        
        return {"sensor_readings": np.array(state, dtype=np.float32), "image": resized_image}, {}

    def step(self, action):
        # Execute one time step within the environment

        if action == 0:
            steer_left(self.robot)
        elif action == 1:
            steer_right(self.robot)
        elif action == 2:
            gas(self.robot)
        elif action == 3 and self.has_red_object == False:
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

    def calculate_distance(self, pos1, pos2):
        return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5

    def calc_red_reward(self, red_ori_percentage):
        reward = 0
        # TODO: check if Robobo has red object
        # TODO: if touching the red object - keep certain percentage of red in view?


        # Spotting reward and moving toward, penalize if no food in image
        if red_ori_percentage > 0.:
            reward +=  50 * (red_ori_percentage / 100)
        else:
            reward -= 5

        # Food reward
        if self.robot.nr_food_collected() > self.current_num_food:
            reward += 50
            self.current_num_food = self.robot.nr_food_collected()
            self.has_red_object = True

        return reward

    def calc_green_reward(self, green_ori_percentage, action):
        reward = 0
        
        if self.has_red_object == True:
            if action == 3:
                reward -=10
        
        # Spotting reward and moving toward, penalize if no base in image
        if green_ori_percentage > 0.:
            reward += 50 * (green_ori_percentage / 100)
        else:
            reward -= 5

        robot_pos = self.robot.get_position()

        current_distance_to_base = self.calculate_distance(robot_pos, self.robot.base_position())

        # if self.previous_distance_to_base is not None and current_distance_to_base < self.previous_distance_to_base:
        #     reward += 1
        # elif self.previous_distance_to_base is not None and current_distance_to_base >= self.previous_distance_to_base:
        #     reward -= 1

        self.previous_distance_to_base = current_distance_to_base

        # Food reward
        if self.robot.base_detects_food():
            reward += 100

        return reward

    def compute_reward(self, next_state, action_taken, black_percentage, original_percentage):
        red_ori_percentage = original_percentage["red"]
        green_ori_percentage = original_percentage["green"]
        
        # Give time
        time_diff = time.time() - self.start_time
        if time_diff >= 180:
            return 0, True

        # Max distance from obstacle, excluding FrontC (index 4)
        max_distance = max(v for i, v in enumerate(next_state["sensor_readings"]) if i != 4 and v is not None)
        if max_distance >= 300 and black_percentage >= 99.:  # Threshold distance for being too close to an obstacle
            return -100, True
        
        has_object = False

        if next_state["sensor_readings"][4] > 300:
            has_object = True

        found_base = False

        if has_object:
            # task b - keep red, find green, approach
            green_reward = self.calc_green_reward(green_ori_percentage, action_taken)
            found_base = True
        else:
            red_reward = self.calc_red_reward(red_ori_percentage)
            green_reward = 0

        # Base rewards for actions
        base_reward = 2 if action_taken == 2 else -1

        # print(base_reward, food_reward, spot_reward)
        total_reward = base_reward + red_reward + green_reward

        if found_base:
            total_reward, True

        # End after 3 mins
        if self.robot.get_sim_time() >= 180:
            # print("TIMES UP")
            return total_reward, True

        return total_reward, False

    def render(self, mode='human', close=False):
        pass  # No rendering required for this example

# # original - smaller boundaries
def process_image(image, colors={'red1': [np.array([0, 70, 50]), np.array([10, 255, 255])],
                                 'red2': [np.array([170, 70, 50]), np.array([180, 255, 255])],
                                 'green': [np.array([33, 19, 105]), np.array([77, 255, 255])]}, min_area=2000):

# larger upper and lower bound - larger boundaries
# def process_image(image, colors={'green': [np.array([30, 20, 100]), np.array([90, 255, 255])]}, min_area=5000):
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
    black_mask = np.zeros(frame.shape[:2], dtype="uint8")  # Create a black mask for all colors
    color_masks = {color: np.zeros(frame.shape[:2], dtype="uint8") for color in colors}

    for name, clr in colors.items():  # For each color in colors
        contours_info = find_colors(hsv, clr)
        for c in contours_info:
            area = cv.contourArea(c)  # Calculate contour area
            if area > min_area:  # Draw filled contour only if contour is big enough
                cv.drawContours(black_mask, [c], -1, 255, -1)  # Draw filled contour on the total mask
                cv.drawContours(color_masks[name], [c], -1, 255, -1)  # Draw filled contour on the color-specific mask

    # Bitwise AND operation to keep only the region inside the contour
    result = cv.bitwise_and(frame, frame, mask=black_mask)

    # Calculate the percentages of each color and black content
    total_pixels = frame.shape[0] * frame.shape[1]
    black_pixels = total_pixels - cv.countNonZero(black_mask)

    color_percentages = {}
    for name, color_mask in color_masks.items():
        color_pixels = cv.countNonZero(color_mask)
        color_percentages[name] = (color_pixels / total_pixels) * 100

    black_percentage = (black_pixels / total_pixels) * 100

     # Merge the two red percentages into one
    if 'red1' in color_percentages and 'red2' in color_percentages:
        color_percentages['red'] = color_percentages.pop('red1') + color_percentages.pop('red2')

    return result, black_percentage, color_percentages

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
        for i in range(1, 15):
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
        model = DQN.load(f"{FIGRURES_DIR}/slow_model/1000.zip")

        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
        
        rob.set_phone_tilt(109, 100)

        count = 0

        while True:
            # Get new observation
            sensor_readings = rob.read_irs()
            # img = rob.get_image_front()
            cv.imwrite(f"{FIGRURES_DIR}/images_testing/image{count}.jpg", rob.get_image_front()) 
            processed_image, black_percentage, original_percentage = process_image(rob.get_image_front())
            cv.imwrite(f"{FIGRURES_DIR}/images_testing/processed_image{count}.jpg", processed_image) 
            resized_image = cv.resize(processed_image, (64,64), cv.INTER_AREA)
            cv.imwrite(f"{FIGRURES_DIR}/images_testing/resized_image{count}.jpg", processed_image) 
            next_state = {"sensor_readings": np.array(sensor_readings, dtype=np.float32), "image": resized_image}

            # print("Red: ", original_percentage["red"])
            # print("Green: ", original_percentage["green"])

            gas(rob)

            print(sensor_readings)
   

            # action = model.predict(next_state)[0]
            # if action == 0:
            #     steer_left(rob)
            # elif action == 1:
            #     steer_right(rob)
            # elif action == 2:
            #     gas(rob)
            # elif action == 3:
            #     reverse(rob)
            # time.sleep(0.5)

            count += 1

