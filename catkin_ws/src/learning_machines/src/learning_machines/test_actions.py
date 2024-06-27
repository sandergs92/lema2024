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

COLORS = {'red': ([np.array([0, 120, 70]), np.array([10, 255, 255])], (0, 0, 255)),
          'green': ([np.array([33, 19, 105]), np.array([77, 255, 255])], (0, 255, 0))}


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
        self.red_object_count = 0
        self.reached_green_base_with_red = 0
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
        self.current_num_food = 0
        self.start_time = time.time()
        self.robot.set_phone_tilt(109, 100)
        processed_image, black_percentage, original_percentage = process_image(self.robot.get_image_front())
        resized_image = cv.resize(processed_image, (64,64), cv.INTER_AREA)

        # Initialize the previous distance to the base
        robot_pos = self.robot.get_position()
        base_pos = self.robot.base_position()
        self.previous_distance_to_base = self.calculate_distance(robot_pos, base_pos)

        self.explored_positions = set()

        self.has_red_object = False
        self.red_object_count = 0
        self.reached_green_base_with_red = 0
        
        return {"sensor_readings": np.array(state, dtype=np.float32), "image": resized_image}, {}

    def step(self, action):
        # Execute one time step within the environment

        if action == 0:
            steer_left(self.robot)
        elif action == 1:
            steer_right(self.robot)
        elif action == 2:
            gas(self.robot)
        # elif action == 3 and self.has_red_object == False:
        elif action == 3:
            reverse(self.robot)

        # Wait for the action to complete
        time.sleep(0.5)

        # Get new observation
        sensor_readings = self.robot.read_irs()
        processed_image, black_percentage, original_percentage = process_image(self.robot.get_image_front())
        cv.imwrite(f"{FIGRURES_DIR}/{time.time()}.jpg", processed_image) 
        resized_image = cv.resize(processed_image, (64,64), cv.INTER_AREA)
        next_state = {"sensor_readings": np.array(sensor_readings, dtype=np.float32), "image": resized_image}

        # Compute reward
        reward, done = self.compute_reward(next_state, action, black_percentage, original_percentage)


        if self.robot.nr_food_collected() > self.red_object_count:
            self.red_object_count = self.robot.nr_food_collected()
            self.has_red_object = True

        if self.robot.base_detects_food() and self.has_red_object:
            self.reached_green_base_with_red += 1
            self.has_red_object = False

        info = {
            "red_object_count": self.red_object_count,
            "reached_green_base_with_red": self.reached_green_base_with_red
        }

        return next_state, reward, done, False, info

    def calculate_distance(self, pos1, pos2):
        return ((round(pos1.x,4) -  round(pos2.x,4)) ** 2 + ( round(pos1.y,4) -  round(pos2.y,4)) ** 2) ** 0.5

    def calc_red_reward(self, red_ori_percentage):
        reward = 0

        # Spotting reward and moving toward, penalize if no food in image
        if red_ori_percentage > 0.:
            reward +=  50 * ((red_ori_percentage) / 100)
        else:
            reward -= 5

        # Food reward
        if self.robot.nr_food_collected() != 0:
            reward += 50
            self.has_red_object = True

        return reward

    def calc_green_reward(self, green_ori_percentage, action):
        reward = 0
        
        if self.has_red_object:
            if action == 3:
                reward -=5
        
        # Spotting reward and moving toward, penalize if no base in image
        if green_ori_percentage > 0.:
            reward += 50 * (green_ori_percentage / 100)
        else:
            reward -= 5

        robot_pos = self.robot.get_position()

        current_distance_to_base = self.calculate_distance(robot_pos, self.robot.base_position())

        if current_distance_to_base < self.previous_distance_to_base:
            reward += 5

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
        if max_distance >= 300 and black_percentage >= 85.:  # Threshold distance for being too close to an obstacle
            return -100, True
    
        reward = 0
        red_reward = 0
        green_reward = 0
        has_object = False

        if self.has_red_object == False:
            red_reward = self.calc_red_reward(red_ori_percentage)

        if next_state["sensor_readings"][4] > 300:
            has_object = True
            reward += 10
        else:
            self.has_red_object = False

        found_base = False

        if self.has_red_object:
            # task b - keep red, find green, approach
            green_reward = self.calc_green_reward(green_ori_percentage, action_taken)
            found_base = True
        else:
            green_reward = 0

        # Base rewards for actions
        base_reward = 2 if action_taken == 2 else -1

        # print(base_reward, food_reward, spot_reward)
        total_reward = base_reward + red_reward + green_reward

        return total_reward, False


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

        # # Load the existing DQN model checkpoint
        # model = DQN.load(f"{FIGRURES_DIR}/redInit_280.zip", env=env, tensorboard_log=str(FIGRURES_DIR))

        # Train the model
        TIMESTEPS = 1000
        for i in range(1, 11):
            # print('RUN: ', str(i))
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            model.save(f"{FIGRURES_DIR}/{TIMESTEPS * i}")
    
    elif dataset == 'validation':
        env = RoboboEnv(rob)
        env = Monitor(env, str(FIGRURES_DIR))

        # List of model paths and timesteps
        model_timesteps = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
        num_episodes = 1000  # Number of episodes to run for each model

        for timestep in model_timesteps:
            model_path = f"{FIGRURES_DIR}/t3/{timestep}.zip"
            model = DQN.load(model_path, env=env)

            if isinstance(rob, SimulationRobobo):
                rob.play_simulation()
        
            rob.set_phone_tilt(109, 100)

            rewards = []
            red_objects_obtained = []
            green_bases_reached = []
            for episode in range(25):
                obs = env.reset()[0]
                total_reward = 0
                done = False
                while not done:
                    action, _states = model.predict(obs)
                    obs, reward, done, test, info = env.step(action)
                    total_reward += reward
                rewards.append(total_reward)
                red_objects_obtained.append(info["red_object_count"])
                green_bases_reached.append(info["reached_green_base_with_red"])

            print(f"Results for model saved at {timestep} timesteps:")
            print(f"Max reward: {max(rewards)}")
            print(f"Min reward: {min(rewards)}")
            print(f"STD reward: {np.std(rewards)}")
            print(f"Avg reward: {sum(rewards) / len(rewards)}")
            print(f"Red objects obtained: {sum(red_objects_obtained) / len(red_objects_obtained)}")
            print(f"Green bases reached with red object: {sum(green_bases_reached) / len(green_bases_reached)}")
            print(rewards)
    
    elif dataset == 'testing':
        model = DQN.load(f"{FIGRURES_DIR}/slow_model/1000.zip")

        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()

        print("INIT FOOD: ", rob.nr_food_collected())
        
        rob.set_phone_tilt(109, 100)

        count = 0

        while True:
            # Get new observation
            sensor_readings = rob.read_irs()
            # img = rob.get_image_front()
            # cv.imwrite(f"{FIGRURES_DIR}/images_testing/image{count}.jpg", rob.get_image_front()) 
            processed_image, black_percentage, original_percentage = process_image(rob.get_image_front())
            # cv.imwrite(f"{FIGRURES_DIR}/images_testing/processed_image{count}.jpg", processed_image) 
            resized_image = cv.resize(processed_image, (64,64), cv.INTER_AREA)
            # cv.imwrite(f"{FIGRURES_DIR}/images_testing/resized_image{count}.jpg", processed_image) 
            next_state = {"sensor_readings": np.array(sensor_readings, dtype=np.float32), "image": resized_image}

            # print("Red: ", original_percentage["red"])
            # print("Green: ", original_percentage["green"])

            gas(rob)

            if sensor_readings[4] > 300:
                print("Food collected")
            # print(sensor_readings)
   

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

