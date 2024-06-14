import cv2
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


def run_all_actions(rob: IRobobo, dataset):

    if dataset == 'train':
        env = RoboboEnv(rob)
        env = Monitor(env, str(FIGRURES_DIR))
        # Create the DQN model
        model = DQN('MlpPolicy', env, verbose=1)

        # Train the model
        TIMESTEPS = 1000
        for i in range(1, 30):
            print('RUN: ', str(i))
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            model.save(f"{FIGRURES_DIR}/{TIMESTEPS * i}")
    elif dataset == 'validation':
        env = RoboboEnv(rob)
        env = Monitor(env, str(FIGRURES_DIR))
        model = DQN.load(f"{FIGRURES_DIR}/15000.zip", env=env)
        obs = env.reset()[0]
        for _ in range(1000):
            action, _states = model.predict(obs)
            obs, reward, done, test, info = env.step(action)
    elif dataset == 'testing':
        model = DQN.load(f"{FIGRURES_DIR}/15000.zip")
        while True:
            next_state = rob.read_irs()
            next_state = np.array(next_state, dtype=np.float32)
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

