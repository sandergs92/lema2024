import cv2
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN

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
        self.action_space = spaces.Discrete(6)  # 6 discrete actions: do nothing, steer left, steer right, gas, brake, reverse
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)  # IR sensor readings
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        if isinstance(self.robot, SimulationRobobo):
            try:
                self.robot.stop_simulation()
            except Exception as e:
                print(f"Error during reset: {e}")
            self.robot.play_simulation()
            state = [0.] * 8
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        # Execute one time step within the environment
        if action == 0:
            do_nothing(self.robot)
        elif action == 1:
            steer_left(self.robot)
        elif action == 2:
            steer_right(self.robot)
        elif action == 3:
            gas(self.robot)
        elif action == 4:
            brake(self.robot)
        elif action == 5:
            reverse(self.robot)

        # Wait for the action to complete
        time.sleep(1)

        next_state = self.robot.read_irs()
        next_state = np.array(next_state, dtype=np.float32)

        # Compute reward
        reward, done = self.compute_reward(next_state, action)

        info = {}

        return next_state, reward, done, False, info

    def compute_reward(self, irs_values, action_taken):
        print(irs_values)
        max_distance = max([v for v in irs_values if v is not None])
        if max_distance >= 70 and max_distance <= 2000:  # Threshold distance for being too close to an obstacle
            print('Robot is too close to an object')
            return -1, False  # Small negative reward for being too close
        elif max_distance >= 2000:
            print('Collision detected.')
            return -10, True
        print('Safe movement!')
        if action_taken == 3:
            return 5, False
        return 1, False  # Positive reward for safe movement

    def render(self, mode='human', close=False):
        pass  # No rendering required for this example


def do_nothing(rob: IRobobo):
    rob.move_blocking(0, 0, 500)

def steer_left(rob: IRobobo):
    rob.move_blocking(-50, 50, 500)

def steer_right(rob: IRobobo):
    rob.move_blocking(50, -50, 500)

def gas(rob: IRobobo):
    rob.move_blocking(100, 100, 500)

def reverse(rob: IRobobo):
    rob.move_blocking(-50, -50, 500)

def brake(rob: IRobobo):
    rob.move_blocking(0, 0, 500)


def run_all_actions(rob: IRobobo):
    env = RoboboEnv(rob)

    # Create the DQN model
    model = DQN('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=1000)

    # Save the model
    model.save("dqn_robobo")
