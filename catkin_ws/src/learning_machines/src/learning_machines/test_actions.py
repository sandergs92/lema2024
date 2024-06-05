import cv2
import time
import random
import numpy as np

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

def move_forward(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)

def turn_left(rob: IRobobo):
    rob.move_blocking(-100, 100, 1000)

def turn_right(rob: IRobobo):
    rob.move_blocking(100, -100, 1000)

def move_backward(rob: IRobobo):
    rob.move_blocking(-100, -100, 1000)

def move_forward_left(rob: IRobobo):
    rob.move_blocking(100, 50, 1000)

def move_forward_right(rob: IRobobo):
    rob.move_blocking(50, 100, 1000)

def move_backward_left(rob: IRobobo):
    rob.move_blocking(-100, -50, 1000)

def move_backward_right(rob: IRobobo):
    rob.move_blocking(-50, -100, 1000)

def stop(rob: IRobobo):
    rob.move_blocking(0, 0, 500)

def stop_and_move(rob: IRobobo, move_func, message: str):
    rob.talk(message)
    stop(rob)
    time.sleep(0.5)
    move_func(rob)
    time.sleep(1)

# Mapping of binary codes to actions for octagonal movement
action_map = {
    "000": move_forward,
    "001": turn_left,
    "010": turn_right,
    "011": move_backward,
    "100": move_forward_left,
    "101": move_forward_right,
    "110": move_backward_left,
    "111": move_backward_right
}

def execute_next_action(rob: IRobobo, sequence: str):
    if len(sequence) <= 0:
        return None
    next_action = sequence[:3]
    print(next_action)
    if next_action in action_map:
        action_map[next_action](rob)
    else:
        raise ValueError(f"Unknown action code: {next_action}")
    sequence = sequence[3:]
    print(sequence)
    return sequence

def generate_random_sequence(length: int) -> str:
    return ''.join(random.choice("01") for _ in range(length * 3))

def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    encountered_walls = 0

    sequence = generate_random_sequence(16)
    print(sequence)
    
    while True:
        if encountered_walls == 5:
            break

        ir_readings = rob.read_irs()
        
        # Define some threshold for detecting an object
        threshold = 25

        # [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
        front_sensors = np.take(ir_readings, [2, 3, 4, 5, 7])
        back_sensors = np.take(ir_readings, [0, 1, 6])

        # Check if any of the front sensors detect a wall
        if any(sensor >= threshold for sensor in front_sensors if sensor is not None):
            encountered_walls += 1
            stop_and_move(rob, move_backward, 'Oh no an object!')
        elif any(sensor >= threshold for sensor in back_sensors if sensor is not None):
            encountered_walls += 1
            stop_and_move(rob, move_forward, 'Oh no an object!')
        else:
            # Execute the current sequence of actions
            sequence = execute_next_action(rob, sequence)
            if sequence is None:
                break
            time.sleep(1)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
