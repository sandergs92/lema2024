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

# def run_all_actions(rob: IRobobo):
#     if isinstance(rob, SimulationRobobo):
#         rob.play_simulation()
    
#     encountered_walls = 0

#     sequence = generate_random_sequence(16)
#     sequence = '000000000000000000000000000000000000000000000'
#     print(sequence)
#     while True:
#         if encountered_walls == 5:
#             break

#         ir_readings = rob.read_irs()
#         print(ir_readings)
        
#         # Define some threshold for detecting an object
#         threshold = 200

#         # [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
#         front_sensors = np.take(ir_readings, [2, 3, 4, 5, 7])
#         back_sensors = np.take(ir_readings, [0, 1, 6])

#         # Check if any of the front sensors detect a wall
#         if any(sensor >= threshold for sensor in ir_readings if sensor is not None):
#             stop(rob)
#             encountered_walls += 1
#             # Assuming equal weight for each sensor, calculate the weighted average
#             # If weights are different, replace np.ones(len(...)) with the actual weights
#             front_weights = np.ones(len(front_sensors))
#             back_weights = np.ones(len(back_sensors))

#             # Calculate weighted averages
#             front_weighted_average = np.average(front_sensors, weights=front_weights)
#             back_weighted_average = np.average(back_sensors, weights=back_weights)

#             # Compare the weighted averages
#             if front_weighted_average > back_weighted_average:
#                 print("Front sensors have a higher weighted average.")
#                 stop_and_move(rob, move_backward_right, 'Oh no an object!')
#                 stop_and_move(rob, move_backward_right, 'Oh no an object!')
#             elif front_weighted_average < back_weighted_average:
#                 print("Back sensors have a higher weighted average.")
#                 stop_and_move(rob, move_forward_left, 'Oh no an object!')
#                 stop_and_move(rob, move_forward_left, 'Oh no an object!')
#             else:
#                 print("Both front and back sensors have the same weighted average.")
#                 sequence = execute_next_action(rob, sequence)
#         else:
#             # Execute the current sequence of actions
#             sequence = execute_next_action(rob, sequence)
#             if sequence is None:
#                 break

#     if isinstance(rob, SimulationRobobo):
#         rob.stop_simulation()

def run_all_actions(rob: IRobobo, trial_data):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    encountered_walls = 0
    count = 0

    sequence = generate_random_sequence(16)
    sequence = '000000000000000000000000000000000000000000000'
    print(sequence)

    fall_off_readings = [3.8179993109585295e-11, 3.8179993109585295e-11, 3.8179993109585295e-11, 
                3.8179993109585295e-11, 3.8179993109585295e-11, 3.8179993109585295e-11, 
                3.8179993109585295e-11, 3.8179993109585295e-11]

    while True:
        if encountered_walls == 5:
            break

        ir_readings = rob.read_irs()
        print(ir_readings)

        if np.allclose(ir_readings, fall_off_readings, atol=1e-12):
            rob.stop_simulation()
            break

        
        if count != 0:
            trial_data.append(ir_readings)  
        
        # Sensor threshold
        threshold = 200

        # [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
        front_sensors = np.take(ir_readings, [2, 3, 4, 5, 7])
        back_sensors = np.take(ir_readings, [0, 1, 6])

        # Check if any of the front sensors detect a wall
        if any(sensor >= threshold for sensor in ir_readings if sensor is not None):
            stop(rob)
            encountered_walls += 1
            # Assuming equal weight for each sensor, calculate the weighted average
            # If weights are different, replace np.ones(len(...)) with the actual weights
            front_weights = np.ones(len(front_sensors))
            back_weights = np.ones(len(back_sensors))

            # Calculate weighted averages
            front_weighted_average = np.average(front_sensors, weights=front_weights)
            back_weighted_average = np.average(back_sensors, weights=back_weights)

            # Compare the weighted averages
            if front_weighted_average > back_weighted_average:
                print("Front sensors have a higher weighted average.")
                stop_and_move(rob, move_backward_right, 'Oh no an object!')
                stop_and_move(rob, move_backward_right, 'Oh no an object!')
            elif front_weighted_average < back_weighted_average:
                print("Back sensors have a higher weighted average.")
                stop_and_move(rob, move_forward_left, 'Oh no an object!')
                stop_and_move(rob, move_forward_left, 'Oh no an object!')
            else:
                print("Both front and back sensors have the same weighted average.")
                sequence = execute_next_action(rob, sequence)
        else:
            # Execute the current sequence of actions
            sequence = execute_next_action(rob, sequence)
            if sequence is None:
                break

        count += 1

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
