#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions


if __name__ == "__main__":
    # Check nr of arguments
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")


    # Nr of specified trails
    num_trials = int(sys.argv[2])
    all_trial_data = []

    for trial in range(num_trials):
        print(f"Starting trial {trial + 1}")
        trial_data = []
        run_all_actions(rob, trial_data)
        all_trial_data.append(trial_data)

    # Convert each trial's data to a numpy
    trial_arrays = [np.array(trial) for trial in all_trial_data]

    # Max time stamps of all trials
    max_length = max(len(trial) for trial in trial_arrays)

    # Pad the arrays with NaN values to ensure consistent shape
    padded_data = np.array([np.pad(trial, ((0, max_length - len(trial)), (0, 0)), mode='constant', constant_values=np.nan) for trial in trial_arrays])

    # Average for each sensor across trials and timesteps
    average_sensor_values = np.nanmean(padded_data, axis=0)

    print("All sensor values: ", trial_arrays)

    print("Average sensor values: ", average_sensor_values)

    # # File paths
    # average_sensor_file = "/Users/amberhawkins/Desktop/lema2024/average_sensor_data.txt"
    # all_trial_file = "/Users/amberhawkins/Desktop/lema2024/all_trial_data.txt"

    # # Save average sensor data
    # print("Saving average sensor data to:", average_sensor_file)
    # np.savetxt(average_sensor_file, average_sensor_values, delimiter=',')

    # # Save all trial data
    # print("Saving all trial data to:", all_trial_file)
    # with open(all_trial_file, 'w') as file:
    #     for trial_data in all_trial_data:
    #         for row in trial_data:
    #             file.write(','.join(map(str, row)) + '\n')
    #         file.write('\n')
