#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

from data_files import FIGRURES_DIR
from robobo_interface import SimulationRobobo, HardwareRobobo, Position, Orientation
# from learning_machines import test_hardware
from ddpg import OUActionNoise, Buffer, DPPG

def check_collision(irs_val):
        max_distance = max(v for v in irs_val if v is not None)
        if max_distance >= 1000:
            return True
        return False

def calculate_distance(x, y):
    return np.sqrt(x**2 + y**2)

def calculate_position_reward(curr_x, curr_y, target_x, target_y, reward, eval=False):
    
    distance_to_target = calculate_distance((curr_x - target_x), (curr_y - target_y))

    reward += 1 / (1 + distance_to_target)

    target_found = False
    if distance_to_target <= 1.5:
        reward += 1
    if distance_to_target <= 1:
        reward += 1
    if distance_to_target <= 0.5:
        reward += 1
    if distance_to_target <= 0.25:
        reward += 1
    if distance_to_target <= 0.1:
        reward += 20
        
        target_found = True

    return reward, target_found

def forward_movement_reward(wheel_speed_left, wheel_speed_right, reward):
    if (wheel_speed_left >= 0 and wheel_speed_right >= 0) or (wheel_speed_left <= 0 and wheel_speed_right <= 0):
        if (wheel_speed_left >= 50 and wheel_speed_right >= 50) or (wheel_speed_left <= -50 and wheel_speed_right <= -50):
            reward += 2
        # Calculate difference in wheel speeds
        wheel_speed_difference = abs(wheel_speed_left - wheel_speed_right)

        # Calculate the average of absolute values of wheel speeds
        average_wheel_speed = (abs(wheel_speed_left) + abs(wheel_speed_right)) / 2

        # Reward straight movement (similar wheel speeds)
        if wheel_speed_difference / average_wheel_speed < 0.2:
            reward += 1  # Adjust the reward amount based on the degree of straightness
    else:
        reward -= 0.25

    return reward

def check_insufficient_movement(prev_positions, sim_time, reward):
    total_distance = 0
    for i in range(1, len(prev_positions)):
        dx = prev_positions[i][0] - prev_positions[i - 1][0]
        dy = prev_positions[i][1] - prev_positions[i - 1][1]
        total_distance += calculate_distance(dx, dy)

    # Check if the total distance is less than 2 units
    if ((len(prev_positions) == 50 and total_distance < 1) and sim_time):
        reward -= 10
        return reward, True
    return reward, False

def create_tf_input(irs_values, position):
    combined_readings_state = np.concatenate((irs_values, [position[0], position[1]]))

    sensor_data = combined_readings_state[:-2]  # Exclude last two elements (position data)
    normalized_sensor_data = (sensor_data - np.mean(sensor_data)) / np.std(sensor_data)

    # Combine normalized sensor data with position data
    normalized_combined_state = np.concatenate((normalized_sensor_data, combined_readings_state[-2:]))

    # Convert to TensorFlow tensor and expand dimensions
    state = tf.expand_dims(
        tf.convert_to_tensor(normalized_combined_state), 0
    )

    return state

def validate_model(actor_model, rob):

    val_predefined_positions = [
            (-6.425, -4.473),
            (-8.525, -4.273),
            (-10.925, -4.423),
            (-6.450, -2.198),
            (-8.625, -2.198),
            (-10.875, -2.223)
        ]

    val_target_positions = [
        (-7.775, -2.798),
        (-10.125, -2.823),
        (-12.350, -2.848),
        (-7.875, -0.523),
        (-10.150, -0.523),
        (-12.350, -0.548)
    ]

    avg_reward_list = []

    # Set initial position and orientation
    random_pos = random.choice(val_predefined_positions)
    random_x, random_y = random_pos
    rob.set_position(Position(x=random_x, y=random_y, z=0.03743), 
                        Orientation(yaw=-1.5719101704887524, pitch=-1.5144899542889299, roll=-1.5719103944826311))

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # Calculate reward based on distance to corresponding target position
    index = val_predefined_positions.index((random_x, random_y))
    target_x, target_y = val_target_positions[index]

    episodic_reward = 0

    sensor_data = rob.read_irs()
    if np.isinf(sensor_data).any():
        rob.move_blocking(10, 10, 100)
        sensor_data = rob.read_irs()

    tf_state = tf.expand_dims(
        tf.convert_to_tensor(sensor_data), 0
    )
    # tf_state = create_tf_input(sensor_data, [rob.get_position().x, rob.get_position().y])

    prev_positions = [(rob.get_position().x, rob.get_position().y)]

    collision = False

    while True:
        action = actor_model(tf_state).numpy()
        rob.move_blocking(action[0][0], action[0][1], 500)

        next_state = rob.read_irs()
        position_state = [rob.get_position().x, rob.get_position().y]
        prev_positions.append((position_state[0], position_state[1]))
        tf_state = tf.expand_dims(tf.convert_to_tensor(rob.read_irs()), 0)
        # tf_state = create_tf_input(rob.read_irs(), [position_state[0], position_state[1]])

        # Use the function to calculate reward based on position
        evaluation=False
        episodic_reward, target_found = calculate_position_reward(position_state[0], position_state[1], target_x, target_y, episodic_reward, evaluation)
        episodic_reward = forward_movement_reward(action[0][0], action[0][1], episodic_reward) 
        episodic_reward += 0.25  # Or any reward calculation you prefer

        if check_collision(next_state): 
            episodic_reward -= 20
            collision = True
        if rob.get_sim_time() > 100:
            break

        # Ensure robot moves 
        if len(prev_positions) > 50:
            prev_positions.pop(0)

        # if collision or check_insufficient_movement(prev_positions, rob.get_sim_time(), episodic_reward) or rob.get_sim_time() > 200:
        if collision or rob.get_sim_time() > 300 or target_found:
            break

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    avg_reward_list.append(episodic_reward)

    return np.mean(avg_reward_list)

def training(total_episodes, file_path):

    # nr sensors
    num_states = 8
    # num_states = env.observation_space.shape[0]
    num_actions = 2

    # Retrieve the upper and lower bounds of the action space from the environment
    upper_bound = 100
    lower_bound = -100

    print("Size of State Space ->  {}".format(num_states))
    print("Size of Action Space ->  {}".format(num_actions))
    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    # Set standard deviation for the exploration noise
    std_dev = 0.2

    # Learning rates for the critic and actor networks
    critic_lr = 0.002
    actor_lr = 0.001

    # Discount factor for future rewards in the Bellman equation
    gamma = 0.99

    # Smoothing factor for updating the target networks
    tau = 0.005

    # Maximum capacity of the replay buffer
    buffer_capacity = 5000

    # Number of experiences to sample in each training batch
    batch_size = 64

    # Initialize the Ornstein-Uhlenbeck process for action exploration noise
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    # Initialize the Deep Deterministic Policy Gradient (DPPG) agent
    rl = DPPG(num_states, num_actions, upper_bound, lower_bound)

    # Create the main actor and critic models
    # Actor_model is the policy network
    actor_model = rl.get_actor()
    # Critic_model is the critic network
    critic_model = rl.get_critic()

    # Create target networks for the actor and critic which are initially the same as the main networks
    target_actor = rl.get_actor()
    target_critic = rl.get_critic()

    # Set initial weights of target networks to match the main networks
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Define the optimizers for updating the actor and critic networks
    critic_optimizer = keras.optimizers.Adam(critic_lr)
    actor_optimizer = keras.optimizers.Adam(actor_lr)

    # Initialize the experience replay buffer with specified configurations
    buffer = Buffer(num_states,
                    num_actions,
                    target_actor,
                    target_critic,
                    actor_model,
                    critic_model,
                    critic_optimizer,
                    actor_optimizer,
                    gamma,
                    buffer_capacity,
                    batch_size)

    predefined_positions = [
    (-2.500, 0.177),
    (0.00001, -0.07441),
    (2.20001, -0.07441),
    (4.37501, -0.04941),
    (4.37501, -2.29941),
    (2.17501, -2.24941),
    (-0.09999, -2.24941),
    (-2.27499, -2.27441),
    (-2.27499, -4.54941),
    (-0.02499, -4.54941),
    (2.05001, -4.27441),
    (4.35001, -4.54941),
    (4.40001, -6.74941),
    (2.20001, -6.77441),
    (-0.04999, -6.82441),
    (-2.22499, -6.79941)
    ]

    target_positions = [
    (-3.925, 1.67644),
    (-1.54999, 1.72559),
    (0.62501, 1.70059),
    (3.10001, 1.72559),
    (2.92501, -0.52441),
    (0.62501, -0.57441),
    (-1.59999, -0.57441),
    (-3.94999, -0.59941),
    (-3.94999, -2.82441),
    (-1.67499, -2.77441),
    (0.70001, -2.77441),
    (2.82501, -2.89941),
    (2.82501, -5.07441),
    (0.57501, -5.07441),
    (-1.69999, -5.04941),
    (-3.87499, -5.07441)
    ]
    
    # To store reward history and average reward history of each episode
    ep_reward_list = []
    avg_reward_list = []
    validation_reward_list = []
    penalties_list = []
    sim_step_list = []
    actor_loss_list = []
    critic_loss_list = []
    distance_list = []

    action_none = False

    for ep in range(total_episodes):
        # Set initial position and orientation
        random_pos = random.choice(predefined_positions)
        random_x, random_y = random_pos

        # TODO: randomize

        rob.set_position(Position(x=random_x, y=random_y, z=0.03743), Orientation(yaw=-1.5719101704887524, pitch=-1.5144899542889299, roll=-1.5719103944826311))

        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
        episodic_reward = 0

        # Set to 1 to avoid zero division issues
        sim_step = 1
        distance = 1

        # Read sensor so initial not inf
        tf_prev_state = tf.expand_dims(
                tf.convert_to_tensor(rob.read_irs()), 0
            )
        
        prev_x, prev_y = random_x, random_y
        # Calculate reward based on distance to corresponding target position
        index = predefined_positions.index((prev_x, prev_y))
        target_x, target_y = target_positions[index]

        print("episode: ", ep)

        positions_at_intervals =[(prev_x, prev_y)]
        prev_positions = [(prev_x, prev_y)]
        total_distance = 0

        while True:
            reward = 0
            penalties = 0

            # State before action
            if np.isinf(rob.read_irs()).any():
                rob.move_blocking(10, 10, 100)

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(rob.read_irs()), 0)
            # tf_prev_state = create_tf_input(rob.read_irs(), [prev_x, prev_y])

            action = rl.policy(tf_prev_state, ou_noise, actor_model)
            action = np.array(action)

            if np.any(np.isnan(action)):
                print("NaN")
                print("state: ", rob.read_irs(), tf_prev_state)
                action = rl.policy(tf_prev_state, OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1)), actor_model)
                action = np.array(action)

                action_none = True
                rob.reset_wheels()
                break
            else:
                rob.move_blocking(action[0][0], action[0][1], 500)

            # Current state
            curr_pos = rob.get_position()
            curr_x, curr_y = curr_pos.x, curr_pos.y

            state = tf.expand_dims(tf.convert_to_tensor(rob.read_irs()), 0)
            # state = create_tf_input(rob.read_irs(), [curr_x, curr_y])

            ## REWARDS

            collision = False

            # Collision penalty
            if check_collision(rob.read_irs()):
                # Penalize collision by subtracting from the reward
                reward -= 20
                penalties -= 20
                collision = True

            # Use the function to calculate reward based on position
            evaluation=False
            reward, target_found = calculate_position_reward(curr_x, curr_y, target_x, target_y, reward, evaluation)

            reward = forward_movement_reward(action[0][0], action[0][1], reward)                

            # Ensure robot moves 
            prev_positions.append((curr_x, curr_y))
            if len(prev_positions) > 50:
                prev_positions.pop(0)

            _, end_sim = check_insufficient_movement(prev_positions, rob.get_sim_time(), reward)

            print("reward ; tot_dist - ", reward, total_distance)
            distance_list.append(total_distance)
            episodic_reward += reward

            buffer.record((tf_prev_state, action, reward, state))
            
            critic_loss, actor_loss = buffer.learn()

            critic_loss_list.append(critic_loss)
            actor_loss_list.append(actor_loss)

            rl.update_target(target_actor, actor_model, tau)
            rl.update_target(target_critic, critic_model, tau)

            tf_prev_state = state
            sim_step += 1

            if collision or end_sim or rob.get_sim_time() > 200 or target_found:
                # print("collision")
                break

        if action_none:
            if isinstance(rob, SimulationRobobo):
                rob.stop_simulation()

        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()


        ep_reward_list.append(episodic_reward)
        penalties_list.append(penalties)
        sim_step_list.append(sim_step)

        print(ep_reward_list)
        avg_reward = np.mean(ep_reward_list)

        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        validation_avg_reward = validate_model(actor_model, rob)
        validation_reward_list.append(validation_avg_reward)

    # Save results
    np.savetxt(file_path / "training_reward.txt", avg_reward_list)
    np.savetxt(file_path / "validation.txt", validation_reward_list)
    with open(file_path / "episode_data.txt", "w") as f:
        f.write("Episode Reward,Penalties,Simulation Steps,Validation Reward,Distance Travelled\n")
        for ep in range(total_episodes):
            f.write(f"{ep_reward_list[ep]},{penalties_list[ep]},{sim_step_list[ep]},{validation_reward_list[ep]}, {distance_list[ep]}\n")


    # np.savetxt(file_path /"validation_reward.txt", validation_reward_list)
    plt.plot(validation_reward_list, label='Validation Rewards')
    plt.plot(ep_reward_list, label='Training Rewards')
    plt.plot(avg_reward_list, label='Average Episode Rewards')      
    # plt.plot(np.arange(validation_interval, total_episodes + 1, validation_interval), validation_reward_list, label='Validation')
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig(file_path / "training_and_validation_graph.pdf")
    plt.clf()
    # plt.show()

    plt.plot(penalties_list, label='Penalties')   
    # plt.plot(np.arange(validation_interval, total_episodes + 1, validation_interval), validation_reward_list, label='Validation')
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    plt.legend()
    plt.savefig(file_path / "penalties_graph.pdf")
    plt.clf()

    plt.plot(sim_step_list, label='Nr Steps')      
    # plt.plot(np.arange(validation_interval, total_episodes + 1, validation_interval), validation_reward_list, label='Validation')
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    plt.legend()
    plt.savefig(file_path / "steps_graph.pdf")
    plt.clf()

    np.savetxt(file_path / "actor_loss.txt", actor_loss_list)
    np.savetxt(file_path / "critic_loss.txt", critic_loss_list)

    plt.plot(actor_loss_list, label='Actor Loss')
    plt.plot(critic_loss_list, label='Critic Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Time')
    plt.savefig(file_path / "loss_curves_plot.pdf")
    plt.clf()

    actor_model = rl.get_actor()
    critic_model = rl.get_critic()

    return actor_model, critic_model

def evaluate_model(actor_model, rob, num_episodes=10):

    predefined_positions = [
        (-14.475, -4.373),
        (-16.575, -4.498),
        (-14.39766, -2.221),
        (-16.625, -2.223)
        ]

    target_positions = [
        (-15.975, -2.823),
        (-10.125, -2.823),
        (-15.975, -0.498),
        (-18.200, -0.498)
        ]
    
    if not actor_model._is_compiled:
        actor_model.compile(optimizer='adam', loss='mse')
    
    # Open a file to write rewards
    with open(file_path /"evaluation_rewards.txt", "w") as file:
        
        collision_count = 0

        for ep in range(num_episodes):
            # Set initial position and orientation
            random_pos = random.choice(predefined_positions)
            random_x, random_y = random_pos
            rob.set_position(Position(x=random_x, y=random_y, z=0.03743), 
                             Orientation(yaw=-1.5719101704887524, pitch=-1.5144899542889299, roll=-1.5719103944826311))


            if isinstance(rob, SimulationRobobo):
                rob.play_simulation()

            # Calculate reward based on distance to corresponding target position
            index = predefined_positions.index((random_x, random_y))
            target_x, target_y = target_positions[index]

            episodic_reward = 0

            sensor_data = rob.read_irs()
            if np.isinf(sensor_data).any():
                rob.move_blocking(10, 10, 100)
                sensor_data = rob.read_irs()

            tf_state = tf.expand_dims(tf.convert_to_tensor(sensor_data, dtype=tf.float32), axis=0)
            # tf_state = create_tf_input(sensor_data, [rob.get_position().x, rob.get_position().y])

            prev_positions = [(rob.get_position().x, rob.get_position().y)]

            collision = False

            while True:
                action = actor_model(tf_state).numpy()
                rob.move_blocking(action[0][0], action[0][1], 500)

                next_state = rob.read_irs()
                position_state = [rob.get_position().x, rob.get_position().y]
                prev_positions.append((position_state[0], position_state[1]))
                tf_state = tf.expand_dims(tf.convert_to_tensor(sensor_data, dtype=tf.float32), axis=0)
                # tf_state = create_tf_input(rob.read_irs(), [position_state[0], position_state[1]])

                episodic_reward = forward_movement_reward(action[0][0], action[0][1], episodic_reward)
                evaluation = True
                episodic_reward, target_found = calculate_position_reward(position_state[0], position_state[1], target_x, target_y, episodic_reward, evaluation)

                episodic_reward += 0.25  # Or any reward calculation you prefer

                if check_collision(next_state): 
                    collision_count += 1
                    episodic_reward -= 20
                    break
                if rob.get_sim_time() > 300 or target_found:
                    break

            if isinstance(rob, SimulationRobobo):
                rob.stop_simulation()

            print(f"Episode {ep + 1}: Total Reward: {episodic_reward}")
            file.write(f"{episodic_reward}\n")

def hardware_evaluate_model(actor_model, rob):
    
    print("Running hardware")
    
    collision_count = 0 

    if isinstance(rob, HardwareRobobo):
        rob.read_phone_battery()
        rob.read_robot_battery()
        
    episodic_reward = 0

    sensor_data = rob.read_irs()

    # if np.isinf(sensor_data).any():
    #     rob.move_blocking(10, 10, 100)
    #     sensor_data = rob.read_irs()

    tf_state = tf.expand_dims(tf.convert_to_tensor(sensor_data, dtype=tf.float32), axis=0)

    collision = False

    while True:
        print("sensors: ", tf_state)
        action = actor_model(tf_state)
        rob.move_blocking(action[0][0], action[0][1], 500)

        next_state = rob.read_irs()
        tf_state = tf.expand_dims(tf.convert_to_tensor(sensor_data, dtype=tf.float32), axis=0)


        episodic_reward = forward_movement_reward(action[0][0], action[0][1], episodic_reward)

        episodic_reward += 0.25  # Or any reward calculation you prefer

        if check_collision(next_state): 
            collision_count += 1
            episodic_reward -= 10
            rob.talk("crashed")

    return collision_count

if __name__ == "__main__":
    # Check nr of arguments
    if len(sys.argv) < 3:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
        file_name = sys.argv[2]
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        file_name = sys.argv[2]
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")


    training_episodes = 75
    eval_episodes = 1

    file_path = full_path = FIGRURES_DIR / file_name
    file_path.mkdir(parents=True, exist_ok=True)

    # actor_model, critic_model = training(training_episodes, file_path)

    # actor_model.compile(optimizer='adam', loss='mse')
    # actor_model.save(file_path /"actor_model.h5", include_optimizer=True)

    actor_model = load_model(file_path /"actor_model.h5")

    if sys.argv[1] == "--hardware":
        hardware_evaluate_model(actor_model, rob)
    else:
        evaluate_model(actor_model, rob, eval_episodes)