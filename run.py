import matplotlib.pyplot as plt
from ddpg import OUActionNoise, Buffer, DPPG
import keras
import numpy as np
import gymnasium as gym

# Create the environment
# TODO: Add in our robot simulation here
env = gym.make("Pendulum-v1", render_mode="human")

# TODO: Add in the correct values for the sensors/camera
# TODO: Rework the ANNs to function for the respective sensors
# TODO: CNN for both? Might resolve the filtering on the infra-reds
# TODO: Repeat learning in the real world

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# Retrieve the upper and lower bounds of the action space from the environment
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Size of State Space ->  {}".format(num_states))
print("Size of Action Space ->  {}".format(num_actions))
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

# Set standard deviation for the exploration noise
std_dev = 0.2

# Learning rates for the critic and actor networks
critic_lr = 0.002
actor_lr = 0.001

# Total number of episodes to train
total_episodes = 100

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

# To store reward history and average reward history of each episode
ep_reward_list = []
avg_reward_list = []

for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    # TODO: Adapt the loop for running the robot simulation
    while True:
        tf_prev_state = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(prev_state), 0
        )

        action = rl.policy(tf_prev_state, ou_noise, actor_model)
        state, reward, done, truncated, _ = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        # TODO: we need to create our own reward function
        # Robots don't have touch sensors, how can a robot detect collision?
        # Use domain randomisation
        # Repeat experiments with different seeds and starting states


        buffer.learn()

        rl.update_target(target_actor, actor_model, tau)
        rl.update_target(target_critic, critic_model, tau)

        if done or truncated:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Rolling average of the episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
