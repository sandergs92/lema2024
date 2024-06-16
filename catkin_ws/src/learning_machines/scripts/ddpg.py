from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """
        :param mean: The mean towards which the noise will revert.
        :param std_deviation: The standard deviation of the noise.
        :param theta: The rate of mean reversion.
        :param dt: The time step for the noise calculation.
        :param x_initial: The initial value for the noise.
        """

        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """
        Update and return the noise value based on the Ornstein-Uhlenbeck process.
        This method allows the instance to be called as a function.

        :return: Updated noise value as a NumPy array
        """

        # Calculate the noise
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )

        # Update x_prev for use in the next step
        self.x_prev = x
        return x

    def reset(self):
        """
        Reset the noise to the initial state or zero if no initial state was provided.
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, num_states, num_actions,
                 target_actor, target_critic,
                 actor_model, critic_model,
                 critic_optimizer, actor_optimizer,
                 gamma, buffer_capacity=100000, batch_size=64):

        """
        :param num_states: The dimensionality of the state space of the environment.
        :param num_actions: The dimensionality of the action space of the environment.
        :param target_actor: The target actor model used for delayed policy updates.
        :param target_critic: The target critic model used for delayed policy updates.
        :param actor_model: The main actor model that selects actions.
        :param critic_model: The main critic model that evaluates action-state pairs.
        :param critic_optimizer: The optimizer for the critic model.
        :param actor_optimizer: The optimizer for the actor model.
        :param gamma: The discount factor for future rewards.
        :param buffer_capacity: The maximum size of the buffer for storing experiences.
        :param batch_size: The number of experiences to sample from the buffer during training.
        """

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.critic_model = critic_model
        self.actor_model = actor_model
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.buffer_counter = 0

        # Initialize numpy arrays to store experience components
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))

    def record(self, obs_tuple):
        """
        Record a new experience to the buffer.

        :param obs_tuple: A tuple (state, action, reward, next_state) representing the experience.
        """
        index = self.buffer_counter % self.buffer_capacity

        # Store the experience data
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        """
        Perform the weight update for both actor and critic networks.

        :param state_batch: Batch of states.
        :param action_batch: Batch of actions.
        :param reward_batch: Batch of rewards.
        :param next_state_batch: Batch of next states.
        """

        # Update critic model
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        # Update actor model
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

        return critic_loss, actor_loss

    def learn(self):
        """
        Sample a batch of experiences from the buffer and perform learning (update models).
        """
        # Ensure we have enough samples
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert numpy arrays to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype="float32")
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices]
        )

        # Update the models
        critic_loss, actor_loss = self.update(state_batch, action_batch, reward_batch, next_state_batch)

        return critic_loss, actor_loss

class DPPG:
    def __init__(self, num_states, num_actions, upper_bound, lower_bound):
        """
        :param num_states: The number of states in the environment.
        :param num_actions: The number of actions the agent can take.
        :param upper_bound: The upper bound of the action space.
        :param lower_bound: The lower bound of the action space.
        """

        self.num_states = num_states
        self.num_actions = num_actions
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def get_actor(self):
        """
        Creates the actor model which outputs the action to be taken, given a state.
        """

        # Initialize weights of the last layer of the actor to a small random value
        last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        # Define input and hidden layers
        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh", kernel_initializer=last_init)(out)

        # Define output layer with tanh activation function scaled by the action bound
        outputs = outputs * self.upper_bound
        model = keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        """
        Creates the critic model which outputs the Q-value given a state and an action.

        """

        # State as input
        state_input = layers.Input(shape=(self.num_states,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Combine state and action into one stream
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)

        # Output a single Q-value
        outputs = layers.Dense(1)(out)

        model = keras.Model([state_input, action_input], outputs)

        return model

    def policy(self, state, noise_object, actor_model):
        """
        Generates an action given a state as per current policy and adds noise for exploration.

        :param state: The current state.
        :param noise_object: An instance of noise class to add exploration noise.
        :param actor_model: The actor model to predict the action.
        :return: The action to be taken after clipping to ensure it's within bounds.
        """

        sampled_actions = tf.squeeze(actor_model(state))
        noise = noise_object()
        sampled_actions = sampled_actions.numpy() + noise

        # Clip the actions to ensure they are within the allowed action space limits
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]
        # return legal_action

    @staticmethod
    def update_target(target, original, tau):
        """
        Soft update the target model parameters using the weights from the original model.

        :param target: The target model to be updated.
        :param original: The original model providing the weights.
        :param tau: The interpolation factor indicating how much weight to take from the original model.
        """

        target_weights = target.get_weights()
        original_weights = original.get_weights()

        # Update the target weights with a blend of original and target weights
        for i in range(len(target_weights)):
            target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

        target.set_weights(target_weights)
