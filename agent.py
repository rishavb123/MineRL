import json
import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer
from dqn import DQN


class Agent:
    def __init__(
        self,
        alpha,
        gamma,
        num_actions,
        epsilon,
        batch_size,
        input_shape,
        model,
        model_file,
        data_file,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        copy_period=300,
        mem_size=200,
    ):
        self.action_space = np.arange(num_actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = model_file
        self.data_file = data_file
        self.copy_period = copy_period

        self.memory = ReplayBuffer(mem_size, input_shape, num_actions, discrete=True)
        self.dqn = DQN(model, learning_rate=alpha)
        self.target_dqn = self.dqn.create_target_network()
        self.learn_counter = 0
        self.metrics = {}
        self.temp = {}

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        return (
            np.random.choice(self.action_space)
            if np.random.random() < self.epsilon
            else np.argmax(self.dqn.get_model().predict(state))
        )

    def learn(self):
        self.learn_counter += 1
        if self.memory.mem_counter < self.batch_size:
            return
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer(
            self.batch_size
        )

        qs = self.dqn.get_model().predict(states)
        qs_next = self.target_dqn.get_model().predict(next_states)

        batch_index = np.arange(self.batch_size)

        qs_target = qs.copy()
        qs_target[batch_index, actions] = (
            rewards + self.gamma * np.max(qs_next, axis=1) * terminals
        )

        self.dqn.get_model().fit(states, qs_target, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

        if self.learn_counter % self.copy_period == 0:
            self.target_dqn.copy_from(self.dqn)

    def save_model(self):
        self.dqn.save_model(self.model_file)

    def load_model(self, model_file=None):
        if model_file == None or model_file == "":
            model_file = self.model_file
        self.dqn.load_model(model_file)
        self.target_dqn.copy_from(self.dqn)

    def save_data(self, metric_file=None):
        if metric_file == None or metric_file == "":
            metric_file = self.metric_file
        with open(metric_file, "w") as out_file:
            json.dump(self.metrics, out_file)