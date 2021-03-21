import numpy as np

class ReplayBuffer:
    def __init__(self, mem_size, input_shape, n_actions, discrete=False):
        self.mem_size = mem_size
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size,) + input_shape)
        self.new_state_memory = np.zeros_like(self.state_memory)
        if discrete:
            self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        else:
            self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.mem_counter = 0

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.action_memory[index] = action
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminals