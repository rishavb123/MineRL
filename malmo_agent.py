from agent import Agent

import numpy as np
import tensorflow as tf


class MalmoAgent(Agent):

    actions = [
        "move 1",
        "move 0",
        "move -1",
        "strafe 1",
        "strafe -1",
        "turn -0.7",
        "turn 0.7",
    ]
    finish_actions = [
        "move 0",
        "move 0",
        "move 0",
        "strafe 0",
        "strafe 0",
        "turn 0",
        "turn 0",
    ]
    rewards = {"per_kill": 100, "per_health_lost": -4, "per_step": 0.05, "per_hit": 10}

    def __init__(
        self,
        alpha,
        gamma,
        epsilon,
        batch_size,
        input_shape,
        model,
        model_file,
        metric_file,
        agent_host,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        copy_period=300,
        mem_size=200,
    ):
        super(MalmoAgent, self).__init__(
            alpha,
            gamma,
            len(MalmoAgent.actions),
            epsilon,
            batch_size,
            input_shape,
            model,
            model_file,
            metric_file,
            epsilon_decay,
            epsilon_min,
            copy_period,
            mem_size,
        )
        self.metrics["cumulative_rewards"] = []
        self.metrics["kills"] = []
        self.reset_temp_data()
        self.temp["total_kills"] = -1
        self.agent_host = agent_host
        self.input_shape = input_shape

    def reset_temp_data(self):
        self.temp["kills"] = 0
        self.temp["cumulative_reward"] = 0
        self.temp["health"] = -1
        self.temp["last_action"] = 0

    def finished_episode(self):
        self.metrics["cumulative_rewards"].append(self.temp["cumulative_reward"])
        self.metrics["kills"].append(self.temp["kills"])
        self.reset_temp_data()

    def choose_and_take_action(self, state):
        action = self.choose_action(state)
        self.agent_host.sendCommand(MalmoAgent.finish_actions[self.temp["last_action"]])
        self.agent_host.sendCommand(MalmoAgent.actions[action])
        self.temp["last_action"] = action
        return action

    def process_observation(self, obs):
        if "MobsKilled" in obs and "LineOfSight" in obs:
            reward = 0
            if self.temp["total_kills"] == -1:
                self.temp["total_kills"] = obs["MobsKilled"]
            if self.temp["health"] == -1:
                self.temp["health"] = obs["Life"]
            reward += (
                obs["MobsKilled"] - self.temp["total_kills"]
            ) * MalmoAgent.rewards["per_kill"]
            if self.temp["total_kills"] < obs["MobsKilled"]:
                self.agent_host.sendCommand(
                    "chat /summon Zombie 5.5 6 5.5 {Equipment:[{},{},{},{},{id:minecraft:stone_button}], HealF:10.0f}"
                )
                self.temp["kills"] += obs["MobsKilled"] - self.temp["total_kills"]
                self.temp["total_kills"] = obs["MobsKilled"]
            reward += (self.temp["health"] - obs["Life"]) * MalmoAgent.rewards[
                "per_health_lost"
            ]
            if self.temp["health"] - obs["Life"]:
                self.temp["health"] = obs["Life"]
            reward += MalmoAgent.rewards["per_step"]
            if (
                obs["LineOfSight"]["hitType"] == "entity"
                and obs["LineOfSight"]["inRange"]
            ):
                reward += MalmoAgent.rewards["per_hit"]
            self.temp["cumulative_reward"] += reward
            return reward, self.temp["health"] <= 0
        return 0, False

    def process_frame(self, frame):
        pixels = np.array(frame.pixels, dtype=np.uint8)
        frame_shape = (frame.height, frame.width, frame.channels)
        image = pixels.reshape(frame_shape)
        if self.input_shape != frame_shape:
            image = tf.image.resize(image, (224, 224))
        return image