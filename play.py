import json
import atexit
from model import make_model, make_baseline_model
from agent import Agent
from savgol_filter import savgol_filter

import malmo.MalmoPython as MalmoPython
import argparse
from pathlib import Path
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

# Constants
xml_file = "./envs/zombie_fight.xml"
action_space = [
    "move 1",
    "move 0",
    "move -1",
    "strafe 1",
    "strafe -1",
    "turn -0.7",
    "turn 0.7",
]
finish_action = [
    "move 0",
    "move 0",
    "move 0",
    "strafe 0",
    "strafe 0",
    "turn 0",
    "turn 0",
]
episodes = 1000


def process_observation(obs, kills, health, agent_host):
    reward = 0
    if "MobsKilled" in obs and "LineOfSight" in obs:
        reward += (obs["MobsKilled"] - kills) * 40
        if kills < obs["MobsKilled"]:
            agent_host.sendCommand(
                "chat /summon Zombie 5.5 6 5.5 {Equipment:[{},{},{},{},{id:minecraft:stone_button}], HealF:10.0f}"
            )
        reward += (health - obs["Life"]) * -5
        reward += 0.03
        if obs["LineOfSight"]["hitType"] == "entity" and obs["LineOfSight"]["inRange"]:
            reward += 2.5
    return reward, obs["MobsKilled"], obs["Life"]


if __name__ == "__main__":
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print("ERROR:", e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    agent_host.setObservationsPolicy(
        MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY
    )
    agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

    xml = Path(xml_file).read_text()
    xml = xml.replace("{{width}}", str(640)).replace("{{height}}", str(480))

    mission = MalmoPython.MissionSpec(xml, True)
    record = MalmoPython.MissionRecordSpec()

    for ep in range(episodes):
        max_retries = 3
        for retry in range(max_retries):
            try:
                agent_host.startMission(mission, record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)

        i = 0
        kills = -1
        health = -1

        agent_host.sendCommand(
            "chat /summon Zombie 5.5 6 5.5 {Equipment:[{},{},{},{},{id:minecraft:stone_button}], HealF:10.0f}"
        )
        agent_host.sendCommand("chat /gamerule naturalRegeneration false")
        agent_host.sendCommand("chat /difficulty 1")

        action = 0
        agent_host.sendCommand("turn 1")
        while world_state.is_mission_running:
            agent_host.sendCommand("attack 1")
            world_state = agent_host.getWorldState()
            time.sleep(0.02)
            if len(world_state.observations) and len(world_state.video_frames):
                obs = json.loads(world_state.observations[-1].text)
                frame = world_state.video_frames[0].pixels
                if i == 0:
                    if "MobsKilled" in obs:
                        kills = obs["MobsKilled"]
                        health = obs["Life"]
                        i += 1
                    continue

                agent_host.sendCommand(finish_action[action])
                action = np.random.choice(len(action_space))
                agent_host.sendCommand(action_space[action])

                obs = json.loads(world_state.observations[-1].text)
                reward, kills, health = process_observation(
                    obs, kills, health, agent_host
                )

                i += 1

                for error in world_state.errors:
                    print("Error:", error.text)