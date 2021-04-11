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
    "move -1",
    "jumpmove 1",
    "jumpmove -1",
    "strafe 1",
    "strafe -1",
    "turn -0.1",
    "turn 0.02",
    "turn 0",
    "jump 1"
]

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

if __name__ == "__main__":
    xml = Path(xml_file).read_text()

    agent_host = MalmoPython.AgentHost()
    mission = MalmoPython.MissionSpec(xml, True)
    record = MalmoPython.MissionRecordSpec()

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

    agent_host.sendCommand("attack 1")


    while world_state.is_mission_running:
        time.sleep(0.1)
        # agent_host.sendCommand(np.random.choice(action_space))
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission Ended")