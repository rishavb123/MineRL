import os

def get_last_stamp():
    return int(os.listdir("models")[-1].split("_")[-1][:-3])

def get_starting_episode(agent):
    return len(agent.metrics["kills"]) + 1