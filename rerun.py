import os

def get_last_stamp():
    return int(os.listdir("models")[-1].split("_")[-1][:-3])

def get_stamps():
    return [int(os.listdir("models")[i + 1].split("_")[-1][:-3]) for i in range(len(os.listdir("models")) - 1)]

def get_starting_episode(agent):
    assert len(agent.metrics["kills"]) == len(agent.metrics["times"])
    assert len(agent.metrics["kills"]) == len(agent.metrics["cumulative_rewards"])
    return len(agent.metrics["kills"]) + 1