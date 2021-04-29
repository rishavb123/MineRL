from rerun import get_last_stamp, get_starting_episode
from model import make_model, make_baseline_model
from malmo_agent import MalmoAgent

import malmo.MalmoPython as MalmoPython
from pathlib import Path
import json
import sys
import time
import numpy as np

# Constants
xml_file = "./envs/zombie_fight.xml"
episodes = 5000
baseline = False
video_shape = (480, 640, 3)
input_shape = (84, 112, 3)
save = True
load_last_trained = False
load_model = None
start_episode = 1
max_steps_per_episode = 1000
running_average_length = episodes // 20
num_zombies = 2
agent_cfg = {
    "alpha": 0.0005,
    "gamma": 0.92,
    "batch_size": 64,
    "epsilon": 1,
    "epsilon_decay": 0.998,
    "epsilon_min": 0.05,
    "copy_period": 300,
    "mem_size": 10000
}


if __name__ == "__main__":
    # Runtime Generated Constants
    stamp = int(time.time()) if not load_last_trained else get_last_stamp()
    env_name = xml_file.split("/")[-1].split(".")[0]
    if baseline:
        env_name += "_baseline"
    env_name += "_" + str(num_zombies)
    model_file = f"models/{env_name}_{stamp}.h5"
    metric_file = f"metrics/{env_name}_{stamp}.json"
    h, w, d = video_shape
    input_shape = (224, 224, 3) if baseline else input_shape
    n = len(MalmoAgent.actions)
    r = lambda x: np.around(x, decimals=3)

    # Environment Setup
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
    xml = xml.replace("{{width}}", str(w)).replace("{{height}}", str(h))

    mission = MalmoPython.MissionSpec(xml, True)
    record = MalmoPython.MissionRecordSpec()

    # Agent Setup
    model = (
        make_baseline_model(video_shape, n) if baseline else make_model(input_shape, n)
    )
    print(model.summary())
    agent = MalmoAgent(
        alpha=agent_cfg["alpha"],
        gamma=agent_cfg["gamma"],
        batch_size=agent_cfg["batch_size"],
        epsilon=agent_cfg["epsilon"],
        epsilon_decay=agent_cfg["epsilon_decay"],
        epsilon_min=agent_cfg["epsilon_min"],
        copy_period=agent_cfg["copy_period"],
        mem_size=agent_cfg["mem_size"],
        model=model,
        model_file=model_file,
        metric_file=metric_file,
        input_shape=input_shape,
        agent_host=agent_host,
    )
    if load_last_trained:
        agent.load_model()
        agent.load_data()
        start_episode = get_starting_episode(agent)
        agent.epsilon = agent.epsilon_decay ** (start_episode - 1)
    elif load_model:
        agent.load_model(load_model)

    # Episode Loop
    for ep in range(start_episode - 1, episodes):

        # Mission Setup
        first = True
        working = False
        while not working:
            if not first:
                print("Had to restart the mission since the data was not sent to the server!")
                time.sleep(30)
            first = False
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
            i = 0
            while not world_state.has_mission_begun and i < 500:
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:", error.text)
                i += 1
            working = i < 500

        for _ in range(num_zombies):
            agent_host.sendCommand(
                "chat /summon Zombie "
                + str(np.random.randint(-10, 11))
                + " 4.5 "
                + str(np.random.randint(-10, 11))
                + " {HealF:10.0f}"
            )
        agent_host.sendCommand("chat /gamerule naturalRegeneration false")
        agent_host.sendCommand("chat /difficulty 1")

        # Step Loop
        step = 0
        start_time = time.time()

        state = None
        action = 0
        reward = 0
        done = False

        while world_state.is_mission_running and step < max_steps_per_episode:
            # agent_host.sendCommand("attack 1")
            time.sleep(0.02)
            world_state = agent_host.getWorldState()

            if len(world_state.observations) and len(world_state.video_frames):
                obs = json.loads(world_state.observations[-1].text)
                frame = world_state.video_frames[0]

                next_state = agent.process_frame(frame)
                reward, done = agent.process_observation(obs)
                if state is not None:
                    agent.remember(state, action, reward, next_state, done)
                    agent.learn()

                state = next_state
                action = agent.choose_and_take_action(state)

                step += 1
                time_elapsed = time.time() - start_time
                print(
                    f"Step {1 + step}; Reward {r(reward)}; Score {r(agent.temp['cumulative_reward'])}; Kills {r(agent.temp['kills'])}; Epsilon {r(agent.epsilon)}; Time Elapsed {int(time_elapsed)}s"
                    + " " * 20,
                    end="\r",
                )
        agent.finished_episode()
        agent.metrics["times"].append(time_elapsed)
        avg_score = np.mean(
            agent.metrics["cumulative_rewards"][
                max(0, ep - running_average_length) : ep + 1
            ]
        )
        print(
            f"Episode {ep + 1} of {episodes}; Score {r(agent.metrics['cumulative_rewards'][-1])}; Kills {r(agent.metrics['kills'][-1])}; Average Score {r(avg_score)}; Episode Time {int(time_elapsed)}s"
            + " " * 20
        )
        if save:
            agent.save_model()
            agent.save_data()