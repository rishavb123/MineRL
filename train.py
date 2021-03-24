import atexit
from model import make_model, make_baseline_model
from agent import Agent
from savgol_filter import savgol_filter

import malmoenv
import argparse
from pathlib import Path
import time
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="envs/mobchase_single_agent.xml",
        help="the mission xml",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=9000, help="the mission server port"
    )
    parser.add_argument(
        "-s",
        "--server",
        type=str,
        default="127.0.0.1",
        help="the mission server DNS or IP address",
    )
    parser.add_argument(
        "-eps",
        "--episodes",
        type=int,
        default=100,
        help="the number of resets to perform - default is 100",
    )
    parser.add_argument(
        "-ms", "--episode-max-steps", type=int, default=0, help="max number of steps per episode"
    )
    parser.add_argument(
        "-si", "--save-image-steps", type=int, default=0, help="save an image every N steps"
    )

    parser.add_argument(
        "-lm", "--load-model", type=str, default="", help="the file to load the model from before training"
    )
    parser.add_argument(
        "-sm", "--save-model", type=str, default="", help="the file to save the model from after training"
    )

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser.add_argument(
        "-b", "--baseline", type=str2bool, default=False, help="Whether or not to use the baseline model."
    )

    return parser

if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()
    xml = Path(args.env).read_text()
    env = malmoenv.make()
    env.init(
        xml,
        args.port,
        server=args.server,
        server2=args.server,
        port2=args.port,
        role=0,
        episode=0,
    )

    env_name = args.env.split("/")[1].split(".")[0]
    stamp = int(time.time())
    model_file = f"models/{env_name}_{stamp}.h5" if args.save_model == "" else args.save_model
    h, w, d = env.observation_space.shape
    input_shape = (224, 224, 3) if args.baseline else (h, w, d)
    model = make_baseline_model((h, w, d), env.action_space.n) if args.baseline else make_model(input_shape, env.action_space.n)
    print(model.summary())
    agent = Agent(
        alpha=0.0005,
        gamma=0.99,
        num_actions=env.action_space.n,
        batch_size=16,
        epsilon=1,
        input_shape=input_shape,
        model=model,
        model_file=model_file,
        epsilon_decay=0.999,
        epsilon_min=0.01
    )

    if len(args.load_model) > 0:
        agent.load_model(args.load_model)
        
    if args.save_image_steps > 0:
        os.mkdir(f"images/{stamp}")

    scores = []

    def process_state(state):
        state = state.reshape(h, w, d)
        if args.baseline:
            state = tf.image.resize(state, (224, 224))
        return state

    def onexit():
        if len(scores) >= 5:
            try:
                plt.plot(scores, label="Scores Over Episodes")
                plt.plot(savgol_filter(scores, args.episodes / 2, 4), label="Savgol Filter Smoothing")
                plt.legend()
                plt.savefig("./graphs/" + agent.model_file.split("/")[1][:-3] + "-scores.png")
            except KeyboardInterrupt:
                onexit()

    atexit.register(onexit)

    for ep in range(args.episodes):
        done = False
        state = process_state(env.reset())
        steps = 0
        score = 0
        start_time = time.time()

        while not done and (args.episode_max_steps <= 0 or steps < args.episode_max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            if len(next_state) == 0:
                continue
            next_state = process_state(next_state)
            
            steps += 1
            score += reward
            time_elapsed = time.time() - start_time

            print(f"Step {steps}, Reward {reward}, Done {done}, Score {score} Time Elapsed {int(time_elapsed)}s" + " "*10, end="\r")

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.learn()

            if args.save_image_steps > 0 and steps % args.save_image_steps == 0:
                img = Image.fromarray(state.reshape(h, w, d))
                img.save(f"images/{stamp}/ep_{ep}_step_{steps}.png")
    

        scores.append(score)
        avg_score = np.mean(scores[max(0, ep - 100): ep + 1])
        print(f"Episode {ep + 1} Score {score} Average Score {avg_score} Episode Time {int(time_elapsed)}s" + " " * 10)

        agent.save_model()

    agent.save_model()
    env.close()
