import malmoenv
import argparse
from pathlib import Path
import time
from PIL import Image
import os


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
        default=1,
        help="the number of resets to perform - default is 1",
    )
    parser.add_argument(
        "-ms", "--episodemaxsteps", type=int, default=0, help="max number of steps per episode"
    )
    parser.add_argument(
        "-si", "--saveimagesteps", type=int, default=0, help="save an image every N steps"
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

    stamp = int(time.time())
    if args.saveimagesteps > 0:
        os.mkdir(f"images/{stamp}")

    for ep in range(args.episodes):
        print("reset " + str(ep))
        obs = env.reset()

        steps = 0
        done = False
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            steps += 1
            print("reward: " + str(reward))
            # print("done: " + str(done))
            print("obs: " + str(obs))
            # print("info" + info)
            if args.saveimagesteps > 0 and steps % args.saveimagesteps == 0:
                h, w, d = env.observation_space.shape
                img = Image.fromarray(obs.reshape(h, w, d))
                img.save(f"images/{stamp}/ep_{ep}_step_{step}.png")

            time.sleep(0.05)

    env.close()