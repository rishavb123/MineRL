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
        default="envs/zombie_fight.xml",
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
        action_filter={
            "move",
            # "jumpmove",
            "strafe",
            # "jumpstrafe",
            "turn",
            # "movenorth",
            # "moveeast",
            # "movesouth",
            # "movewest",
            # "jumpnorth",
            # "jumpeast",
            # "jumpsouth",
            # "jumpwest",
            # "jump",
            # "look",
            # "attack",
            # "use",
            # "jumpuse",
        },
    )
    env.action_space.actions.append("turn 0")
    env.action_space.n += 1

    stamp = int(time.time())
    if args.saveimagesteps > 0:
        os.mkdir(f"images/{stamp}")

    for ep in range(args.episodes):
        print("reset " + str(ep))
        obs = env.reset()

        steps = 0
        done = False
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            # action = env.action_space.sample()
            action = int(input(f"Choose a number between 0 and {env.action_space.n - 1}: "))
            print(action)
            obs, reward, done, info = env.step(action)
            steps += 1
            time.sleep(0.05)

    env.close()