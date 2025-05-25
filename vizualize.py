import os

import gym

from ppo import PPO
from plot_graph import save_graph

if __name__ == "__main__":
    env_name = "Craftium/SpidersAttack-v0"
    env = gym.make(env_name, frameskip=3, render_mode="human")

    ppo = PPO(env, env_name, 500, 300000, 100000)

    ppo.load_checkpoint(os.path.join("checkpoints", env_name, "PPO_CartPole-v1_2.pth"))
    ppo.test(1)
