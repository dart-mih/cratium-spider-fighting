import gymnasium as gym
import craftium

from ppo import PPO
from plot_graph import save_graph

if __name__ == "__main__":
    env_name = "Craftium/SpidersAttack-v0"

    env = gym.make(env_name, frameskip=3)

    ppo = PPO(env, env_name, 500, 300000, 100000)

    ppo.train()
    ppo.test(10)

    save_graph(env_name)
