import os

import gymnasium as gym
import craftium

from ppo import PPO
from plot_graph import save_graph

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    env_name = "Craftium/SpidersAttack-v0"

    # env = gym.make(env_name, frameskip=3)

    # ppo = PPO(
    #     env=env,
    #     env_name=env_name,
    #     max_ep_len=4000,
    #     max_train_timesteps=512000,
    #     save_model_freq=64000,
    #     update_timestep_mult=1.5,
    #     print_freq_mult=1.5,
    #     log_freq_mult=1,
    # )

    # ppo.train()
    # ppo.test(10)

    # env.close()

    save_graph(env_name)
