import os

import gymnasium as gym
import craftium

from ppo import PPO

if __name__ == "__main__":
    env_name = "Craftium/SpidersAttack-v0"
    env = gym.make(env_name, frameskip=3, render_mode="human")

    ppo = PPO(
        env=env,
        env_name=env_name,
        max_ep_len=4000,
        max_train_timesteps=64000,
        save_model_freq=32000,
        update_timestep_mult=2,
        print_freq_mult=2,
        log_freq_mult=1,
    )

    ppo.load_checkpoint(
        "checkpoints/Craftium-SpidersAttack-v0/PPO_Craftium-SpidersAttack-v0_1.pth"
    )
    ppo.test(1)

    env.close()
