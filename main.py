import os

from dotenv import load_dotenv
import gymnasium as gym
import craftium

from ppo import PPO
from plot_graph import save_graph

load_dotenv("parameters.env")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    # Загружаем параметры из конфига.
    env_name = os.getenv("ENV_NAME")
    max_ep_len = int(os.getenv("MAX_EP_LEN"))

    max_train_timesteps = int(os.getenv("MAX_TRAIN_TIMESTEPS"))
    update_timestep_mult = float(os.getenv("UPDATE_TIMESTEP_MULT"))
    print_freq_mult = float(os.getenv("PRINT_FREQ_MULT"))
    log_freq_mult = float(os.getenv("LOG_FREQ_MULT"))

    k_epochs = int(os.getenv("K_EPOCHS"))
    eps_clip = float(os.getenv("EPS_CLIP"))
    gamma = float(os.getenv("GAMMA"))
    lr_actor = float(os.getenv("LR_ACTOR"))
    lr_critic = float(os.getenv("LR_CRITIC"))

    save_model_freq = int(os.getenv("SAVE_MODEL_FREQ"))

    env = gym.make(env_name, frameskip=3)

    ppo = PPO(
        env=env,
        env_name=env_name,
        max_ep_len=max_ep_len,
        max_train_timesteps=max_train_timesteps,
        update_timestep_mult=update_timestep_mult,
        print_freq_mult=print_freq_mult,
        log_freq_mult=log_freq_mult,
        k_epochs=k_epochs,
        eps_clip=eps_clip,
        gamma=gamma,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        save_model_freq=save_model_freq,
    )

    ppo.train()
    ppo.test(10)

    env.close()

    save_graph(env_name)
