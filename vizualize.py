import os

from dotenv import load_dotenv
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import craftium

from ppo import PPO
from utility import delete_minetest_run_folders, make_env

load_dotenv("parameters.env")


if __name__ == "__main__":
    # Загружаем параметры из конфига.
    env_name = os.getenv("ENV_NAME")
    max_ep_len = int(os.getenv("MAX_EP_LEN"))

    max_train_timesteps = int(os.getenv("MAX_TRAIN_TIMESTEPS"))
    update_min_samples = int(os.getenv("UPDATE_MIN_SAMPLES"))
    print_freq_mult = float(os.getenv("PRINT_FREQ_MULT"))
    log_freq_mult = float(os.getenv("LOG_FREQ_MULT"))

    frame_stack_size = int(os.getenv("FRAME_STACK_SIZE"))
    grayscale = os.getenv("GRAYSCALE") == "True"
    obs_w = int(os.getenv("OBS_W"))
    obs_h = int(os.getenv("OBS_H"))

    k_epochs = int(os.getenv("K_EPOCHS"))
    eps_clip = float(os.getenv("EPS_CLIP"))
    gamma = float(os.getenv("GAMMA"))
    lr_actor = float(os.getenv("LR_ACTOR"))
    lr_critic = float(os.getenv("LR_CRITIC"))

    save_model_freq = int(os.getenv("SAVE_MODEL_FREQ"))
    model_weights = os.getenv("MODEL_WEIGHTS_PATH")

    delete_minetest_run_folders(".")

    env = SyncVectorEnv(
        [
            make_env(
                env_name=env_name,
                obs_height=obs_h,
                obs_width=obs_w,
                grayscale=grayscale,
                render_mode="human",
            )
            for _ in range(1)
        ]
    )

    ppo = PPO(
        env=env,
        env_count=1,
        env_name=env_name,
        max_ep_len=max_ep_len,
        max_train_timesteps=max_train_timesteps,
        update_min_samples=update_min_samples,
        print_freq_mult=print_freq_mult,
        log_freq_mult=log_freq_mult,
        k_epochs=k_epochs,
        eps_clip=eps_clip,
        gamma=gamma,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        save_model_freq=save_model_freq,
    )

    ppo.load_checkpoint(model_weights)
    ppo.test(1)

    env.close()
