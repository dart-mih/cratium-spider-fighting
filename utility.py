import os
import shutil

import gymnasium as gym
import craftium


def make_env(
    env_name, frameskip=3, obs_width=128, obs_height=128, render_mode="rgb_array"
):
    def _make():
        env = gym.make(
            env_name,
            frameskip=frameskip,
            obs_width=obs_width,
            obs_height=obs_height,
            rgb_observations=False,
            gray_scale_keepdim=True,
            render_mode=render_mode,
            sync_mode=True,
        )
        return env

    return _make


def delete_minetest_run_folders(root_dir):
    """
    Удаляет папки прошлых запусков minetest.

    :param root_dir: Путь к директории, в которой нужно искать и удалять папки
    """
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item.startswith("minetest-run-"):
            try:
                shutil.rmtree(item_path)
                print(f"Успешно удалено: {item_path}")
            except Exception as e:
                print(f"Ошибка при удалении {item_path}: {e}")
