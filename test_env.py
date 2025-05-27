import time
import os

from dotenv import load_dotenv

from utility import make_env, delete_minetest_run_folders

load_dotenv("parameters.env")
delete_minetest_run_folders(".")

env_name = os.getenv("ENV_NAME")

env = make_env(env_name, obs_height=512, obs_width=512, render_mode="human")()
iters = 1000

observation, info = env.reset()

ep_ret = 0
start = time.time()
for i in range(iters):
    action = 0
    observation, reward, terminated, truncated, _info = env.step(action)

    ep_ret += reward
    print(i, reward, terminated, truncated, ep_ret)

    if terminated or truncated:
        observation, info = env.reset()
        ep_ret = 0

end = time.time()
print(f"** {iters} frames in {end-start}s => {(end-start)/iters} per frame")

env.close()
