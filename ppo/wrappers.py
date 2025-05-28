from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np


class FrameStackWrapper(ObservationWrapper):
    def __init__(self, env, num_stacks=4):
        super().__init__(env)
        self.num_stacks = num_stacks
        self.frames = []

        assert isinstance(env.observation_space, Box)
        assert len(env.observation_space.shape) == 3

        # Новое пространство наблюдений
        low = env.observation_space.low.repeat(num_stacks, axis=-1)
        high = env.observation_space.high.repeat(num_stacks, axis=-1)
        self.observation_space = Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.frames = [observation] * self.num_stacks
        return self._get_stacked_frames(), info

    def observation(self, observation):
        self.frames.pop(0)
        self.frames.append(observation)
        return self._get_stacked_frames()

    def _get_stacked_frames(self):
        return np.concatenate(self.frames, axis=-1)


class NormalizeWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, Box)

        # Определяем новый диапазон [0, 1]
        self.observation_space = Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

        # Вычисляем масштаб для нормализации
        original_low = env.observation_space.low
        original_high = env.observation_space.high
        self.scale = original_high - original_low
        self.min = original_low

    def observation(self, observation):
        normalized = (observation - self.min) / self.scale
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)
