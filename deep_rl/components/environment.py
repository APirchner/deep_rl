from typing import Tuple

import numpy as np
import torch
from torchvision import transforms as T
import gym
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class PermuteObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(PermuteObservation, self).__init__(env)

        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def _permute(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.tensor(observation.copy())
        obs = torch.permute(obs, (2, 0, 1))
        return obs

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self._permute(observation)


class GrayScaleResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: int):
        super(GrayScaleResizeObservation, self).__init__(env)
        self._size = size

        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation: np.ndarray):
        transform = T.Compose([
            T.Resize((self._size, self._size)),
            T.Grayscale(),
        ])
        return transform(observation).squeeze(0)


def get_environment(frame_size: int, path: str, random_stages: bool = True, seed: int = 170990) -> gym.Env:
    if random_stages:
        env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
    else:
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = gym.wrappers.Monitor(env, path)
    env = SkipFrame(env, 4)
    env = PermuteObservation(env)
    env = GrayScaleResizeObservation(env, frame_size)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(seed)
    return env
