from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import gym.wrappers
import numpy as np
import torch
import gym

from deep_rl.components.buffers import ReplayBuffer, Transition


class Agent(ABC):
    def __init__(
            self,
            env: gym.Env,
            observation_size: Tuple[int],
            buffer_size: int,
            batch_size: int,
            gamma: float,
            eps: float,
            eps_min: float,
            eps_decay: float,
            test: bool = False,
            cuda: bool = True
    ):
        self.env = env
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.observation_size = observation_size
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.test = test
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and cuda else 'cpu'
        )

    @abstractmethod
    def _loss(self, sample_steps: Transition) -> torch.Tensor:
        raise NotImplementedError('_loss')

    @abstractmethod
    def _train_step(self):
        raise NotImplementedError('_train_step')

    @abstractmethod
    def _select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError('_select_action')

    def _write_to_buffer(
            self,
            state: torch.Tensor,
            next_state: torch.Tensor,
            action: int,
            reward: float,
            done: bool
    ) -> None:
        self.replay_buffer.add(
            torch.tensor(state, dtype=torch.float),
            torch.tensor(next_state, dtype=torch.float),
            torch.tensor(action),
            torch.tensor(reward),
            torch.tensor(done)
        )

    def action(self, state: torch.Tensor) -> int:
        if self.eps > np.random.random():
            # exploration
            action = self.env.action_space.sample()
        else:
            # exploitation
            action = self._select_action(state)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return action

    def step(self, state: torch.Tensor) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = self.action(state)
        state_new, reward, is_done, info = self.env.step(action)
        self._write_to_buffer(state, state_new.__array__(), action, reward, is_done) if not self.test else None
        return state_new, reward, is_done, info

    def train(self, frames: int):
        self.test = False

        rewards = []
        score = 0

        state = self.env.reset()
        for _ in range(frames):
            state_next, reward, is_done, info = self.step(torch.tensor(state.__array__(), dtype=torch.float))
            rewards.append(reward)
            score += reward
            if is_done or info['flag_get']:
                state = self.env.reset()
                score = 0
            if len(self.replay_buffer) < self.replay_buffer.batch_size:
                continue
            batch = self.replay_buffer.sample()
            loss = self._loss(batch)
            # TODO: agent update when memory buffer is at least batch size
