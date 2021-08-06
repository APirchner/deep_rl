import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import logging

import numpy as np
import torch
import gym
from torch.utils.tensorboard import SummaryWriter

from deep_rl.components.buffers import ReplayBuffer, Transition

log = logging.getLogger(__name__)


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
            update_steps: int,
            test: bool = False,
            cuda: bool = True
    ):
        self.env = env
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.observation_size = observation_size
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.update_steps = update_steps
        self.test = test
        self.tb_writer = SummaryWriter()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and cuda else 'cpu'
        )

    @abstractmethod
    def _loss(self, sample_steps: Transition) -> torch.Tensor:
        raise NotImplementedError('_loss')

    @abstractmethod
    def _train_step(self, sample_steps: Transition) -> float:
        raise NotImplementedError('_train_step')

    @abstractmethod
    def _select_action(self, state: torch.Tensor) -> int:
        raise NotImplementedError('_select_action')

    @abstractmethod
    def _update_target(self):
        raise NotImplementedError('_update_target')

    @abstractmethod
    def _save_state(self, path: str):
        raise NotImplementedError('_save_state')

    def _write_to_buffer(
            self,
            state: np.ndarray,
            next_state: np.ndarray,
            action: int,
            reward: float,
            done: bool
    ) -> None:
        self.replay_buffer.add(
            state=torch.tensor(state, dtype=torch.float),
            next_state=torch.tensor(next_state, dtype=torch.float),
            action=torch.tensor(action),
            reward=torch.tensor(reward, dtype=torch.float),
            done=torch.tensor(done, dtype=torch.float)
        )

    def action(self, state: np.ndarray) -> int:
        if not self.test and self.eps > np.random.random():
            # exploration
            action = self.env.action_space.sample()
        else:
            # exploitation
            action = self._select_action(torch.tensor(state, dtype=torch.float))
        self.eps = max(self.eps_min, self.eps)
        return action

    def step(self, state: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state = state.__array__()
        action = self.action(state)
        state_new, reward, is_done, info = self.env.step(action)
        self._write_to_buffer(
            state, state_new.__array__(), action, reward, is_done
        ) if not self.test else None
        return state_new, reward, is_done, info

    def train(self, frames: int):
        self.test = False
        eps_decay = (self.eps - self.eps_min) / frames
        rewards = []
        scores = []
        losses = []
        score = 0

        state = self.env.reset()
        for i in range(frames):
            state_next, reward, is_done, info = self.step(state.__array__())
            self.eps = self.eps - eps_decay
            rewards.append(reward)
            score += reward
            if is_done:
                state = self.env.reset()
                scores.append(score)
                score = 0
            if len(self.replay_buffer) < self.replay_buffer.batch_size:
                continue
            batch = self.replay_buffer.sample()
            loss = self._train_step(batch)
            losses.append(loss)
            if i % self.update_steps == 0:
                self._update_target()
            if i % 1000 == 0:
                mean_score = np.array(scores).mean() if len(scores) > 0 else 0
                mean_loss = np.array(losses).mean() if len(losses) > 0 else 0
                log.info(f'Step {i} - Score: {mean_score} | Loss: {mean_loss}')
                self.tb_writer.add_scalar('Score', mean_score, i)
                self.tb_writer.add_scalar('Loss', mean_loss, i)
                self.tb_writer.add_scalar('Eps', self.eps, i)
                losses = []
            if i % 100000 == 0:
                self._save_state(os.path.join(os.getcwd(), f'checkpoint_step_{i}'))
