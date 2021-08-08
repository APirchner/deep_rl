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
            learn_period: int,
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
        self.learn_period = learn_period
        self.test = test
        self.tb_writer = SummaryWriter()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and cuda else 'cpu'
        )

    @abstractmethod
    def _loss(self, sample_steps: Transition) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('_loss')

    @abstractmethod
    def _train_step(self, sample_steps: Transition) -> Tuple[float, float]:
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

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def _eval(self):
        pass

    def _write_to_buffer(
            self,
            state: np.ndarray,
            next_state: np.ndarray,
            action: int,
            reward: float,
            done: bool
    ) -> None:
        self.replay_buffer.add(
            state=torch.tensor(state, dtype=torch.uint8),
            next_state=torch.tensor(next_state, dtype=torch.uint8),
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
        eps_decay = 5 * (self.eps - self.eps_min) / frames
        scores = []
        losses = []
        mean_qs = []
        score = 0

        state = self.env.reset()
        for i in range(frames):
            state_next, reward, is_done, info = self.step(state)
            state = state_next
            score += reward
            self.eps = self.eps - eps_decay

            if is_done:
                # end of episode
                scores.append(score)
                score = 0
                state = self.env.reset()

            if len(self.replay_buffer) < 10_000:
                continue

            if i % self.learn_period == 0:
                # training
                batch = self.replay_buffer.sample()
                loss, mean_q = self._train_step(batch)
                losses.append(loss)
                mean_qs.append(mean_q)

            # target update
            if i % self.update_steps == 0:
                self._update_target()

            # logging
            if i % 1_000 == 0:
                scores = scores[max(0, len(scores) - 10):]
                mean_score = np.array(scores).mean()
                mean_loss = np.array(losses).mean()
                mean_q = np.array(mean_qs).mean()
                log.info(f'Step {i} - Score: {mean_score} | Loss: {mean_loss}')
                self.tb_writer.add_scalar('Score (last 10 episodes)', mean_score, i)
                self.tb_writer.add_scalar('Loss', mean_loss, i)
                self.tb_writer.add_scalar('Q estimate', mean_q, i)
                self.tb_writer.add_scalar('Eps', self.eps, i)
                losses = []
                mean_qs = []
            if i % 100_000 == 0:
                self._save_state(os.path.join(os.getcwd(), f'checkpoint_step_{i}'))
        self.env.close()

    def evaluate(self) -> float:
        eps_train = self.eps
        eps_min = self.eps_min
        self.eps = 0.001
        self.eps_min = 0.
        self._eval()
        state = self.env.reset()
        score = 0
        is_done = False
        while not is_done:
            state, reward, is_done, _ = self.step(state)
            score += reward
        self.eps = eps_train
        self.eps_min = eps_min
        self._train()
        self.env.reset()
        return score
