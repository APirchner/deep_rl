from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import gym

from deep_rl.agents.agent import Agent
from deep_rl.components.nets import QNet
from deep_rl.components.buffers import Transition


class DoubleDQN(Agent):
    def __init__(self, env: gym.Env, observation_size: Tuple[int], optimizer_conf: DictConfig, **kwargs):
        super(DoubleDQN, self).__init__(env, observation_size, **kwargs)

        self.dqn = QNet(input_dim=self.observation_size, num_actions=self.env.action_space.n).to(self.device)
        self.dqn_target = QNet(input_dim=self.observation_size, num_actions=self.env.action_space.n).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = hydra.utils.instantiate(
            optimizer_conf, params=self.dqn.parameters()
        )  # type: torch.optim.Optimizer

    def _select_action(self, state: torch.Tensor) -> int:
        state_tensor = torch.unsqueeze(state, 0).to(self.device)
        action = self.dqn(state_tensor).argmax().detach().cpu().item()
        return action

    def _loss(self, sample_steps: Transition) -> torch.Tensor:
        state = sample_steps.state.to(self.device)
        next_state = sample_steps.next_state.to(self.device)
        action = sample_steps.action.reshape(-1, 1).to(self.device)
        reward = sample_steps.reward.reshape(-1, 1).to(self.device)
        done = sample_steps.done.reshape(-1, 1).to(self.device)

        # required for parameter update - eq (1) in paper
        q_current = self.dqn(state).gather(1, action)  # get action values
        q_next = self._target_forward(next_state)
        target = (reward + self.gamma * (1 - done) * q_next).float().to(self.device)
        loss = F.smooth_l1_loss(q_current, target).float()
        return loss

    @torch.no_grad()
    def _target_forward(self, next_state: torch.Tensor) -> torch.Tensor:
        q_next = self.dqn_target(next_state).gather(
            1, self.dqn(next_state).argmax(dim=1, keepdim=True))
        return q_next

    def _train_step(self, sample_steps: Transition) -> float:
        self.dqn.zero_grad()
        loss = self._loss(sample_steps)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()

    def _update_target(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _save_state(self, path: str):
        torch.save({
            'model_state_dict': self.dqn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
