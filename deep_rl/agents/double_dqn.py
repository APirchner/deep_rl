from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
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

        self.optimizer = hydra.utils.instantiate(optimizer_conf, params=self.dqn.parameters())

    def _loss(self, sample_steps: Transition) -> torch.Tensor:
        state = sample_steps.state.to(self.device)
        next_state = sample_steps.next_state.to(self.device)


    def _train_step(self):
        pass

    def _select_action(self, state: torch.Tensor) -> int:
        state_tensor = torch.unsqueeze(state, 0).to(self.device)
        action = self.dqn(state_tensor).argmax().detach().cpu().item()
        return action
