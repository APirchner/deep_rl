from collections import namedtuple, deque
import random

import torch

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity: int = 1024, batch_size: int = 128):
        self.buffer = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def add(self, state, next_state, action, reward, done):
        self.buffer.append(
            Transition(state=state, next_state=next_state, action=action, reward=reward, done=done)
        )

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return Transition(
            state=torch.stack([b.state for b in batch]),
            next_state=torch.stack([b.next_state for b in batch]),
            action=torch.stack([b.action for b in batch]),
            reward=torch.stack([b.reward for b in batch]),
            done=torch.stack([b.done for b in batch])
        )

    def __len__(self):
        return len(self.buffer)
