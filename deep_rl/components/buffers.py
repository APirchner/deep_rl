from collections import namedtuple, deque
import random

import torch

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity: int = 1024, batch_size: int = 128):
        self.buffer = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return

    def __len__(self):
        return len(self.buffer)
