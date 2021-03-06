import os

from omegaconf import DictConfig
import hydra
import numpy as np
import torch

from deep_rl.components.environment import get_environment


@hydra.main(config_path='deep_rl/conf', config_name='config.yaml')
def test(cfg: DictConfig) -> None:
    def seed_torch(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(cfg.trainer.seed)
    seed_torch(cfg.trainer.seed)
    env = get_environment(cfg.trainer.observation_size, os.getcwd(), cfg.trainer.random_stage, cfg.trainer.seed)

    agent = hydra.utils.instantiate(
        cfg.agent,
        _args_=[
            env,
            (cfg.trainer.stack_frames, cfg.trainer.observation_size, cfg.trainer.observation_size),
            cfg.optim
        ],
        _recursive_=False
    )
    agent.load(cfg.tester.path)
    agent.test_episode()


if __name__ == '__main__':
    test()
