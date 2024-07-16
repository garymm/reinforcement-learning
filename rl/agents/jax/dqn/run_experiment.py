import logging
import sys

import dm_env_wrappers
import gymnasium as gym
import numpy as np
from corax.jax import experiments

from rl.agents.jax.dqn.builder import DQNBuilder
from rl.agents.jax.dqn.config import DQNConfig
from rl.agents.jax.dqn.networks import make_networks
from rl.utils.loggers import mlflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))


class Int32DiscreteSpaceWrapper(gym.spaces.Discrete):
    """Jax doesn't like 64 bit numbers by default."""

    def __init__(self, space: gym.Space):
        if not isinstance(space, gym.spaces.Discrete):
            raise ValueError("Only Discrete action spaces are supported.")
        assert space.n < np.iinfo(np.int32).max
        self.n = np.int32(space.n)
        self.dtype = np.int32


def _make_environment(seed: int):
    del seed
    # From paper section 5:
    # We use k = 4 for all games except Space Invaders where we noticed that using
    # k = 4 makes the lasers invisible because of the period at which they blink.
    # We used k = 3 to make the lasers visible and
    # this change was the only difference in hyperparameter values between any of
    # the games.
    # TODO: obs_type="grayscale" to match the paper
    # Probably use acme/wrappers/atari_wrapper.py
    gym_env = gym.make("BreakoutDeterministic-v4", frameskip=4)
    gym_env.action_space = Int32DiscreteSpaceWrapper(gym_env.action_space)
    return dm_env_wrappers.GymnasiumWrapper(gym_env)


def main():
    builder = DQNBuilder(config=DQNConfig())
    config = experiments.ExperimentConfig(
        builder,
        max_num_actor_steps=2000,
        seed=0,
        network_factory=make_networks,
        environment_factory=_make_environment,
        logger_factory=mlflow.make_factory("garymm-dqn-breakout"),
        checkpointing=None,
    )
    logger.info("running experiment")
    experiments.run_experiment(config)


if __name__ == "__main__":
    main()
