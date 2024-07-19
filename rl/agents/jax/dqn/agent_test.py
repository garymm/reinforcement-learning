import unittest

import numpy as np

from rl import fake_deps  # noqa # isort: skip noqa
from acme import specs
from acme.jax.experiments.config import ExperimentConfig
from acme.jax.experiments.run_experiment import run_experiment
from acme.testing import fakes
from acme.utils import loggers
from dm_env import specs as env_specs

from rl.agents.jax.dqn.builder import DQNBuilder
from rl.agents.jax.dqn.config import DQNConfig
from rl.agents.jax.dqn.networks import make_networks

env_episode_length = 10


def _make_empty_experiment_logger(*args, **kwargs):
    del args, kwargs
    return loggers.TerminalLogger(time_delta=10.0)


def _make_fake_environment(seed: int):
    del seed
    environment = fakes.Environment(
        specs.EnvironmentSpec(
            observations=env_specs.BoundedArray((84, 84, 9), np.uint8, 0, 255),
            actions=env_specs.DiscreteArray(
                num_values=3,
                # dm_env lacks type hints.
                dtype=np.uint8,  # type: ignore
            ),
            rewards=env_specs.Array((), np.float32),
            discounts=env_specs.BoundedArray((), np.float32, 0, 1),
        ),
        episode_length=env_episode_length,
    )
    return environment


class DQNTest(unittest.TestCase):
    def test_agent(self):
        builder = DQNBuilder(
            config=DQNConfig(
                # trying to force learning to happen during the test.
                # these get used by
                # experiments.run_experiment._LearningActor
                # to determine when there's enough observations to learn.
                replay_buffer_size=env_episode_length,
                batch_size=2,
                num_stacked_observations=2,
            )
        )
        config = ExperimentConfig(
            builder,
            max_num_actor_steps=2 * env_episode_length,
            seed=0,
            network_factory=make_networks,
            environment_factory=_make_fake_environment,
            logger_factory=_make_empty_experiment_logger,
            checkpointing=None,
        )
        run_experiment(config)


if __name__ == "__main__":
    unittest.main()
