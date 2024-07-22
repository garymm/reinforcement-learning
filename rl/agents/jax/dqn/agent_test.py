import unittest
from typing import Optional

import numpy as np

from rl import fake_deps  # noqa # isort: skip noqa
from acme import specs
from acme.jax.experiments.config import ExperimentConfig
from acme.testing import fakes
from acme.utils import loggers
from acme.utils.loggers import base as loggers_base
from dm_env import specs as env_specs

from rl.agents.jax.dqn.builder import DQNBuilder
from rl.agents.jax.dqn.config import DQNConfig
from rl.agents.jax.dqn.networks import make_networks
from rl.experiments.run_experiment import run_experiment

env_episode_length = 10


def logger_factory(
    label: loggers_base.LoggerLabel,
    steps_key: Optional[loggers_base.LoggerStepsKey] = None,
    instance: Optional[loggers_base.TaskInstance] = None,
) -> loggers_base.Logger:
    return loggers.TerminalLogger(label, time_delta=10.0)


def environment_factory(seed: int):
    del seed
    environment = fakes.Environment(
        specs.EnvironmentSpec(
            # stack, height, width, channel
            observations=env_specs.BoundedArray((2, 96, 96, 3), np.uint8, 0, 255),
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
            )
        )

        config = ExperimentConfig(
            builder,
            max_num_actor_steps=2 * env_episode_length,
            seed=0,
            network_factory=make_networks,
            environment_factory=environment_factory,
            logger_factory=logger_factory,
            checkpointing=None,
        )
        run_experiment(config)


if __name__ == "__main__":
    unittest.main()
