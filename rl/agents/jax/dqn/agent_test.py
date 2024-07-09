import unittest

import numpy as np
from corax import specs
from corax.jax import experiments
from corax.testing import fakes
from corax.utils import loggers
from dm_env import specs as env_specs

from rl.agents.jax.dqn.builder import DQNBuilder
from rl.agents.jax.dqn.config import DQNConfig
from rl.agents.jax.dqn.networks import make_networks


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
        )
    )
    return environment


class DrQV2Test(unittest.TestCase):
    def test_agent(self):
        builder = DQNBuilder(config=DQNConfig())
        config = experiments.ExperimentConfig(
            builder,
            max_num_actor_steps=20,
            seed=0,
            network_factory=make_networks,
            environment_factory=_make_fake_environment,
            logger_factory=_make_empty_experiment_logger,
            checkpointing=None,
        )
        experiments.run_experiment(config)


if __name__ == "__main__":
    unittest.main()
