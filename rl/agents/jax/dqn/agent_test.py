import unittest

import dm_env
import numpy as np
import tree
from corax import specs
from corax.jax import experiments
from corax.testing import fakes
from corax.utils import loggers
from dm_env import specs as env_specs
from dm_env_wrappers._src.base import EnvironmentWrapper

from rl.agents.jax.dqn.builder import DQNBuilder
from rl.agents.jax.dqn.config import DQNConfig
from rl.agents.jax.dqn.networks import make_networks


def _make_empty_experiment_logger(*args, **kwargs):
    del args, kwargs
    return loggers.TerminalLogger(time_delta=10.0)


# Working around https://github.com/kevinzakka/dm_env_wrappers/pull/3
class CanonicalSpecWrapper(EnvironmentWrapper):
    """Wrapper which converts environments to use canonical action specs.

    This only affects action specs of type `specs.BoundedArray`.

    For bounded action specs, we refer to a canonical action spec as the bounding
    box [-1, 1]^d where d is the dimensionality of the spec. So the shape and
    dtype of the spec is unchanged, while the maximum/minimum values are set
    to +/- 1.
    """

    def __init__(self, environment: dm_env.Environment, clip: bool = False):
        super().__init__(environment)
        self._action_spec = environment.action_spec()
        self._clip = clip

    def step(self, action) -> dm_env.TimeStep:
        scaled_action = _scale_nested_action(action, self._action_spec, self._clip)
        return self._environment.step(scaled_action)

    def action_spec(self):
        return _convert_spec(self._environment.action_spec())


def _convert_spec(nested_spec):
    """Converts all bounded specs in nested spec to the canonical scale."""

    def _convert_single_spec(spec):
        """Converts a single spec to canonical if bounded."""
        if isinstance(spec, specs.BoundedArray) and not isinstance(
            spec, specs.DiscreteArray
        ):
            return spec.replace(
                minimum=-np.ones(spec.shape), maximum=np.ones(spec.shape)
            )
        else:
            return spec

    return tree.map_structure(_convert_single_spec, nested_spec)


def _scale_nested_action(nested_action, nested_spec, clip: bool):
    """Converts a canonical nested action back to the given nested action spec."""

    def _scale_action(action: np.ndarray, spec: specs.Array):
        """Converts a single canonical action back to the given action spec."""
        if isinstance(spec, specs.BoundedArray) and not isinstance(
            spec, specs.DiscreteArray
        ):
            # Get scale and offset of output action spec.
            scale = spec.maximum - spec.minimum
            offset = spec.minimum

            # Maybe clip the action.
            if clip:
                action = np.clip(action, -1.0, 1.0)

            # Map action to [0, 1].
            action = 0.5 * (action + 1.0)

            # Map action to [spec.minimum, spec.maximum].
            action *= scale
            action += offset

        return action

    return tree.map_structure(_scale_action, nested_action, nested_spec)


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
    return CanonicalSpecWrapper(environment, clip=True)


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
