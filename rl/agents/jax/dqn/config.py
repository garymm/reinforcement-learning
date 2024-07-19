import dataclasses
import typing

from optax import ScalarOrSchedule


def _is_valid_rate(x: ScalarOrSchedule):
    return (isinstance(x, float) and x > 0) or (isinstance(x, typing.Callable))


@dataclasses.dataclass
class DQNConfig:
    """Configuration options for DQN."""

    batch_size: int = 32
    discount: float = 0.99
    policy_epsilon: ScalarOrSchedule = 0.01
    learning_rate: ScalarOrSchedule = 2e-4
    replay_buffer_size: int = 1_000_000
    seed: int = 0
    # How often to update the the actor from the learner.
    variable_update_period: int = 1

    num_stacked_observations: int = 1

    def __post_init__(self):
        assert self.batch_size > 0
        assert 0 <= self.discount <= 1
        assert _is_valid_rate(self.policy_epsilon)
        assert _is_valid_rate(self.learning_rate)
        assert self.replay_buffer_size > self.batch_size
        assert self.seed >= 0
