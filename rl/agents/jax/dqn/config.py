import dataclasses


@dataclasses.dataclass
class DQNConfig:
    """Configuration options for DQN."""

    batch_size: int = 32
    discount: float = 0.99
    epsilon: float = 0.01  # TODO: support schedule
    learning_rate: float = 2e-4  # TODO: support optax.Schedule
    policy_epsilon: float = 0.01
    replay_buffer_size: int = 1_000_000
    seed: int = 0
    # How often to update the the actor from the learner.
    variable_update_period: int = 1
    device: str = "cpu"
    # TODO: maybe frame skip, frame stacking, but I guess that maybe
    # done inside the environment?

    def __post_init__(self):
        assert self.batch_size > 0
        assert 0 <= self.discount <= 1
        assert self.learning_rate > 0
        assert self.replay_buffer_size > self.batch_size
        assert self.seed >= 0
