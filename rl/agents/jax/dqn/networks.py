import dataclasses

import corax.jax.networks.base as base_networks
import equinox as eqx
from corax import specs


@dataclasses.dataclass
class DQNNetworks:
    q_network: base_networks.QNetwork


def make_networks(
    spec: specs.EnvironmentSpec,
) -> DQNNetworks:
    # TODO: need to know if observations are channel first or last
    obs_shape = spec.observations.shape
    return DQNNetworks(q_network=eqx.nn.Conv())
