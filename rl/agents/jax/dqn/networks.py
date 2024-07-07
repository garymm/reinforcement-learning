import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from corax import specs
from corax.jax import types as jax_types
from corax.jax.networks import base as networks_base

# # Adding a few types here because the corax types seem to assume haiku
# # (which has separate .init() and .apply()  methods) and I'm using equinox.
# # Inputs can be numpy or Jax, outputs are always Jax.
# CallableFeedForwardNetwork = Callable[[jaxtyping.Array | np.ndarray], jaxtyping.Array]
# CallableFeedForwardPolicy = Callable[
#     [jax_types.PRNGKey, jaxtyping.Array | np.ndarray], jaxtyping.Array
# ]


@dataclasses.dataclass
class DQNNetworks:
    # TODO: having to conform to TypedFeedForwardNetwork is awkward.
    # seems needed for Haiku, but I'm using Equinoxs, o the init and apply
    # is annoying.
    q_network: networks_base.TypedFeedForwardNetwork


def _conv_output_shape(conv: eqx.nn.Conv, input_shape: tuple[int]) -> tuple[int]:
    sizes = []
    for i, input_size in enumerate(input_shape):
        sizes.append((input_size - conv.out_channels) // conv.stride[i] + 1)
    return tuple(sizes)


class _QNetwork(eqx.Module):
    # From the paper §4.1:
    # The input to the neural network consists is an 84 × 84 × 4 image produced by φ.
    # The first hidden layer convolves 16 8 × 8 filters with stride 4 with the input
    # image and applies a rectifier nonlinearity.
    # The second hidden layer convolves 32 4 × 4 filters with stride 2, again
    # followed by a rectifier nonlinearity.
    # The final hidden layer is fully-connected and consists of 256 rectifier units.
    # The output layer is a fully-connected linear layer with a single output for
    # each valid action.
    # TODO: probably more efficient to do the transformations φ inside the network,
    # since I can compile them.
    _submodule: eqx.nn.Sequential
    _batch_size: int

    def __init__(self, spec: specs.EnvironmentSpec, key: jax_types.PRNGKey):
        obs_shape = spec.observations.shape
        num_spatial_dims = len(obs_shape) - 1
        self._batch_size = obs_shape[
            -1
        ]  # TODO: batch size may not be specified this way
        keys = jax.random.split(key, 4)
        num_filters = (16, 32)
        hid_0 = eqx.nn.Conv(num_spatial_dims, 1, num_filters[0], 8, 4, key=keys[0])
        hid_0_out_shape = _conv_output_shape(hid_0, obs_shape[:-1])
        hid_1 = eqx.nn.Conv(
            num_spatial_dims, num_filters[0], num_filters[1], 4, 2, key=keys[1]
        )
        hid_1_out_shape = _conv_output_shape(hid_1, hid_0_out_shape)
        hid_2_in_shape = num_filters[1] * int(np.prod(hid_1_out_shape))
        hid_2_out_shape = 256
        assert isinstance(spec.actions, specs.DiscreteArray)

        self._submodule = eqx.nn.Sequential(
            (
                hid_0,
                jax.nn.relu,
                hid_1,
                jax.nn.relu,
                # Flatten the output for the fully connected layer
                lambda x: jnp.reshape(x, (x.shape[0], hid_2_in_shape)),
                eqx.nn.Linear(hid_2_in_shape, hid_2_out_shape, key=keys[2]),
                jax.nn.relu,
                eqx.nn.Linear(hid_2_out_shape, spec.actions.num_values, key=keys[3]),
            )
        )

    def __call__(self, x: jaxtyping.Array | np.ndarray) -> jaxtyping.Array:
        # TODO: handle batch size.
        if isinstance(x, np.ndarray):
            x = jnp.array(x)
        return self._submodule(x)


def make_networks(
    spec: specs.EnvironmentSpec,
) -> DQNNetworks:
    def _apply_fn(params: _QNetwork, observation, *args, is_training, key=None):
        return params(observation)

    typed_ffn = networks_base.TypedFeedForwardNetwork(
        init=lambda key: _QNetwork(spec, key),
        apply=_apply_fn,
    )
    return DQNNetworks(typed_ffn)
