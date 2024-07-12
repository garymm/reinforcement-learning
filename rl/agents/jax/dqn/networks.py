import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from corax import specs
from corax.jax import types as jax_types
from corax.jax.networks import base as networks_base


@dataclasses.dataclass
class DQNNetworks:
    # TODO: having to conform to TypedFeedForwardNetwork is awkward.
    # seems needed for Haiku, but I'm using Equinoxs, so the init and apply
    # is annoying.
    q_network: networks_base.TypedFeedForwardNetwork


# def _conv_output_shape(conv: eqx.nn.Conv, input_shape: tuple[int]) -> tuple[int]:
#     sizes = []
#     for i, input_size in enumerate(input_shape):
#         sizes.append((input_size - conv.out_channels) // conv.stride[i] + 1)
#     return tuple(sizes)


def _conv_output_shape(conv: eqx.nn.Conv, input_shape: tuple[int]) -> tuple[int]:
    sizes = []
    for i, input_size in enumerate(input_shape):
        filter_size = (
            conv.kernel_size[i]
            if isinstance(conv.kernel_size, (tuple, list))
            else conv.kernel_size
        )
        stride = (
            conv.stride[i] if isinstance(conv.stride, (tuple, list)) else conv.stride
        )
        sizes.append((input_size - filter_size) // stride + 1)
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
    # TODO: Do the transformations!
    # probably more efficient to do the transformations φ inside the network,
    # since I can compile them.
    _submodule: eqx.nn.Sequential

    def __init__(self, spec: specs.EnvironmentSpec, key: jax_types.PRNGKey):
        obs_shape = spec.observations.shape
        obs_rank = len(obs_shape)
        if obs_rank < 3 or obs_rank > 4:
            raise ValueError("Expected obs format BHWC or HWC. Got rank %d" % obs_rank)

        batched_inputs = obs_rank == 4
        if batched_inputs:
            obs_shape = obs_shape[1:]
            obs_rank -= 1

        num_spatial_dims = obs_rank - 1
        keys = jax.random.split(key, 4)
        num_filters = (16, 32)
        hid_0 = eqx.nn.Conv(
            num_spatial_dims, obs_shape[-1], num_filters[0], 8, 4, key=keys[0]
        )
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
                eqx.nn.Lambda(jax.nn.relu),
                hid_1,
                eqx.nn.Lambda(jax.nn.relu),
                # Flatten the output for the fully connected layer
                eqx.nn.Lambda(jnp.ravel),
                eqx.nn.Linear(hid_2_in_shape, hid_2_out_shape, key=keys[2]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(hid_2_out_shape, spec.actions.num_values, key=keys[3]),
            )
        )

    @eqx.filter_jit
    def __call__(self, x: jaxtyping.Array | np.ndarray) -> jaxtyping.Array:
        if isinstance(x, np.ndarray):
            x = jnp.array(x)
        x = x.astype(jnp.float32)
        # channel last -> channel first
        # Assuming channel last is the input format because of this:
        # https://github.com/google-deepmind/acme/blob/bea6d6b27c366cd07dd5202356f372e02c1f3f9b/acme/jax/networks/atari.py#L56
        # Would be more efficient if the observations could match what my convs expect.
        if x.ndim == 3:
            x = x.transpose((2, 0, 1))
            return self._submodule(x)
        elif x.ndim == 4:
            x = x.transpose((0, 3, 1, 2))
            return jax.vmap(self._submodule)(x)
        else:
            raise ValueError("Expected input to have rank 3 or 4. Got rank %d" % x.ndim)


def make_networks(
    environment_spec: specs.EnvironmentSpec,
) -> DQNNetworks:
    # This seems awkward and verbose.
    # relates to the TODO in the DQNNetworks class above.
    q_network_static_pytree: jaxtyping.PyTree | None = None

    def _apply(params: jaxtyping.PyTree, observation, *args, is_training, key=None):
        q_network = eqx.combine(params, q_network_static_pytree)
        return q_network(observation)

    def _init(key: jax_types.PRNGKey) -> networks_base.Params:
        q_network = _QNetwork(environment_spec, key)
        arrays, static = eqx.partition(q_network, eqx.is_array)
        nonlocal q_network_static_pytree
        q_network_static_pytree = static
        return arrays

    typed_ffn = networks_base.TypedFeedForwardNetwork(
        init=_init,
        apply=_apply,
    )
    return DQNNetworks(typed_ffn)
