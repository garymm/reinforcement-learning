import dataclasses
from collections import abc

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


def _conv_output_shape(
    conv: eqx.nn.Conv, input_shape: tuple[int, ...]
) -> tuple[int, ...]:
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
    _submodule: eqx.nn.Sequential

    def __init__(self, spec: specs.EnvironmentSpec, key: jax_types.PRNGKey):
        super().__init__()
        obs_shape = tuple(spec.observations.shape)
        assert all(isinstance(x, int) for x in obs_shape)
        obs_rank = len(obs_shape)
        if obs_rank != 3:
            raise ValueError(
                "Expected obs format (Height, Width, Channel). Got rank %d" % obs_rank
            )

        # TODO: Support (or require?) frame stacking

        # TODO: Ideally we want to use jax to do the transformations, since it can be
        # compiled and run on the GPU, but we also don't want to do the transformation
        # again for observations that are being replayed. So we should move the
        # pre-processing to the environment.

        # image resizing. TODO: make this configurable
        resize_image_to = (84, 84)
        need_resize = False
        if (obs_shape[0], obs_shape[1]) != resize_image_to:
            need_resize = True

        num_spatial_dims = 2
        keys = jax.random.split(key, 4)
        num_filters = (16, 32)
        hid_0 = eqx.nn.Conv(
            num_spatial_dims,
            1,  # in_channels=1 since we convert to grayscale below.
            num_filters[0],
            8,
            4,
            key=keys[0],
        )
        hid_0_out_shape = _conv_output_shape(hid_0, resize_image_to)
        hid_1 = eqx.nn.Conv(
            num_spatial_dims, num_filters[0], num_filters[1], 4, 2, key=keys[1]
        )
        hid_1_out_shape = _conv_output_shape(hid_1, hid_0_out_shape)
        hid_2_in_shape = num_filters[1] * int(np.prod(hid_1_out_shape))
        hid_2_out_shape = 256
        assert isinstance(spec.actions, specs.DiscreteArray)

        layers: list[abc.Callable] = [
            # channel last -> channel first
            # Would be more efficient if the observations could match what my convs expect.
            eqx.nn.Lambda(lambda x: jnp.transpose(x, (2, 0, 1))),
            # grayscale TODO: make this configurable
            eqx.nn.Lambda(lambda x: jnp.mean(x, 0, keepdims=True)),
        ]

        if need_resize:
            layers.append(
                eqx.nn.Lambda(
                    lambda x: jax.image.resize(
                        x, (1,) + resize_image_to, jax.image.ResizeMethod.NEAREST
                    )
                )
            )
        layers.extend(
            [
                hid_0,
                eqx.nn.Lambda(jax.nn.relu),
                hid_1,
                eqx.nn.Lambda(jax.nn.relu),
                # Flatten the output for the fully connected layer
                eqx.nn.Lambda(jnp.ravel),
                eqx.nn.Linear(hid_2_in_shape, hid_2_out_shape, key=keys[2]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(hid_2_out_shape, spec.actions.num_values, key=keys[3]),
            ]
        )

        self._submodule = eqx.nn.Sequential(layers)

    @eqx.filter_jit
    def __call__(self, x: jaxtyping.Array | np.ndarray) -> jaxtyping.Array:
        if x.ndim != 3:
            raise ValueError("Expected input to have rank 3. Got rank %d" % x.ndim)
        if isinstance(x, np.ndarray):
            x = jnp.array(x)
        x = x.astype(jnp.float32)
        return self._submodule(x)


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
