"""DQN learner."""

import collections
import time
from typing import Any, Iterator, Sequence

import corax
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import optax
import reverb
from corax.jax import networks as networks_lib
from corax.jax.utils import fetch_devicearray
from corax.utils import counting, loggers

from rl.agents.jax.dqn.networks import (
    DQNNetworks,
)

# eqx.filter_jit can automatically handle namedtuple but not dataclass.
# TODO: Try to use chex.dataclass.
_TrainingState = collections.namedtuple(
    "_TrainingState", ("q_optimizer_state", "q_params", "target_q_params", "random_key")
)


# TODO: maybe I can use corax.agents.jax.DefaultJaxLearner
class DQNLearner(corax.Learner):
    def __init__(
        self,
        networks: DQNNetworks,
        random_key: networks_lib.PRNGKey,
        iterator: Iterator[reverb.ReplaySample],
        q_optimizer: optax.GradientTransformation,
        counter: counting.Counter | None = None,
        logger: loggers.Logger | None = None,
    ):
        self._networks = networks
        self._iterator = iterator
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            "learner",
            asynchronous=True,
            serialize_fn=fetch_devicearray,
            steps_key=self._counter.get_steps_key(),
        )
        self._timestamp: float = 0.0

        key_q, random_key = jax.random.split(random_key, 2)

        q_params = networks.q_network.init(key_q)
        q_optimizer_state = q_optimizer.init(eqx.filter(q_params, eqx.is_array))

        self._state = _TrainingState(
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=networks.q_network.init(key_q),
            random_key=random_key,
        )

        def q_loss_single(
            q_params: networks_lib.Params,
            target_q_params: networks_lib.Params,
            transition: corax.types.Transition,
        ):
            target = transition.reward + transition.discount * networks.q_network.apply(
                target_q_params, transition.next_observation, is_training=True
            )
            estimate = networks.q_network.apply(
                q_params, transition.observation, is_training=True
            )
            return jnp.mean(jnp.square(target - estimate))

        q_loss_batched = jax.vmap(q_loss_single, in_axes=(None, None, 0))

        def q_loss(
            q_params: networks_lib.Params,
            target_q_params: networks_lib.Params,
            transition: corax.types.Transition,
        ):
            return jnp.mean(q_loss_batched(q_params, target_q_params, transition))

        q_loss_grad = eqx.filter_value_and_grad(q_loss)

        def update_step(
            state: _TrainingState,
            transition: corax.types.Transition,
        ) -> tuple[_TrainingState, dict[str, jaxtyping.Array]]:
            loss, loss_grad = q_loss_grad(
                state.q_params, state.target_q_params, transition
            )
            q_update, q_optimizer_state = q_optimizer.update(
                loss_grad, state.q_optimizer_state, state.q_params
            )
            metrics = {
                "q_loss": loss,
                "q_loss_grad_l2_norm": optax.tree_utils.tree_l2_norm(loss_grad),
            }
            if self._counter._prefix:
                # A bit hacky, but I want to ensure that all the metrics have the same prefix.
                # The instantiator of this class specified the prefix only in the counter.
                metrics = {
                    f"{self._counter._prefix}_{k}": v for k, v in metrics.items()
                }
            return (
                _TrainingState(
                    q_optimizer_state=q_optimizer_state,
                    q_params=eqx.apply_updates(state.q_params, q_update),
                    target_q_params=q_params,
                    random_key=state.random_key,
                ),
                metrics,
            )

        # TODO: donate="all" should work here but it doesn't.
        self._update_step = eqx.filter_jit(update_step)

    def step(self):
        sample = next(self._iterator)
        transitions = corax.types.Transition(*sample.data)

        self._state, metrics = self._update_step(self._state, transitions)

        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        counts = self._counter.increment(steps=1, walltime=elapsed_time)

        self._logger.write({**metrics, **counts})

    def get_variables(self, names: Sequence[str]) -> list[Any]:
        variables = {"q": eqx.filter(self._state.q_params, eqx.is_array)}
        return [variables[name] for name in names]

    def save(self) -> _TrainingState:
        return self._state

    def restore(self, state: _TrainingState):
        self._state = state
