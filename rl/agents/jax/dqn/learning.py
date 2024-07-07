"""DQN learner."""

import dataclasses
import time
from typing import Any, Iterator, Sequence

import corax
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


@dataclasses.dataclass
class _TrainingState:
    q_optimizer_state: optax.OptState
    q_params: networks_lib.Params
    target_q_params: networks_lib.Params
    random_key: networks_lib.PRNGKey


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

        key_q, random_key = jax.random.split(random_key, 2)

        q_params = networks.q_network.init(key_q)
        q_optimizer_state = q_optimizer.init(q_params)

        self.state = _TrainingState(
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=q_params,
            random_key=random_key,
        )

        def q_loss(
            q_params: networks_lib.Params,
            target_q_params: networks_lib.Params,
            transitions: corax.types.Transition,
        ):
            target = (
                transitions.reward
                + transitions.discount
                * networks.q_network.apply(
                    target_q_params, transitions.next_observation, is_training=True
                )
            )
            estimate = networks.q_network.apply(
                q_params, transitions.observation, is_training=True
            )
            return jnp.mean(jnp.square(target - estimate))

        q_loss_grad = jax.value_and_grad(q_loss)

        def update_step(
            state: _TrainingState,
            transitions: corax.types.Transition,
        ) -> tuple[_TrainingState, dict[str, jaxtyping.Array]]:
            loss, loss_grad = q_loss_grad(
                state.q_params, state.target_q_params, transitions
            )
            q_update, q_optimizer_state = q_optimizer.update(
                loss_grad, state.q_optimizer_state, state.q_params
            )
            metrics = {"q_loss": loss}
            return (
                _TrainingState(
                    q_optimizer_state=q_optimizer_state,
                    q_params=optax.apply_updates(state.q_params, q_update),
                    target_q_params=q_params,
                    random_key=state.random_key,
                ),
                metrics,
            )

        # TODO: do I need to vmap to handle batched transitions?
        self._update_step = jax.jit(update_step)

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
        variables = {"q": self._state.q_params}
        return [variables[name] for name in names]

    def save(self) -> _TrainingState:
        return self._state

    def restore(self, state: _TrainingState):
        self._state = state
