from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import optax
import reverb
from corax import adders, core, specs
from corax.adders import reverb as adders_reverb
from corax.adders.reverb.base import DEFAULT_PRIORITY_TABLE
from corax.agents.jax import actor_core as actor_core_lib
from corax.agents.jax import actors, builders
from corax.datasets import reverb as datasets_reverb
from corax.jax import networks as networks_lib
from corax.jax import types as jax_types
from corax.jax import utils, variable_utils
from corax.jax.experiments.savers import Checkpointer
from corax.utils import counting, loggers
from jax.dtypes import issubdtype
from reverb import rate_limiters

from rl.agents.jax.dqn.config import DQNConfig
from rl.agents.jax.dqn.learning import DQNLearner
from rl.agents.jax.dqn.networks import (
    DQNNetworks,
)


class DQNBuilder(
    builders.ActorLearnerBuilder[
        DQNNetworks, actor_core_lib.FeedForwardPolicy, reverb.ReplaySample
    ]
):
    def __init__(self, config: DQNConfig):
        self._config = config

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: actor_core_lib.FeedForwardPolicy,
    ) -> list[reverb.Table]:
        """Create tables to insert data into.

        Args:
          environment_spec: A container for all relevant environment specs.
          policy: Agent's policy which can be used to extract the extras_spec.

        Returns:
          The replay tables used to store the experience the agent uses to train.
        """
        signature = adders_reverb.NStepTransitionAdder.signature(environment_spec)
        return [
            reverb.Table(
                name=DEFAULT_PRIORITY_TABLE,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self._config.replay_buffer_size,
                rate_limiter=rate_limiters.Queue(self._config.replay_buffer_size),
                signature=signature,
            )
        ]

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        dataset = datasets_reverb.make_reverb_dataset(
            table=DEFAULT_PRIORITY_TABLE,
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
        )
        return utils.device_put(dataset.as_numpy_iterator(), jax.devices()[0])

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec],
        policy: Optional[builders.Policy],
    ) -> Optional[adders.Adder]:
        """Create an adder which records data generated by the actor/environment.

        Args:
          replay_client: Reverb Client which points to the replay server.
          environment_spec: specs of the environment.
          policy: Agent's policy which can be used to extract the extras_spec.
        """
        # paper section 4:
        # we store the agent’s experiences at each time-step, e_t = (s_t, a_t, r_t, s_{t+1})
        return adders_reverb.NStepTransitionAdder(
            replay_client, 1, self._config.discount
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: actor_core_lib.FeedForwardPolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> core.Actor:
        assert variable_source
        actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
        variable_client = variable_utils.VariableClient(
            variable_source,
            "q",
            update_period=self._config.variable_update_period,
        )

        return actors.GenericActor(
            actor_core,
            random_key,
            variable_client,
            adder,
            backend=None,
        )

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: DQNNetworks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional["reverb.Client"] = None,
        counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        """Creates an instance of the learner.

        Args:
          random_key: A key for random number generation.
          networks: struct describing the networks needed by the learner; this can
            be specific to the learner in question.
          dataset: iterator over samples from replay.
          logger_fn: factory providing loggers used for logging progress.
          environment_spec: A container for all relevant environment specs.
          replay_client: client which allows communication with replay. Note that
            this is only intended to be used for updating priorities. Samples should
            be obtained from `dataset`.
          counter: a Counter which allows for recording of counts (learner steps,
            actor steps, etc.) distributed throughout the agent.
        """
        learner = DQNLearner(
            networks=networks,
            random_key=random_key,
            iterator=dataset,
            q_optimizer=optax.sgd(self._config.learning_rate),
            counter=counter,
            logger=logger_fn("learner"),
        )
        if self._config.restore_from_checkpoint:
            checkpointer = Checkpointer(
                {"learner": learner}, directory=self._config.restore_from_checkpoint
            )
            # TODO: this silently fails.
            # checkpointer.restore() should check the status returned by self._checkpoint.restore,
            # or at least return it.
            checkpointer.restore()
        return learner

    def make_policy(
        self,
        networks: DQNNetworks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> actor_core_lib.FeedForwardPolicy:
        assert networks.q_network
        action_dtype = environment_spec.actions.dtype
        assert issubdtype(action_dtype, jnp.integer)

        def _greedy_policy(
            params: jaxtyping.PyTree,
            key: jax_types.PRNGKey,
            obs: jaxtyping.Array | np.ndarray,
        ) -> jaxtyping.Array:
            q = networks.q_network.apply(params, obs, is_training=True)
            res = jnp.argmax(q)
            return res.astype(action_dtype)

        def _random_action(
            params: jaxtyping.PyTree,
            key: jax_types.PRNGKey,
            obs: jaxtyping.Array | np.ndarray,
        ) -> jaxtyping.Array:
            return jax.random.randint(
                key, (), 0, environment_spec.actions.maximum + 1, dtype=action_dtype
            )

        def _epsilon_greedy_policy(
            params: jaxtyping.PyTree,
            key: jax_types.PRNGKey,
            obs: jaxtyping.Array | np.ndarray,
        ) -> jaxtyping.Array:
            # From the paper algorithm 1, epsilon greedy policy.
            return jax.lax.cond(
                jax.random.uniform(key) < self._config.epsilon,
                _random_action,
                _greedy_policy,
                params,
                key,
                obs,
            )

        policy = _epsilon_greedy_policy
        if evaluation:
            policy = _greedy_policy
        return jax.vmap(policy, in_axes=(None, None, 0))
