import abc
import argparse
from functools import partial
from typing import NamedTuple, Tuple, TypeAlias

import equinox as eqx
import gymnasium
import gymnax
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from gymnax.environments.environment import Environment as GymnaxEnvironment

AgentState: TypeAlias = jaxtyping.PyTree
EnvState: TypeAlias = jaxtyping.PyTree


class EnvTimestep(NamedTuple):
    obs: jaxtyping.Array
    reward: jaxtyping.Array
    terminated: jaxtyping.Array


class Agent(abc.ABC):
    @abc.abstractmethod
    def initial_state(self, key: jaxtyping.PRNGKeyArray) -> AgentState:
        """Initializes agent state."""

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def select_action_batched(
        self, agent_state: AgentState, env_timestep: EnvTimestep
    ) -> Tuple[AgentState, jnp.ndarray]:
        """Selects a batch of actions for a timestep."""


@eqx.filter_jit
def gymnax_loop(
    env: GymnaxEnvironment, num_envs: int, agent: Agent, num_steps: int
) -> Tuple[EnvState, AgentState]:
    key = jax.random.PRNGKey(0)
    agent_key, env_key, scan_key = jax.random.split(key, 3)
    env_keys = jax.random.split(env_key, num_envs)
    agent_state = agent.initial_state(agent_key)
    obs, env_state = jax.vmap(env.reset)(env_keys)

    def step_time(
        carry: Tuple[EnvTimestep, EnvState, AgentState, jaxtyping.PRNGKeyArray], _
    ) -> Tuple[Tuple[EnvTimestep, EnvState, AgentState, jaxtyping.PRNGKeyArray], None]:
        env_timestep, env_state, agent_state, key = carry
        agent_state, action = agent.select_action_batched(agent_state, env_timestep)
        env_key, key = jax.random.split(key)
        env_keys = jax.random.split(env_key, num_envs)
        obs, env_state, reward, done, info = jax.vmap(env.step)(
            env_keys, env_state, action
        )
        next_timestep = EnvTimestep(obs, reward, done)
        return (
            (next_timestep, env_state, agent_state, key),
            None,
        )

    env_timestep = EnvTimestep(
        obs, jnp.zeros((num_envs,)), jnp.zeros((num_envs,), dtype=jnp.bool_), None
    )
    (env_timestep, env_state, agent_state, _), ys = jax.lax.scan(
        step_time,
        init=(env_timestep, env_state, agent_state, scan_key),
        xs=None,
        length=num_steps,
    )

    return env_state, agent_state


def gymnasium_loop(
    env: gymnasium.vector.VectorEnv, agent: Agent, num_steps: int
) -> AgentState:
    key = jax.random.PRNGKey(0)
    agent_key, key = jax.random.split(key)
    env_keys = jax.random.split(key, env.num_envs)
    agent_state = agent.initial_state(agent_key)
    with env:
        obs, info = env.reset(seed=[int(k[0]) for k in env_keys])
        env_timestep = EnvTimestep(
            obs,
            jnp.zeros((env.num_envs,)),
            jnp.zeros((env.num_envs,), dtype=jnp.bool),
        )

        for _ in range(num_steps):
            agent_state, action = agent.select_action_batched(agent_state, env_timestep)
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            # TODO: not sure about the logical or on terminated | truncated
            env_timestep = EnvTimestep(
                obs, jnp.array(reward), jnp.array(terminated | truncated)
            )


class _StupidState(NamedTuple):
    key: jaxtyping.PRNGKeyArray


class StupidAgent(Agent):
    def initial_state(self, key: jaxtyping.PRNGKeyArray) -> AgentState:
        return _StupidState(key)

    @partial(jax.jit, static_argnums=(0,))
    def select_action_batched(
        self, agent_state: _StupidState, env_timestep: EnvTimestep
    ) -> Tuple[_StupidState, jnp.ndarray]:
        key, action_key = jax.random.split(agent_state.key)
        actions = jax.random.randint(action_key, (env_timestep.obs.shape[0],), 0, 2)
        return _StupidState(key), actions


parser = argparse.ArgumentParser()
parser.add_argument(
    "--loop", type=str, choices=["gymnasium", "gymnax"], default="gymnax"
)

if __name__ == "__main__":
    args = parser.parse_args()
    agent = StupidAgent()
    num_steps = 100
    num_envs = 2
    if args.loop == "gymnasium":
        env = gymnasium.vector.AsyncVectorEnv(
            [partial(gymnasium.make, "CartPole-v1") for _ in range(num_envs)]
        )
        gymnasium_loop(env, agent, num_steps)
    else:
        # TODO: probably want to pass in the params
        env, params = gymnax.make("CartPole-v1")
        gymnax_loop(env, num_envs, agent, num_steps)
