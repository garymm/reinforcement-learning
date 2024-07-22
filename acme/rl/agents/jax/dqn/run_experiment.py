import argparse
import logging
import os
import pickle
import sys
from typing import Optional

import dm_env_wrappers
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStack as GymFrameStack

from rl import fake_deps  # noqa # isort: skip noqa
import datetime

import optax
from acme.jax import experiments
from acme.utils.loggers import AsyncLogger, Dispatcher, TerminalLogger
from acme.utils.loggers import base as loggers_base
from gymnasium.envs.classic_control import cartpole

from rl.agents.jax.dqn.builder import DQNBuilder
from rl.agents.jax.dqn.config import DQNConfig
from rl.agents.jax.dqn.networks import make_networks
from rl.experiments.run_experiment import run_experiment
from rl.utils.loggers import mlflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

obs_num_stack = 4


class Int32DiscreteSpaceWrapper(gym.spaces.Discrete):
    """Jax doesn't like 64 bit numbers by default."""

    def __init__(self, space: gym.Space):
        if not isinstance(space, gym.spaces.Discrete):
            raise ValueError("Only Discrete action spaces are supported.")
        assert space.n < np.iinfo(np.int32).max
        self.n = np.int32(space.n)
        self.dtype = np.int32


class FrameStack(GymFrameStack):
    def observation(self, observation):
        return np.stack(list(self.frames), axis=0)


def _environment_factory_breakout(seed: int):
    del seed
    # From paper section 5:
    # We use k = 4 for all games except Space Invaders where we noticed that using
    # k = 4 makes the lasers invisible because of the period at which they blink.
    # We used k = 3 to make the lasers visible and
    # this change was the only difference in hyperparameter values between any of
    # the games.
    # TODO: use atari_wrapper to handle obs stacking
    env = gym.make("BreakoutDeterministic-v4", frameskip=4)
    env.action_space = Int32DiscreteSpaceWrapper(env.action_space)
    env = FrameStack(env, obs_num_stack)
    return dm_env_wrappers.GymnasiumWrapper(env)


class _CartPoleObsToRenderWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.unwrapped, cartpole.CartPoleEnv):
            raise ValueError("Need a CartPoleEnv")
        if env.render_mode != "rgb_array":
            raise ValueError('need render_mode="rgb_array"')

        self.observation_space = gym.spaces.Box(
            0,
            255,
            shape=(env.unwrapped.screen_height, env.unwrapped.screen_width, 3),
            dtype=np.uint8,
        )

    def observation(self, observation) -> np.ndarray:
        rendered_obs = self.env.render()
        assert isinstance(rendered_obs, np.ndarray)
        return rendered_obs


def _environment_factory_cart_pole(seed: int):
    del seed
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space = Int32DiscreteSpaceWrapper(env.action_space)
    env = _CartPoleObsToRenderWrapper(env)
    env = FrameStack(env, obs_num_stack)
    return dm_env_wrappers.GymnasiumWrapper(env)


def main(args):
    if args.env == "breakout":
        environment_factory = _environment_factory_breakout
    elif args.env == "cart_pole":
        environment_factory = _environment_factory_cart_pole
    else:
        sys.exit("invalid --env")

    mlflow_logger_factory = mlflow.make_factory(f"garymm-dqn-{args.env}")

    def logger_factory(
        label: loggers_base.LoggerLabel,
        steps_key: Optional[loggers_base.LoggerStepsKey] = None,
        instance: Optional[loggers_base.TaskInstance] = None,
    ) -> loggers_base.Logger:
        return Dispatcher(
            (
                AsyncLogger(mlflow_logger_factory(label, steps_key, instance)),
                TerminalLogger(label, time_delta=10.0),
            )
        )

    max_num_actor_steps = 2_000_000

    dqn_config = DQNConfig(
        learning_rate=optax.cosine_onecycle_schedule(
            transition_steps=max_num_actor_steps,
            peak_value=1e-3,
        ),
    )

    builder = DQNBuilder(config=dqn_config)

    timestamp = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")
    run_dir = f"runs/{args.env}/{timestamp}"

    logger.info("saving to %s", run_dir)

    config = experiments.ExperimentConfig(
        builder,
        max_num_actor_steps=max_num_actor_steps,
        seed=0,
        network_factory=make_networks,
        environment_factory=environment_factory,
        logger_factory=logger_factory,
    )
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir)
    logger.info("running experiment")

    trained = run_experiment(
        config,
        eval_every=1_000,
        eval_video_dir=run_dir,  # VideoWrapper adds a subdir automatically
        checkpoint_dir=checkpoint_dir,
    )
    logger.info("done")
    model_state = trained.save()
    filename = f"{run_dir}/model_state_final.pkl"
    with open(filename, "wb") as f:
        f.write(pickle.dumps(model_state))


argparser = argparse.ArgumentParser()
argparser.add_argument("--env")
if __name__ == "__main__":
    main(argparser.parse_args())
