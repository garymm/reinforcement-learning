# Based on https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/examples/pytorch/pg_math/1_simple_pg.py
import equinox as eqx
import gymnasium
import jax
import jax.numpy as np
import optax
import pygame
from gymnasium.spaces import Box, Discrete
from jaxtyping import Array, Float, Int


def make_mlp(layer_dims: list[int], prng_key: Array) -> eqx.Module:
    layers: list[eqx.Module] = []
    for i in range(len(layer_dims) - 1):
        layers.append(eqx.nn.Linear(layer_dims[i], layer_dims[i + 1], key=prng_key))
        if i < len(layer_dims) - 2:
            layers.append(eqx.nn.Lambda(np.tanh))
        else:
            layers.append(eqx.nn.Identity())
    return eqx.nn.Sequential(layers)


def compute_loss(
    logits_net: eqx.Module,
    obs: Float[Array, "batch obs"],
    acts: Int[Array, "batch"],
    weights: Float[Array, "batch t"],
) -> float:
    """loss function whose gradient, for the right data, is policy gradient"""
    logits = jax.vmap(logits_net)(obs)
    log_probs = jax.vmap(jax.nn.log_softmax)(logits)
    log_probs_for_actions = log_probs[np.arange(log_probs.shape[0]), acts]
    return -np.mean(weights * log_probs_for_actions)


def train(
    env_name: str,
    render: bool,
    lr: float,
    hidden_sizes=[32],
    epochs=50,
    batch_size=5000,
):
    # make environment, check spaces, get obs / act dims
    env = gymnasium.make(env_name, render_mode="rgb_array")
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, Discrete
    ), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = make_mlp([obs_dim] + hidden_sizes + [n_acts], jax.random.key(0))

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(model: eqx.Module, obs: Float[Array, str(obs_dim)], prng_key) -> int:
        logits = model(obs)
        return jax.random.categorical(prng_key, logits).item()

    # make optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(logits_net, eqx.is_array))

    surface = None
    if render:
        env.reset()
        pixels = env.render().transpose(1, 0, 2)
        pygame.init()
        surface = pygame.display.set_mode((pixels.shape[0], pixels.shape[1]))

    # for training policy
    # TODO: @eqx.filter_jit
    def train_one_epoch(
        model: eqx.Module, opt_state: optax.OptState, prng_key: Array
    ) -> tuple:
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()[0]  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                pixels = env.render().transpose(1, 0, 2)
                pygame.pixelcopy.array_to_surface(surface, pixels)
                pygame.display.flip()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            prng_key, subkey = jax.random.split(prng_key)
            act = get_action(model, np.asarray(obs, dtype=np.float32), subkey)
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                (obs, _), done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        batch_loss, grads = eqx.filter_value_and_grad(compute_loss)(
            model,
            np.asarray(batch_obs, dtype=np.float32),
            np.asarray(batch_acts, dtype=np.int32),
            np.asarray(batch_weights, dtype=np.float32),
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, batch_loss, np.asarray(batch_rets), np.asarray(
            batch_lens
        ), prng_key

    # training loop
    prng_key = jax.random.key(0)
    for i in range(epochs):
        (
            logits_net,
            opt_state,
            batch_loss,
            batch_rets,
            batch_lens,
            prng_key,
        ) = train_one_epoch(logits_net, opt_state, prng_key)
        print(
            "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    print("\nUsing simplest formulation of policy gradient.\n")
    train(env_name=args.env_name, render=args.render, lr=args.lr)
