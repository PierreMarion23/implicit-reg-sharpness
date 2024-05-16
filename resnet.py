import jax.numpy as jnp
from jax import jit, vmap
from jax import random


def init_resnet(d, L, scale, key):
    """Initialize the deep linear network with identity + rescaled Gaussians."""
    keys = random.split(key, L)
    return [
        jnp.eye(d) + scale / jnp.sqrt(L * d) * random.normal(keys[k], (d, d))
        for k in range(L)
    ]


@jit
def linear_network_proj(params, inputs, w):
    """Implementation of a deep linear network with a final projection w."""
    h = inputs
    for W in params:
        h = jnp.dot(W, h)
    return jnp.dot(w, h)


batched_linear_network = vmap(linear_network_proj, in_axes=(None, 0, None))


def loss_fn_resnet(params, args):
    X, y, w = args
    outputs = batched_linear_network(params, X, w)
    return jnp.mean((outputs.flatten() - y) ** 2)


@jit
def square_distance_to_minimizer_resnet(params, args):
    (w_star, w) = args
    w_prod = params[0]
    for W in params[1:]:
        w_prod = W @ w_prod
    w_prod = w @ w_prod
    return jnp.mean((w_prod - w_star) ** 2)
