import jax.numpy as jnp
from jax import jit, vmap
from jax import random


def init_mlp(d, L, scale, key):
    """Initialize the deep linear network with Gaussians."""
    keys = random.split(key, L)
    return [scale * random.normal(keys[k], (d, d)) for k in range(L - 1)] + [
        scale * random.normal(keys[-1], (1, d))
    ]


@jit
def linear_network(params, inputs):
    """Implementation of a deep linear network."""
    h = inputs
    for W in params:
        h = jnp.dot(W, h)
    return h


batched_linear_network = vmap(linear_network, in_axes=(None, 0))


def loss_fn_mlp(params, args):
    X, y, Xtest, ytest = args
    outputs = batched_linear_network(params, X)
    outputs_test = batched_linear_network(params, Xtest)
    return jnp.mean((outputs.flatten() - y) ** 2), jnp.mean(
        (outputs_test.flatten() - ytest) ** 2
    )


@jit
def square_distance_to_minimizer_mlp(params, args):
    (w_star,) = args
    w_prod = params[0]
    for W in params[1:]:
        w_prod = W @ w_prod
    return jnp.mean((w_prod - w_star) ** 2)
