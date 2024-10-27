from functools import partial

import jax.numpy as jnp
import jax
import jax.flatten_util
from jax import grad, jvp, value_and_grad
from jax import random


def generate_data(n, d, noise_variance, key):
    """Generates random Gaussian data with noise."""
    key, subkey = random.split(key)
    X = random.normal(subkey, (n, d))
    key, subkey = random.split(key)
    Xtest = random.normal(subkey, (n, d))
    key, subkey = random.split(key)
    w_true = random.normal(subkey, (d,))
    key, subkey = random.split(key)
    noise = random.normal(subkey, (n,))
    y = X @ w_true + noise_variance * noise
    key, subkey = random.split(key)
    noise2 = random.normal(subkey, (n,))
    ytest = Xtest @ w_true + noise_variance * noise2
    w_star = jnp.linalg.pinv(X.T @ X) @ X.T @ y
    norm_factor = jnp.linalg.norm(w_star)
    w_star = w_star / norm_factor
    y = y / norm_factor
    ytest = ytest / norm_factor
    return (X, y, Xtest, ytest, w_star)


# from https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-using-both-forward-and-reverse-mode
def hessian_vector_product(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]


@partial(jax.jit, static_argnames=["unravel_fn", "loss_fn"])
def param_hessian_vector_product(vec, args, params, unravel_fn, loss_fn):
    delta = unravel_fn(vec)
    f = lambda p: loss_fn(p, args)[0]
    return jax.flatten_util.ravel_pytree(
        hessian_vector_product(f, (params,), (delta,))
    )[0]


@partial(jax.jit, static_argnames=["dim", "n_iter", "unravel_fn", "loss_fn"])
def largest_eigenvalue(args, params, dim, key, n_iter, unravel_fn, loss_fn):
    """Returns the largest eigenvalue of the hessian of the loss, and an estimate of the error."""
    new_eigv = 1.0
    vec = random.normal(key, (dim,))
    vec = vec / jnp.linalg.norm(vec)
    for k in range(n_iter):
        eigv = new_eigv
        new_vec = param_hessian_vector_product(vec, args, params, unravel_fn, loss_fn)
        new_eigv = jnp.linalg.norm(new_vec)
        vec = new_vec / new_eigv
    return eigv, jnp.abs(new_eigv - eigv)


@partial(jax.jit, static_argnames=["loss_fn"])
def update(params, args, step_size, loss_fn):
    (loss, test_loss), grads = value_and_grad(loss_fn, has_aux=True)(params, args)
    return [p - step_size * dp for p, dp in zip(params, grads)], loss, test_loss, grads
