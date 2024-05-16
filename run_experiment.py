import os

import jax.numpy as jnp
import jax.flatten_util
from jax import random
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

from utils import generate_data, largest_eigenvalue, update
from mlp import init_mlp, loss_fn_mlp, square_distance_to_minimizer_mlp
from resnet import init_resnet, loss_fn_resnet, square_distance_to_minimizer_resnet


def run_full_experiment(
    d,
    L,
    dim,
    args_loss_fn,
    args_distance_minimizer_fn,
    step_sizes,
    num_stepss,
    repss,
    scales,
    init_fn,
    loss_fn,
    square_distance_to_minimizer_fn,
    name,
    key,
    num_iter_compute_sharpness=20,
    compute_sharpness_every_step=5,
):
    os.makedirs("logs", exist_ok=True)

    number_configs = len(step_sizes)
    main_keys = random.split(key, number_configs)
    result = {
        "step_size": [],
        "num_steps": [],
        "rep": [],
        "init_scale": [],
        "init_sharpness": [],
        "final_sharpness": [],
        "init_distance": [],
        "final_distance": [],
    }

    idx_config = 0
    for step_size, num_steps, reps in zip(step_sizes, num_stepss, repss):
        print(step_size)
        key = main_keys[idx_config]
        for i in range(reps):
            keys = random.split(key, reps)
            print(i)
            subkeys = random.split(keys[i], len(scales))
            for j, scale in enumerate(scales):
                subsubkeys = random.split(subkeys[j], num_steps + 2)
                params = init_fn(d, L, scale, subsubkeys[-1])
                result["step_size"].append(step_size)
                result["num_steps"].append(num_steps)
                result["rep"].append(i)
                result["init_scale"].append(scale)
                result["init_distance"].append(
                    square_distance_to_minimizer_fn(params, args_distance_minimizer_fn)
                )
                eigv, _ = largest_eigenvalue(
                    args_loss_fn,
                    params,
                    dim,
                    subsubkeys[-2],
                    num_iter_compute_sharpness,
                    unravel_fn,
                    loss_fn,
                )
                result["init_sharpness"].append(eigv)
                for step in range(num_steps):
                    params, _, _ = update(params, args_loss_fn, step_size, loss_fn)
                    if step % compute_sharpness_every_step == 0:
                        eigv, _ = largest_eigenvalue(
                            args_loss_fn,
                            params,
                            dim,
                            subsubkeys[step],
                            num_iter_compute_sharpness,
                            unravel_fn,
                            loss_fn,
                        )
                result["final_sharpness"].append(eigv)
                result["final_distance"].append(
                    square_distance_to_minimizer_fn(params, args_distance_minimizer_fn)
                )
        idx_config += 1

    df = pd.DataFrame(result)
    df["init_scale"] = df["init_scale"].astype(float).round(3)
    df["init_sharpness"] = df["init_sharpness"].astype(float)
    df["final_sharpness"] = df["final_sharpness"].astype(float)
    df["init_distance"] = df["init_distance"].astype(float)
    df["final_distance"] = df["final_distance"].astype(float)
    df.to_pickle("logs/exp_{}_df.pkl".format(name))


def make_plots_full_experiment(
    name, scales, lower_bound_sharpness_minimizers, upper_bound_sharpness_mlp=None
):
    os.makedirs("figures", exist_ok=True)

    df = pd.read_pickle("logs/exp_{}_df.pkl".format(name))

    palette = sns.color_palette("flare", 5)
    xmin = min(scales)
    xmax = max(scales)

    df_without_small_lr = df[df["step_size"] < 0.09]
    df_without_small_lr["step_size"] = "l. rate: " + df_without_small_lr[
        "step_size"
    ].astype(str)

    plt.figure(figsize=(7, 5))
    sns.lineplot(
        df_without_small_lr,
        x="init_scale",
        y="final_sharpness",
        hue="step_size",
        palette=palette,
        style="step_size",
        markers=True,
        dashes=False,
    )
    for k, step_size in enumerate(step_sizes[:-2]):
        plt.hlines(
            [2 / step_size],
            xmin=xmin,
            xmax=xmax,
            colors=palette[k],
            linestyles="dashed",
        )
    plt.hlines(
        [lower_bound_sharpness_minimizers],
        xmin=xmin,
        xmax=xmax,
        colors="black",
        linestyles="dotted",
    )
    if upper_bound_sharpness_mlp:
        plt.hlines(
            [upper_bound_sharpness_mlp],
            xmin=xmin,
            xmax=xmax,
            colors="black",
            linestyles="dotted",
        )
    plt.legend(loc="upper right")
    plt.xlabel("Scale of initialization")
    plt.ylabel("Sharpness")
    plt.savefig("figures/sharpness_{}.png".format(name), bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    sns.lineplot(df, x="init_scale", y="init_sharpness", label="Initialization")
    sns.lineplot(
        df_without_small_lr,
        x="init_scale",
        y="final_sharpness",
        hue="step_size",
        palette=palette,
        style="step_size",
        markers=True,
        dashes=False,
    )
    for k, step_size in enumerate(step_sizes[:-2]):
        plt.hlines(
            [2 / step_size],
            xmin=xmin,
            xmax=xmax,
            colors=palette[k],
            linestyles="dashed",
        )
    plt.hlines(
        [lower_bound_sharpness_minimizers],
        xmin=xmin,
        xmax=xmax,
        colors="black",
        linestyles="dotted",
    )
    if upper_bound_sharpness_mlp:
        plt.hlines(
            [upper_bound_sharpness_mlp],
            xmin=xmin,
            xmax=xmax,
            colors="black",
            linestyles="dotted",
        )
    plt.legend(loc="lower right")
    plt.xlabel("Scale of initialization")
    plt.ylabel("Sharpness")
    plt.yscale("log")
    plt.ylim([0.1, 3 * 10**4])
    plt.savefig("figures/sharpness_{}_log.png".format(name), bbox_inches="tight")

    df["step_size"] = "l. rate: " + df["step_size"].astype(str)

    plt.figure(figsize=(7, 5))
    sns.lineplot(df, x="init_scale", y="init_distance", label="Initialization")
    sns.lineplot(
        df,
        x="init_scale",
        y="final_distance",
        hue="step_size",
        palette=palette,
        style="step_size",
        markers=True,
        dashes=False,
    )
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Scale of initialization")
    plt.ylabel("Squared distance to optimal regressor")
    plt.savefig("figures/squared_distance_{}.png".format(name), bbox_inches="tight")

    df["failed"] = df["final_sharpness"].isna()
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        df,
        x="init_scale",
        y="failed",
        hue="step_size",
        palette=palette,
        style="step_size",
        markers=True,
        dashes=False,
        errorbar=None,
    )
    plt.legend()
    plt.xlabel("Scale of initialization")
    plt.ylabel("Probability of divergence")
    plt.savefig(
        "figures/divergence_probability_{}.png".format(name), bbox_inches="tight"
    )


def run_sub_experiment(
    d,
    L,
    dim,
    args_loss_fn,
    args_distance_minimizer_fn,
    step_sizes,
    scales,
    num_stepss,
    palette_idxs,
    init_fn,
    loss_fn,
    square_distance_to_minimizer_fn,
    name,
    key,
    num_iter_compute_sharpness=20,
    compute_sharpness_every_step=5,
):
    os.makedirs("figures", exist_ok=True)

    palette = sns.color_palette("flare", 5)

    idx_config = 0
    for step_size, scale, num_steps, palette_idx in zip(
        step_sizes, scales, num_stepss, palette_idxs
    ):
        keys = random.split(key, num_steps + 1)
        params = init_fn(d, L, scale, keys[-1])
        square_distances_to_min = []
        sharpness = []
        eigv, _ = largest_eigenvalue(
            args_loss_fn,
            params,
            dim,
            keys[-1],
            num_iter_compute_sharpness,
            unravel_fn,
            loss_fn,
        )
        sharpness.append(eigv)
        distance = square_distance_to_minimizer_fn(params, args_distance_minimizer_fn)
        square_distances_to_min.append(distance)
        for step in range(num_steps):
            params, _, _ = update(params, args_loss_fn, step_size, loss_fn)
            if step % compute_sharpness_every_step == 4:
                eigv, _ = largest_eigenvalue(
                    args_loss_fn,
                    params,
                    dim,
                    keys[step],
                    num_iter_compute_sharpness,
                    unravel_fn,
                    loss_fn,
                )
                sharpness.append(eigv)
            distance = square_distance_to_minimizer_fn(
                params, args_distance_minimizer_fn
            )
            square_distances_to_min.append(distance)

        fig, axs = plt.subplots(1, 2)
        fig.set_figwidth(10)
        xmin = 0
        xmax = num_steps

        axs[0].plot(
            compute_sharpness_every_step
            * jnp.arange(len(square_distances_to_min[::compute_sharpness_every_step])),
            square_distances_to_min[::compute_sharpness_every_step],
            color=palette[palette_idx],
        )
        axs[0].set_yscale("log")
        axs[0].set_xlabel("Training steps")
        axs[0].set_ylabel("Squared distance to optimal regressor")

        axs[1].plot(
            compute_sharpness_every_step * jnp.arange(len(sharpness)),
            sharpness,
            color=palette[palette_idx],
            label="sharpness",
        )
        axs[1].hlines(
            [2 / step_size],
            xmin=xmin,
            xmax=xmax,
            colors=palette[palette_idx],
            label="2/lr",
            linestyles="dashed",
        )
        axs[1].hlines(
            [lower_bound_sharpness_minimizers],
            xmin=xmin,
            xmax=xmax,
            colors="black",
            linestyles="dotted",
        )
        if name == "mlp":
            axs[1].hlines(
                [upper_bound_sharpness_mlp],
                xmin=xmin,
                xmax=xmax,
                colors="black",
                linestyles="dotted",
            )
        axs[1].legend(loc="lower right")
        axs[1].set_yscale("log")
        # To have the same scale as in the main plot.
        axs[1].set_ylim([0.1, 3 * 10**4])
        axs[1].set_xlabel("Training steps")
        axs[1].set_ylabel("Sharpness")

        plt.savefig(
            "figures/evolution_sharp_distance_{}_{}.png".format(str(step_size), name),
            bbox_inches="tight",
        )

        idx_config += 1


if __name__ == "__main__":  # Around one hour in total.
    n = 50
    d = 5
    L = 10

    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    X, y, w_star = generate_data(n, d, subkey)
    lambd = jnp.real(min(jnp.linalg.eigvals(X.T @ X / n)))
    Lambd = jnp.real(min(jnp.linalg.eigvals(X.T @ X / n)))
    lower_bound_sharpness_minimizers = 2 * L * lambd
    upper_bound_sharpness_mlp = 8 * L * Lambd

    num_iter_compute_sharpness = 20
    compute_sharpness_every_step = 5

    # MLP initialization
    key, subkey = random.split(key)
    params = init_mlp(d, L, 0.25, subkey)
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    dim = flat_params.shape[0]
    step_sizes = [0.005, 0.02, 0.07, 0.1, 0.2]
    num_stepss = [40_000, 10_000, 4_000, 4_000, 2_000]
    repss = [20, 20, 20, 20, 80]
    scales = jnp.linspace(0.25, 0.6, 10)
    args_loss_fn = (X, y)
    args_distance_minimizer_fn = (w_star,)
    key, subkey = random.split(key)
    run_full_experiment(
        d,
        L,
        dim,
        args_loss_fn,
        args_distance_minimizer_fn,
        step_sizes,
        num_stepss,
        repss,
        scales,
        init_mlp,
        loss_fn_mlp,
        square_distance_to_minimizer_mlp,
        "mlp",
        subkey,
    )
    make_plots_full_experiment(
        "mlp", scales, lower_bound_sharpness_minimizers, upper_bound_sharpness_mlp
    )

    step_sizes = [0.02, 0.1]
    scales = [0.3, 0.3]
    num_stepss = [400, 400]
    palette_idxs = [1, 3]
    key, subkey = random.split(key)
    run_sub_experiment(
        d,
        L,
        dim,
        args_loss_fn,
        args_distance_minimizer_fn,
        step_sizes,
        scales,
        num_stepss,
        palette_idxs,
        init_mlp,
        loss_fn_mlp,
        square_distance_to_minimizer_mlp,
        "mlp",
        subkey,
    )

    # Resnets initialization
    key, subkey = random.split(key)
    params = init_resnet(d, L, 0.25, subkey)
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    dim = flat_params.shape[0]
    key, subkey = random.split(key)
    w = random.normal(subkey, (d,))
    w = w / jnp.linalg.norm(w)
    step_sizes = [0.005, 0.02, 0.07, 0.1, 0.2]
    num_stepss = [4_000, 4_000, 4_000, 4_000, 4_000]
    repss = [20, 20, 20, 20, 20]
    scales = jnp.linspace(0.0, 1.7, 10)
    args_loss_fn = (X, y, w)
    args_distance_minimizer_fn = (w_star, w)
    key, subkey = random.split(key)
    run_full_experiment(
        d,
        L,
        dim,
        args_loss_fn,
        args_distance_minimizer_fn,
        step_sizes,
        num_stepss,
        repss,
        scales,
        init_resnet,
        loss_fn_resnet,
        square_distance_to_minimizer_resnet,
        "resnet",
        subkey,
    )
    make_plots_full_experiment("resnet", scales, lower_bound_sharpness_minimizers)

    step_sizes = [0.02, 0.1]
    scales = [0.5, 0.5]
    num_stepss = [400, 400]
    palette_idxs = [1, 3]
    key, subkey = random.split(key)
    run_sub_experiment(
        d,
        L,
        dim,
        args_loss_fn,
        args_distance_minimizer_fn,
        step_sizes,
        scales,
        num_stepss,
        palette_idxs,
        init_resnet,
        loss_fn_resnet,
        square_distance_to_minimizer_resnet,
        "resnet",
        subkey,
    )
