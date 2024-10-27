import os
import pickle
import sys

import jax.numpy as jnp
import jax.flatten_util
from jax import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

sns.set_theme(style="darkgrid")

from utils import generate_data, largest_eigenvalue, update
from mlp import (
    init_mlp,
    loss_fn_linear_mlp,
    loss_fn_non_linear_mlp,
    square_distance_to_minimizer_mlp,
)
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
        "final_train_loss": [],
        "final_test_loss": [],
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
                    params, train_loss, test_loss, _ = update(
                        params, args_loss_fn, step_size, loss_fn
                    )
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
                result["final_train_loss"].append(train_loss)
                result["final_test_loss"].append(test_loss)
        idx_config += 1

    df = pd.DataFrame(result)
    df["init_scale"] = df["init_scale"].astype(float).round(3)
    df["init_sharpness"] = df["init_sharpness"].astype(float)
    df["final_sharpness"] = df["final_sharpness"].astype(float)
    df["init_distance"] = df["init_distance"].astype(float)
    df["final_distance"] = df["final_distance"].astype(float)
    df["final_train_loss"] = df["final_train_loss"].astype(float)
    df["final_test_loss"] = df["final_test_loss"].astype(float)
    df.to_pickle("logs/exp_{}_df.pkl".format(name))


def make_plots_full_experiment(
    name,
    scales,
    step_sizes,
    lower_bound_sharpness_minimizers,
    upper_bound_sharpness_mlp=None,
):
    os.makedirs("figures", exist_ok=True)

    df = pd.read_pickle("logs/exp_{}_df.pkl".format(name))

    palette = sns.color_palette("flare", 5)
    xmin = min(scales)
    xmax = max(scales)

    df_without_small_lr = df[df["step_size"] < 0.09]
    step_sizes = [s for s in step_sizes if s < 0.09]
    df_without_small_lr["step_size"] = "l. rate: " + df_without_small_lr[
        "step_size"
    ].astype(str)

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
    for k, step_size in enumerate(step_sizes):
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
    plt.xlim([xmin - 0.01, xmax + 0.01])
    plt.ylim([0.1, 3 * 10**4])
    plt.savefig("figures/sharpness_{}_log.pdf".format(name), bbox_inches="tight")

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
    plt.savefig("figures/squared_distance_{}.pdf".format(name), bbox_inches="tight")

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
        "figures/divergence_probability_{}.pdf".format(name), bbox_inches="tight"
    )


def make_plot_learning_rate(
    name,
    lower_bound_sharpness_minimizers,
    depths,
):
    os.makedirs("figures", exist_ok=True)

    dfs = []
    for L in depths:
        df = pd.read_pickle("logs/exp_{}_{}_df.pkl".format(name, L))
        df["L"] = L
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["step_size"] = full_df["step_size"].astype(float)

    palette = sns.color_palette("flare", 4)

    full_df["L"] = "depth: " + full_df["L"].astype(str)

    plt.figure(figsize=(7, 5))
    ymin = -0.01
    ymax = 0.21
    sns.lineplot(
        full_df,
        x="step_size",
        y="final_distance",
        hue="L",
        palette=palette,
        style="L",
        markers=True,
        dashes=False,
    )
    for k, L in enumerate(depths):
        plt.vlines(
            [2 / (lower_bound_sharpness_minimizers * L)],
            ymin=ymin,
            ymax=ymax,
            colors=palette[k],
            linestyles="dashed",
        )
    plt.legend(loc="upper left")
    plt.xlabel("Learning rate")
    plt.ylabel("Squared distance to optimal regressor")
    plt.xscale("log")
    plt.ylim([ymin, ymax])
    plt.savefig("figures/distance_lr_{}.pdf".format(name), bbox_inches="tight")


def make_plots_underdetermined(
    name,
    scales,
    step_sizes,
    lower_bound_sharpness_minimizers,
    upper_bound_sharpness_mlp=None,
):
    os.makedirs("figures", exist_ok=True)

    df = pd.read_pickle("logs/exp_{}_df.pkl".format(name))

    palette = sns.color_palette("flare", 3)
    xmin = min(scales)
    xmax = max(scales)

    df["step_size"] = "l. rate: " + df["step_size"].astype(str)

    plt.figure(figsize=(7, 5))
    sns.lineplot(df, x="init_scale", y="init_sharpness", label="Initialization")
    sns.lineplot(
        df,
        x="init_scale",
        y="final_sharpness",
        hue="step_size",
        palette=palette,
        style="step_size",
        markers=True,
        dashes=False,
    )
    for k, step_size in enumerate(step_sizes):
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
    plt.legend(loc="lower right")
    plt.xlabel("Scale of initialization")
    plt.ylabel("Sharpness")
    plt.yscale("log")
    plt.ylim([0.1, 3 * 10**4])
    plt.savefig("figures/sharpness_{}_log.pdf".format(name), bbox_inches="tight")

    df["generalization_gap"] = df["final_test_loss"] - df["final_train_loss"]
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    df["log_final_sharpness"] = np.log10(df["final_sharpness"])
    df["log_generalization_gap"] = np.log10(df["generalization_gap"])
    only_one_lr_df = df[df["step_size"] == "l. rate: 0.001"].dropna()
    sns.regplot(
        only_one_lr_df,
        x="log_final_sharpness",
        y="log_generalization_gap",
        order=1,
    )
    plt.xlabel("Sharpness after training")
    plt.ylabel("Generalization gap")
    formatter = lambda x, pos: f"{int(10 ** x):g}"
    ax.get_xaxis().set_major_formatter(formatter)
    formatter = lambda x, pos: f"{np.round(10 ** x, 2):g}"
    ax.get_yaxis().set_major_formatter(formatter)
    plt.savefig("figures/scatter_plot_{}.pdf".format(name), bbox_inches="tight")

    lin_reg_res = stats.linregress(
        np.array(only_one_lr_df["log_final_sharpness"]),
        np.array(only_one_lr_df["log_generalization_gap"]),
        alternative="less",
    )
    print("slope: {}".format(lin_reg_res.slope))
    print("intercept: {}".format(lin_reg_res.intercept))
    print("rvalue squared: {}".format(lin_reg_res.rvalue**2))
    print("pvalue: {}".format(lin_reg_res.pvalue))
    print("stderr: {}".format(lin_reg_res.stderr))
    print("intercept stderr: {}".format(lin_reg_res.intercept_stderr))
    with open("logs/lin_reg_{}.pkl".format(name), "wb") as file:
        pickle.dump(lin_reg_res, file)


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
            params, _, _, _ = update(params, args_loss_fn, step_size, loss_fn)
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
        if name == "linear_mlp":
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
            "figures/evolution_sharp_distance_{}_{}.pdf".format(str(step_size), name),
            bbox_inches="tight",
        )

        idx_config += 1


if __name__ == "__main__":

    if sys.argv[1] == "linear":
        n = 50
        d = 5
        L = 10
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        X, y, Xtest, ytest, w_star = generate_data(n, d, 1.0, subkey)
        cov_matrix = X.T @ X / n
        a = (w_star.T) @ cov_matrix @ w_star
        Lambd = jnp.real(max(jnp.linalg.eigvals(X.T @ X / n)))
        lower_bound_sharpness_minimizers = 2 * L * a
        upper_bound_sharpness_mlp = 8 * L * Lambd

        num_iter_compute_sharpness = 20
        compute_sharpness_every_step = 5

        # MLP linear initialization
        key, subkey = random.split(key)
        params = init_mlp(d, L, 0.25, subkey)
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        dim = flat_params.shape[0]
        step_sizes = [0.005, 0.02, 0.07, 0.1, 0.2]
        num_stepss = [40_000, 10_000, 4_000, 4_000, 2_000]
        repss = [20, 20, 20, 20, 80]
        scales = jnp.linspace(0.25, 0.6, 10)
        args_loss_fn = (X, y, Xtest, ytest)
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
            loss_fn_linear_mlp,
            square_distance_to_minimizer_mlp,
            "linear_mlp",
            subkey,
        )
        make_plots_full_experiment(
            "linear_mlp",
            scales,
            step_sizes,
            lower_bound_sharpness_minimizers,
            upper_bound_sharpness_mlp,
        )

        step_sizes = [0.02, 0.1]
        scales = [0.35, 0.35]
        num_stepss = [450, 450]
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
            loss_fn_linear_mlp,
            square_distance_to_minimizer_mlp,
            "linear_mlp",
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
        make_plots_full_experiment(
            "resnet", scales, step_sizes, lower_bound_sharpness_minimizers
        )

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

        # Resnets initialization - learning rate as a function of depth
        lower_bound_sharpness_minimizers = 2 * a
        depths = [5, 10, 20, 40]
        step_sizes = jnp.logspace(-2, -0.3, num=25)
        num_stepss = [1_000] * len(step_sizes)
        repss = [50] * len(step_sizes)
        scales = [0.25]

        for L in depths:
            print("Start exp with L={}".format(L))
            key, subkey = random.split(key)
            params = init_resnet(d, L, 0.25, subkey)
            flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
            dim = flat_params.shape[0]
            key, subkey = random.split(key)
            w = random.normal(subkey, (d,))
            w = w / jnp.linalg.norm(w)
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
                "resnet_several_depths_{}".format(L),
                subkey,
            )

        make_plot_learning_rate(
            "resnet_several_depths",
            lower_bound_sharpness_minimizers,
            depths,
        )

        # MLP in underdetermined case
        n = 15
        d = 20
        L = 5
        key, subkey = random.split(key)
        X, y, Xtest, ytest, w_star = generate_data(n, d, 1.0, subkey)
        cov_matrix = X.T @ X / n
        a = (w_star.T) @ cov_matrix @ w_star
        Lambd = jnp.real(max(jnp.linalg.eigvals(X.T @ X / n)))
        lower_bound_sharpness_minimizers = 2 * L * a
        upper_bound_sharpness_mlp = 8 * L * Lambd
        key, subkey = random.split(key)
        params = init_mlp(d, L, 0.25, subkey)
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        dim = flat_params.shape[0]
        step_sizes = [0.001, 0.004, 0.01]
        num_stepss = [40_000, 10_000, 4_000]
        repss = [100, 20, 20]
        scales = jnp.linspace(0.1, 0.36, 9)
        args_loss_fn = (X, y, Xtest, ytest)
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
            loss_fn_linear_mlp,
            square_distance_to_minimizer_mlp,
            "mlp_underdetermined",
            subkey,
        )
        make_plots_underdetermined(
            "mlp_underdetermined",
            scales,
            step_sizes,
            lower_bound_sharpness_minimizers,
            upper_bound_sharpness_mlp,
        )

    elif sys.argv[1] == "nonlinear":
        # MLP non-linear initialization
        n = 50
        d = 5
        L = 10
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        X, y, Xtest, ytest, w_star = generate_data(n, d, 0.0, subkey)
        cov_matrix = X.T @ X / n
        a = (w_star.T) @ cov_matrix @ w_star
        Lambd = jnp.real(max(jnp.linalg.eigvals(X.T @ X / n)))
        lower_bound_sharpness_minimizers = 2 * L * a
        upper_bound_sharpness_mlp = 8 * L * Lambd
        key, subkey = random.split(key)
        params = init_mlp(d, L, 0.25, subkey)
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        dim = flat_params.shape[0]
        step_sizes = [0.005, 0.02, 0.07]
        num_stepss = [160_000, 40_000, 16_000]
        repss = [20, 20, 20]
        scales = jnp.linspace(0.428, 0.7, 8)
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
            loss_fn_non_linear_mlp,
            square_distance_to_minimizer_mlp,
            "non_linear",
            subkey,
        )
        make_plots_full_experiment(
            "non_linear",
            scales,
            step_sizes,
            lower_bound_sharpness_minimizers,
            upper_bound_sharpness_mlp,
        )
