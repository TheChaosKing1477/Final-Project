import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hedge_core import hedge_update


# Figure Directory & Utilities
#
# These functions create output folders and save figures.
# The plotting backend is set to Agg so that figures are written directly
# to disk when the code is executed on the cluster.


def ensure_directory(path):
    if path is None or path == "":
        return

    os.makedirs(path, exist_ok=True)


def save_current_figure(path, dpi=200):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# Plot toggles
#
# Every plot described in the computational section is enabled by default.
# A user can manually disable any plot by setting the corresponding toggle
# to False before running the batch save routines.


plot_toggles = {
    "stationary_weight_trajectories": True,
    "stationary_entropy_trajectories": True,
    "stationary_average_loss_trajectories": True,
    "stationary_continuous_time_error_trajectories": True,
    "stationary_simplex_trajectories": True,
    "stationary_phase_portraits": True,
    "stationary_spectral_radius_summaries": True,
    "stationary_jacobian_difference_summaries": True,
    "stationary_eigenvalue_discrepancy_summaries": True,
    "stationary_linearization_error_summaries": True,
    "stationary_final_entropy_summaries": True,
    "stationary_final_average_loss_summaries": True,
    "stationary_mean_continuous_time_error_summaries": True,
    "periodic_weight_trajectories": True,
    "periodic_entropy_trajectories": True,
    "periodic_average_loss_trajectories": True,
    "periodic_continuous_time_error_trajectories": True,
    "periodic_margin_trajectories": True,
    "periodic_step_size_trajectories": True,
    "periodic_simplex_trajectories": True,
    "periodic_tracking_accuracy_summaries": True,
    "periodic_step_size_summaries": True,
    "periodic_continuous_time_error_summaries": True,
    "periodic_final_entropy_summaries": True,
    "periodic_final_average_loss_summaries": True,
}


# Simplex geometry for three-expert plots
#
# A point w = (w_1, w_2, w_3) in the simplex is mapped to the plane by
# barycentric coordinates using the triangle vertices
# v_1 = (0, 0), v_2 = (1, 0), v_3 = (1 / 2, sqrt(3) / 2).


def simplex_vertices():
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
        ],
        dtype=float,
    )


def simplex_to_cartesian(weights):
    weights = np.asarray(weights, dtype=float)
    vertices = simplex_vertices()
    return weights @ vertices


def draw_simplex_boundary(ax):
    vertices = simplex_vertices()
    closed = np.vstack([vertices, vertices[0]])
    ax.plot(closed[:, 0], closed[:, 1], linewidth=1.5)

    ax.text(vertices[0, 0] - 0.03, vertices[0, 1] - 0.025, r"$e_1$", fontsize=13, clip_on=False)
    ax.text(vertices[1, 0] + 0.01, vertices[1, 1] - 0.025, r"$e_2$", fontsize=13, clip_on=False)
    ax.text(vertices[2, 0], vertices[2, 1] + 0.03, r"$e_3$", fontsize=13, ha="center", clip_on=False)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.045, np.sqrt(3.0) / 2.0 + 0.06)
    ax.set_aspect("equal")

    # Keep visible Cartesian axes for the simplex embedding so the plots are
    # easier to interpret when viewed outside the notebook.
    ax.set_xlabel("Simplex x-coordinate")
    ax.set_ylabel("Simplex y-coordinate")
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, np.sqrt(3.0) / 2.0, 5))
    ax.grid(True, alpha=0.18)


def plot_weight_trajectories(simulation, title=None, save_path=None):
    weights = simulation["weights"]
    times = np.arange(simulation["horizon"] + 1)

    plt.figure(figsize=(8, 4.5))
    for i in range(weights.shape[1]):
        plt.plot(times, weights[:, i], label=fr"$w_{{t,{i+1}}}$")

    plt.xlabel("t")
    plt.ylabel("Weight")
    if title is not None:
        plt.title(title)
    plt.legend()

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_entropy_trajectory(simulation, title=None, save_path=None):
    entropies = simulation["entropies"]
    times = np.arange(simulation["horizon"] + 1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, entropies)
    plt.xlabel("t")
    plt.ylabel("Entropy")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_average_loss_trajectory(simulation, title=None, save_path=None):
    average_losses = simulation["average_losses"]
    times = np.arange(simulation["horizon"])

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, average_losses)
    plt.xlabel("t")
    plt.ylabel(r"$\langle w_t,\ell_t\rangle$")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_continuous_time_error_trajectory(simulation, title=None, save_path=None):
    errors = simulation["continuous_time_errors"]
    times = np.arange(simulation["horizon"])

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, errors)
    plt.xlabel("t")
    plt.ylabel("Continuous-time comparison error")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


# Periodic tracking diagnostics in time
#
# The preferred-weight margin is
# preferred_weight_t - opposite_weight_t,
#
# where preferred_weight_t is the weight assigned to the currently lower-loss
# expert among the first two experts, and opposite_weight_t is the weight on
# the other alternating expert.
#
# The step size is
# ||w_{t+1} - w_t||,
#
# which measures how sharply the weight vector changes from one round to the next.


def periodic_margin_sequence(simulation):
    weights = simulation["weights"][:-1]
    losses = simulation["losses"]

    better_expert = np.argmin(losses[:, :2], axis=1)
    preferred_weight = np.zeros(losses.shape[0], dtype=float)
    opposite_weight = np.zeros(losses.shape[0], dtype=float)

    for t in range(losses.shape[0]):
        preferred_weight[t] = weights[t, better_expert[t]]
        opposite_weight[t] = weights[t, 1 - better_expert[t]]

    return preferred_weight - opposite_weight


def step_size_sequence(simulation):
    weights = simulation["weights"]
    return np.linalg.norm(weights[1:] - weights[:-1], axis=1)


def plot_periodic_margin(simulation, title=None, save_path=None):
    margin = periodic_margin_sequence(simulation)
    times = np.arange(simulation["horizon"])

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, margin)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("t")
    plt.ylabel("Preferred-weight margin")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_step_sizes(simulation, title=None, save_path=None):
    steps = step_size_sequence(simulation)
    times = np.arange(simulation["horizon"])

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, steps)
    plt.xlabel("t")
    plt.ylabel(r"$\|w_{t+1} - w_t\|$")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


# Simplex Trajectory Plots
#
# These are used in the three-expert cases to show how trajectories move
# across the simplex.


def plot_simplex_trajectory(simulation, title=None, save_path=None):
    weights = simulation["weights"]

    if weights.shape[1] != 3:
        raise ValueError("Simplex trajectory plots require exactly three experts.")

    xy = simplex_to_cartesian(weights)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    draw_simplex_boundary(ax)
    ax.plot(xy[:, 0], xy[:, 1], marker="o", markersize=2)
    ax.scatter(xy[0, 0], xy[0, 1], s=40)
    ax.scatter(xy[-1, 0], xy[-1, 1], s=40)

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


# Phase Portraits on the Simplex
#
# For a grid of simplex points, I compute the one-step arrow field
# F(w) - w
#
# under stationary losses. This provides a discrete-time phase portrait in the
# three-expert cases.


def simplex_grid_points(grid_size=15, tol=1.0e-12):
    points = []

    for i in range(grid_size + 1):
        for j in range(grid_size + 1 - i):
            k = grid_size - i - j
            w = np.array([i, j, k], dtype=float) / grid_size
            if np.all(w >= -tol):
                points.append(w)

    return np.array(points, dtype=float)


def plot_stationary_phase_portrait(ell, eta, grid_size=15, title=None, save_path=None):
    ell = np.asarray(ell, dtype=float)

    if ell.size != 3:
        raise ValueError("Phase portraits are implemented for three-expert cases only.")

    points = simplex_grid_points(grid_size=grid_size)
    images = np.array([hedge_update(w, ell, eta) for w in points])
    xy = simplex_to_cartesian(points)
    xy_next = simplex_to_cartesian(images)
    arrows = xy_next - xy

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    draw_simplex_boundary(ax)
    ax.quiver(
        xy[:, 0],
        xy[:, 1],
        arrows[:, 0],
        arrows[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
    )

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


# Summary plots as a function of eta
#
# These plots show how key diagnostics change as the learning rate eta varies.
# They complement the time-dependent trajectory plots by displaying one summary
# value per eta.


def plot_stationary_spectral_radius_summary(case_runs, title=None, save_path=None):
    etas = []
    radii = []

    for run in case_runs:
        for item in run["jacobian_data"]:
            etas.append(run["eta"])
            radii.append(item["spectral_radius"])

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, radii, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Spectral radius")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_jacobian_difference_summary(case_runs, title=None, save_path=None):
    etas = []
    values = []

    for run in case_runs:
        for item in run["jacobian_data"]:
            etas.append(run["eta"])
            values.append(item["comparison"]["difference_norm"])

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Jacobian difference norm")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_eigenvalue_discrepancy_summary(case_runs, title=None, save_path=None):
    etas = []
    values = []

    for run in case_runs:
        for item in run["jacobian_data"]:
            etas.append(run["eta"])
            values.append(item["eigen_comparison"]["max_difference"])

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Maximum eigenvalue discrepancy")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_linearization_error_summary(case_runs, title=None, save_path=None):
    etas = []
    values = []

    for run in case_runs:
        if len(run["linearization_data"]) > 0:
            mean_error = np.mean([item["error"] for item in run["linearization_data"]])
            etas.append(run["eta"])
            values.append(mean_error)

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Mean one-step linearization error")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_final_entropy_summary(case_runs, title=None, save_path=None):
    etas = [run["eta"] for run in case_runs]
    values = [run["simulation"]["final_entropy"] for run in case_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Final entropy")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_final_average_loss_summary(case_runs, title=None, save_path=None):
    etas = [run["eta"] for run in case_runs]
    values = [run["simulation"]["final_average_loss"] for run in case_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Final weighted average loss")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_mean_continuous_time_error_summary(case_runs, title=None, save_path=None):
    etas = [run["eta"] for run in case_runs]
    values = [run["simulation"]["continuous_time_error_summary"]["mean"] for run in case_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Mean continuous-time comparison error")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_periodic_tracking_summary(periodic_runs, title=None, save_path=None):
    etas = [run["eta"] for run in periodic_runs]
    accuracies = [run["diagnostic"]["tracking_accuracy"] for run in periodic_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, accuracies, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Tracking accuracy")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_periodic_step_summary(periodic_runs, title=None, save_path=None):
    etas = [run["eta"] for run in periodic_runs]
    step_sizes = [run["diagnostic"]["average_step"] for run in periodic_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, step_sizes, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Average step size")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_periodic_continuous_time_error_summary(periodic_runs, title=None, save_path=None):
    etas = [run["eta"] for run in periodic_runs]
    errors = [run["simulation"]["continuous_time_error_summary"]["mean"] for run in periodic_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, errors, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Mean continuous-time comparison error")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_periodic_final_entropy_summary(periodic_runs, title=None, save_path=None):
    etas = [run["eta"] for run in periodic_runs]
    values = [run["simulation"]["final_entropy"] for run in periodic_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Final entropy")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_periodic_final_average_loss_summary(periodic_runs, title=None, save_path=None):
    etas = [run["eta"] for run in periodic_runs]
    values = [run["simulation"]["final_average_loss"] for run in periodic_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Final weighted average loss")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


# Batch figure generation
#
# These functions save all figures required by the computational section:
# - time-dependent trajectories for each eta,
# - simplex trajectory plots in the three-expert cases,
# - phase portraits in the stationary three-expert cases,
# - eta-summary diagnostics for stationary and periodic simulations.


def save_stationary_case_figures(case_name, case_runs, output_dir):
    ensure_directory(output_dir)

    for run in case_runs:
        eta = run["eta"]
        simulation = run["simulation"]

        eta_tag = f"eta_{eta:.2f}".replace(".", "p")
        base = os.path.join(output_dir, f"{case_name}_{eta_tag}")

        # Figure: Stationary weight trajectories
        # This plot shows how the expert weights evolve over time for one eta.
        if plot_toggles["stationary_weight_trajectories"]:
            plot_weight_trajectories(
                simulation,
                title=f"{case_name}: weight trajectories, eta = {eta:.2f}",
                save_path=base + "_weights.png",
            )

        # Figure: Stationary entropy trajectory
        # This plot shows how concentration develops in time through H(w_t).
        if plot_toggles["stationary_entropy_trajectories"]:
            plot_entropy_trajectory(
                simulation,
                title=f"{case_name}: entropy, eta = {eta:.2f}",
                save_path=base + "_entropy.png",
            )

        # Figure: Stationary weighted average loss trajectory
        # This plot shows how <w_t, ell> evolves across the simulation.
        if plot_toggles["stationary_average_loss_trajectories"]:
            plot_average_loss_trajectory(
                simulation,
                title=f"{case_name}: weighted average loss, eta = {eta:.2f}",
                save_path=base + "_average_loss.png",
            )

        # Figure: Stationary continuous-time comparison error
        # This plot shows how closely the discrete dynamics follow the small-eta continuous-time mechanism.
        if plot_toggles["stationary_continuous_time_error_trajectories"]:
            plot_continuous_time_error_trajectory(
                simulation,
                title=f"{case_name}: continuous-time comparison error, eta = {eta:.2f}",
                save_path=base + "_continuous_time_error.png",
            )

        # Figure: Stationary simplex trajectory
        # This plot shows the path of the three-expert dynamics inside the simplex.
        if plot_toggles["stationary_simplex_trajectories"] and simulation["weights"].shape[1] == 3:
            plot_simplex_trajectory(
                simulation,
                title=f"{case_name}: simplex trajectory, eta = {eta:.2f}",
                save_path=base + "_simplex.png",
            )

        # Figure: Stationary phase portrait
        # This plot shows the discrete vector field w -> F(w) - w on the simplex.
        if plot_toggles["stationary_phase_portraits"] and simulation["weights"].shape[1] == 3:
            plot_stationary_phase_portrait(
                run["case_data"]["loss"],
                eta,
                title=f"{case_name}: phase portrait, eta = {eta:.2f}",
                save_path=base + "_phase_portrait.png",
            )

    # Figure: Stationary spectral radius summary
    # This plot shows how the spectral radius changes as eta varies.
    if plot_toggles["stationary_spectral_radius_summaries"]:
        plot_stationary_spectral_radius_summary(
            case_runs,
            title=f"{case_name}: spectral radius vs eta",
            save_path=os.path.join(output_dir, f"{case_name}_spectral_radius_summary.png"),
        )

    # Figure: Stationary Jacobian difference summary
    # This plot shows how closely the numerical Jacobian matches the analytical Jacobian as eta varies.
    if plot_toggles["stationary_jacobian_difference_summaries"]:
        plot_stationary_jacobian_difference_summary(
            case_runs,
            title=f"{case_name}: Jacobian difference norm vs eta",
            save_path=os.path.join(output_dir, f"{case_name}_jacobian_difference_summary.png"),
        )

    # Figure: Stationary eigenvalue discrepancy summary
    # This plot shows how closely the numerical eigenvalues match the predicted eigenvalues as eta varies.
    if plot_toggles["stationary_eigenvalue_discrepancy_summaries"]:
        plot_stationary_eigenvalue_discrepancy_summary(
            case_runs,
            title=f"{case_name}: eigenvalue discrepancy vs eta",
            save_path=os.path.join(output_dir, f"{case_name}_eigenvalue_discrepancy_summary.png"),
        )

    # Figure: Stationary linearization error summary
    # This plot shows how accurately the linearized one-step prediction matches the nonlinear update as eta varies.
    if plot_toggles["stationary_linearization_error_summaries"]:
        plot_stationary_linearization_error_summary(
            case_runs,
            title=f"{case_name}: linearization error vs eta",
            save_path=os.path.join(output_dir, f"{case_name}_linearization_error_summary.png"),
        )

    # Figure: Stationary final entropy summary
    # This plot shows how the final concentration level depends on eta.
    if plot_toggles["stationary_final_entropy_summaries"]:
        plot_stationary_final_entropy_summary(
            case_runs,
            title=f"{case_name}: final entropy vs eta",
            save_path=os.path.join(output_dir, f"{case_name}_final_entropy_summary.png"),
        )

    # Figure: Stationary final weighted average loss summary
    # This plot shows how the terminal weighted average loss depends on eta.
    if plot_toggles["stationary_final_average_loss_summaries"]:
        plot_stationary_final_average_loss_summary(
            case_runs,
            title=f"{case_name}: final weighted average loss vs eta",
            save_path=os.path.join(output_dir, f"{case_name}_final_average_loss_summary.png"),
        )

    # Figure: Stationary mean continuous-time comparison error summary
    # This plot shows how the average discrete-versus-continuous discrepancy depends on eta.
    if plot_toggles["stationary_mean_continuous_time_error_summaries"]:
        plot_stationary_mean_continuous_time_error_summary(
            case_runs,
            title=f"{case_name}: mean continuous-time comparison error vs eta",
            save_path=os.path.join(output_dir, f"{case_name}_mean_continuous_time_error_summary.png"),
        )


def save_periodic_figures(periodic_runs, output_dir):
    ensure_directory(output_dir)

    for run in periodic_runs:
        eta = run["eta"]
        simulation = run["simulation"]

        eta_tag = f"eta_{eta:.2f}".replace(".", "p")
        base = os.path.join(output_dir, f"periodic_{eta_tag}")

        # Figure: Periodic weight trajectories
        # This plot shows how the expert weights respond over time to the periodic loss sequence.
        if plot_toggles["periodic_weight_trajectories"]:
            plot_weight_trajectories(
                simulation,
                title=f"Periodic case: weight trajectories, eta = {eta:.2f}",
                save_path=base + "_weights.png",
            )

        # Figure: Periodic entropy trajectory
        # This plot shows whether the periodic dynamics remain spread out or become concentrated.
        if plot_toggles["periodic_entropy_trajectories"]:
            plot_entropy_trajectory(
                simulation,
                title=f"Periodic case: entropy, eta = {eta:.2f}",
                save_path=base + "_entropy.png",
            )

        # Figure: Periodic weighted average loss trajectory
        # This plot shows how the current weighted loss evolves under time-varying losses.
        if plot_toggles["periodic_average_loss_trajectories"]:
            plot_average_loss_trajectory(
                simulation,
                title=f"Periodic case: weighted average loss, eta = {eta:.2f}",
                save_path=base + "_average_loss.png",
            )

        # Figure: Periodic continuous-time comparison error
        # This plot shows how closely the discrete periodic dynamics follow the continuous-time mechanism.
        if plot_toggles["periodic_continuous_time_error_trajectories"]:
            plot_continuous_time_error_trajectory(
                simulation,
                title=f"Periodic case: continuous-time comparison error, eta = {eta:.2f}",
                save_path=base + "_continuous_time_error.png",
            )

        # Figure: Periodic preferred-weight margin trajectory
        # This plot shows how strongly the algorithm favors the currently better expert over time.
        if plot_toggles["periodic_margin_trajectories"]:
            plot_periodic_margin(
                simulation,
                title=f"Periodic case: preferred-weight margin, eta = {eta:.2f}",
                save_path=base + "_margin.png",
            )

        # Figure: Periodic step-size trajectory
        # This plot shows how sharply the weight vector changes from one round to the next.
        if plot_toggles["periodic_step_size_trajectories"]:
            plot_step_sizes(
                simulation,
                title=f"Periodic case: step sizes, eta = {eta:.2f}",
                save_path=base + "_step_sizes.png",
            )

        # Figure: Periodic simplex trajectory
        # This plot shows the path of the three-expert periodic dynamics inside the simplex.
        if plot_toggles["periodic_simplex_trajectories"] and simulation["weights"].shape[1] == 3:
            plot_simplex_trajectory(
                simulation,
                title=f"Periodic case: simplex trajectory, eta = {eta:.2f}",
                save_path=base + "_simplex.png",
            )

    # Figure: Periodic tracking-accuracy summary
    # This plot shows how often the algorithm aligns with the currently lower-loss expert as eta varies.
    if plot_toggles["periodic_tracking_accuracy_summaries"]:
        plot_periodic_tracking_summary(
            periodic_runs,
            title="Periodic case: tracking accuracy vs eta",
            save_path=os.path.join(output_dir, "periodic_tracking_accuracy_summary.png"),
        )

    # Figure: Periodic average-step-size summary
    # This plot shows how the typical sharpness of the updates changes with eta.
    if plot_toggles["periodic_step_size_summaries"]:
        plot_periodic_step_summary(
            periodic_runs,
            title="Periodic case: average step size vs eta",
            save_path=os.path.join(output_dir, "periodic_step_size_summary.png"),
        )

    # Figure: Periodic mean continuous-time comparison error summary
    # This plot shows how the average discrete-versus-continuous discrepancy changes with eta.
    if plot_toggles["periodic_continuous_time_error_summaries"]:
        plot_periodic_continuous_time_error_summary(
            periodic_runs,
            title="Periodic case: continuous-time comparison error vs eta",
            save_path=os.path.join(output_dir, "periodic_continuous_time_error_summary.png"),
        )

    # Figure: Periodic final entropy summary
    # This plot shows how the final concentration level depends on eta in the periodic case.
    if plot_toggles["periodic_final_entropy_summaries"]:
        plot_periodic_final_entropy_summary(
            periodic_runs,
            title="Periodic case: final entropy vs eta",
            save_path=os.path.join(output_dir, "periodic_final_entropy_summary.png"),
        )

    # Figure: Periodic final weighted average loss summary
    # This plot shows how the terminal weighted average loss depends on eta in the periodic case.
    if plot_toggles["periodic_final_average_loss_summaries"]:
        plot_periodic_final_average_loss_summary(
            periodic_runs,
            title="Periodic case: final weighted average loss vs eta",
            save_path=os.path.join(output_dir, "periodic_final_average_loss_summary.png"),
        )


def save_all_figures(all_results, output_dir="figures"):
    ensure_directory(output_dir)

    stationary_dir = os.path.join(output_dir, "stationary")
    periodic_dir = os.path.join(output_dir, "periodic")

    ensure_directory(stationary_dir)
    ensure_directory(periodic_dir)

    for case_name, case_runs in all_results["stationary"].items():
        case_dir = os.path.join(stationary_dir, case_name)
        save_stationary_case_figures(case_name, case_runs, case_dir)

    save_periodic_figures(all_results["periodic"], periodic_dir)


def _label_times(horizon):
    raw = [0, 1, 2, 5, 10, 20, horizon]
    return sorted(set(t for t in raw if 0 <= t <= horizon))


def _vector_label(vector, digits=2):
    vector = np.asarray(vector, dtype=float)
    return "[" + ", ".join(f"{x:.{digits}f}" for x in vector) + "]"


def _fixed_point_label(point):
    point = np.asarray(point, dtype=float)
    return r"$w^*=$" + _vector_label(point, digits=2)


def _eta_title(eta):
    return rf"$\eta = {eta:.2f}$"


def _panel_axes(n_panels, base_width=5.0, base_height=3.6, max_cols=3):
    ncols = min(max_cols, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(base_width * ncols, base_height * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, axes[:n_panels]


def _configure_time_axis(ax, ylabel=None, ylim=None):
    ax.set_xlabel("t")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)


def stationary_preference_plot_data(run):
    preference = run.get("stationary_preference")
    if preference is None:
        return None

    return {
        "tracking_sequence": np.asarray(preference["tracking_sequence"], dtype=float),
        "margin_sequence": np.asarray(preference["margin_sequence"], dtype=float),
        "tracking_label": preference.get("tracking_label", "Tracking indicator"),
        "margin_label": preference.get("margin_label", "Preferred margin"),
        "mean_tracking_accuracy": float(preference.get("mean_tracking_accuracy", np.mean(preference["tracking_sequence"]))),
        "mean_margin": float(preference.get("mean_margin", np.mean(preference["margin_sequence"]))),
        "mode": preference.get("mode", ""),
    }


def plot_time_series(values, ylabel, title=None, save_path=None, ylim=None):
    values = np.asarray(values, dtype=float)
    times = np.arange(values.size)
    plt.figure(figsize=(8, 4.5))
    plt.plot(times, values)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_scalar_eta_panel(runs, sequence_getter, ylabel, panel_title, save_path=None, ylim=None):
    fig, axes = _panel_axes(len(runs))
    for ax, run in zip(axes, runs):
        values = np.asarray(sequence_getter(run), dtype=float)
        times = np.arange(values.size)
        ax.plot(times, values)
        ax.set_title(_eta_title(run["eta"]))
        _configure_time_axis(ax, ylabel=ylabel, ylim=ylim)
    fig.suptitle(panel_title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_weight_eta_panel(runs, panel_title, save_path=None):
    fig, axes = _panel_axes(len(runs), base_width=5.2, base_height=3.8)
    for ax, run in zip(axes, runs):
        simulation = run["simulation"]
        weights = simulation["weights"]
        times = np.arange(simulation["horizon"] + 1)
        for i in range(weights.shape[1]):
            ax.plot(times, weights[:, i], label=fr"$w_{{t,{i+1}}}$")
        ax.set_title(_eta_title(run["eta"]))
        _configure_time_axis(ax, ylabel="Weight", ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=8)
    fig.suptitle(panel_title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_simplex_trajectory(simulation, title=None, save_path=None, special_points=None, special_point_labels=None):
    weights = simulation["weights"]

    if weights.shape[1] != 3:
        raise ValueError("Simplex trajectory plots require exactly three experts.")

    xy = simplex_to_cartesian(weights)
    horizon = simulation["horizon"]
    constant_path = np.max(np.linalg.norm(weights - weights[0], axis=1)) < 1.0e-10

    plt.figure(figsize=(6.5, 6.5))
    ax = plt.gca()
    draw_simplex_boundary(ax)

    ax.plot(xy[:, 0], xy[:, 1], marker="o", markersize=2.5, linewidth=1.3, label="trajectory")

    if constant_path:
        ax.scatter(xy[0, 0], xy[0, 1], s=60, zorder=5, label=r"$w_0 = w_T$")
        ax.annotate(r"$w_0 = w_T$\n" + _vector_label(weights[0], 2), (xy[0, 0], xy[0, 1]), textcoords="offset points", xytext=(8, 8), fontsize=8)
    else:
        ax.scatter(xy[0, 0], xy[0, 1], s=60, zorder=5, label=r"start $w_0$")
        ax.scatter(xy[-1, 0], xy[-1, 1], s=60, zorder=5, label=rf"end $w_{{{horizon}}}$")
        ax.annotate(r"$w_0$\n" + _vector_label(weights[0], 2), (xy[0, 0], xy[0, 1]), textcoords="offset points", xytext=(-5, -18), ha="left", fontsize=8)
        ax.annotate(rf"$w_{{{horizon}}}$\n" + _vector_label(weights[-1], 2), (xy[-1, 0], xy[-1, 1]), textcoords="offset points", xytext=(8, 8), ha="left", fontsize=8)

        for t in _label_times(horizon):
            if t in (0, horizon):
                continue
            x_t, y_t = xy[t]
            ax.scatter(x_t, y_t, s=18, zorder=4)
            ax.annotate(f"t={t}", (x_t, y_t), textcoords="offset points", xytext=(4, 4), fontsize=7)

    if special_points is not None:
        if special_point_labels is None:
            special_point_labels = [f"point {k+1}" for k in range(len(special_points))]
        for point, label in zip(special_points, special_point_labels):
            point = np.asarray(point, dtype=float)
            point_xy = simplex_to_cartesian(point[None, :])[0]
            ax.scatter(point_xy[0], point_xy[1], marker="*", s=90, zorder=6)
            ax.annotate(label, (point_xy[0], point_xy[1]), textcoords="offset points", xytext=(6, -12), fontsize=8)

    ax.legend(loc="best", fontsize=8)

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_simplex_eta_panel(runs, panel_title, save_path=None, special_points_getter=None):
    fig, axes = _panel_axes(len(runs), base_width=4.5, base_height=4.3)
    for ax, run in zip(axes, runs):
        simulation = run["simulation"]
        weights = simulation["weights"]
        xy = simplex_to_cartesian(weights)
        draw_simplex_boundary(ax)
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.1, marker="o", markersize=1.8)
        ax.scatter(xy[0, 0], xy[0, 1], s=28, zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=28, zorder=5)
        ax.annotate(r"$w_0$", (xy[0, 0], xy[0, 1]), textcoords="offset points", xytext=(-3, -10), fontsize=7)
        ax.annotate(r"$w_T$", (xy[-1, 0], xy[-1, 1]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        if special_points_getter is not None:
            points, labels = special_points_getter(run)
            for point, label in zip(points, labels):
                pxy = simplex_to_cartesian(np.asarray(point, dtype=float)[None, :])[0]
                ax.scatter(pxy[0], pxy[1], marker="*", s=55, zorder=6)
                ax.annotate(label, (pxy[0], pxy[1]), textcoords="offset points", xytext=(4, -10), fontsize=6)
        ax.set_title(_eta_title(run["eta"]))
    fig.suptitle(panel_title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_phase_eta_panel(case_runs, panel_title, save_path=None):
    fig, axes = _panel_axes(len(case_runs), base_width=4.6, base_height=4.2)
    for ax, run in zip(axes, case_runs):
        ell = np.asarray(run["case_data"]["loss"], dtype=float)
        eta = run["eta"]
        points = simplex_grid_points(grid_size=13)
        images = np.array([hedge_update(w, ell, eta) for w in points])
        xy = simplex_to_cartesian(points)
        xy_next = simplex_to_cartesian(images)
        arrows = xy_next - xy
        draw_simplex_boundary(ax)
        ax.quiver(
            xy[:, 0],
            xy[:, 1],
            arrows[:, 0],
            arrows[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
        )
        ax.set_title(_eta_title(eta))
    fig.suptitle(panel_title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_preference_eta_panel(case_runs, quantity, panel_title, save_path=None):
    filtered_runs = [run for run in case_runs if stationary_preference_plot_data(run) is not None]
    if not filtered_runs:
        return

    if quantity == "margin":
        getter = lambda run: stationary_preference_plot_data(run)["margin_sequence"]
        ylabel = "Preferred margin"
        ylim = None
    else:
        getter = lambda run: stationary_preference_plot_data(run)["tracking_sequence"]
        ylabel = "Tracking indicator"
        ylim = (-0.02, 1.02)

    plot_scalar_eta_panel(filtered_runs, getter, ylabel, panel_title, save_path=save_path, ylim=ylim)


def plot_stationary_spectral_radius_panel(case_runs, panel_title, save_path=None):
    num_panels = len(case_runs[0]["jacobian_data"])
    fig, axes = _panel_axes(num_panels, base_width=5.2, base_height=3.8)
    fixed_points = [item["fixed_point"] for item in case_runs[0]["jacobian_data"]]
    etas = [run["eta"] for run in case_runs]
    for ax, idx in zip(axes, range(num_panels)):
        values = [run["jacobian_data"][idx]["spectral_radius"] for run in case_runs]
        ax.plot(etas, values, marker="o")
        ax.axhline(1.0, linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel("Spectral radius")
        ax.set_title(_fixed_point_label(fixed_points[idx]), fontsize=9)
        ax.grid(True, alpha=0.25)
    fig.suptitle(panel_title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_average_step_summary(case_runs, title=None, save_path=None):
    etas = [run["eta"] for run in case_runs]
    values = []
    for run in case_runs:
        weights = np.asarray(run["simulation"]["weights"], dtype=float)
        steps = np.linalg.norm(weights[1:] - weights[:-1], axis=1)
        values.append(float(np.mean(steps)) if steps.size else 0.0)
    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Average step size")
    plt.grid(True, alpha=0.25)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_tracking_accuracy_summary(case_runs, title=None, save_path=None):
    filtered = [(run["eta"], stationary_preference_plot_data(run)) for run in case_runs]
    filtered = [(eta, data) for eta, data in filtered if data is not None]
    if not filtered:
        return
    etas = [eta for eta, _ in filtered]
    values = [data["mean_tracking_accuracy"] for _, data in filtered]
    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Tracking accuracy")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_stationary_margin_summary(case_runs, title=None, save_path=None):
    filtered = [(run["eta"], stationary_preference_plot_data(run)) for run in case_runs]
    filtered = [(eta, data) for eta, data in filtered if data is not None]
    if not filtered:
        return
    etas = [eta for eta, _ in filtered]
    values = [data["mean_margin"] for _, data in filtered]
    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, values, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Mean preferred margin")
    plt.grid(True, alpha=0.25)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def plot_periodic_soft_tracking_summary(periodic_runs, title=None, save_path=None):
    etas = [run["eta"] for run in periodic_runs]
    scores = [run["diagnostic"].get("soft_tracking_accuracy", run["diagnostic"]["tracking_accuracy"]) for run in periodic_runs]
    plt.figure(figsize=(8, 4.5))
    plt.plot(etas, scores, marker="o")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Mean preferred-expert share")
    plt.ylim(0.48, 1.02)
    plt.grid(True, alpha=0.25)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        save_current_figure(save_path)
    else:
        plt.show()


def save_stationary_case_figures(case_name, case_runs, output_dir):
    ensure_directory(output_dir)

    def stationary_special_points(run):
        fixed_points = run["case_data"]["fixed_points"]
        labels = [_fixed_point_label(point) for point in fixed_points]
        return fixed_points, labels

    for run in case_runs:
        eta = run["eta"]
        simulation = run["simulation"]
        preference_data = stationary_preference_plot_data(run)

        eta_tag = f"eta_{eta:.2f}".replace(".", "p")
        base = os.path.join(output_dir, f"{case_name}_{eta_tag}")

        if plot_toggles["stationary_weight_trajectories"]:
            plot_weight_trajectories(simulation, title=f"{case_name}: weight trajectories, eta = {eta:.2f}", save_path=base + "_weights.png")

        if plot_toggles["stationary_entropy_trajectories"]:
            plot_entropy_trajectory(simulation, title=f"{case_name}: entropy, eta = {eta:.2f}", save_path=base + "_entropy.png")

        if plot_toggles["stationary_average_loss_trajectories"]:
            plot_average_loss_trajectory(simulation, title=f"{case_name}: weighted average loss, eta = {eta:.2f}", save_path=base + "_average_loss.png")

        if plot_toggles["stationary_continuous_time_error_trajectories"]:
            plot_continuous_time_error_trajectory(simulation, title=f"{case_name}: continuous-time comparison error, eta = {eta:.2f}", save_path=base + "_continuous_time_error.png")

        if plot_toggles["stationary_simplex_trajectories"] and simulation["weights"].shape[1] == 3:
            points, labels = stationary_special_points(run)
            plot_simplex_trajectory(simulation, title=f"{case_name}: simplex trajectory, eta = {eta:.2f}", save_path=base + "_simplex.png", special_points=points, special_point_labels=labels)

        if plot_toggles["stationary_phase_portraits"] and simulation["weights"].shape[1] == 3:
            plot_stationary_phase_portrait(run["case_data"]["loss"], eta, title=f"{case_name}: phase portrait, eta = {eta:.2f}", save_path=base + "_phase_portrait.png")

        if preference_data is not None:
            plot_time_series(preference_data["margin_sequence"], ylabel="Margin", title=f"{case_name}: preferred-support margin, eta = {eta:.2f}", save_path=base + "_preferred_margin.png")
            plot_time_series(preference_data["tracking_sequence"], ylabel="Tracking indicator", title=f"{case_name}: tracking indicator, eta = {eta:.2f}", save_path=base + "_tracking_indicator.png", ylim=(-0.02, 1.02))

    plot_weight_eta_panel(case_runs, panel_title=f"{case_name}: weight trajectories across eta", save_path=os.path.join(output_dir, f"{case_name}_weight_eta_panel.png"))
    plot_scalar_eta_panel(case_runs, lambda run: run["simulation"]["entropies"], "Entropy", f"{case_name}: entropy across eta", save_path=os.path.join(output_dir, f"{case_name}_entropy_eta_panel.png"))
    plot_scalar_eta_panel(case_runs, lambda run: run["simulation"]["average_losses"], r"$\langle w_t,\ell\rangle$", f"{case_name}: weighted average loss across eta", save_path=os.path.join(output_dir, f"{case_name}_average_loss_eta_panel.png"))
    plot_scalar_eta_panel(case_runs, lambda run: run["simulation"]["continuous_time_errors"], "CT error", f"{case_name}: continuous-time comparison error across eta", save_path=os.path.join(output_dir, f"{case_name}_continuous_time_error_eta_panel.png"))

    if case_runs[0]["simulation"]["weights"].shape[1] == 3:
        plot_simplex_eta_panel(case_runs, panel_title=f"{case_name}: simplex trajectories across eta", save_path=os.path.join(output_dir, f"{case_name}_simplex_eta_panel.png"), special_points_getter=stationary_special_points)
        plot_stationary_phase_eta_panel(case_runs, panel_title=f"{case_name}: phase portraits across eta", save_path=os.path.join(output_dir, f"{case_name}_phase_eta_panel.png"))

    if any(stationary_preference_plot_data(run) is not None for run in case_runs):
        plot_stationary_preference_eta_panel(case_runs, "margin", panel_title=f"{case_name}: preferred-support margin across eta", save_path=os.path.join(output_dir, f"{case_name}_preferred_margin_eta_panel.png"))
        plot_stationary_preference_eta_panel(case_runs, "tracking", panel_title=f"{case_name}: tracking indicator across eta", save_path=os.path.join(output_dir, f"{case_name}_tracking_eta_panel.png"))

    plot_stationary_spectral_radius_panel(case_runs, panel_title=f"{case_name}: spectral radius by representative fixed point", save_path=os.path.join(output_dir, f"{case_name}_spectral_radius_panel.png"))

    plot_stationary_spectral_radius_summary(case_runs, title=f"{case_name}: spectral radius vs eta", save_path=os.path.join(output_dir, f"{case_name}_spectral_radius_summary.png"))
    plot_stationary_jacobian_difference_summary(case_runs, title=f"{case_name}: Jacobian difference norm vs eta", save_path=os.path.join(output_dir, f"{case_name}_jacobian_difference_summary.png"))
    plot_stationary_eigenvalue_discrepancy_summary(case_runs, title=f"{case_name}: eigenvalue discrepancy vs eta", save_path=os.path.join(output_dir, f"{case_name}_eigenvalue_discrepancy_summary.png"))
    plot_stationary_linearization_error_summary(case_runs, title=f"{case_name}: linearization error vs eta", save_path=os.path.join(output_dir, f"{case_name}_linearization_error_summary.png"))
    plot_stationary_final_entropy_summary(case_runs, title=f"{case_name}: final entropy vs eta", save_path=os.path.join(output_dir, f"{case_name}_final_entropy_summary.png"))
    plot_stationary_final_average_loss_summary(case_runs, title=f"{case_name}: final weighted average loss vs eta", save_path=os.path.join(output_dir, f"{case_name}_final_average_loss_summary.png"))
    plot_stationary_mean_continuous_time_error_summary(case_runs, title=f"{case_name}: mean continuous-time comparison error vs eta", save_path=os.path.join(output_dir, f"{case_name}_mean_continuous_time_error_summary.png"))
    plot_stationary_average_step_summary(case_runs, title=f"{case_name}: average step size vs eta", save_path=os.path.join(output_dir, f"{case_name}_average_step_summary.png"))
    plot_stationary_tracking_accuracy_summary(case_runs, title=f"{case_name}: tracking accuracy vs eta", save_path=os.path.join(output_dir, f"{case_name}_tracking_accuracy_summary.png"))
    plot_stationary_margin_summary(case_runs, title=f"{case_name}: preferred-support margin vs eta", save_path=os.path.join(output_dir, f"{case_name}_preferred_margin_summary.png"))


def save_periodic_figures(periodic_runs, output_dir):
    ensure_directory(output_dir)

    for run in periodic_runs:
        eta = run["eta"]
        simulation = run["simulation"]

        eta_tag = f"eta_{eta:.2f}".replace(".", "p")
        base = os.path.join(output_dir, f"periodic_{eta_tag}")

        if plot_toggles["periodic_weight_trajectories"]:
            plot_weight_trajectories(simulation, title=f"Periodic case: weight trajectories, eta = {eta:.2f}", save_path=base + "_weights.png")
        if plot_toggles["periodic_entropy_trajectories"]:
            plot_entropy_trajectory(simulation, title=f"Periodic case: entropy, eta = {eta:.2f}", save_path=base + "_entropy.png")
        if plot_toggles["periodic_average_loss_trajectories"]:
            plot_average_loss_trajectory(simulation, title=f"Periodic case: weighted average loss, eta = {eta:.2f}", save_path=base + "_average_loss.png")
        if plot_toggles["periodic_continuous_time_error_trajectories"]:
            plot_continuous_time_error_trajectory(simulation, title=f"Periodic case: continuous-time comparison error, eta = {eta:.2f}", save_path=base + "_continuous_time_error.png")
        if plot_toggles["periodic_margin_trajectories"]:
            plot_periodic_margin(simulation, title=f"Periodic case: preferred-weight margin, eta = {eta:.2f}", save_path=base + "_margin.png")
        if plot_toggles["periodic_step_size_trajectories"]:
            plot_step_sizes(simulation, title=f"Periodic case: step sizes, eta = {eta:.2f}", save_path=base + "_step_sizes.png")
        if plot_toggles["periodic_simplex_trajectories"] and simulation["weights"].shape[1] == 3:
            plot_simplex_trajectory(simulation, title=f"Periodic case: simplex trajectory, eta = {eta:.2f}", save_path=base + "_simplex.png")

    plot_weight_eta_panel(periodic_runs, panel_title="Periodic case: weight trajectories across eta", save_path=os.path.join(output_dir, "periodic_weight_eta_panel.png"))
    plot_scalar_eta_panel(periodic_runs, lambda run: run["simulation"]["entropies"], "Entropy", "Periodic case: entropy across eta", save_path=os.path.join(output_dir, "periodic_entropy_eta_panel.png"))
    plot_scalar_eta_panel(periodic_runs, lambda run: run["simulation"]["average_losses"], r"$\langle w_t,\ell_t\rangle$", "Periodic case: weighted average loss across eta", save_path=os.path.join(output_dir, "periodic_average_loss_eta_panel.png"))
    plot_scalar_eta_panel(periodic_runs, lambda run: run["simulation"]["continuous_time_errors"], "CT error", "Periodic case: continuous-time comparison error across eta", save_path=os.path.join(output_dir, "periodic_continuous_time_error_eta_panel.png"))
    plot_scalar_eta_panel(periodic_runs, lambda run: periodic_margin_sequence(run["simulation"]), "Margin", "Periodic case: preferred-weight margin across eta", save_path=os.path.join(output_dir, "periodic_margin_eta_panel.png"))
    plot_scalar_eta_panel(periodic_runs, lambda run: step_size_sequence(run["simulation"]), r"$\|w_{t+1} - w_t\|$", "Periodic case: step sizes across eta", save_path=os.path.join(output_dir, "periodic_step_size_eta_panel.png"))

    if periodic_runs[0]["simulation"]["weights"].shape[1] == 3:
        plot_simplex_eta_panel(periodic_runs, panel_title="Periodic case: simplex trajectories across eta", save_path=os.path.join(output_dir, "periodic_simplex_eta_panel.png"))

    plot_periodic_tracking_summary(periodic_runs, title="Periodic case: hard tracking accuracy vs eta", save_path=os.path.join(output_dir, "periodic_tracking_accuracy_summary.png"))
    plot_periodic_soft_tracking_summary(periodic_runs, title="Periodic case: mean preferred-expert share vs eta", save_path=os.path.join(output_dir, "periodic_preferred_share_summary.png"))
    plot_periodic_step_summary(periodic_runs, title="Periodic case: average step size vs eta", save_path=os.path.join(output_dir, "periodic_step_size_summary.png"))
    plot_periodic_continuous_time_error_summary(periodic_runs, title="Periodic case: continuous-time comparison error vs eta", save_path=os.path.join(output_dir, "periodic_continuous_time_error_summary.png"))
    plot_periodic_final_entropy_summary(periodic_runs, title="Periodic case: final entropy vs eta", save_path=os.path.join(output_dir, "periodic_final_entropy_summary.png"))
    plot_periodic_final_average_loss_summary(periodic_runs, title="Periodic case: final weighted average loss vs eta", save_path=os.path.join(output_dir, "periodic_final_average_loss_summary.png"))


def save_all_figures(all_results, output_dir="figures"):
    ensure_directory(output_dir)
    stationary_dir = os.path.join(output_dir, "stationary")
    periodic_dir = os.path.join(output_dir, "periodic")
    ensure_directory(stationary_dir)
    ensure_directory(periodic_dir)

    for case_name, case_runs in all_results["stationary"].items():
        case_dir = os.path.join(stationary_dir, case_name)
        save_stationary_case_figures(case_name, case_runs, case_dir)

    save_periodic_figures(all_results["periodic"], periodic_dir)
