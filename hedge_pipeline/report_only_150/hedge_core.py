import numpy as np


DEFAULT_HORIZON = 50


# Simplex
#
# The probability simplex is
# Delta^n = {w in R^n : w_i >= 0 for all i, sum_i w_i = 1}.
#
# The Hedge iterates remain in the simplex because each update preserves
# nonnegativity and normalizes the coordinates to sum to 1.


def normalize_to_simplex(w):
    # Normalize a nonnegative vector to the simplex.
    w = np.asarray(w, dtype=float)

    if np.any(w < 0):
        raise ValueError("Weights must be nonnegative.")

    total = np.sum(w)

    if total <= 0:
        raise ValueError("Weight vector must have positive sum.")

    return w / total


# Shannon entropy
#
# H(w) = -sum_i w_i log w_i
#
# This measures how spread out the weights are across experts.
# Smaller entropy corresponds to greater concentration of mass.


def entropy(w, tol=1.0e-12):
    w = np.asarray(w, dtype=float)
    mask = w > tol
    return float(-np.sum(w[mask] * np.log(w[mask])))


# Weighted average loss
#
# bar(ell)(w) = <w, ell> = sum_i w_i ell_i
#
# In the stationary case, this is the quantity that appears in the
# continuous-time limit.


def weighted_average_loss(w, ell):
    w = np.asarray(w, dtype=float)
    ell = np.asarray(ell, dtype=float)
    return float(np.dot(w, ell))


# Hedge update
#
# F_i(w) = w_i exp(-eta ell_i) / sum_j w_j exp(-eta ell_j)
#
# Under stationary losses, the dynamics are given by repeated iteration
# of this map on the simplex.


def hedge_update(w, ell, eta):
    w = np.asarray(w, dtype=float)
    ell = np.asarray(ell, dtype=float)

    scores = np.exp(-eta * ell)
    numerator = w * scores
    denominator = np.sum(numerator)

    if denominator <= 0:
        raise ValueError("Normalization denominator must be positive.")

    return normalize_to_simplex(numerator / denominator)


# Loss sequence construction
#
# In the stationary regime, ell_t = ell for all t.


def stationary_loss_sequence(ell, horizon, horizon_limit=200000):
    ell = np.asarray(ell, dtype=float)

    if horizon < 1:
        raise ValueError("Horizon must be positive.")

    if horizon > horizon_limit:
        raise ValueError("Horizon exceeds the allowed limit.")

    return np.repeat(ell[None, :], horizon, axis=0)


# Periodic loss sequence
#
# ell_t = (
#   0.5 + 0.4 sin(2 pi t / P),
#   0.5 - 0.4 sin(2 pi t / P),
#   1.0
# )
#
# The first two experts alternate in relative quality.
# The third expert is uniformly worse because the first two losses remain
# in the interval [0.1, 0.9], while the third loss is always 1.


def periodic_loss_sequence(horizon, period, horizon_limit=200000):
    if horizon < 1:
        raise ValueError("Horizon must be positive.")

    if horizon > horizon_limit:
        raise ValueError("Horizon exceeds the allowed limit.")

    if period <= 0:
        raise ValueError("Period must be positive.")

    losses = np.zeros((horizon, 3), dtype=float)

    for t in range(horizon):
        phase = 2.0 * np.pi * t / period
        losses[t, 0] = 0.5 + 0.4 * np.sin(phase)
        losses[t, 1] = 0.5 - 0.4 * np.sin(phase)
        losses[t, 2] = 1.0

    return losses


# Sequence summaries
#
# For any time-indexed quantity q_t, it is useful to record:
# - the mean value,
# - the maximum value and the time at which it occurs,
# - the minimum value and the time at which it occurs.
#
# This is especially useful for entropy, weighted average loss,
# continuous-time comparison error, tracking margins, and step sizes.


def sequence_summary(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return {
            "mean": np.nan,
            "max_value": np.nan,
            "max_time": None,
            "min_value": np.nan,
            "min_time": None,
        }

    max_index = int(np.argmax(values))
    min_index = int(np.argmin(values))

    return {
        "mean": float(np.mean(values)),
        "max_value": float(values[max_index]),
        "max_time": max_index,
        "min_value": float(values[min_index]),
        "min_time": min_index,
    }


# Simulation routines


def run_hedge_simulation(w0, losses, eta, horizon_limit=200000):
    # Simulate Hedge for a given initial weight vector, loss sequence, and eta.
    losses = np.asarray(losses, dtype=float)
    w0 = normalize_to_simplex(w0)

    horizon = losses.shape[0]
    n = w0.size

    if horizon > horizon_limit:
        raise ValueError("Horizon exceeds the allowed limit.")

    weights = np.zeros((horizon + 1, n), dtype=float)
    average_losses = np.zeros(horizon, dtype=float)
    entropies = np.zeros(horizon + 1, dtype=float)

    w = w0.copy()
    weights[0] = w
    entropies[0] = entropy(w)

    for t in range(horizon):
        ell_t = losses[t]
        average_losses[t] = weighted_average_loss(w, ell_t)
        w = hedge_update(w, ell_t, eta)
        weights[t + 1] = w
        entropies[t + 1] = entropy(w)

    return {
        "weights": weights,
        "losses": losses,
        "average_losses": average_losses,
        "entropies": entropies,
        "average_loss_summary": sequence_summary(average_losses),
        "entropy_summary": sequence_summary(entropies),
        "eta": eta,
        "horizon": horizon,
        "final_weight": weights[-1],
        "final_entropy": entropies[-1],
        "final_average_loss": average_losses[-1] if horizon > 0 else np.nan,
    }


# Analytical Jacobian in the stationary-loss regime
#
# dF_i / dw_k =
# [delta_ik exp(-eta ell_i) sum_j w_j exp(-eta ell_j)
#  - w_i exp(-eta ell_i) exp(-eta ell_k)]
# / [sum_j w_j exp(-eta ell_j)]^2


def analytic_jacobian_stationary(w, ell, eta):
    w = np.asarray(w, dtype=float)
    ell = np.asarray(ell, dtype=float)

    n = w.size
    a = np.exp(-eta * ell)
    z = np.dot(w, a)

    if z <= 0:
        raise ValueError("Normalization factor must be positive.")

    jacobian = np.zeros((n, n), dtype=float)

    for i in range(n):
        for k in range(n):
            delta = 1.0 if i == k else 0.0
            jacobian[i, k] = (delta * a[i] * z - w[i] * a[i] * a[k]) / (z ** 2)

    return jacobian


# Tangent basis for the simplex
#
# The simplex constraint sum_i w_i = 1 implies that tangent perturbations
# must satisfy sum_i v_i = 0.


def tangent_basis(w):
    # Feasible basis directions for the simplex at the point w.
    #
    # I choose an anchor coordinate with positive weight and define the
    # columns as e_i - e_anchor for i != anchor.
    #
    # These directions span the tangent space sum_i v_i = 0, and they are
    # feasible in the positive direction for sufficiently small step size.
    w = normalize_to_simplex(w)
    n = w.size

    support = np.where(w > 1.0e-14)[0]
    if support.size == 0:
        raise ValueError("Weight vector must have at least one positive coordinate.")

    anchor = int(support[0])
    basis = np.zeros((n, n - 1), dtype=float)

    column = 0
    for i in range(n):
        if i == anchor:
            continue
        basis[i, column] = 1.0
        basis[anchor, column] = -1.0
        column += 1

    return basis



# Numerical Jacobian restricted to the simplex
#
# I approximate the derivative using finite differences along tangent
# directions whose coordinates sum to zero.


def numerical_jacobian_tangent(w, ell, eta, h=1.0e-7):
    w = normalize_to_simplex(w)
    ell = np.asarray(ell, dtype=float)

    n = w.size
    basis = tangent_basis(w)
    jacobian_numerical = np.zeros((n, n - 1), dtype=float)

    f0 = hedge_update(w, ell, eta)

    for k in range(n - 1):
        direction = basis[:, k]
        step = h

        # Use a feasible forward difference along the basis direction.
        # Each basis vector is chosen so that the positive direction remains
        # inside the simplex for sufficiently small step size.
        wp = w + step * direction

        while np.any(wp < 0):
            step *= 0.5

            if step < 1.0e-14:
                raise ValueError("Could not construct a valid tangent perturbation.")

            wp = w + step * direction

        fp = hedge_update(wp, ell, eta)
        jacobian_numerical[:, k] = (fp - f0) / step

    return jacobian_numerical, basis



# Compare analytical and numerical Jacobians in tangent coordinates
#
# The comparison includes:
# - the full analytical Jacobian in ambient coordinates,
# - the analytical tangent action,
# - the numerical tangent Jacobian,
# - a norm of the difference between the tangent Jacobians.
#
# This checks whether the numerical derivative reproduces the analytical formula.


def compare_jacobians_stationary(w, ell, eta, h=1.0e-7):
    jacobian_analytic = analytic_jacobian_stationary(w, ell, eta)
    jacobian_numerical, basis = numerical_jacobian_tangent(w, ell, eta, h=h)
    jacobian_analytic_tangent = jacobian_analytic @ basis

    difference_norm = float(np.linalg.norm(jacobian_analytic_tangent - jacobian_numerical))
    eigenvalues_analytic = np.linalg.eigvals(jacobian_analytic)

    # In basis coordinates, the linear map A satisfies
    # J B = B A.
    # I recover A from the numerical tangent action using the pseudoinverse.
    coordinate_matrix_numerical = np.linalg.pinv(basis) @ jacobian_numerical
    eigenvalues_numerical = np.linalg.eigvals(coordinate_matrix_numerical)

    return {
        "jacobian_analytic": jacobian_analytic,
        "jacobian_analytic_tangent": jacobian_analytic_tangent,
        "jacobian_numerical_tangent": jacobian_numerical,
        "difference_norm": difference_norm,
        "eigenvalues_analytic": eigenvalues_analytic,
        "eigenvalues_numerical": eigenvalues_numerical,
    }



# Spectral radius of a matrix
#
# ρ(J) = max |lambda_i|
#
# This determines whether the linearized dynamics are contractive,
# unstable, or neutral in the relevant directions.


def spectral_radius(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))


# Local linearization error
#
# I compare the exact one-step change
# F(w* + eps v) - F(w*)
#
# with the linear prediction
# J(w*) (eps v).
#
# The reported quantity is
# ||F(w* + eps v) - F(w*) - J(w*) (eps v)|| / ||eps v||.
#
# This measures how accurately the linearized map approximates one nonlinear
# update near the fixed point.


def one_step_linearization_error(w_star, ell, eta, v, epsilon):
    w_star = normalize_to_simplex(w_star)
    ell = np.asarray(ell, dtype=float)
    v = np.asarray(v, dtype=float)

    if not np.isclose(np.sum(v), 0.0):
        raise ValueError("Perturbation vector must be tangent to the simplex.")

    step = epsilon
    wp = w_star + step * v

    while np.any(wp < 0):
        step *= 0.5

        if step < 1.0e-14:
            raise ValueError("Perturbation leaves the simplex.")

        wp = w_star + step * v

    jacobian = analytic_jacobian_stationary(w_star, ell, eta)
    true_change = hedge_update(wp, ell, eta) - hedge_update(w_star, ell, eta)
    linear_prediction = jacobian @ (step * v)
    denominator = np.linalg.norm(step * v)

    if denominator <= 1.0e-14:
        raise ValueError("Perturbation norm is too small.")

    return float(np.linalg.norm(true_change - linear_prediction) / denominator)


# Continuous-time comparison
#
# The continuous-time limit is
# dw_i / d tau = w_i (bar(ell)(w) - ell_i).
#
# A discrete comparison is obtained by comparing
# (w_{t+1} - w_t) / eta
#
# with
# w_t * (bar(ell)(w_t) - ell).
#
# The resulting error sequence measures how closely the discrete update
# matches the continuous-time mechanism for a given eta.


def continuous_time_comparison_errors(weights, losses, eta):
    weights = np.asarray(weights, dtype=float)
    losses = np.asarray(losses, dtype=float)

    horizon = losses.shape[0]
    errors = np.zeros(horizon, dtype=float)

    for t in range(horizon):
        w_t = weights[t]
        w_next = weights[t + 1]
        ell_t = losses[t]
        bar_ell = weighted_average_loss(w_t, ell_t)

        discrete_velocity = (w_next - w_t) / eta
        continuous_velocity = w_t * (bar_ell - ell_t)
        errors[t] = np.linalg.norm(discrete_velocity - continuous_velocity)

    return errors


# Stationary cases I consider (see Computational Section).


def stationary_cases(horizon=None):
    if horizon is None:
        horizon = DEFAULT_HORIZON
    horizon = int(horizon)

    return {
        "unique_minimum": {
            "loss": np.array([0.0, 1.0, 2.0], dtype=float),
            "w0": np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float),
            "horizon": horizon,
            "fixed_points": [
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            ],
            "linearization_vectors": [
                np.array([1.0, -1.0, 0.0], dtype=float),
                np.array([1.0, 0.0, -1.0], dtype=float),
            ],
        },
        "equal_minimum_pair": {
            "loss": np.array([0.0, 0.0, 1.0], dtype=float),
            "w0": np.array([0.2, 0.8, 0.0], dtype=float),
            "horizon": horizon,
            "fixed_points": [
                np.array([0.4, 0.6, 0.0], dtype=float),
                np.array([0.5, 0.5, 0.0], dtype=float),
            ],
            "linearization_vectors": [
                np.array([1.0, -1.0, 0.0], dtype=float),
                np.array([0.5, 0.5, -1.0], dtype=float),
            ],
        },
        "all_equal": {
            "loss": np.array([1.0, 1.0, 1.0], dtype=float),
            "w0": np.array([0.2, 0.3, 0.5], dtype=float),
            "horizon": horizon,
            "fixed_points": [
                np.array([0.2, 0.3, 0.5], dtype=float),
                np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float),
            ],
            "linearization_vectors": [
                np.array([1.0, -1.0, 0.0],
                dtype=float),
                np.array([1.0, 0.0, -1.0], dtype=float),
            ],
        },
    }


def shared_eta_values(
):
    return [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def stationary_eta_values(use_shared=True):
    if use_shared:
        return shared_eta_values()

    return [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def periodic_eta_values(use_shared=True):
    if use_shared:
        return shared_eta_values()

    return [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


# Predicted eigenvalues from the stationary stability classification
#
# For a fixed point supported on a set with common loss c:
# - tangent directions inside the active equal-loss set have eigenvalue 1,
# - inactive directions have eigenvalues exp(-eta (ell_i - c)).
#
# These are the analytical eigenvalues that the numerical tangent Jacobian
# should reproduce.


def stationary_predicted_eigenvalues(loss, fixed_point, eta):
    loss = np.asarray(loss, dtype=float)
    fixed_point = np.asarray(fixed_point, dtype=float)

    support = np.where(fixed_point > 1.0e-12)[0]

    if support.size == 0:
        raise ValueError("Fixed point support must be nonempty.")

    c = float(loss[support[0]])
    predicted = []

    if support.size >= 2:
        predicted.extend([1.0] * (support.size - 1))

    inactive = [i for i in range(loss.size) if i not in support]

    for i in inactive:
        predicted.append(np.exp(-eta * (loss[i] - c)))

    return np.array(predicted, dtype=complex)


# Analytical stability classification for a stationary fixed point.


def classify_stationary_expectation(loss, fixed_point, eta, tol=1.0e-10):
    predicted = stationary_predicted_eigenvalues(loss, fixed_point, eta)

    if np.any(np.abs(predicted) > 1.0 + tol):
        return "Analytical expectation: unstable, because at least one relevant eigenvalue has magnitude greater than 1"

    if np.any(np.abs(predicted - 1.0) <= tol):
        return "Analytical expectation: neutral in at least one relevant direction, because at least one tangent or inactive eigenvalue is equal to 1"

    return "Analytical expectation: contractive in the relevant directions, because all relevant eigenvalues have magnitude less than 1"


# Sort eigenvalues by real part, then imaginary part.


def _sorted_eigenvalues(values):
    values = [complex(v) for v in values]
    values.sort(key=lambda z: (round(z.real, 12), round(z.imag, 12)))
    return values


# Format real or complex values for text output.


def _format_real_or_complex(value, digits=6):
    value = complex(value)

    if abs(value.imag) < 1.0e-12:
        return f"{value.real:.{digits}f}"

    sign = "+" if value.imag >= 0 else "-"
    return f"{value.real:.{digits}f} {sign} {abs(value.imag):.{digits}f}i"


# Eigenvalue discrepancy summary
#
# For the predicted and numerical eigenvalues, it is useful to record:
# - the maximum discrepancy,
# - the minimum discrepancy,
# - the mean discrepancy.
#
# There is no associated time index here because the comparison is made at
# one selected fixed point rather than along a trajectory.


def eigenvalue_discrepancy_summary(predicted, numerical):
    pair_count = min(len(predicted), len(numerical))

    if pair_count == 0:
        return {
            "max_difference": 0.0,
            "min_difference": 0.0,
            "mean_difference": 0.0,
        }

    differences = np.array(
        [abs(predicted[k] - numerical[k]) for k in range(pair_count)],
        dtype=float,
    )

    return {
        "max_difference": float(np.max(differences)),
        "min_difference": float(np.min(differences)),
        "mean_difference": float(np.mean(differences)),
    }


# Compare predicted and numerical tangent eigenvalues.


def compare_predicted_and_numerical_eigenvalues(loss, fixed_point, eta, h=1.0e-7):
    comparison = compare_jacobians_stationary(fixed_point, loss, eta, h=h)
    predicted = _sorted_eigenvalues(stationary_predicted_eigenvalues(loss, fixed_point, eta))
    numerical = _sorted_eigenvalues(comparison["eigenvalues_numerical"])
    discrepancy_summary = eigenvalue_discrepancy_summary(predicted, numerical)

    return {
        "predicted": predicted,
        "numerical": numerical,
        "max_difference": discrepancy_summary["max_difference"],
        "min_difference": discrepancy_summary["min_difference"],
        "mean_difference": discrepancy_summary["mean_difference"],
    }


# Periodic tracking diagnostics
#
# The lower-loss expert among the first two changes over time.
# These quantities measure how well the weights follow that changing preference.
#
# Tracking accuracy is the fraction of time steps at which the algorithm assigns
# at least as much weight to the currently lower-loss expert as to its alternating
# competitor.
#
# The preferred-weight margin is
# preferred_weight_t - opposite_weight_t,
#
# where preferred_weight_t is the weight on the currently lower-loss expert and
# opposite_weight_t is the weight on the other alternating expert.
#
# The step size is
# ||w_{t+1} - w_t||,
#
# which measures how sharply the weight vector moves from one round to the next.


def tracking_diagnostic_periodic(simulation):
    weights = simulation["weights"][:-1]
    losses = simulation["losses"]

    better_expert = np.argmin(losses[:, :2], axis=1)
    preferred_weight = np.zeros(losses.shape[0], dtype=float)
    opposite_weight = np.zeros(losses.shape[0], dtype=float)

    for t in range(losses.shape[0]):
        preferred_weight[t] = weights[t, better_expert[t]]
        opposite_weight[t] = weights[t, 1 - better_expert[t]]

    margin = preferred_weight - opposite_weight
    tracking_accuracy = float(np.mean(preferred_weight >= opposite_weight))
    average_margin = float(np.mean(margin))
    minimum_margin = float(np.min(margin))
    maximum_margin = float(np.max(margin))
    minimum_margin_time = int(np.argmin(margin))
    maximum_margin_time = int(np.argmax(margin))

    step_sizes = np.linalg.norm(weights[1:] - weights[:-1], axis=1)
    step_summary = sequence_summary(step_sizes)

    return {
        "tracking_accuracy": tracking_accuracy,
        "average_margin": average_margin,
        "minimum_margin": minimum_margin,
        "minimum_margin_time": minimum_margin_time,
        "maximum_margin": maximum_margin,
        "maximum_margin_time": maximum_margin_time,
        "average_step": step_summary["mean"],
        "maximum_step": step_summary["max_value"],
        "maximum_step_time": step_summary["max_time"],
        "minimum_step": step_summary["min_value"],
        "minimum_step_time": step_summary["min_time"],
        "preferred_weight_margin_summary": sequence_summary(margin),
        "step_size_summary": step_summary,
    }


# Selected sample times for printed trajectory summaries.


def sample_times(horizon):
    raw = [0, 1, 2, 5, 10, 20, horizon]
    times = [t for t in raw if 0 <= t <= horizon]
    unique_times = sorted(set(times))
    return unique_times



def stationary_fixed_point_note(case_name):
    if case_name == "equal_minimum_pair":
        return (
            "Mathematical note: the full fixed-point set is the edge {w in simplex : w_3 = 0}. "
            "The listed fixed points are representative samples used for Jacobian evaluation."
        )
    if case_name == "all_equal":
        return (
            "Mathematical note: every point of the simplex is a fixed point when all losses are equal. "
            "The listed fixed points are representative samples used for Jacobian evaluation."
        )
    return (
        "Mathematical note: the stationary fixed points listed below are exactly the simplex vertices. "
        "The unique minimizer e_1 is stable, while the other vertices are unstable for eta > 0."
    )


def stationary_preference_diagnostic(case, simulation):
    ell = np.asarray(case["loss"], dtype=float)
    minima = np.flatnonzero(np.isclose(ell, np.min(ell)))
    weights = np.asarray(simulation["weights"][:-1], dtype=float)
    n = weights.shape[1]

    if minima.size == n:
        return None

    if minima.size == 1:
        idx = int(minima[0])
        competitor_idx = [j for j in range(n) if j != idx]
        competitor = np.max(weights[:, competitor_idx], axis=1)
        preferred = weights[:, idx]
        tracking_label = f"fraction of times expert {idx + 1} leads"
        margin_label = f"expert {idx + 1} lead margin"
        mode = "unique_best"
    else:
        preferred = np.sum(weights[:, minima], axis=1)
        competitor = 1.0 - preferred
        tracking_label = "fraction of times the minimum-loss set carries at least half the mass"
        margin_label = "minimum-set support margin"
        mode = "minimum_set"

    margin = preferred - competitor
    tracking = preferred >= competitor

    return {
        "mode": mode,
        "tracking_label": tracking_label,
        "margin_label": margin_label,
        "margin_sequence": margin,
        "tracking_sequence": tracking.astype(float),
        "mean_margin": float(np.mean(margin)),
        "mean_tracking_accuracy": float(np.mean(tracking)),
        "minimum_margin": float(np.min(margin)),
        "minimum_margin_time": int(np.argmin(margin)),
        "maximum_margin": float(np.max(margin)),
        "maximum_margin_time": int(np.argmax(margin)),
    }


# Numerical interpretation for the stationary simulation.


def interpret_stationary_simulation(case_name, simulation):
    final_weight = simulation["final_weight"]

    if case_name == "unique_minimum":
        if final_weight[0] > 0.99:
            return "Numerical observation: the trajectory concentrates on the unique minimum-loss expert, which is consistent with the expected attracting behavior of the minimum-loss vertex"
        return "Numerical observation: the trajectory does not yet fully concentrate on the unique minimum-loss expert"

    if case_name == "equal_minimum_pair":
        if final_weight[2] < 1.0e-6:
            return "Numerical observation: the trajectory remains on or moves toward the equal-loss support of the first two experts, which is consistent with the predicted neutral behavior within that set"
        return "Numerical observation: the trajectory retains noticeable weight on the higher-loss expert"

    if case_name == "all_equal":
        if np.linalg.norm(simulation["weights"][-1] - simulation["weights"][0]) < 1.0e-10:
            return "Numerical observation: the trajectory remains unchanged, which is consistent with the identity-map behavior when all losses are equal"
        return "Numerical observation: the trajectory changes despite equal losses"

    return "Numerical observation: no interpretation available"


# Numerical interpretation for the periodic simulation.


def interpret_periodic_simulation(diagnostic):
    if diagnostic["tracking_accuracy"] > 0.8 and diagnostic["average_margin"] > 0:
        return "Numerical observation: the weights track the currently better expert well overall"

    if diagnostic["tracking_accuracy"] > 0.6:
        return "Numerical observation: the weights track the changing lower-loss expert moderately well, but the preference is not consistently strong"

    return "Numerical observation: the weights do not track the changing lower-loss expert reliably"


# Run one named stationary case for one eta.



def run_stationary_case(case_name, eta, horizon=None):
    cases = stationary_cases(horizon=horizon)

    if case_name not in cases:
        raise ValueError("Unknown stationary case.")

    case = cases[case_name]
    losses = stationary_loss_sequence(case["loss"], case["horizon"])
    simulation = run_hedge_simulation(case["w0"], losses, eta)
    simulation["continuous_time_errors"] = continuous_time_comparison_errors(
        simulation["weights"],
        simulation["losses"],
        eta,
    )
    simulation["continuous_time_error_summary"] = sequence_summary(simulation["continuous_time_errors"])

    jacobian_data = []

    for fixed_point in case["fixed_points"]:
        comparison = compare_jacobians_stationary(fixed_point, case["loss"], eta)
        eigen_comparison = compare_predicted_and_numerical_eigenvalues(
            case["loss"],
            fixed_point,
            eta,
        )

        jacobian_data.append(
            {
                "fixed_point": fixed_point,
                "comparison": comparison,
                "eigen_comparison": eigen_comparison,
                "spectral_radius": spectral_radius(comparison["jacobian_analytic"]),
                "expectation": classify_stationary_expectation(case["loss"], fixed_point, eta),
            }
        )

    linearization_data = []

    for fixed_point in case["fixed_points"]:
        for direction in case["linearization_vectors"]:
            if np.isclose(np.sum(direction), 0.0):
                try:
                    error = one_step_linearization_error(
                        fixed_point,
                        case["loss"],
                        eta,
                        direction,
                        epsilon=1.0e-4,
                    )
                    linearization_data.append(
                        {
                            "fixed_point": fixed_point,
                            "direction": direction,
                            "error": error,
                        }
                    )
                except ValueError:
                    pass

    return {
        "case_name": case_name,
        "case_data": case,
        "eta": eta,
        "simulation": simulation,
        "jacobian_data": jacobian_data,
        "linearization_data": linearization_data,
        "numerical_interpretation": interpret_stationary_simulation(case_name, simulation),
        "stationary_preference": stationary_preference_diagnostic(case, simulation),
    }


def run_all_stationary_cases(use_shared_eta=True, horizon=None, T=None):
    if horizon is None:
        horizon = T

    results = {}

    for case_name in stationary_cases(horizon=horizon):
        results[case_name] = []

        for eta in stationary_eta_values(use_shared=use_shared_eta):
            results[case_name].append(run_stationary_case(case_name, eta, horizon=horizon))

    return results


def run_periodic_case(period=20, horizon=None, w0=None, use_shared_eta=True, T=None):
    if horizon is None:
        horizon = T

    if horizon is None:
        horizon = DEFAULT_HORIZON

    if w0 is None:
        w0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)

    losses = periodic_loss_sequence(horizon, period)
    results = []

    for eta in periodic_eta_values(use_shared=use_shared_eta):
        simulation = run_hedge_simulation(w0, losses, eta)
        simulation["continuous_time_errors"] = continuous_time_comparison_errors(
            simulation["weights"],
            simulation["losses"],
            eta,
        )
        simulation["continuous_time_error_summary"] = sequence_summary(simulation["continuous_time_errors"])
        diagnostic = tracking_diagnostic_periodic(simulation)

        if eta <= 0.1:
            expectation = "Analytical expectation: smaller eta should produce smoother tracking and closer agreement with the continuous-time mechanism"
        else:
            expectation = "Analytical expectation: larger eta should produce sharper reweighting and greater sensitivity to the periodic oscillation"

        results.append(
            {
                "eta": eta,
                "simulation": simulation,
                "period": period,
                "w0": normalize_to_simplex(w0),
                "diagnostic": diagnostic,
                "expectation": expectation,
                "numerical_interpretation": interpret_periodic_simulation(diagnostic),
            }
        )

    return results


def run_all_cases(use_shared_eta=True, horizon=None, T=None):
    if horizon is None:
        horizon = T

    if horizon is None:
        horizon = DEFAULT_HORIZON

    return {
        "stationary": run_all_stationary_cases(use_shared_eta=use_shared_eta, horizon=horizon),
        "periodic": run_periodic_case(use_shared_eta=use_shared_eta, horizon=horizon),
    }


def format_vector(
vector, digits=6):
    vector = np.asarray(vector, dtype=float)
    return "[" + ", ".join(f"{x:.{digits}f}" for x in vector) + "]"


def format_matrix(matrix, digits=6):
    matrix = np.asarray(matrix, dtype=float)
    rows = []

    for row in matrix:
        rows.append("[" + ", ".join(f"{x:.{digits}f}" for x in row) + "]")

    return "[\n  " + ",\n  ".join(rows) + "\n]"



# Directional Interpretation of Stationary Fixed Points
#
# This explains why a fixed point is contractive or unstable by identifying
# whether inactive directions correspond to higher-loss experts, which are
# suppressed, or lower-loss experts, which grow and destabilize the point.


def stationary_directional_interpretation(loss, fixed_point, eta, tol=1.0e-12):
    loss = np.asarray(loss, dtype=float)
    fixed_point = np.asarray(fixed_point, dtype=float)

    support = np.where(fixed_point > tol)[0]
    inactive = np.where(fixed_point <= tol)[0]

    if support.size == 0:
        return "Directional interpretation: No active coordinates were detected."

    c = float(loss[support[0]])
    messages = []

    if support.size >= 2:
        messages.append(
            "Along the active equal-loss support, tangent directions are neutral because the active experts have the same loss."
        )

    for i in inactive:
        if loss[i] > c + tol:
            lam = np.exp(-eta * (loss[i] - c))
            messages.append(
                f"Inactive expert {i + 1} has higher loss than the active loss level, so perturbations into that coordinate are suppressed; the corresponding factor is exp(-eta (ell_{i + 1} - c)) = {lam:.6f} < 1."
            )
        elif loss[i] < c - tol:
            lam = np.exp(-eta * (loss[i] - c))
            messages.append(
                f"Inactive expert {i + 1} has lower loss than the active loss level, so perturbations into that coordinate grow and destabilize the fixed point; the corresponding factor is exp(-eta (ell_{i + 1} - c)) = {lam:.6f} > 1."
            )
        else:
            lam = np.exp(-eta * (loss[i] - c))
            messages.append(
                f"Inactive expert {i + 1} has the same loss as the active loss level, so that direction is neutral; the corresponding factor is exp(-eta (ell_{i + 1} - c)) = {lam:.6f}."
            )

    if not messages:
        messages.append("There are no inactive directions at this fixed point.")

    return "Directional interpretation: " + " ".join(messages)


# Stationary Case Report
#
# The report prints both the numerical value of each quantity and a short
# explanation of why that quantity matters for the analytical comparison.


def stationary_report(results):
    lines = []
    lines.append("Stationary Hedge simulations")
    lines.append("")

    for case_name, case_runs in results.items():
        lines.append(f"Case: {case_name}")
        lines.append("")

        case_data = case_runs[0]["case_data"]
        lines.append(f"Loss vector = {format_vector(case_data['loss'])}")
        lines.append(f"Initial weight = {format_vector(case_data['w0'])}")
        lines.append(f"Horizon = {case_data['horizon']}")
        lines.append(stationary_fixed_point_note(case_name))
        lines.append("")

        for run in case_runs:
            simulation = run["simulation"]
            times = sample_times(simulation["horizon"])
            entropy_summary = simulation["entropy_summary"]
            average_loss_summary = simulation["average_loss_summary"]
            continuous_time_summary = simulation["continuous_time_error_summary"]
            preference = run.get("stationary_preference")

            lines.append(f"Eta = {run['eta']:.2f}")
            lines.append(run["numerical_interpretation"])
            lines.append("")

            lines.append("Selected weight values w_t")
            for t in times:
                lines.append(f"t = {t}: {format_vector(simulation['weights'][t])}")
            lines.append("Significance: These values show how the probability mass moves over time.")
            lines.append("")

            lines.append("Selected entropy values H(w_t)")
            for t in times:
                lines.append(f"t = {t}: {simulation['entropies'][t]:.6e}")
            lines.append(f"Final entropy = {simulation['final_entropy']:.6e}")
            lines.append(f"Maximum entropy = {entropy_summary['max_value']:.6e} at t = {entropy_summary['max_time']}")
            lines.append(f"Minimum entropy = {entropy_summary['min_value']:.6e} at t = {entropy_summary['min_time']}")
            lines.append("Significance: Smaller entropy indicates stronger concentration of weight.")
            lines.append("")

            lines.append("Selected weighted average losses <w_t, ell>")
            for t in times:
                if t < simulation["horizon"]:
                    lines.append(f"t = {t}: {simulation['average_losses'][t]:.6e}")
            lines.append(f"Final weighted average loss = {simulation['final_average_loss']:.6e}")
            lines.append(f"Maximum weighted average loss = {average_loss_summary['max_value']:.6e} at t = {average_loss_summary['max_time']}")
            lines.append(f"Minimum weighted average loss = {average_loss_summary['min_value']:.6e} at t = {average_loss_summary['min_time']}")
            lines.append("Significance: In the stationary case, this should move toward the loss level favored by the dynamics.")
            lines.append("")

            ct_errors = simulation["continuous_time_errors"]
            lines.append("Continuous-time comparison error")
            for t in times:
                if t < simulation["horizon"]:
                    lines.append(f"t = {t}: {ct_errors[t]:.6e}")
            lines.append(f"Mean continuous-time comparison error = {continuous_time_summary['mean']:.6e}")
            lines.append(f"Maximum continuous-time comparison error = {continuous_time_summary['max_value']:.6e} at t = {continuous_time_summary['max_time']}")
            lines.append(f"Minimum continuous-time comparison error = {continuous_time_summary['min_value']:.6e} at t = {continuous_time_summary['min_time']}")
            lines.append("Significance: Smaller values indicate closer agreement with the continuous-time approximation.")
            lines.append("")

            if preference is not None:
                lines.append(f"Mean tracking accuracy = {preference['mean_tracking_accuracy']:.6f}")
                lines.append(f"Significance: {preference['tracking_label']}.")
                lines.append("")
                lines.append(f"Mean preferred margin = {preference['mean_margin']:.6e}")
                lines.append(f"Maximum preferred margin = {preference['maximum_margin']:.6e} at t = {preference['maximum_margin_time']}")
                lines.append(f"Minimum preferred margin = {preference['minimum_margin']:.6e} at t = {preference['minimum_margin_time']}")
                lines.append(f"Significance: {preference['margin_label']} compares the preferred support with its competitor.")
                lines.append("")

            lines.append("Jacobian and spectral comparisons")
            for item in run["jacobian_data"]:
                fixed_point = item["fixed_point"]
                comparison = item["comparison"]
                eigen_comparison = item["eigen_comparison"]

                lines.append(f"Fixed point = {format_vector(fixed_point)}")
                lines.append(item["expectation"])
                lines.append(stationary_directional_interpretation(case_data["loss"], fixed_point, run["eta"]))
                lines.append("Analytical Jacobian =")
                lines.append(format_matrix(comparison["jacobian_analytic"]))
                lines.append("Numerical Jacobian on the simplex =")
                lines.append(format_matrix(comparison["jacobian_numerical_tangent"]))
                lines.append(f"Jacobian difference norm = {comparison['difference_norm']:.6e}")
                lines.append("Significance: A small difference norm indicates close agreement between the analytical derivative and the numerical finite-difference approximation.")
                lines.append("")

                lines.append(
                    "Predicted tangent eigenvalues = ["
                    + ", ".join(_format_real_or_complex(x) for x in eigen_comparison["predicted"])
                    + "]"
                )
                lines.append(
                    "Numerical tangent eigenvalues = ["
                    + ", ".join(_format_real_or_complex(x) for x in eigen_comparison["numerical"])
                    + "]"
                )
                lines.append(f"Maximum eigenvalue discrepancy = {eigen_comparison['max_difference']:.6e}")
                lines.append(f"Minimum eigenvalue discrepancy = {eigen_comparison['min_difference']:.6e}")
                lines.append(f"Mean eigenvalue discrepancy = {eigen_comparison['mean_difference']:.6e}")
                lines.append("Significance: Small discrepancy supports the analytical stability classification.")
                lines.append("")

                lines.append(f"Spectral radius = {item['spectral_radius']:.6f}")
                if item["spectral_radius"] < 1.0 - 1.0e-10:
                    lines.append("Significance: The spectral radius is less than 1, so the linearized dynamics are contractive in the relevant directions.")
                elif item["spectral_radius"] > 1.0 + 1.0e-10:
                    lines.append("Significance: The spectral radius exceeds 1, so the fixed point is linearly unstable.")
                else:
                    lines.append("Significance: The spectral radius is approximately 1, so at least one relevant direction is neutral.")
                lines.append("")

            if run["linearization_data"]:
                lines.append("One-step linearization error")
                for item in run["linearization_data"]:
                    lines.append(
                        "Fixed point = "
                        + format_vector(item["fixed_point"])
                        + ", direction = "
                        + format_vector(item["direction"])
                        + f", error = {item['error']:.6e}"
                    )
                lines.append("Significance: Smaller values indicate that the linearized map approximates one nonlinear update accurately near the fixed point.")
                lines.append("")

            lines.append("-" * 72)
            lines.append("")

    return "\n".join(lines)


# Periodic Case Report
#
# The report prints both the numerical value of each quantity and a short
# explanation of why that quantity matters for the tracking analysis.


def periodic_report(results):
    lines = []
    lines.append("Periodic Hedge simulations")
    lines.append("")

    for run in results:
        simulation = run["simulation"]
        diagnostic = run["diagnostic"]
        times = sample_times(simulation["horizon"])
        entropy_summary = simulation["entropy_summary"]
        average_loss_summary = simulation["average_loss_summary"]
        continuous_time_summary = simulation["continuous_time_error_summary"]

        lines.append(f"Eta = {run['eta']:.2f}")
        lines.append(f"Period = {run['period']}")
        lines.append(f"Initial weight = {format_vector(run['w0'])}")
        lines.append(run["expectation"])
        lines.append(run["numerical_interpretation"])
        lines.append("")

        lines.append("Selected weight values w_t")
        for t in times:
            lines.append(f"t = {t}: {format_vector(simulation['weights'][t])}")
        lines.append("Significance: These values show how the weights shift as the better expert changes over time.")
        lines.append("")

        lines.append("Selected entropy values H(w_t)")
        for t in times:
            lines.append(f"t = {t}: {simulation['entropies'][t]:.6e}")
        lines.append(f"Final entropy = {simulation['final_entropy']:.6e}")
        lines.append(f"Maximum entropy = {entropy_summary['max_value']:.6e} at t = {entropy_summary['max_time']}")
        lines.append(f"Minimum entropy = {entropy_summary['min_value']:.6e} at t = {entropy_summary['min_time']}")
        lines.append("Significance: The entropy shows whether the periodic reweighting remains spread out or becomes concentrated.")
        lines.append("")

        lines.append("Selected weighted average losses <w_t, ell_t>")
        for t in times:
            if t < simulation["horizon"]:
                lines.append(f"t = {t}: {simulation['average_losses'][t]:.6e}")
        lines.append(f"Final weighted average loss = {simulation['final_average_loss']:.6e}")
        lines.append(f"Maximum weighted average loss = {average_loss_summary['max_value']:.6e} at t = {average_loss_summary['max_time']}")
        lines.append(f"Minimum weighted average loss = {average_loss_summary['min_value']:.6e} at t = {average_loss_summary['min_time']}")
        lines.append("Significance: This records the current average loss against which each expert is being compared.")
        lines.append("")

        ct_errors = simulation["continuous_time_errors"]
        lines.append("Continuous-time comparison error")
        for t in times:
            if t < simulation["horizon"]:
                lines.append(f"t = {t}: {ct_errors[t]:.6e}")
        lines.append(f"Mean continuous-time comparison error = {continuous_time_summary['mean']:.6e}")
        lines.append(f"Maximum continuous-time comparison error = {continuous_time_summary['max_value']:.6e} at t = {continuous_time_summary['max_time']}")
        lines.append(f"Minimum continuous-time comparison error = {continuous_time_summary['min_value']:.6e} at t = {continuous_time_summary['min_time']}")
        lines.append("Significance: Smaller values indicate closer agreement with the continuous-time mechanism.")
        lines.append("")

        lines.append(f"Tracking accuracy = {diagnostic['tracking_accuracy']:.6f}")
        lines.append("Significance: This is the fraction of time steps at which the algorithm assigns at least as much weight to the currently lower-loss expert as to its alternating competitor. In the symmetric periodic construction this number can remain near 0.5 even when the soft tracking measure is informative.")
        lines.append("")

        lines.append(f"Mean preferred-expert share = {diagnostic.get('soft_tracking_accuracy', diagnostic['tracking_accuracy']):.6f}")
        lines.append("Significance: This averages the share of weight placed on the currently better expert among the first two experts. It is often more informative than the hard accuracy when the leader does not actually switch.")
        lines.append("")

        lines.append(f"Average preferred-weight margin = {diagnostic['average_margin']:.6e}")
        lines.append("Significance: This measures how strongly the algorithm favors the currently better expert on average.")
        lines.append("")

        lines.append(f"Maximum preferred-weight margin = {diagnostic['maximum_margin']:.6e} at t = {diagnostic['maximum_margin_time']}")
        lines.append(f"Minimum preferred-weight margin = {diagnostic['minimum_margin']:.6e} at t = {diagnostic['minimum_margin_time']}")
        lines.append("Significance: The maximum and minimum margins identify the strongest and weakest moments of tracking. A negative minimum means that at some time the algorithm favored the currently worse expert.")
        lines.append("")

        lines.append(f"Average step size = {diagnostic['average_step']:.6e}")
        lines.append(f"Maximum step size = {diagnostic['maximum_step']:.6e} at t = {diagnostic['maximum_step_time']}")
        lines.append(f"Minimum step size = {diagnostic['minimum_step']:.6e} at t = {diagnostic['minimum_step_time']}")
        lines.append("Significance: Step sizes measure how sharply the weight vector changes from one round to the next. Larger values indicate more abrupt reweighting.")
        lines.append("")
        lines.append("-" * 72)
        lines.append("")

    return "\n".join(lines)


# Combine all report sections.


def full_report(all_results):
    parts = []
    parts.append(stationary_report(all_results["stationary"]))
    parts.append(periodic_report(all_results["periodic"]))
    return "\n\n".join(parts)


# Write the report to a text file when executed in the cluster.


def write_report(all_results, path="hedge_results.txt"):
    report_text = full_report(all_results)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    return path
