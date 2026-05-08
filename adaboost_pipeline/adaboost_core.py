import math
from pathlib import Path

import numpy as np


# Default random seed used for synthetic datasets.
default_seed = 7


# Return a normalized probability vector.
def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)

    if total <= 0:
        raise ValueError("Weight vector must have positive total mass.")

    return weights / total


# Return Shannon entropy H(w) = - sum_i w_i log w_i for a sample-weight vector.
def entropy(weights):
    weights = normalize_weights(weights)
    positive = weights[weights > 0]

    if positive.size == 0:
        return 0.0

    return float(-np.sum(positive * np.log(positive)))


# Return the effective sample size 1 / sum_i w_i^2.
# This is a concentration diagnostic.
def effective_sample_size(weights):
    weights = normalize_weights(weights)
    return float(1.0 / np.sum(weights ** 2))


# Return the weighted classification error.
def weighted_error(weights, y_true, y_pred):
    weights = normalize_weights(weights)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mistakes = (y_true != y_pred).astype(float)
    return float(np.sum(weights * mistakes))


# Return the edge gamma = 1/2 - epsilon for a weak learner with weighted error epsilon.
def weak_learner_edge(error):
    return float(0.5 - error)


# Return sample margins y_i F_t(x_i).
def margin_values(y, score_values):
    y = np.asarray(y, dtype=float)
    score_values = np.asarray(score_values, dtype=float)
    return y * score_values


# Return a summary dictionary for an array-valued time series.
def sequence_summary(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return {
            "mean": 0.0,
            "min_value": 0.0,
            "min_time": 0,
            "max_value": 0.0,
            "max_time": 0,
        }

    return {
        "mean": float(np.mean(values)),
        "min_value": float(np.min(values)),
        "min_time": int(np.argmin(values)),
        "max_value": float(np.max(values)),
        "max_time": int(np.argmax(values)),
    }


# Generate a two-class Gaussian dataset in R^2.
# Labels are encoded as {-1, +1}, which is the standard AdaBoost convention.
def make_gaussian_dataset(
    n_per_class=80,
    class_1_mean=(-1.0, -1.0),
    class_2_mean=(1.0, 1.0),
    covariance=None,
    seed=default_seed,
):
    rng = np.random.default_rng(seed)

    if covariance is None:
        covariance = np.array([[0.55, 0.0], [0.0, 0.55]], dtype=float)
    else:
        covariance = np.asarray(covariance, dtype=float)

    x_neg = rng.multivariate_normal(np.asarray(class_1_mean, dtype=float), covariance, size=n_per_class)
    x_pos = rng.multivariate_normal(np.asarray(class_2_mean, dtype=float), covariance, size=n_per_class)

    y_neg = -np.ones(n_per_class, dtype=float)
    y_pos = np.ones(n_per_class, dtype=float)

    X = np.vstack([x_neg, x_pos])
    y = np.concatenate([y_neg, y_pos])

    return X, y


# Generate a two-moons-style synthetic dataset without external dependencies.
# This two-moon dataset was determined to be too simple as it saturates
# too quickly under the Ada Boost decision-stump weak learner.
def make_two_moons_dataset(n_per_class=80, noise=0.12, seed=default_seed):
    rng = np.random.default_rng(seed)

    angles = rng.uniform(0.0, math.pi, size=n_per_class)

    moon_1 = np.column_stack([np.cos(angles), np.sin(angles)])
    moon_2 = np.column_stack([1.0 - np.cos(angles), -np.sin(angles) - 0.5])

    moon_1 += noise * rng.normal(size=moon_1.shape)
    moon_2 += noise * rng.normal(size=moon_2.shape)

    y_1 = np.ones(n_per_class, dtype=float)
    y_2 = -np.ones(n_per_class, dtype=float)

    X = np.vstack([moon_1, moon_2])
    y = np.concatenate([y_1, y_2])

    return X, y


# Generate an XOR-style checkerboard dataset in R^2.
# Labels are determined by the signs of the coordinates after adding noise.
# This is the favorable alternative to two-moon.
def make_xor_checkerboard_dataset(n_per_quadrant=45, noise=0.22, seed=default_seed):
    rng = np.random.default_rng(seed)

    centers = [
        (-1.0, -1.0, 1.0),
        (-1.0,  1.0, -1.0),
        ( 1.0, -1.0, -1.0),
        ( 1.0,  1.0, 1.0),
    ]

    X_blocks = []
    y_blocks = []

    for cx, cy, label in centers:
        block = np.column_stack(
            [
                cx + noise * rng.normal(size=n_per_quadrant),
                cy + noise * rng.normal(size=n_per_quadrant),
            ]
        )
        X_blocks.append(block)
        y_blocks.append(np.full(n_per_quadrant, label, dtype=float))

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)

    return X, y


# Return a dictionary of default synthetic datasets used in the extension section.
def default_datasets():
    gaussian_easy = make_gaussian_dataset(
        n_per_class=80,
        class_1_mean=(-1.4, -1.0),
        class_2_mean=(1.4, 1.0),
        covariance=np.array([[0.45, 0.0], [0.0, 0.45]], dtype=float),
        seed=default_seed,
    )

    gaussian_overlap = make_gaussian_dataset(
        n_per_class=80,
        class_1_mean=(-0.8, -0.8),
        class_2_mean=(0.8, 0.8),
        covariance=np.array([[0.75, 0.15], [0.15, 0.75]], dtype=float),
        seed=default_seed + 1,
    )

    xor_checkerboard = make_xor_checkerboard_dataset(
        n_per_quadrant=45,
        noise=0.22,
        seed=default_seed + 2,
    )

    return {
        "gaussian_easy": {
            "X": gaussian_easy[0],
            "y": gaussian_easy[1],
            "description": "Two well-separated Gaussian classes in R^2.",
        },
        "gaussian_overlap": {
            "X": gaussian_overlap[0],
            "y": gaussian_overlap[1],
            "description": "Two overlapping Gaussian classes in R^2.",
        },
        "xor_checkerboard": {
            "X": xor_checkerboard[0],
            "y": xor_checkerboard[1],
            "description": "An XOR-style checkerboard dataset in R^2.",
        },
    }


# Predict with a one-dimensional decision stump.
# The rule is: if x_j <= threshold then left_label else right_label.
def stump_predict(X, feature_index, threshold, left_label, right_label):
    X = np.asarray(X, dtype=float)
    prediction = np.where(X[:, feature_index] <= threshold, left_label, right_label)
    return prediction.astype(float)


# Fit a weighted decision stump.
# The candidate thresholds are midpoints between sorted distinct feature values.
def fit_decision_stump(X, y, weights):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = normalize_weights(weights)

    n_samples, n_features = X.shape

    best = {
        "feature_index": None,
        "threshold": None,
        "left_label": None,
        "right_label": None,
        "predictions": None,
        "weighted_error": np.inf,
        "edge": None,
    }

    for feature_index in range(n_features):
        feature_values = X[:, feature_index]
        unique_values = np.unique(feature_values)

        if unique_values.size == 1:
            thresholds = unique_values
        else:
            thresholds = 0.5 * (unique_values[:-1] + unique_values[1:])

        for threshold in thresholds:
            for left_label, right_label in [(-1.0, 1.0), (1.0, -1.0)]:
                predictions = stump_predict(X, feature_index, threshold, left_label, right_label)
                error = weighted_error(weights, y, predictions)

                if error < best["weighted_error"]:
                    best = {
                        "feature_index": int(feature_index),
                        "threshold": float(threshold),
                        "left_label": float(left_label),
                        "right_label": float(right_label),
                        "predictions": predictions,
                        "weighted_error": float(error),
                        "edge": weak_learner_edge(error),
                    }

    if best["feature_index"] is None:
        raise RuntimeError("Decision stump fitting failed.")

    return best


# Apply one AdaBoost weight update:
# D_{t+1}(i) = D_t(i) exp(- alpha_t y_i h_t(x_i)) / Z_t.
def adaboost_weight_update(weights, y, predictions, alpha):
    weights = normalize_weights(weights)
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    updated = weights * np.exp(-alpha * y * predictions)
    normalization = float(np.sum(updated))

    if normalization <= 0:
        raise ValueError("Normalization constant must be positive in the AdaBoost update.")

    return updated / normalization, normalization


# Run AdaBoost with decision stumps on a fixed dataset.
# This records the following quantities:
# edge, margins, entropy, effective sample size, and concentration on hard examples.
def run_adaboost(X, y, rounds=25):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("AdaBoost labels must be encoded as -1 and +1.")

    n_samples = X.shape[0]
    weights = np.full(n_samples, 1.0 / n_samples, dtype=float)

    stumps = []
    alpha_values = []
    weight_history = [weights.copy()]
    entropy_history = [entropy(weights)]
    effective_sample_size_history = [effective_sample_size(weights)]
    normalization_constants = []
    weighted_errors = []
    edge_history = []
    hard_example_mass_history = []

    score_history = [np.zeros(n_samples, dtype=float)]
    margin_history = [margin_values(y, score_history[-1])]
    training_error_history = [float(np.mean(np.sign(np.where(score_history[-1] == 0.0, 1.0, score_history[-1])) != y))]

    for _ in range(rounds):
        stump = fit_decision_stump(X, y, weights)
        error = min(max(stump["weighted_error"], 1.0e-12), 1.0 - 1.0e-12)
        alpha = 0.5 * math.log((1.0 - error) / error)

        updated_weights, normalization = adaboost_weight_update(weights, y, stump["predictions"], alpha)

        new_scores = score_history[-1] + alpha * stump["predictions"]
        new_margins = margin_values(y, new_scores)
        hard_mass = float(np.sum(updated_weights[new_margins <= 0.0]))

        stumps.append(stump)
        alpha_values.append(float(alpha))
        normalization_constants.append(float(normalization))
        weighted_errors.append(float(error))
        edge_history.append(float(weak_learner_edge(error)))
        hard_example_mass_history.append(hard_mass)

        weights = updated_weights

        weight_history.append(weights.copy())
        entropy_history.append(entropy(weights))
        effective_sample_size_history.append(effective_sample_size(weights))
        score_history.append(new_scores.copy())
        margin_history.append(new_margins.copy())

        classifier = np.sign(np.where(new_scores == 0.0, 1.0, new_scores))
        training_error_history.append(float(np.mean(classifier != y)))

    return {
        "X": X,
        "y": y,
        "rounds": int(rounds),
        "stumps": stumps,
        "alpha_values": np.asarray(alpha_values, dtype=float),
        "weight_history": np.asarray(weight_history, dtype=float),
        "entropy_history": np.asarray(entropy_history, dtype=float),
        "effective_sample_size_history": np.asarray(effective_sample_size_history, dtype=float),
        "normalization_constants": np.asarray(normalization_constants, dtype=float),
        "weighted_errors": np.asarray(weighted_errors, dtype=float),
        "edge_history": np.asarray(edge_history, dtype=float),
        "hard_example_mass_history": np.asarray(hard_example_mass_history, dtype=float),
        "score_history": np.asarray(score_history, dtype=float),
        "margin_history": np.asarray(margin_history, dtype=float),
        "training_error_history": np.asarray(training_error_history, dtype=float),
    }


# Run AdaBoost on the default collection of synthetic datasets.
def run_default_adaboost_experiments(rounds=25):
    results = {}

    for name, dataset in default_datasets().items():
        results[name] = {
            "description": dataset["description"],
            "result": run_adaboost(dataset["X"], dataset["y"], rounds=rounds),
        }

    return results


# Format a vector for text output.
def format_vector(values, decimals=6):
    values = np.asarray(values, dtype=float)
    return "[" + ", ".join(f"{x:.{decimals}f}" for x in values) + "]"


# This generates a compact text report for the AdaBoost extension.
# Note: The AdaBoost code is intended to be a minor relative to
# the main Hedge report.
def adaboost_report(results):
    lines = []
    lines.append("AdaBoost extension")
    lines.append("")

    for name, payload in results.items():
        result = payload["result"]

        lines.append(f"Dataset: {name}")
        lines.append(payload["description"])
        lines.append(f"Rounds = {result['rounds']}")
        lines.append("")

        edge_summary = sequence_summary(result["edge_history"])
        entropy_summary = sequence_summary(result["entropy_history"])
        hard_mass_summary = sequence_summary(result["hard_example_mass_history"])
        training_error_summary = sequence_summary(result["training_error_history"])
        min_margin_summary = sequence_summary(np.min(result["margin_history"], axis=1))

        lines.append(f"Final weighted weak-learner error = {result['weighted_errors'][-1]:.6e}")
        lines.append(f"Final weak-learner edge = {result['edge_history'][-1]:.6e}")
        lines.append(f"Final alpha = {result['alpha_values'][-1]:.6e}")
        lines.append(f"Final training error = {result['training_error_history'][-1]:.6e}")
        lines.append(f"Final entropy of sample weights = {result['entropy_history'][-1]:.6e}")
        lines.append(f"Final effective sample size = {result['effective_sample_size_history'][-1]:.6e}")
        lines.append(f"Final hard-example mass = {result['hard_example_mass_history'][-1]:.6e}")
        lines.append("")

        lines.append(f"Mean edge = {edge_summary['mean']:.6e}")
        lines.append(f"Minimum edge = {edge_summary['min_value']:.6e} at round {edge_summary['min_time'] + 1}")
        lines.append(f"Maximum edge = {edge_summary['max_value']:.6e} at round {edge_summary['max_time'] + 1}")
        lines.append("")

        lines.append(f"Minimum entropy = {entropy_summary['min_value']:.6e} at round {entropy_summary['min_time']}")
        lines.append(f"Maximum entropy = {entropy_summary['max_value']:.6e} at round {entropy_summary['max_time']}")
        lines.append(f"Maximum hard-example mass = {hard_mass_summary['max_value']:.6e} at round {hard_mass_summary['max_time'] + 1}")
        lines.append(f"Minimum training error = {training_error_summary['min_value']:.6e} at round {training_error_summary['min_time']}")
        lines.append(f"Minimum sample margin = {min_margin_summary['min_value']:.6e} at round {min_margin_summary['min_time']}")
        lines.append("")

        lines.append("Interpretation:")
        lines.append("The edge sequence records whether each weak learner performs better than random under the current sample weights.")
        lines.append("The entropy and effective sample size track concentration of weight on difficult training examples.")
        lines.append("The hard-example mass measures how much sample weight remains on examples with nonpositive margin.")
        lines.append("The minimum-margin sequence tracks whether the combined classifier is improving the worst classified examples.")
        lines.append("")
        lines.append("-" * 72)
        lines.append("")

    return "\n".join(lines)


# Write the AdaBoost report to disk.
def write_adaboost_report(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(adaboost_report(results), encoding="utf-8")
    return path
