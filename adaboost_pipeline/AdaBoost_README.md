# AdaBoost Read-Me

## Overview

This file explains how to use the AdaBoost code.

The AdaBoost code is intentionally separate from the Hedge pipeline. The main Hedge analysis is the core of the project, while AdaBoost is treated as a smaller extension that studies a related multiplicative-weights update on training examples rather than on experts.

The AdaBoost extension is implemented in:

- `adaboost_core.py`
- `adaboost_run.py`
- `adaboost_run.ipynb`

---

## File roles

### `adaboost_core.py`

This file contains the mathematical and computational logic for the AdaBoost extension.

It includes:

- synthetic dataset generation,
- weighted decision-stump fitting,
- AdaBoost sample-weight updates,
- edge computation,
- entropy and effective-sample-size diagnostics,
- margin computation,
- hard-example-mass diagnostics,
- compact text report generation.

### `adaboost_run.py`

This is the command-line driver for the AdaBoost extension.

It:

- locates the project folder automatically,
- runs the default AdaBoost experiments,
- writes the AdaBoost report to disk,
- saves the AdaBoost diagnostic figures,
- prints a report preview to the terminal.

### `adaboost_run.ipynb`

This is the interactive notebook for the AdaBoost extension.

It:

- locates the project folder,
- runs sanity checks,
- runs the default experiments,
- writes the AdaBoost report,
- saves and displays diagnostic plots.

---

## How to run AdaBoost

### Command-line use

From the project folder:

```
python adaboost_run.py
```

This runs the default experiments for `25` boosting rounds and writes:

- `results_adaboost/adaboost_results.txt`
- diagnostic figures under `results_adaboost/figures/`

To choose a different number of boosting rounds:

```
python adaboost_run.py 40
```

To choose a different output directory:

```
python adaboost_run.py 40 --output-dir my_adaboost_results
```

### Notebook use

Open `adaboost_run.ipynb`, then:

1. run the setup cell,
2. set the number of boosting rounds,
3. run the experiment cell,
4. inspect the report preview and diagnostic plots.

---

## Datasets used by default

The current AdaBoost extension runs on three synthetic datasets:

### `gaussian_easy`

Two well-separated Gaussian classes in `R^2`.

Expected behavior:
- strong positive weak-learner edge,
- rapid drop in training error,
- strong concentration of sample weights.

### `gaussian_overlap`

Two overlapping Gaussian classes in `R^2`.

Expected behavior:
- smaller weak-learner edge,
- slower error reduction,
- less extreme weight concentration,
- persistent hard-example mass.

### `xor_checkerboard`

An XOR-style checkerboard dataset in `R^2`.

Why it is preferable here:
- it is not solvable by a single axis-aligned stump,
- AdaBoost has to combine several simple rules to improve performance,
- the extension remains informative without immediate saturation,
- the resulting edge, margin, and concentration diagnostics are easier to interpret as a genuine boosting phenomenon.

---

## Main quantities computed

### Weighted weak-learner error

This is the weighted classification error of the current decision stump under the current sample weights.

### Weak-learner edge

This is

```math
\gamma_t =
\frac{1}{2}
-
\varepsilon_t.
```

where `\varepsilon_t` is the weighted weak-learner error.

Interpretation:
- positive edge means the weak learner performs better than random,
- larger edge produces a larger AdaBoost coefficient.

### AdaBoost coefficient

This is

```math
\alpha_t =
\frac{1}{2}
\log
\frac{1-\varepsilon_t}{\varepsilon_t}.
```

Interpretation:
- better weak learners receive larger coefficients in the ensemble.

### Entropy of sample weights

This measures concentration of mass on the training examples.

Interpretation:
- lower entropy means the sample weights are concentrating on a smaller set of examples.

### Effective sample size

This is

```math
\frac{1}{\sum_i D_t(i)^2}.
```

Interpretation:
- smaller effective sample size means the weights are concentrating more strongly.

### Hard-example mass

This is the total sample weight assigned to examples with nonpositive current margin.

Interpretation:
- larger values mean the algorithm is still focusing substantial mass on examples that are currently hard to classify.

### Margins

The sample margins are

```math
y_i F_t(x_i).
```

where `F_t` is the current ensemble score.

Interpretation:
- positive margins correspond to correct classification with confidence,
- negative margins correspond to misclassification,
- growth in the minimum margin suggests improvement on the hardest examples.

---

## Output interpretation

Typical reading order:

1. weighted weak-learner error,
2. edge,
3. training error,
4. entropy / effective sample size,
5. hard-example mass,
6. minimum margin.

A coherent AdaBoost run usually looks like this:

- weighted error stays below `1/2`,
- edge stays positive,
- training error decreases,
- entropy decreases,
- effective sample size decreases,
- hard-example mass eventually shrinks,
- margins become more positive.

If the weighted error becomes essentially zero immediately and stays there, the dataset is too easy for the current weak learner and the extension is no longer very informative.

---

## Cluster

For cluster use, run:

```python adaboost_run.py
```

or

```python adaboost_run.py 40
```

`adaboost_core.py` by itself defines functions but does not execute them on import. The separate driver `adaboost_run.py` is the correct script for standalone cluster execution.

---

## Final Note

The AdaBoost extension is intended to remain minor relative to the main Hedge analysis. The cleanest use is:

- keep Hedge as the main body of the project,
- use AdaBoost only as a short extension section,
- report a small set of interpretable diagnostics,
- and prefer a dataset that does not saturate immediately under a single decision stump.
