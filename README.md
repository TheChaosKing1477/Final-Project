# Final-Project

To run this project, choose whether you want to execute the **Hedge pipeline**, the **AdaBoost extension**, or both. The Hedge analysis is the main body of the project. AdaBoost is included as a smaller extension that studies a related multiplicative-weights update on training-example weights.

This repository contains:

- a full **Hedge** simulation and verification pipeline,
- a separate **AdaBoost** extension,
- the specific run files for each pipeline,
- Jupyter notebooks for interactive inspection,
- text reports,
- saved figures,
- and detailed computational diagnostics aligned with the analytical report.

---

## Overview

This project has two connected computational components.

### 1. Hedge pipeline

The main pipeline studies the discrete Hedge update under two regimes:

1. **Stationary losses**, where the same loss vector is used at every time step.
2. **Periodic structured losses**, where the relative quality of the first two experts changes over time.

The Hedge code is organized so that you can:

- run the simulations,
- compute analytical and numerical stability diagnostics,
- write text reports,
- save all figures automatically,
- and inspect the results in either a Python script or a Jupyter notebook.

The central Hedge map is:

`F_i(w) = w_i exp(-eta * ell_i) / sum_j [w_j exp(-eta * ell_j)]`

The weights always remain on the simplex:

`Delta^n = { w in R^n : w_i >= 0 and sum_i w_i = 1 }`

### 2. AdaBoost extension

The AdaBoost code is intentionally separate from the Hedge pipeline. The main Hedge analysis is the core of the project, while AdaBoost is treated as a smaller extension that studies a related multiplicative-weights update on training examples rather than on experts.

The AdaBoost code is organized so that you can:

- generate synthetic datasets,
- fit weighted decision stumps,
- run AdaBoost for a specified number of rounds,
- write text reports,
- save AdaBoost diagnostic figures,
- and inspect the results in either a Python script or a Jupyter notebook.

---

## Repository structure

### Hedge files

- `hedge_core.py`  
  This file contains the mathematical and reporting logic. It includes simplex utilities, entropy and weighted-average-loss computations, the Hedge update, stationary and periodic loss constructions, stationary Jacobian and eigenvalue calculations, numerical finite-difference Jacobian approximations on the simplex, continuous-time comparison diagnostics, stationary and periodic simulation routines, and text reports.

- `hedge_plots.py`  
  This file contains all plotting logic. It includes time-dependent trajectory plots, simplex trajectory plots, phase portraits, eta-summary plots, eta-panel subplot figures, and batch routines that save all requested figures to disk.

- `hedge_run.py`  
  This is the code that runs the Hedge pipeline. It locates the project folder, runs all simulations, writes the text report, saves all figures, and prints a short console summary.

- `hedge_run.ipynb`  
  This notebook provides an interactive way to run the pipeline. It locates the project folder, lets the user set a horizon `T`, runs all simulations, writes the report, saves all figures, and displays chosen saved figures. A subset of representative figures are presented by default.

- `Hedge_README.md`  
  This file contains additional Hedge-specific details on how to run the Hedge code and how to interpret the Hedge outputs.

### AdaBoost files

- `adaboost_core.py`  
  This file contains the mathematical and computational logic for the AdaBoost extension. It includes synthetic dataset generation, weighted decision-stump fitting, AdaBoost sample-weight updates, edge computation, entropy and effective-sample-size diagnostics, margin computation, hard-example-mass diagnostics, and compact text reports.

- `adaboost_run.py`  
  This is the code that runs the AdaBoost pipeline. It locates the project folder automatically, runs the default AdaBoost experiments, writes the AdaBoost report to disk, saves the AdaBoost diagnostic figures, and prints a report preview to the terminal.

- `adaboost_run.ipynb`  
  This is the interactive notebook for the AdaBoost extension. It locates the project folder, runs sanity checks, runs the default experiments, writes the AdaBoost report, and saves and displays diagnostic plots.

- `AdaBoost_README.md`  
  This file contains additional AdaBoost-specific details on how to run the AdaBoost code and how to interpret the AdaBoost outputs.

---

## How to run the project on Adroit

Run the respective pipeline through the Adroit cluster using the included `job.slurm` files.

The pipeline/job naming convention is:

- **`adaboost`** for the AdaBoost case,
- **`plot_only_150`** for a figure-only Hedge job with `T = 150`,
- **`report_only_150`** for a report-only Hedge job,
- **`default_50`** for the Hedge run with `T = 50` that writes both figures and reports.

Submit the corresponding configuration through Adroit with `sbatch job.slurm`. The attached `job.slurm` configurations are intended to cover those cases.

The two main pipeline files are:

- `hedge_run.py` for Hedge,
- `adaboost_run.py` for AdaBoost.

The notebook files are for interactive inspection and verification, not the primary cluster workflow.

---

## Hedge pipeline: core metrics and functions

### Shannon entropy

**Function:**
- `entropy(w)`

**Formula:** `H(w) = -sum_i w_i log w_i`

**Interpretation:**
- larger entropy means the weight is more spread out,
- smaller entropy means the weight is more concentrated.

### Weighted average loss

**Function:**
- `weighted_average_loss(w, ell)`

**Formula:** `<w, ell> = sum_i w_i ell_i`

**Interpretation:**
- in the stationary case, this is the quantity that appears in the continuous-time limit,
- it measures the current average loss induced by the weight vector.

### Hedge update

**Function:**
- `hedge_update(w, ell, eta)`

**Formula:** `F_i(w) = w_i exp(-eta * ell_i) / sum_j [w_j exp(-eta * ell_j)]`

**Interpretation:**
- experts with lower loss receive relatively more weight,
- the learning rate `eta` controls how aggressively the reweighting occurs.

### Analytical stationary Jacobian

**Function:**
- `analytic_jacobian_stationary(w, ell, eta)`

**Formula:**

`dF_i/dw_k = [delta_ik exp(-eta ell_i) sum_j w_j exp(-eta ell_j) - w_i exp(-eta ell_i) exp(-eta ell_k)] / [sum_j w_j exp(-eta ell_j)]^2`

### Numerical Jacobian on the simplex

**Function:**
- `numerical_jacobian_tangent(w, ell, eta, h=1.0e-7)`

**Interpretation:**
- this approximates the Jacobian using simplex-compatible tangent perturbations,
- it uses a boundary-safe one-sided or central difference when needed.

### Spectral radius

**Function:**
- `spectral_radius(matrix)`

**Formula:** `rho(J) = max_i |lambda_i|`

**Interpretation:**
- `rho(J) < 1` suggests contraction in the relevant directions,
- `rho(J) > 1` suggests linear instability,
- `rho(J) = 1` indicates at least one neutral direction.

### One-step linearization error

**Function:**
- `one_step_linearization_error(w_star, ell, eta, v, epsilon)`

**Formula:**

`||F(w* + epsilon v) - F(w*) - J(w*)(epsilon v)|| / ||epsilon v||`

**Interpretation:**
- smaller values mean the local linearization accurately describes one nonlinear step near the fixed point.

### Continuous-time comparison error

**Function:**
- `continuous_time_comparison_errors(weights, losses, eta)`

**Continuous-time model:**

`dw_i / d tau = w_i ( <w, ell> - ell_i )`

**Discrete comparison:**

`(w_{t+1} - w_t) / eta`

versus

`w_t ( <w_t, ell_t> - ell_t )`

**Interpretation:**
- smaller values mean the discrete dynamics more closely match the continuous-time approximation,
- this is most meaningful for small `eta`.

### Periodic tracking diagnostics

**Function:**
- `tracking_diagnostic_periodic(simulation)`

**Computed quantities:**
- hard tracking accuracy,
- mean preferred-expert share,
- preferred-weight margin,
- step sizes.

**Interpretation:**
- hard tracking accuracy is a strict switching diagnostic,
- mean preferred-expert share is usually the more informative soft diagnostic.

### Stationary support-preference diagnostics

**Function:**
- `stationary_preference_diagnostic(case, simulation)`

**Computed quantities:**
- mean tracking accuracy,
- mean preferred margin,
- min/max preferred margin with times.

**Interpretation:**
- in the unique-minimum case, these measure dominance of the single best expert,
- in the equal-minimum case, they measure support concentration on the equal-loss face.

---

## Hedge pipeline: default cases

### `unique_minimum`

**Loss vector:** `(0, 1, 2)`

**Initial weight:** `(1/3, 1/3, 1/3)`

**Interpretation:**
- expert 1 is uniquely best,
- one expects convergence toward the vertex `e1`,
- the vertices `e2` and `e3` are unstable.

### `equal_minimum_pair`

**Loss vector:** `(0, 0, 1)`

**Initial weight:** `(0.2, 0.8, 0.0)`

**Interpretation:**
- experts 1 and 2 share the same minimum loss,
- the full fixed-point set is the edge `w3 = 0`,
- the code evaluates representative points on that edge.

### `all_equal`

**Loss vector:** `(1, 1, 1)`

**Initial weight:** `(0.2, 0.3, 0.5)`

**Interpretation:**
- all points are fixed points,
- the Hedge map is the identity,
- the trajectory should remain unchanged.

### Periodic case

**Losses vary periodically with default period `20`:**

`ell_t = (0.5 + 0.4 sin(2 pi t / P), 0.5 - 0.4 sin(2 pi t / P), 1.0)`

**Interpretation:**
- experts 1 and 2 alternate in relative quality,
- expert 3 remains uniformly worse.

---

## Hedge pipeline: default eta grid and horizon

### Default eta grid

Shared eta values:
- `0.02`
- `0.05`
- `0.10`
- `0.20`
- `0.50`
- `1.00`
- `2.00`

By default, the stationary and periodic regimes use the same eta grid.

### Default horizon

The default horizon is:

- `DEFAULT_HORIZON = 50`

---

## Hedge saved figures and what they measure

The Hedge plotting code writes a large set of diagnostic figures. The list below explains not only which plots are saved, but also what each plotted quantity measures and how it should be interpreted.

### Stationary figures

#### Time-dependent figures for each stationary case and each `eta`

- **Weight trajectories**  
  These plot the coordinates of the weight vector `w_t` against time `t`. They show how mass is redistributed among the experts over time.

- **Entropy trajectories**  
  These plot the Shannon entropy `H(w_t) = -sum_i w_{t,i} log w_{t,i}`. They measure how spread out or concentrated the weights are. Lower entropy means stronger concentration on fewer experts.

- **Weighted average loss trajectories**  
  These plot `<w_t, ell>` in the stationary regime. They show the loss currently induced by the weight vector and connect directly to the continuous-time limit.

- **Continuous-time comparison error trajectories**  
  These plot the discrepancy between the discrete Hedge update and the corresponding continuous-time approximation. Smaller values mean closer agreement.

- **Simplex trajectories**  
  These show the full geometric path of `w_t` in the simplex. They are useful for seeing whether trajectories move toward a vertex, remain on an edge, or stay fixed.

- **Phase portraits**  
  These show the vector field induced by one Hedge step across the simplex. They visualize local attraction, repulsion, and neutral directions.

- **Preferred-support margin trajectories**, when relevant  
  These plot the weight advantage of the lower-loss side over its competitor. Positive values mean the lower-loss side carries more mass; negative values mean the opposite.

- **Lower-loss-expert share / indicator trajectories**, when relevant  
  These record either the share of weight placed on the lower-loss side or a binary indicator of whether that side currently carries the larger weight.

#### Across-`eta` stationary figures

- **Weight `eta` panels**
- **Simplex `eta` panels**
- **Phase `eta` panels**
- **Entropy `eta` panels**
- **Average-loss `eta` panels**
- **Continuous-time-error `eta` panels**
- **Preferred-margin `eta` panels**, when relevant
- **Lower-loss-expert share / indicator `eta` panels**, when relevant
- **Spectral-radius panels by representative fixed point**
- **Spectral radius vs `eta`**
- **Jacobian difference norm vs `eta`**
- **Eigenvalue discrepancy vs `eta`**
- **Linearization error vs `eta`**
- **Average step size vs `eta`**
- **Final entropy vs `eta`**
- **Final weighted average loss vs `eta`**
- **Mean continuous-time comparison error vs `eta`**

### Periodic figures

#### Time-dependent figures for each `eta`

- **Weight trajectories**
- **Entropy trajectories**
- **Weighted average loss trajectories**
- **Continuous-time comparison error trajectories**
- **Preferred-weight margin trajectories**
- **Step-size trajectories**
- **Simplex trajectories**

#### Across-`eta` periodic figures

- **Weight `eta` panels**
- **Simplex `eta` panels**
- **Preferred-weight margin `eta` panels**
- **Step-size `eta` panels**
- **Entropy `eta` panels**
- **Average-loss `eta` panels**
- **Continuous-time-error `eta` panels**
- **Hard tracking accuracy vs `eta`**
- **Mean preferred-expert share vs `eta`**
- **Average step size vs `eta`**
- **Final entropy vs `eta`**
- **Final weighted average loss vs `eta`**

### How to interpret the main Hedge plot quantities

- **Weight trajectory:** the sequence of expert weights `w_t` over time.
- **Simplex trajectory:** the same sequence viewed geometrically in the simplex.
- **Phase portrait:** the local update direction across the simplex.
- **Spectral radius:** the modulus of the largest eigenvalue of the Jacobian.
- **Jacobian difference:** the discrepancy between the analytical and numerical Jacobian.
- **Eigenvalue discrepancy:** the discrepancy between analytically predicted and numerically computed eigenvalues.
- **Linearization error:** the difference between the true one-step update and its Jacobian-based linear approximation.
- **Continuous-time error:** the discrepancy between the discrete Hedge update and the corresponding continuous-time approximation.
- **Preferred margin / preferred-weight margin:** the signed weight advantage of the lower-loss side over its competitor. The sign indicates which side dominates; the magnitude measures how strong that dominance is.
- **Preferred-expert share:** the weight assigned to the currently lower-loss expert, averaged over time in the periodic setting.
- **Step size:** the size of the one-step reweighting move, measured by `||w_{t+1} - w_t||`.

---

## How to use the Hedge notebook effectively

The notebook includes a `subplot_only` toggle.

Set `subplot_only = True` to display only eta-panel / subplot figures.

Set `subplot_only = False` to additionally display a smaller set of single representative figures.

A useful first-pass Hedge figure review is:

1. `unique_minimum_weight_eta_panel.png`
2. `unique_minimum_simplex_eta_panel.png`
3. `unique_minimum_spectral_radius_panel.png`
4. `equal_minimum_pair_simplex_eta_panel.png`
5. `all_equal_weight_eta_panel.png`
6. `periodic_weight_eta_panel.png`
7. `periodic_margin_eta_panel.png`
8. `periodic_preferred_share_summary.png`

---

## Hedge interpretation cautions

The periodic regime should be interpreted carefully.

1. **Hard tracking accuracy should not be overinterpreted.**  
   In the symmetric periodic setup it can stay near `0.5` even when the soft tracking response is meaningful.

2. **Continuous-time comparison error should not be described as strictly monotone in eta across whole trajectories.**  
   The small-eta theory is local; full-trajectory empirical averages can reflect transient and saturation effects.

So the most reliable periodic conclusion is:

- larger `eta` generally produces sharper and more oscillatory reweighting,
- but not necessarily better tracking in every diagnostic sense.

---

## AdaBoost extension: datasets, metrics, and outputs

### Default datasets

The AdaBoost extension runs on three synthetic datasets:

#### `gaussian_easy`

Two well-separated Gaussian classes in `R^2`.

**Expected behavior:**
- strong positive weak-learner edge,
- rapid drop in training error,
- strong concentration of sample weights.

#### `gaussian_overlap`

Two overlapping Gaussian classes in `R^2`.

**Expected behavior:**
- smaller weak-learner edge,
- slower error reduction,
- less extreme weight concentration,
- persistent hard-example mass.

#### `xor_checkerboard`

An XOR-style checkerboard dataset in `R^2`.

**Why it is preferable here:**
- it is not solvable by a single axis-aligned stump,
- AdaBoost has to combine several simple rules to improve performance,
- the extension remains informative without immediate saturation,
- the resulting edge, margin, and concentration diagnostics are easier to interpret as a genuine boosting phenomenon.

### Main AdaBoost quantities

#### Weighted weak-learner error

This is the weighted classification error of the current decision stump under the current sample weights.

#### Weak-learner edge

**Formula:** `gamma_t = 1/2 - epsilon_t`

where `epsilon_t` is the weighted weak-learner error.

**Interpretation:**
- positive edge means the weak learner performs better than random,
- larger edge produces a larger AdaBoost coefficient.

#### AdaBoost coefficient

**Formula:** `alpha_t = (1/2) log((1 - epsilon_t) / epsilon_t)`

**Interpretation:**
- better weak learners receive larger coefficients in the ensemble.

#### Entropy of sample weights

This measures concentration of mass on the training examples.

**Interpretation:**
- lower entropy means the sample weights are concentrating on a smaller set of examples.

#### Effective sample size

**Formula:** `1 / sum_i D_t(i)^2`

**Interpretation:**
- smaller effective sample size means the weights are concentrating more strongly.

#### Hard-example mass

This is the total sample weight assigned to examples with nonpositive current margin.

**Interpretation:**
- larger values mean the algorithm is still focusing substantial mass on examples that are currently hard to classify.

#### Margins

**Formula:** `y_i F_t(x_i)`

where `F_t` is the current ensemble score.

**Interpretation:**
- positive margins correspond to correct classification with confidence,
- negative margins correspond to misclassification,
- growth in the minimum margin suggests improvement on the hardest examples.

### AdaBoost output interpretation

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

### AdaBoost saved figures

The standalone AdaBoost pipeline saves four figures for each default dataset:

- **weighted weak-learner error by round**  
  Shows whether the stump remains a valid weak learner over time.

- **entropy of the sample-weight distribution by round**  
  Shows how strongly the sample weights concentrate.

- **hard-example mass by round**  
  Shows how much weight remains on examples with nonpositive margin.

- **minimum sample margin by round**  
  Shows whether the worst-classified examples are improving.

---

## Output locations

### Hedge outputs

After a successful Hedge run:
- report: `results/hedge_results.txt`
- figures: `results/figures/`

### AdaBoost outputs

After a successful AdaBoost run:
- report: `results_adaboost/adaboost_results.txt`
- figures: `results_adaboost/figures/`

---

## Recommended workflow

### Full main analysis

1. Run the Hedge pipeline.
2. Inspect the Hedge report and representative Hedge figures.
3. Use the Hedge notebook if you want interactive figure review.

### AdaBoost extension

1. Run the AdaBoost pipeline.
2. Inspect the AdaBoost report and figures.
3. Use the AdaBoost notebook if you want sanity tests and interactive plots.

### Full repository workflow

1. Run the Hedge pipeline through the relevant Hedge run file on Adroit.
2. Run the AdaBoost pipeline through the AdaBoost run file on Adroit.
3. Review `results/` and `results_adaboost/`.
4. Use the notebooks if you want interactive verification after the main pipeline outputs are written.

---

## Additional detailed readmes

This file is the repository-level entry point.

For additional specific details on how to run each component and how to interpret its outputs, see:

- `Hedge_README.md`
- `AdaBoost_README.md`

---

## Final note

The Hedge pipeline is the main form of analysis in this project. The AdaBoost component is a smaller extension designed to illustrate how a related multiplicative-weights perspective appears in boosting. Together, the repository is intended to function as a reproducible computational project and as a numerical analogue of the analytical expectations described in the report.
