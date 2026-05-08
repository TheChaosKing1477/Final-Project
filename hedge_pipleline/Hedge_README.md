# Hedge Project Read-Me

## Overview

This project studies the discrete Hedge update under two regimes:

1. **Stationary losses**, where the same loss vector is used at every time step.
2. **Periodic structured losses**, where the relative quality of the first two experts changes over time.

The code is organized so that I can:

- run the simulations,
- compute analytical and numerical stability diagnostics,
- write a text report,
- save all figures automatically,
- and inspect the results in either a Python script or a Jupyter notebook.

The central object is the Hedge map

```math
F_i(w) =
\frac{
w_i e^{-\eta \ell_i}
}{
\sum_{j=1}^{n} w_j e^{-\eta \ell_j}
}
```

together with the simplex constraint

```math
\Delta^n =
\left\{
w \in \mathbb{R}^n :
w_i \ge 0,\;
\sum_i w_i = 1
\right\}.
```

---

## File structure

### `hedge_core.py`

This file contains the mathematical and reporting logic.

It includes:

- simplex utilities,
- entropy and weighted-average-loss computations,
- the Hedge update,
- stationary and periodic loss constructions,
- stationary Jacobian and eigenvalue calculations,
- numerical finite-difference Jacobian approximations on the simplex,
- continuous-time comparison diagnostics,
- stationary and periodic simulation routines,
- text report generation.

### `hedge_plots.py`

This file contains all plotting logic.

It includes:

- time-dependent trajectory plots,
- simplex trajectory plots,
- phase portraits,
- eta-summary plots,
- eta-panel subplot figures,
- batch routines that save all requested figures to disk.

### `hedge_run.py`

This is the command-line driver.

It:

- locates the project folder,
- runs all simulations,
- writes the text report,
- saves all figures,
- prints a short console summary.

### `hedge_run.ipynb`

This notebook provides an interactive notebook file.

It:

- locates the project folder,
- lets the user set a horizon `T`,
- runs all simulations,
- writes the report,
- saves all figures,
- displays representative saved figures.

---

## Core metrics and their functions

### Shannon entropy

Function:
- `entropy(w)`

Formula:

```math
H(w) =
-\sum_i w_i \log w_i.
```

Interpretation:
- larger entropy means the weight is more spread out,
- smaller entropy means the weight is more concentrated.

### Weighted average loss

Function:
- `weighted_average_loss(w, ell)`

Formula:

```math
\langle w,\ell\rangle =
\sum_i w_i \ell_i.
```

Interpretation:
- in the stationary case, this is the quantity that appears in the continuous-time limit,
- it measures the current average loss induced by the weight vector.

### Hedge update

Function:
- `hedge_update(w, ell, eta)`

Formula:

```math
F_i(w) =
\frac{
w_i e^{-\eta \ell_i}
}{
\sum_j w_j e^{-\eta \ell_j}
}.
```

Interpretation:
- experts with lower loss receive relatively more weight,
- the learning rate `eta` controls how aggressively the reweighting occurs.

### Analytical stationary Jacobian

Function:
- `analytic_jacobian_stationary(w, ell, eta)`

Formula:

```math
\frac{\partial F_i}{\partial w_k}
=
\frac{
\delta_{ik} e^{-\eta \ell_i}
\sum_j w_j e^{-\eta \ell_j}
-
w_i e^{-\eta \ell_i} e^{-\eta \ell_k}
}{
\left(\sum_j w_j e^{-\eta \ell_j}\right)^2
}.
```

### Numerical Jacobian on the simplex

Function:
- `numerical_jacobian_tangent(w, ell, eta, h=1.0e-7)`

Interpretation:
- this approximates the Jacobian using simplex-compatible tangent perturbations,
- it uses a boundary-safe one-sided or central difference when needed.

### Spectral radius

Function:
- `spectral_radius(matrix)`

Formula:

```math
\rho(J) =
\max_i |\lambda_i|.
```

Interpretation:
- `rho(J) < 1` suggests contraction in the relevant directions,
- `rho(J) > 1` suggests linear instability,
- `rho(J) = 1` indicates at least one neutral direction.

### One-step linearization error

Function:
- `one_step_linearization_error(w_star, ell, eta, v, epsilon)`

Formula:

```math
\frac{
\left\|
F(w^{*}+\varepsilon v)
-
F(w^{*})
-
J(w^{*})(\varepsilon v)
\right\|
}{
\left\|
\varepsilon v
\right\|
}.
```

Interpretation:
- smaller values mean the local linearization accurately describes one nonlinear step near the fixed point.

### Continuous-time comparison error

Function:
- `continuous_time_comparison_errors(weights, losses, eta)`

Continuous-time model:

```math
\frac{dw_i}{d\tau}
=
w_i\bigl(\langle w,\ell\rangle-\ell_i\bigr).
```

Discrete comparison:

```math
\frac{w_{t+1}-w_t}{\eta}
```

versus

```math
w_t\bigl(\langle w_t,\ell_t\rangle-\ell_t\bigr).
```

Interpretation:
- smaller values mean the discrete dynamics more closely match the continuous-time approximation,
- this is most meaningful for small `eta`.

### Periodic tracking diagnostics

Function:
- `tracking_diagnostic_periodic(simulation)`

Computed quantities:
- hard tracking accuracy,
- mean preferred-expert share,
- preferred-weight margin,
- step sizes.

Interpretation:
- hard tracking accuracy is a strict switching diagnostic,
- mean preferred-expert share is usually the more informative soft diagnostic.

### Stationary support-preference diagnostics

Function:
- `stationary_preference_diagnostic(case, simulation)`

Computed quantities:
- mean tracking accuracy,
- mean preferred margin,
- min/max preferred margin with times.

Interpretation:
- in the unique-minimum case, these measure dominance of the single best expert,
- in the equal-minimum case, they measure support concentration on the equal-loss face.

---

## Default cases

### `unique_minimum`

Loss vector:

```math
(0,1,2)
```

Initial weight:

```math
\left(\frac{1}{3},\frac{1}{3},\frac{1}{3}\right)
```

Interpretation:
- expert 1 is uniquely best,
- one expects convergence toward the vertex `e1`,
- the vertices `e2` and `e3` are unstable.

### `equal_minimum_pair`

Loss vector:

```math
(0,0,1)
```

Initial weight:

```math
(0.2,0.8,0.0)
```

Interpretation:
- experts 1 and 2 share the same minimum loss,
- the full fixed-point set is the edge `w3 = 0`,
- the code evaluates representative points on that edge.

### `all_equal`

Loss vector:

```math
(1,1,1)
```

Initial weight:

```math
(0.2,0.3,0.5)
```

Interpretation:
- all points are fixed points,
- the Hedge map is the identity,
- the trajectory should remain unchanged.

### Periodic case

Losses vary periodically with default period `20`:

```math
\ell_t =
\left(
0.5 + 0.4\sin\left(\frac{2\pi t}{P}\right),
\;
0.5 - 0.4\sin\left(\frac{2\pi t}{P}\right),
\;
1.0
\right).
```

Interpretation:
- experts 1 and 2 alternate in relative quality,
- expert 3 remains uniformly worse.

---

## Default eta grid

Shared eta values:
- `0.02`
- `0.05`
- `0.10`
- `0.20`
- `0.50`
- `1.00`
- `2.00`

By default, the stationary and periodic regimes use the same eta grid.

---

## Default horizon and how to change it

The default horizon is:

- `DEFAULT_HORIZON = 50`

### Command line

Run:

```bash
python hedge_run.py
```

to use `T = 50`.

Run:

```bash
python hedge_run.py 100
```

to use `T = 100`.

### Notebook

Set:

```python
T = DEFAULT_HORIZON
```

or replace it with another positive integer before running the simulation cell.

---

## Saved figures and what they measure

The plotting code writes a large set of diagnostic figures. The list below explains not only which plots are saved, but also what each plotted quantity measures and how it should be interpreted.

### Stationary figures

#### Time-dependent figures for each stationary case and each `eta`

- **Weight trajectories**  
  These plot the coordinates of the weight vector `w_t` against time `t`. They show how mass is redistributed among the experts over time.

- **Entropy trajectories**  
  These plot the Shannon entropy

  ```math
  H(w_t) =
  -\sum_i w_{t,i}\log w_{t,i}.
  ```

  They measure how spread out or concentrated the weights are. Lower entropy means stronger concentration on fewer experts.

- **Weighted average loss trajectories**  
  These plot

  ```math
  \langle w_t,\ell\rangle
  ```

  in the stationary regime. They show the loss currently induced by the weight vector and connect directly to the continuous-time limit.

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

#### Across-eta stationary figures

- **Weight eta panels**  
  These compare the weight trajectories across several values of `eta`.

- **Simplex eta panels**  
  These compare the geometric trajectories in the simplex across `eta`.

- **Phase eta panels**  
  These compare the vector fields across `eta` and show how the strength of the motion changes with the learning rate.

- **Entropy eta panels**  
  These compare concentration of the weights across `eta`.

- **Average-loss eta panels**  
  These compare the weighted average loss trajectories across `eta`.

- **Continuous-time-error eta panels**  
  These compare the discrete-versus-continuous discrepancy across `eta`.

- **Preferred-margin eta panels**, when relevant  
  These compare the lower-loss-side weight advantage across `eta`.

- **Lower-loss-expert share / indicator eta panels**, when relevant  
  These compare whether and how strongly the lower-loss side is favored across `eta`.

- **Spectral-radius panels by representative fixed point**  
  These plot the spectral radius `rho(J)` of the Jacobian at representative fixed points. The spectral radius is the modulus of the largest eigenvalue; values below `1` indicate contraction, while values above `1` indicate local expansion.

- **Spectral radius vs eta**  
  This summarizes how the local contraction or expansion strength changes with the learning rate.

- **Jacobian difference norm vs eta**  
  This plots the norm of the difference between the analytical Jacobian and the numerically approximated Jacobian.

- **Eigenvalue discrepancy vs eta**  
  This plots the discrepancy between analytically predicted eigenvalues and numerically computed eigenvalues.

- **Linearization error vs eta**  
  This compares the true one-step update with the Jacobian-based linearized prediction near a fixed point.

- **Average step size vs eta**  
  This records the mean size of the one-step changes

  ```math
  \|w_{t+1}-w_t\|
  ```

  across the full trajectory.

- **Final entropy vs eta**  
  This records the final degree of concentration of the weight vector for each `eta`.

- **Final weighted average loss vs eta**  
  This records the final weighted average loss for each `eta`.

- **Mean continuous-time comparison error vs eta**  
  This compresses the full continuous-time error trajectory into a single mean value for each `eta`.

### Periodic figures

#### Time-dependent figures for each `eta`

- **Weight trajectories**  
  These show how the expert weights change over time when the lower-loss expert alternates.

- **Entropy trajectories**  
  These show how concentrated or spread out the weight vector is over time in the periodic regime.

- **Weighted average loss trajectories**  
  These plot

  ```math
  \langle w_t,\ell_t\rangle
  ```

  under the time-varying loss sequence.

- **Continuous-time comparison error trajectories**  
  These show the discrepancy between the periodic discrete dynamics and the corresponding continuous-time mechanism.

- **Preferred-weight margin trajectories**  
  These plot the difference between the weight assigned to the currently lower-loss expert and the weight assigned to its main competitor. They measure not only whether the preferred expert is favored, but by how much.

- **Step-size trajectories**  
  These plot

  ```math
  \|w_{t+1}-w_t\|
  ```

  over time and therefore measure how aggressively the weights move at each step.

- **Simplex trajectories**  
  These show the geometric oscillation of the weight vector in the simplex under the periodic loss sequence.

#### Across-eta periodic figures

- **Weight eta panels**  
  These compare the periodic weight trajectories across `eta`.

- **Simplex eta panels**  
  These compare the geometric oscillatory trajectories across `eta`.

- **Preferred-weight margin eta panels**  
  These compare the size of the lower-loss expert’s weight advantage across `eta`.

- **Step-size eta panels**  
  These compare how aggressively the weight vector moves across `eta`.

- **Entropy eta panels**  
  These compare concentration of the periodic weight vector across `eta`.

- **Average-loss eta panels**  
  These compare the weighted average loss trajectories across `eta`.

- **Continuous-time-error eta panels**  
  These compare the discrete-versus-continuous discrepancy across `eta`.

- **Hard tracking accuracy vs eta**  
  This records the fraction of time steps at which the currently lower-loss expert also has at least as much weight as its main competitor. This is a strict `0/1` diagnostic and should be interpreted cautiously.

- **Mean preferred-expert share vs eta**  
  This records the time-average of the weight placed on the currently lower-loss expert. It is typically more informative than hard tracking accuracy.

- **Average step size vs eta**  
  This summarizes how aggressive the reweighting is for each `eta`.

- **Final entropy vs eta**  
  This records the final concentration of the periodic weight vector for each `eta`.

- **Final weighted average loss vs eta**  
  This records the final weighted average loss for each `eta`.

### How to interpret the main plot quantities

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
- **Step size:** the size of the one-step reweighting move, measured by `||w_{t+1}-w_t||`.

---

## How to run the program

### Script

From the project folder:

```bash
python hedge_run.py
```

or with a custom horizon:

```bash
python hedge_run.py 150
```

This writes:
- `results/hedge_results.txt`
- all figures under `results/figures/`

### Notebook

Open `hedge_run.ipynb`, then:

1. run the setup cell,
2. set `T`,
3. run the simulation cells,
4. inspect the report preview and saved figures.

---

## How to use the notebook effectively

The notebook includes a `subplot_only` toggle.

Set:

```python
subplot_only = True
```

to display only eta-panel / subplot figures.

Set:

```python
subplot_only = False
```

to additionally display a smaller set of single representative figures.

The most useful first-pass review is:

1. `unique_minimum_weight_eta_panel.png`
2. `unique_minimum_simplex_eta_panel.png`
3. `unique_minimum_spectral_radius_panel.png`
4. `equal_minimum_pair_simplex_eta_panel.png`
5. `all_equal_weight_eta_panel.png`
6. `periodic_weight_eta_panel.png`
7. `periodic_margin_eta_panel.png`
8. `periodic_preferred_share_summary.png`

---

## Interpretation

The stationary computations are mathematically consistent with the underlying theory.

In the periodic regime:

1. **Hard tracking accuracy should not be overinterpreted.**
   In the symmetric periodic setup it can stay near `0.5` even when the soft tracking response is meaningful.

2. **Continuous-time comparison error should not be described as strictly monotone in eta across whole trajectories.**
   The small-eta theory is local; full-trajectory empirical averages can reflect transient and saturation effects.

So the most reliable periodic conclusion is:

- larger `eta` generally produces sharper and more oscillatory reweighting,
- but not necessarily better tracking in every diagnostic sense.

---

## Output locations

After a successful run:
- report: `results/hedge_results.txt`
- figures: `results/figures/`

---

## Final note

The code is written so that the comments, function names, and outputs follow the mathematics closely. It is intended to function as an executable simulation tool and as a computational analogue to the analytical report.