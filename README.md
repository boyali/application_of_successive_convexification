# Successive Convexification for Autonomous Vehicle Planning and Control

> **Paper:** *Applications of Successive Convexification in Autonomous Vehicle Planning and Control*
> Ali Boyali, Simon Thompson and David Wong — Tier IV, Inc., Tokyo, Japan
> [arXiv:2010.06276](https://arxiv.org/abs/2010.06276)

---

## Overview

This repository contains the Python implementation accompanying the paper above. The work presents the first systematic
application of **Successive Convexification (SCvx)** algorithms — originally developed for aerospace trajectory
optimization — to autonomous vehicle motion planning and control.

The key contributions are:

- Formulation of **speed planning** and **Model Predictive Control (MPC) path tracking** as successive convexification
  problems
- Use of **arc-length parametrization** in the vehicle kinematic model to linearize obstacle constraints and decouple
  the planning horizon from time
- Demonstration of **logical state-triggered constraints** for evasion maneuvers in a continuous optimization setting
- Comparison of SCvx against classical Sequential Convex Programming (SCP) methods, with emphasis on guaranteed
  convergence

---

## What is Successive Convexification?

SCvx solves non-convex optimal control problems by iteratively linearizing nonlinear dynamics and constraints around the
previous trajectory, then solving the resulting convex sub-problem to full optimality. Key mechanisms that distinguish
SCvx from generic SCP:

| Feature         | Generic SCP      | SCvx                |
|-----------------|------------------|---------------------|
| Trust Region    | Constant         | Scheduled (dynamic) |
| Virtual Force   | No               | Yes                 |
| ODE Integration | Single step      | Multi-step          |
| Propagation     | Single shooting  | Multiple shooting   |
| Convergence     | Rarely addressed | Strong, guaranteed  |

The **virtual control input** prevents artificial infeasibility by making states one-step reachable. The **dynamic trust
region** prevents artificial unboundedness and is adjusted based on the accuracy of each linear approximation.

---

## Vehicle Model

The implementation uses a **single-track kinematic model** with optional side-slip angle β, expressed in road-aligned
error coordinates (lateral error `ey`, heading error `eψ`). The model is re-parametrized by **arc-length `s`** rather
than time, yielding:

```
x'(s) = F(x(s), u(s), κ(s))
```

where `κ(s)` is the road curvature (from an offline lookup table or sensor estimates). States include global pose
`(Xw, Yw, Ψ)`, lateral/heading errors `(ey, eψ)`, longitudinal speed `V`, and steering angle `δf`. Controls are
longitudinal acceleration `u0` and steering rate `u1`.

Discretization uses the **First Order Hold (FOH)** method via the fundamental matrix solution to the LTV system, which
offers the best computational efficiency among standard methods (ZOH, RK4, pseudo-spectral).

---

## Implemented Scenarios

### 1. Speed Planning — No Obstacle

The vehicle decelerates from **20 m/s** to **0.5 m/s** within 50 metres along a curved path, subject to:

- Maximum speed and steering angle limits
- Steering rate limits
- Friction circle constraint (Kamm's circle): `‖[ax, ay]‖₂ ≤ μg`, with `μ = 0.6`
- Lateral path-deviation penalty

### 2. Speed Planning — With Obstacle Avoidance

Same as above, with a virtual obstacle located between discretization steps 20–24 (roughly 25–30 m). The vehicle must
pass the obstacle with at least **0.5 m lateral clearance** while simultaneously meeting the terminal speed target.

### 3. Evasion Maneuver via State-Triggered Constraints

Initial speed is raised to **25 m/s** so the vehicle *cannot* reach the 0.5 m/s terminal target within 50 m. A *
*state-triggered constraint** fires when the terminal speed exceeds 1 m/s, activating an evasion maneuver that forces
the vehicle at least 1 m laterally away from a virtual obstacle at the terminal point. This avoids mixed-integer
programming entirely.

---

## Repository Structure

```
.
├── main.py                        # Main entry point (this file)
├── test_path.txt                  # Reference path waypoints
├── Models/
│   ├── model_scoordinates.py      # Arc-length parametrized vehicle model
│   ├── parameters.py              # Tuning parameters (weights, trust region, etc.)
│   └── utils.py                   # Helper functions
├── Optimization/
│   ├── discretization.py          # First Order Hold integrator & linearization
│   └── mpcproblem.py              # CVXPY problem definition (SCvx sub-problem)
└── Logs/
    ├── logs_pickle_tire_no_obstacle.pickle
    └── logs_pickle_tire_obstacle.pickle
```

---

## Dependencies

| Package  | Purpose                                  |
|----------|------------------------------------------|
| `numpy`  | Numerical arrays and linear algebra      |
| `cvxpy`  | Convex problem modelling                 |
| `ecos`   | SOCP solver backend (via CVXPY)          |
| `scipy`  | ODE integration for FOH discretization   |
| `pickle` | Logging trajectories for post-processing |

Install with:

```bash
pip install numpy cvxpy ecos scipy
```

---

## Running the Code

```bash
python main.py
```

The script runs **MPC time steps**. At each step it:

1. Interpolates the road curvature reference over the current planning horizon
2. Integrates the nonlinear dynamics with the current control sequence
3. Computes FOH discretization matrices `(A, B⁻, B⁺, F, w)`
4. Solves the SCvx convex sub-problem via ECOS
5. Evaluates the trust-region acceptance criterion (`ρ = ΔJ_actual / ΔJ_predicted`)
6. Updates the trajectory and advances the station index `s0`

Logs are saved to `./Logs/logs_pickle_tire_no_obstacle.pickle` (or the obstacle variant). Use a Jupyter notebook to
visualise the pickled trajectories.

To switch between scenarios, toggle the obstacle list in the `Model` initialisation and update the output `file_name` at
the bottom of `main.py`.

---

## Key Tuning Parameters (`Models/parameters.py`)

| Parameter             | Description                                      |
|-----------------------|--------------------------------------------------|
| `K`                   | Number of discretization steps                   |
| `iterations`          | Max SCvx inner iterations per time step          |
| `w_nu`                | Virtual control penalty weight (should be large) |
| `tr_radius`           | Initial trust region radius                      |
| `rho_0, rho_1, rho_2` | Trust region acceptance thresholds               |
| `alpha, beta`         | Trust region shrink / grow factors               |
| `D_x, C_x, D_u, C_u`  | State and control scaling matrices               |

---

## Citation

```bibtex
@article{boyali2020successive,
    title = {Applications of Successive Convexification in Autonomous Vehicle Planning and Control},
    author = {Boyali, Ali and Thompson, Simon and Wong, David},
    journal = {arXiv preprint arXiv:2010.06276},
    year = {2020}
}
```

---

## Acknowledgements

The authors thank Sven Niederberger for his open-source SCvx implementations in Python and
C++ ([github.com/EmbersArc](https://github.com/EmbersArc)), which served as templates for this work. The implementation
is planned for release under the [Autoware](https://github.com/autowarefoundation/autoware) open-source autonomous
driving software stack.

---
 