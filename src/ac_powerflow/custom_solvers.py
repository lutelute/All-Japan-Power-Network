"""Custom PYPOWER-level AC power flow solver implementations.

Provides ~15 custom solvers that operate directly on PYPOWER internal
matrices (Ybus, Sbus, V0) extracted from a pandapower network.  All
solvers follow the unified interface::

    solver(Ybus, Sbus, V0, ref, pv, pq, max_iter=20, tol=1e-8) -> ACMethodResult

This module is organized into four solver categories:

* **Newton-Raphson variants** (7 methods) — ``custom_nr``, ``custom_nr_linesearch``,
  ``custom_nr_iwamoto``, ``custom_nr_rectangular``, ``custom_nr_current``,
  ``custom_nr_dishonest``, ``custom_nr_levenberg``
* **Iterative methods** (4 methods) — ``custom_gs``, ``custom_gs_accelerated``,
  ``custom_jacobi``, ``custom_gs_sor``
* **Decoupled / fast methods** (4 methods) — ``custom_fdpf_bx``,
  ``custom_fdpf_xb``, ``custom_decoupled_nr``, ``custom_nr_continuation``

Usage::

    from src.ac_powerflow.custom_solvers import custom_nr

    result = custom_nr(Ybus, Sbus, V0, ref, pv, pq, max_iter=20, tol=1e-8)
    print(result.summary)
"""

import time
from typing import Any, Dict, List

import numpy as np
from numpy import conj, exp, r_
from numpy.linalg import LinAlgError, norm
from scipy.sparse import (
    vstack as sp_vstack,
    hstack as sp_hstack,
    eye as sp_eye,
    diags as sp_diags,
    csr_matrix,
)
from scipy.sparse.linalg import spsolve

from pandapower.pypower.dSbus_dV import dSbus_dV

from src.ac_powerflow.solver_interface import ACMethodResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Divergence threshold: abort if mismatch grows beyond this factor of initial.
_DIVERGENCE_FACTOR = 10.0


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _compute_mismatch(V, Ybus, Sbus, pv, pq):
    """Compute the active/reactive power mismatch vector.

    The mismatch is ``S_calc - Sbus`` where ``S_calc = V * conj(Ybus @ V)``.
    The returned vector ``F`` is ordered as ``[P_mismatch[pvpq], Q_mismatch[pq]]``
    for use in the Newton-Raphson Jacobian system.

    Args:
        V: Complex voltage vector (all buses).
        Ybus: Bus admittance matrix (sparse).
        Sbus: Complex power injection vector (all buses, per-unit).
        pv: Array of PV bus indices.
        pq: Array of PQ bus indices.

    Returns:
        F: Real mismatch vector ``[dP; dQ]``.
        S_mis: Full complex mismatch vector (all buses).
    """
    pvpq = r_[pv, pq]
    S_calc = V * conj(Ybus @ V)
    S_mis = S_calc - Sbus

    P_mis = S_mis[pvpq].real
    Q_mis = S_mis[pq].imag

    F = r_[P_mis, Q_mis]
    return F, S_mis


def _build_jacobian(dS_dVa, dS_dVm, pv, pq):
    """Build the full Jacobian matrix from dSbus_dV partial derivatives.

    Assembles the 2x2 block Jacobian in polar coordinates::

        J = [ dP/dVa(pvpq,pvpq)   dP/dVm(pvpq,pq) ]
            [ dQ/dVa(pq,pvpq)     dQ/dVm(pq,pq)    ]

    where pvpq = concatenation of pv and pq bus indices.

    Args:
        dS_dVa: Partial derivative of S w.r.t. voltage angle (sparse).
        dS_dVm: Partial derivative of S w.r.t. voltage magnitude (sparse).
        pv: Array of PV bus indices.
        pq: Array of PQ bus indices.

    Returns:
        J: Sparse Jacobian matrix (CSR format).
    """
    pvpq = r_[pv, pq]

    # Extract real parts for P equations and imaginary parts for Q equations.
    # Row selection for pvpq (P equations) and pq (Q equations).
    # Column selection for pvpq (angle unknowns) and pq (magnitude unknowns).
    J11 = dS_dVa[pvpq, :][:, pvpq].real  # dP/dVa
    J12 = dS_dVm[pvpq, :][:, pq].real    # dP/dVm
    J21 = dS_dVa[pq, :][:, pvpq].imag    # dQ/dVa
    J22 = dS_dVm[pq, :][:, pq].imag      # dQ/dVm

    J = sp_vstack([
        sp_hstack([J11, J12]),
        sp_hstack([J21, J22]),
    ], format="csr")

    return J


def _check_numerical_issues(V, iteration):
    """Check for NaN or Inf in the voltage vector.

    Args:
        V: Complex voltage vector.
        iteration: Current iteration number (for logging).

    Returns:
        failure_reason: String describing the issue, or ``None`` if clean.
    """
    if np.isnan(V).any() or np.isinf(V).any():
        return f"NaN/Inf in voltage vector at iteration {iteration}"
    return None


# ---------------------------------------------------------------------------
# Newton-Raphson (standard, polar coordinates)
# ---------------------------------------------------------------------------

def custom_nr(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Standard Newton-Raphson power flow in polar coordinates.

    Implements the full NR method from scratch:

    1. Compute power mismatch ``F = [dP; dQ]``
    2. Evaluate Jacobian via ``dSbus_dV(Ybus, V)``
    3. Solve ``J @ dx = -F`` using sparse LU (``spsolve``)
    4. Update voltage angles and magnitudes
    5. Repeat until ``||F||_inf < tol`` or ``max_iter`` reached

    Includes divergence detection (mismatch grows 10x from initial),
    NaN/Inf detection, and singular Jacobian catching.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of NR iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    # Convert bus index lists to numpy arrays for indexing
    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)

    if npvpq == 0:
        # No unknowns — nothing to solve (all buses are slack).
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    # Working copy of voltage vector
    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    # Compute initial mismatch
    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    # Check if already converged
    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_nr: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        # NaN/Inf check
        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr: %s", issue)
            return result

        # Compute Jacobian
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
        J = _build_jacobian(dS_dVa, dS_dVm, pv, pq)

        # Solve linear system: J @ dx = -F
        try:
            dx = spsolve(J.tocsc(), -F)
        except LinAlgError:
            result.failure_reason = "Singular Jacobian matrix"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr: Singular Jacobian at iteration %d", i + 1)
            return result

        # Check for NaN/Inf in the correction vector
        if np.isnan(dx).any() or np.isinf(dx).any():
            result.failure_reason = (
                f"NaN/Inf in correction vector at iteration {i + 1}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr: %s", result.failure_reason)
            return result

        # Update voltage angles and magnitudes
        Va[pvpq] += dx[:npvpq]
        Vm[pq] += dx[npvpq:]

        # Reconstruct complex voltage
        V = Vm * exp(1j * Va)

        # Recompute mismatch
        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        # Check convergence
        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_nr: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        # Check divergence
        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr: %s", result.failure_reason)
            return result

    # Exhausted max iterations without convergence
    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_nr: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Newton-Raphson with Armijo Backtracking Line Search
# ---------------------------------------------------------------------------

def custom_nr_linesearch(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Newton-Raphson with Armijo backtracking line search.

    Same as the standard NR but after computing the correction ``dx``,
    a backtracking line search finds a step size ``α ∈ (0, 1]``
    satisfying the Armijo sufficient-decrease condition::

        ||F(x + α·dx)||_∞  ≤  (1 − c·α) · ||F(x)||_∞

    This improves robustness for ill-conditioned or heavily-loaded
    networks where the full Newton step may overshoot.

    Line-search parameters (internal):

    * ``c = 1e-4``  — Armijo sufficient-decrease constant.
    * ``ρ = 0.5``   — Step-reduction factor per backtrack.
    * ``max_ls = 10`` — Maximum number of backtracking halvings.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of NR iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_nr_linesearch: converged at iteration 0 (initial)")
        return result

    # Armijo line-search constants
    armijo_c = 1e-4
    armijo_rho = 0.5
    max_ls = 10

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_linesearch: %s", issue)
            return result

        # Compute Jacobian and NR direction
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
        J = _build_jacobian(dS_dVa, dS_dVm, pv, pq)

        try:
            dx = spsolve(J.tocsc(), -F)
        except LinAlgError:
            result.failure_reason = "Singular Jacobian matrix"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning(
                "custom_nr_linesearch: Singular Jacobian at iteration %d", i + 1,
            )
            return result

        if np.isnan(dx).any() or np.isinf(dx).any():
            result.failure_reason = (
                f"NaN/Inf in correction vector at iteration {i + 1}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_linesearch: %s", result.failure_reason)
            return result

        # ----- Armijo backtracking line search -----
        alpha = 1.0
        Va_base = Va.copy()
        Vm_base = Vm.copy()

        for _ls in range(max_ls):
            Va_trial = Va_base.copy()
            Vm_trial = Vm_base.copy()
            Va_trial[pvpq] += alpha * dx[:npvpq]
            Vm_trial[pq] += alpha * dx[npvpq:]
            V_trial = Vm_trial * exp(1j * Va_trial)

            F_trial, _ = _compute_mismatch(V_trial, Ybus, Sbus, pv, pq)
            normF_trial = norm(F_trial, np.inf)

            # Armijo sufficient-decrease condition
            if normF_trial <= (1.0 - armijo_c * alpha) * normF:
                break
            alpha *= armijo_rho

        # Accept the step (even if line search exhausted — take smallest α)
        Va = Va_trial
        Vm = Vm_trial
        V = V_trial
        F = F_trial
        normF = normF_trial
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_nr_linesearch: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_linesearch: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_nr_linesearch: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Newton-Raphson with Iwamoto Optimal Step Multiplier
# ---------------------------------------------------------------------------

def custom_nr_iwamoto(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Newton-Raphson with Iwamoto optimal step multiplier.

    After computing the standard NR correction ``dx``, the Iwamoto
    method determines an optimal scalar multiplier ``μ`` by fitting
    a quadratic to the squared mismatch norm at three trial points
    (``μ = 0, 0.5, 1``):

    .. math::

        g(\\mu) = \\|F(V + \\mu \\, dV)\\|^2 \\approx a\\mu^2 + b\\mu + c

    The optimal multiplier is ``μ* = −b / (2a)`` (clamped to
    ``[0.05, 1.0]``).  If the quadratic has no minimum (``a ≤ 0``),
    the full step (``μ = 1``) is used.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of NR iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_nr_iwamoto: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_iwamoto: %s", issue)
            return result

        # Compute Jacobian and NR direction
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
        J = _build_jacobian(dS_dVa, dS_dVm, pv, pq)

        try:
            dx = spsolve(J.tocsc(), -F)
        except LinAlgError:
            result.failure_reason = "Singular Jacobian matrix"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning(
                "custom_nr_iwamoto: Singular Jacobian at iteration %d", i + 1,
            )
            return result

        if np.isnan(dx).any() or np.isinf(dx).any():
            result.failure_reason = (
                f"NaN/Inf in correction vector at iteration {i + 1}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_iwamoto: %s", result.failure_reason)
            return result

        # ----- Iwamoto optimal multiplier via three-point quadratic fit -----
        # Evaluate mismatch norm² at μ = 0, 0.5, 1.0
        f0_sq = float(np.dot(F, F))

        # Trial at μ = 1.0 (full step)
        Va1 = Va.copy()
        Vm1 = Vm.copy()
        Va1[pvpq] += dx[:npvpq]
        Vm1[pq] += dx[npvpq:]
        V1 = Vm1 * exp(1j * Va1)
        F1, _ = _compute_mismatch(V1, Ybus, Sbus, pv, pq)
        f1_sq = float(np.dot(F1, F1))

        # Trial at μ = 0.5 (half step)
        Va05 = Va.copy()
        Vm05 = Vm.copy()
        Va05[pvpq] += 0.5 * dx[:npvpq]
        Vm05[pq] += 0.5 * dx[npvpq:]
        V05 = Vm05 * exp(1j * Va05)
        F05, _ = _compute_mismatch(V05, Ybus, Sbus, pv, pq)
        f05_sq = float(np.dot(F05, F05))

        # Fit g(μ) = a·μ² + b·μ + c through the three points
        # g(0) = c = f0_sq
        # g(0.5) = 0.25a + 0.5b + c = f05_sq
        # g(1.0) = a + b + c = f1_sq
        c_coeff = f0_sq
        b_coeff = 4.0 * f05_sq - f1_sq - 3.0 * f0_sq
        a_coeff = 2.0 * f1_sq + 2.0 * f0_sq - 4.0 * f05_sq

        if a_coeff > 1e-16:
            mu = np.clip(-b_coeff / (2.0 * a_coeff), 0.05, 1.0)
        else:
            # Quadratic has no minimum — use full step
            mu = 1.0

        # Apply update with optimal multiplier
        Va[pvpq] += mu * dx[:npvpq]
        Vm[pq] += mu * dx[npvpq:]
        V = Vm * exp(1j * Va)

        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_nr_iwamoto: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_iwamoto: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_nr_iwamoto: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Newton-Raphson in Rectangular Coordinates (V = e + jf)
# ---------------------------------------------------------------------------

def _build_rectangular_jacobian(Ybus, V, pv, pq, Vm_spec):
    """Build the Jacobian for NR in rectangular coordinates.

    Assembles the augmented Jacobian for the rectangular formulation
    where the state variables are ``[e(pvpq); f(pvpq)]`` (real and
    imaginary parts of voltage at non-slack buses).

    The equations are:

    * Active-power mismatch ``P(pvpq)``
    * Reactive-power mismatch ``Q(pq)``
    * Voltage-magnitude constraint ``e² + f² − Vm_spec²`` at PV buses

    The resulting block structure is::

        J = [ ∂P/∂e(pvpq,pvpq)   ∂P/∂f(pvpq,pvpq)   ]
            [ ∂Q/∂e(pq,pvpq)     ∂Q/∂f(pq,pvpq)      ]
            [ 2·diag(e)(pv,pvpq) 2·diag(f)(pv,pvpq)   ]

    Args:
        Ybus: Bus admittance matrix (sparse).
        V: Complex voltage vector (all buses).
        pv: PV bus indices.
        pq: PQ bus indices.
        Vm_spec: Specified voltage magnitudes (all buses).

    Returns:
        J: Sparse Jacobian matrix (CSR format).
    """
    pvpq = r_[pv, pq]
    npv = len(pv)
    npvpq = len(pvpq)
    n = len(V)

    e_vec = V.real
    f_vec = V.imag

    I = Ybus @ V
    Ir = I.real
    Ii = I.imag

    G = Ybus.real
    B = Ybus.imag

    # Sparse diagonal matrices
    diag_e = sp_diags(e_vec)
    diag_f = sp_diags(f_vec)
    diag_Ir = sp_diags(Ir)
    diag_Ii = sp_diags(Ii)

    # Full n×n Jacobian sub-blocks (power w.r.t. rectangular coords):
    # ∂P/∂e = diag(Ir) + diag(e)·G + diag(f)·B
    # ∂P/∂f = diag(Ii) − diag(e)·B + diag(f)·G
    # ∂Q/∂e = −diag(Ii) + diag(f)·G − diag(e)·B
    # ∂Q/∂f = diag(Ir) − diag(f)·B − diag(e)·G
    J_Pe = diag_Ir + diag_e @ G + diag_f @ B
    J_Pf = diag_Ii - diag_e @ B + diag_f @ G
    J_Qe = -diag_Ii + diag_f @ G - diag_e @ B
    J_Qf = diag_Ir - diag_f @ B - diag_e @ G

    row1 = sp_hstack([J_Pe[pvpq, :][:, pvpq], J_Pf[pvpq, :][:, pvpq]])
    row2 = sp_hstack([J_Qe[pq, :][:, pvpq], J_Qf[pq, :][:, pvpq]])

    if npv > 0:
        # Voltage-magnitude constraint: h_i = e_i² + f_i² − Vm_spec_i²
        # ∂h/∂e = 2·diag(e),  ∂h/∂f = 2·diag(f)  (only pv rows)
        diag_2e = sp_diags(2.0 * e_vec).tocsr()
        diag_2f = sp_diags(2.0 * f_vec).tocsr()
        row3 = sp_hstack([
            diag_2e[pv, :][:, pvpq],
            diag_2f[pv, :][:, pvpq],
        ])
        J = sp_vstack([row1, row2, row3], format="csr")
    else:
        J = sp_vstack([row1, row2], format="csr")

    return J


def custom_nr_rectangular(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Newton-Raphson power flow in rectangular coordinates.

    Instead of polar variables ``(θ, |V|)``, this method uses
    rectangular voltage components ``V = e + jf`` as the state
    variables.  For PV buses an additional voltage-magnitude
    constraint ``e² + f² = Vm_spec²`` is included, keeping the
    system square.

    The Jacobian is built from analytical expressions involving the
    conductance (``G``) and susceptance (``B``) parts of ``Ybus``.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of NR iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npv = len(pv)
    npq = len(pq)

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    # Working copy — rectangular representation
    V = V0.copy().astype(complex)
    Vm_spec = np.abs(V0)  # specified |V| for PV buses

    # Compute initial mismatch
    S_calc = V * conj(Ybus @ V)
    S_mis = S_calc - Sbus
    P_mis = S_mis[pvpq].real
    Q_mis = S_mis[pq].imag

    if npv > 0:
        h_mis = V.real[pv] ** 2 + V.imag[pv] ** 2 - Vm_spec[pv] ** 2
        F = r_[P_mis, Q_mis, h_mis]
    else:
        F = r_[P_mis, Q_mis]

    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_nr_rectangular: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_rectangular: %s", issue)
            return result

        # Build rectangular Jacobian
        J = _build_rectangular_jacobian(Ybus, V, pv, pq, Vm_spec)

        try:
            dx = spsolve(J.tocsc(), -F)
        except LinAlgError:
            result.failure_reason = "Singular Jacobian matrix"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning(
                "custom_nr_rectangular: Singular Jacobian at iteration %d", i + 1,
            )
            return result

        if np.isnan(dx).any() or np.isinf(dx).any():
            result.failure_reason = (
                f"NaN/Inf in correction vector at iteration {i + 1}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_rectangular: %s", result.failure_reason)
            return result

        # Update rectangular components: dx = [de(pvpq); df(pvpq)]
        de = dx[:npvpq]
        df = dx[npvpq:]

        e_vec = V.real.copy()
        f_vec = V.imag.copy()
        e_vec[pvpq] += de
        f_vec[pvpq] += df
        V = e_vec + 1j * f_vec

        # Recompute mismatch
        S_calc = V * conj(Ybus @ V)
        S_mis = S_calc - Sbus
        P_mis = S_mis[pvpq].real
        Q_mis = S_mis[pq].imag

        if npv > 0:
            h_mis = V.real[pv] ** 2 + V.imag[pv] ** 2 - Vm_spec[pv] ** 2
            F = r_[P_mis, Q_mis, h_mis]
        else:
            F = r_[P_mis, Q_mis]

        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_nr_rectangular: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_rectangular: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_nr_rectangular: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Newton-Raphson with Current-Injection Mismatch Formulation
# ---------------------------------------------------------------------------

def _compute_current_mismatch(V, Ybus, Sbus, pv, pq):
    """Compute the current-injection mismatch vector.

    The mismatch is ``ΔI = I_calc − I_spec`` where
    ``I_calc = Ybus @ V`` and ``I_spec = conj(Sbus / V)``.

    Returns ``F = [Re(ΔI)(pvpq); Im(ΔI)(pq)]`` and the full
    complex mismatch vector ``dI``.

    Args:
        V: Complex voltage vector (all buses).
        Ybus: Bus admittance matrix (sparse).
        Sbus: Complex power injection vector (all buses, per-unit).
        pv: Array of PV bus indices.
        pq: Array of PQ bus indices.

    Returns:
        F: Real mismatch vector ``[dIr; dIi]``.
        dI: Full complex current-mismatch vector (all buses).
    """
    pvpq = r_[pv, pq]
    I_calc = Ybus @ V
    I_spec = conj(Sbus / V)
    dI = I_calc - I_spec

    F = r_[dI[pvpq].real, dI[pq].imag]
    return F, dI


def _build_current_jacobian(Ybus, V, Sbus, pv, pq):
    """Build the Jacobian for current-injection NR in polar coordinates.

    For the current mismatch ``ΔI = Y·V − conj(Sbus/V)``, the
    partial derivatives in polar coordinates are::

        ∂ΔI/∂θ = Y·diag(jV) − diag(j·I_spec)
        ∂ΔI/∂|V| = Y·diag(exp(jθ)) + diag(I_spec / |V|)

    The Jacobian is assembled from real / imaginary parts of these
    matrices in the same block layout as the power-mismatch Jacobian.

    Args:
        Ybus: Bus admittance matrix (sparse).
        V: Complex voltage vector (all buses).
        Sbus: Complex power injection vector (all buses).
        pv: PV bus indices.
        pq: PQ bus indices.

    Returns:
        J: Sparse Jacobian matrix (CSR format).
    """
    pvpq = r_[pv, pq]
    n = len(V)
    Va = np.angle(V)
    Vm = np.abs(V)

    I_spec = conj(Sbus / V)

    # ∂ΔI/∂θ = Y·diag(jV) − diag(j·I_spec)
    diag_jV = sp_diags(1j * V)
    diag_jIspec = sp_diags(1j * I_spec)
    A = Ybus @ diag_jV - diag_jIspec

    # ∂ΔI/∂|V| = Y·diag(exp(jθ)) + diag(I_spec / |V|)
    diag_expjVa = sp_diags(exp(1j * Va))
    diag_IspecVm = sp_diags(I_spec / Vm)
    B_mat = Ybus @ diag_expjVa + diag_IspecVm

    J11 = A[pvpq, :][:, pvpq].real   # ∂(Re ΔI)/∂θ
    J12 = B_mat[pvpq, :][:, pq].real  # ∂(Re ΔI)/∂|V|
    J21 = A[pq, :][:, pvpq].imag     # ∂(Im ΔI)/∂θ
    J22 = B_mat[pq, :][:, pq].imag    # ∂(Im ΔI)/∂|V|

    J = sp_vstack([
        sp_hstack([J11, J12]),
        sp_hstack([J21, J22]),
    ], format="csr")

    return J


def custom_nr_current(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Newton-Raphson with current-injection mismatch formulation.

    Instead of the conventional power mismatch ``ΔS = V·conj(I) − Sbus``,
    this variant uses a current mismatch ``ΔI = Y·V − conj(Sbus/V)``
    and derives the Jacobian accordingly.

    The current-injection form often exhibits better numerical
    conditioning for systems with high-impedance branches.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of NR iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    F, _ = _compute_current_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_nr_current: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_current: %s", issue)
            return result

        # Build current-injection Jacobian
        J = _build_current_jacobian(Ybus, V, Sbus, pv, pq)

        try:
            dx = spsolve(J.tocsc(), -F)
        except LinAlgError:
            result.failure_reason = "Singular Jacobian matrix"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning(
                "custom_nr_current: Singular Jacobian at iteration %d", i + 1,
            )
            return result

        if np.isnan(dx).any() or np.isinf(dx).any():
            result.failure_reason = (
                f"NaN/Inf in correction vector at iteration {i + 1}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_current: %s", result.failure_reason)
            return result

        # Update angles and magnitudes (same update as polar NR)
        Va[pvpq] += dx[:npvpq]
        Vm[pq] += dx[npvpq:]
        V = Vm * exp(1j * Va)

        F, _ = _compute_current_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_nr_current: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_current: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_nr_current: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Dishonest Newton-Raphson (Frozen Jacobian)
# ---------------------------------------------------------------------------

def custom_nr_dishonest(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Dishonest Newton-Raphson with frozen Jacobian.

    A "modified" or "dishonest" NR that recomputes the Jacobian only
    every 3 iterations, reusing the factored matrix for intermediate
    steps.  This reduces the per-iteration cost at the expense of
    convergence rate — useful when Jacobian evaluation is expensive
    or when quadratic convergence is not required.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of NR iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_nr_dishonest: converged at iteration 0 (initial)")
        return result

    # How often to refresh the Jacobian (every N iterations)
    jacobian_refresh_interval = 3
    J_csc = None  # cached Jacobian in CSC format

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_dishonest: %s", issue)
            return result

        # Recompute Jacobian every `jacobian_refresh_interval` iterations
        # (and always on the first iteration)
        if i % jacobian_refresh_interval == 0:
            dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
            J = _build_jacobian(dS_dVa, dS_dVm, pv, pq)
            J_csc = J.tocsc()

        try:
            dx = spsolve(J_csc, -F)
        except LinAlgError:
            result.failure_reason = "Singular Jacobian matrix"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning(
                "custom_nr_dishonest: Singular Jacobian at iteration %d", i + 1,
            )
            return result

        if np.isnan(dx).any() or np.isinf(dx).any():
            result.failure_reason = (
                f"NaN/Inf in correction vector at iteration {i + 1}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_dishonest: %s", result.failure_reason)
            return result

        Va[pvpq] += dx[:npvpq]
        Vm[pq] += dx[npvpq:]
        V = Vm * exp(1j * Va)

        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_nr_dishonest: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_dishonest: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_nr_dishonest: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Levenberg-Marquardt Damped Newton-Raphson
# ---------------------------------------------------------------------------

def custom_nr_levenberg(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Levenberg-Marquardt damped Newton-Raphson.

    Adds a diagonal damping term ``λI`` to the Jacobian before
    solving the linear system::

        (J + λI) · dx = −F

    When ``λ`` is small the method behaves like standard NR;
    when ``λ`` is large the step resembles steepest descent.
    The damping parameter is adapted each iteration:

    * If the step **reduces** the mismatch, ``λ`` is **decreased**
      (×0.1) to accelerate convergence.
    * If the step **increases** the mismatch, ``λ`` is **increased**
      (×10) and the step is rejected.

    Clamped to ``[1e-12, 1e6]`` for numerical stability.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of NR iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)
    ndim = npvpq + npq  # total unknowns

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_nr_levenberg: converged at iteration 0 (initial)")
        return result

    # LM damping parameters
    lam = 1e-3        # initial damping
    lam_min = 1e-12
    lam_max = 1e6
    lam_up = 10.0     # increase factor on failed step
    lam_down = 0.1    # decrease factor on successful step
    max_rejects = 5   # max consecutive step rejections per iteration

    I_sparse = sp_eye(ndim, format="csc")

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_levenberg: %s", issue)
            return result

        # Compute Jacobian
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
        J = _build_jacobian(dS_dVa, dS_dVm, pv, pq)

        # Damped solve with adaptive λ
        step_accepted = False
        for _rej in range(max_rejects):
            J_damped = (J + lam * I_sparse).tocsc()

            try:
                dx = spsolve(J_damped, -F)
            except LinAlgError:
                lam = min(lam * lam_up, lam_max)
                continue

            if np.isnan(dx).any() or np.isinf(dx).any():
                lam = min(lam * lam_up, lam_max)
                continue

            # Trial update
            Va_trial = Va.copy()
            Vm_trial = Vm.copy()
            Va_trial[pvpq] += dx[:npvpq]
            Vm_trial[pq] += dx[npvpq:]
            V_trial = Vm_trial * exp(1j * Va_trial)

            F_trial, _ = _compute_mismatch(V_trial, Ybus, Sbus, pv, pq)
            normF_trial = norm(F_trial, np.inf)

            if normF_trial < normF:
                # Accept step, reduce damping
                Va = Va_trial
                Vm = Vm_trial
                V = V_trial
                F = F_trial
                normF = normF_trial
                lam = max(lam * lam_down, lam_min)
                step_accepted = True
                break
            else:
                # Reject step, increase damping
                lam = min(lam * lam_up, lam_max)

        if not step_accepted:
            # All rejection attempts exhausted — accept last trial anyway
            Va[pvpq] += dx[:npvpq]
            Vm[pq] += dx[npvpq:]
            V = Vm * exp(1j * Va)
            F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
            normF = norm(F, np.inf)

        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_nr_levenberg: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_levenberg: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_nr_levenberg: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Gauss-Seidel (Standard)
# ---------------------------------------------------------------------------

def custom_gs(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Standard Gauss-Seidel power flow with sequential bus voltage updates.

    Iterates through non-slack buses sequentially, updating each bus
    voltage in-place before moving to the next bus::

        V[k] = (conj(Sbus[k] / V[k]) − Σ_{j≠k} Ybus[k,j]·V[j]) / Ybus[k,k]

    For PV buses, reactive power is recalculated from the current
    voltage solution at each step, and voltage magnitude is restored
    to the specified setpoint after each update.

    An incremental current vector is maintained to avoid recomputing
    the full ``Ybus @ V`` product after each single-bus update.

    Handles zero diagonal in Ybus by skipping buses where
    ``|Ybus[k,k]| < 1e-20``.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of GS iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    if len(pvpq) == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Vm_spec = np.abs(V0)  # specified |V| for PV buses
    pv_set = set(pv.tolist())

    # Ensure CSC format for efficient column extraction
    Ybus_csc = Ybus.tocsc()
    Y_diag = np.array(Ybus.diagonal(), dtype=complex)

    # Compute initial mismatch
    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_gs: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        # NaN/Inf check
        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_gs: %s", issue)
            return result

        # Compute current injection vector I = Ybus @ V
        I_bus = np.asarray(Ybus @ V, dtype=complex).ravel()

        # Sequential bus voltage updates
        for k in pvpq:
            # Guard against zero diagonal (isolated bus)
            if abs(Y_diag[k]) < 1e-20:
                continue

            V_old_k = V[k]

            # For PV buses: recalculate Q from current solution
            if k in pv_set:
                S_calc_k = V[k] * conj(I_bus[k])
                Q_k = S_calc_k.imag
                S_eff_k = Sbus[k].real + 1j * Q_k
            else:
                S_eff_k = Sbus[k]

            # GS update: V[k] = (conj(S/V) - sum_{j≠k} Y_kj*V_j) / Y_kk
            V[k] = (conj(S_eff_k / V[k]) - (I_bus[k] - Y_diag[k] * V[k])) / Y_diag[k]

            # For PV buses: restore specified voltage magnitude
            if k in pv_set:
                V_mag = abs(V[k])
                if V_mag > 1e-20:
                    V[k] = Vm_spec[k] * V[k] / V_mag

            # Incrementally update I_bus for subsequent bus updates
            delta_V = V[k] - V_old_k
            if abs(delta_V) > 0:
                col_k = np.asarray(Ybus_csc[:, k].todense()).ravel()
                I_bus += col_k * delta_V

        # Recompute mismatch for convergence check
        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        # Check convergence
        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_gs: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        # Check divergence
        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_gs: %s", result.failure_reason)
            return result

    # Exhausted max iterations without convergence
    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_gs: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Gauss-Seidel with Acceleration Factor
# ---------------------------------------------------------------------------

def custom_gs_accelerated(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Gauss-Seidel power flow with acceleration factor.

    Same as the standard GS but applies an acceleration factor
    ``ω ∈ [1.0, 2.0]`` (default ``ω = 1.6``) to the voltage correction
    at each bus::

        V_gs[k]  = standard GS update
        V[k]     = V_old[k] + ω · (V_gs[k] − V_old[k])

    Over-relaxation (``ω > 1``) amplifies each update to accelerate
    convergence, particularly effective for well-conditioned systems.
    The method degenerates to standard GS when ``ω = 1``.

    PV bus voltage magnitude is restored after applying the
    acceleration to preserve the voltage setpoint.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of GS iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    if len(pvpq) == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Vm_spec = np.abs(V0)
    pv_set = set(pv.tolist())

    # Acceleration factor (typical range 1.0–2.0)
    omega = 1.6

    Ybus_csc = Ybus.tocsc()
    Y_diag = np.array(Ybus.diagonal(), dtype=complex)

    # Compute initial mismatch
    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_gs_accelerated: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_gs_accelerated: %s", issue)
            return result

        I_bus = np.asarray(Ybus @ V, dtype=complex).ravel()

        for k in pvpq:
            if abs(Y_diag[k]) < 1e-20:
                continue

            V_old_k = V[k]

            # Recalculate Q for PV buses
            if k in pv_set:
                S_calc_k = V[k] * conj(I_bus[k])
                Q_k = S_calc_k.imag
                S_eff_k = Sbus[k].real + 1j * Q_k
            else:
                S_eff_k = Sbus[k]

            # Standard GS update
            V_gs = (conj(S_eff_k / V[k]) - (I_bus[k] - Y_diag[k] * V[k])) / Y_diag[k]

            # Apply acceleration: V = V_old + ω·(V_gs − V_old)
            V[k] = V_old_k + omega * (V_gs - V_old_k)

            # Restore PV bus voltage magnitude
            if k in pv_set:
                V_mag = abs(V[k])
                if V_mag > 1e-20:
                    V[k] = Vm_spec[k] * V[k] / V_mag

            # Incrementally update I_bus
            delta_V = V[k] - V_old_k
            if abs(delta_V) > 0:
                col_k = np.asarray(Ybus_csc[:, k].todense()).ravel()
                I_bus += col_k * delta_V

        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_gs_accelerated: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_gs_accelerated: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_gs_accelerated: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Jacobi Iteration (Simultaneous Updates)
# ---------------------------------------------------------------------------

def custom_jacobi(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Jacobi iteration power flow with simultaneous bus voltage updates.

    Unlike Gauss-Seidel, the Jacobi method computes all bus voltage
    updates using the **previous** iteration's voltages, then applies
    them simultaneously::

        V_new[k] = (conj(Sbus[k] / V_old[k]) − Σ_{j≠k} Ybus[k,j]·V_old[j]) / Ybus[k,k]

    This is fully vectorizable and can be parallelized, but typically
    converges more slowly than GS because it does not use the most
    recent updates within the same iteration.

    PV bus voltage magnitudes are restored after each simultaneous
    update step.

    Handles zero diagonal in Ybus by skipping buses where
    ``|Ybus[k,k]| < 1e-20``.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of Jacobi iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    if len(pvpq) == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Vm_spec = np.abs(V0)

    Y_diag = np.array(Ybus.diagonal(), dtype=complex)

    # Compute initial mismatch
    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_jacobi: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_jacobi: %s", issue)
            return result

        # Compute I = Ybus @ V using OLD voltages (Jacobi: all from old V)
        I_bus = np.asarray(Ybus @ V, dtype=complex).ravel()

        # For PV buses: recalculate Q from current voltages
        S_eff = Sbus.copy().astype(complex)
        if len(pv) > 0:
            S_calc_pv = V[pv] * conj(I_bus[pv])
            S_eff[pv] = Sbus[pv].real + 1j * S_calc_pv.imag

        # Vectorized Jacobi update for all pvpq buses simultaneously
        # V_new[k] = (conj(S_eff[k]/V[k]) - (I_bus[k] - Y_diag[k]*V[k])) / Y_diag[k]
        V_new = V.copy()
        for k in pvpq:
            if abs(Y_diag[k]) < 1e-20:
                continue
            V_new[k] = (
                conj(S_eff[k] / V[k]) - (I_bus[k] - Y_diag[k] * V[k])
            ) / Y_diag[k]

        # Restore PV bus voltage magnitudes
        if len(pv) > 0:
            Vm_pv = np.abs(V_new[pv])
            # Guard against zero magnitude
            safe_mask = Vm_pv > 1e-20
            pv_safe = pv[safe_mask]
            if len(pv_safe) > 0:
                V_new[pv_safe] = Vm_spec[pv_safe] * V_new[pv_safe] / np.abs(V_new[pv_safe])

        # Apply simultaneous update (ref buses unchanged)
        V = V_new

        # Recompute mismatch
        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_jacobi: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_jacobi: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_jacobi: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Successive Over-Relaxation (SOR)
# ---------------------------------------------------------------------------

def custom_gs_sor(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Successive Over-Relaxation (SOR) power flow.

    A sequential update method (like GS) with a relaxation factor
    ``ω ∈ (0, 2)`` that blends the old voltage with the GS update::

        V_gs[k]  = standard Gauss-Seidel update
        V[k]     = (1 − ω) · V_old[k]  +  ω · V_gs[k]

    When ``ω = 1`` the method reduces to standard GS.  Values
    ``ω > 1`` (over-relaxation, default ``ω = 1.5``) can accelerate
    convergence, while ``ω < 1`` (under-relaxation) improves
    stability for ill-conditioned systems.

    PV bus voltage magnitude is restored after each relaxed update.

    Handles zero diagonal in Ybus by skipping buses where
    ``|Ybus[k,k]| < 1e-20``.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of SOR iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    if len(pvpq) == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Vm_spec = np.abs(V0)
    pv_set = set(pv.tolist())

    # SOR relaxation factor (ω = 1.0 degenerates to standard GS)
    omega = 1.5

    Ybus_csc = Ybus.tocsc()
    Y_diag = np.array(Ybus.diagonal(), dtype=complex)

    # Compute initial mismatch
    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_gs_sor: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_gs_sor: %s", issue)
            return result

        I_bus = np.asarray(Ybus @ V, dtype=complex).ravel()

        for k in pvpq:
            if abs(Y_diag[k]) < 1e-20:
                continue

            V_old_k = V[k]

            # Recalculate Q for PV buses
            if k in pv_set:
                S_calc_k = V[k] * conj(I_bus[k])
                Q_k = S_calc_k.imag
                S_eff_k = Sbus[k].real + 1j * Q_k
            else:
                S_eff_k = Sbus[k]

            # Standard GS update
            V_gs = (conj(S_eff_k / V[k]) - (I_bus[k] - Y_diag[k] * V[k])) / Y_diag[k]

            # SOR blending: V = (1 − ω)·V_old + ω·V_gs
            V[k] = (1.0 - omega) * V_old_k + omega * V_gs

            # Restore PV bus voltage magnitude
            if k in pv_set:
                V_mag = abs(V[k])
                if V_mag > 1e-20:
                    V[k] = Vm_spec[k] * V[k] / V_mag

            # Incrementally update I_bus
            delta_V = V[k] - V_old_k
            if abs(delta_V) > 0:
                col_k = np.asarray(Ybus_csc[:, k].todense()).ravel()
                I_bus += col_k * delta_V

        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_gs_sor: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_gs_sor: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_gs_sor: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Shared FDPF helper: construct Bp and Bpp matrices from Ybus
# ---------------------------------------------------------------------------

def _build_fdpf_matrices(Ybus, ref, pv, pq, variant="bx"):
    """Construct the constant Bp and Bpp matrices for Fast Decoupled PF.

    The FDPF method approximates the full Jacobian with two constant
    real matrices derived from the imaginary part of ``Ybus``:

    * **Bp** — Used for the P–θ sub-problem.  Rows/columns correspond
      to ``pvpq`` buses.
    * **Bpp** — Used for the Q–|V| sub-problem.  Rows/columns correspond
      to ``pq`` buses only.

    Two construction variants are supported:

    * **BX** ordering (``variant="bx"``):
      - Bp: Ignore shunt elements (zero diagonal contributions) and
        use ``−Im(Ybus)`` for off-diagonal entries.  Neglect
        resistance (set ``G = 0``) in the branch admittance.
      - Bpp: Use the full imaginary part of Ybus.

    * **XB** ordering (``variant="xb"``):
      - Bp: Use the full imaginary part of Ybus.
      - Bpp: Neglect resistance in the branch admittance.

    For simplicity and robustness this implementation uses the imaginary
    part of Ybus directly for both matrices, with the BX/XB distinction
    applied via which matrix gets the shunt-free approximation.

    Args:
        Ybus: Bus admittance matrix (sparse).
        ref: Slack bus indices.
        pv: PV bus indices.
        pq: PQ bus indices.
        variant: ``"bx"`` or ``"xb"``.

    Returns:
        Bp: Sparse matrix for P–θ sub-problem (pvpq × pvpq).
        Bpp: Sparse matrix for Q–|V| sub-problem (pq × pq).

    Raises:
        ValueError: If variant is not ``"bx"`` or ``"xb"``.
    """
    pvpq = r_[pv, pq]

    # Full imaginary part of Ybus (susceptance matrix)
    B_full = Ybus.imag

    if variant == "bx":
        # BX ordering:
        # Bp — remove shunts (zero the diagonal) from -B
        Bp_full = -B_full.copy()
        Bp_full = csr_matrix(Bp_full)
        Bp_full.setdiag(0)
        # Re-add diagonal from off-diagonal row sums (shunt-free)
        diag_vals = -np.array(Bp_full.sum(axis=1)).ravel()
        Bp_full.setdiag(diag_vals)

        Bp = Bp_full[pvpq, :][:, pvpq]

        # Bpp — use full -B(pq,pq)
        Bpp = -B_full[pq, :][:, pq]
    elif variant == "xb":
        # XB ordering:
        # Bp — use full -B(pvpq,pvpq)
        Bp = -B_full[pvpq, :][:, pvpq]

        # Bpp — remove shunts (zero diagonal) from -B
        Bpp_full = -B_full.copy()
        Bpp_full = csr_matrix(Bpp_full)
        Bpp_full.setdiag(0)
        diag_vals = -np.array(Bpp_full.sum(axis=1)).ravel()
        Bpp_full.setdiag(diag_vals)

        Bpp = Bpp_full[pq, :][:, pq]
    else:
        raise ValueError(f"Unknown FDPF variant: {variant!r} (use 'bx' or 'xb')")

    return csr_matrix(Bp), csr_matrix(Bpp)


# ---------------------------------------------------------------------------
# Fast Decoupled Power Flow — BX Ordering
# ---------------------------------------------------------------------------

def custom_fdpf_bx(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Fast Decoupled Power Flow with BX ordering.

    The FDPF method exploits the approximate decoupling between
    active-power/angle (P–θ) and reactive-power/voltage (Q–|V|)
    sub-problems.  Two constant real matrices ``Bp`` and ``Bpp``
    are factored once and reused every iteration, avoiding the
    expensive full Jacobian re-evaluation of standard NR.

    Each iteration consists of two half-steps:

    1. **P–θ half-step**: Solve ``Bp · Δθ = ΔP / |V|``
    2. **Q–|V| half-step**: Solve ``Bpp · Δ|V| = ΔQ / |V|``

    The "BX" variant constructs Bp by neglecting shunt elements and
    resistance, and Bpp from the full bus susceptance matrix.

    If Bp or Bpp is ill-conditioned (singular or near-singular),
    the method reports ``failure_reason`` and returns early.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of FDPF iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    # Construct constant Bp and Bpp matrices (BX variant)
    try:
        Bp, Bpp = _build_fdpf_matrices(Ybus, ref, pv, pq, variant="bx")
        Bp_csc = Bp.tocsc()
        if npq > 0:
            Bpp_csc = Bpp.tocsc()
    except Exception as exc:
        result.failure_reason = f"Failed to construct Bp/Bpp matrices: {exc}"
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.warning("custom_fdpf_bx: %s", result.failure_reason)
        return result

    # Compute initial mismatch
    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_fdpf_bx: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_fdpf_bx: %s", issue)
            return result

        # --- P-θ half-step ---
        S_calc = V * conj(Ybus @ V)
        S_mis = S_calc - Sbus
        dP = S_mis[pvpq].real
        rhs_p = dP / Vm[pvpq]

        try:
            dVa = spsolve(Bp_csc, -rhs_p)
        except LinAlgError:
            result.failure_reason = "Failed to construct Bp/Bpp matrices"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_fdpf_bx: Singular Bp matrix at iteration %d", i + 1)
            return result

        if np.isnan(dVa).any() or np.isinf(dVa).any():
            result.failure_reason = "Failed to construct Bp/Bpp matrices"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning(
                "custom_fdpf_bx: NaN/Inf in Bp solve at iteration %d", i + 1,
            )
            return result

        Va[pvpq] += dVa
        V = Vm * exp(1j * Va)

        # --- Q-|V| half-step (only for PQ buses) ---
        if npq > 0:
            S_calc = V * conj(Ybus @ V)
            S_mis = S_calc - Sbus
            dQ = S_mis[pq].imag
            rhs_q = dQ / Vm[pq]

            try:
                dVm = spsolve(Bpp_csc, -rhs_q)
            except LinAlgError:
                result.failure_reason = "Failed to construct Bp/Bpp matrices"
                result.V = V.copy()
                result.elapsed_sec = time.perf_counter() - start
                logger.warning(
                    "custom_fdpf_bx: Singular Bpp matrix at iteration %d", i + 1,
                )
                return result

            if np.isnan(dVm).any() or np.isinf(dVm).any():
                result.failure_reason = "Failed to construct Bp/Bpp matrices"
                result.V = V.copy()
                result.elapsed_sec = time.perf_counter() - start
                logger.warning(
                    "custom_fdpf_bx: NaN/Inf in Bpp solve at iteration %d", i + 1,
                )
                return result

            Vm[pq] += dVm
            V = Vm * exp(1j * Va)

        # Recompute full mismatch for convergence check
        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_fdpf_bx: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_fdpf_bx: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_fdpf_bx: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Fast Decoupled Power Flow — XB Ordering
# ---------------------------------------------------------------------------

def custom_fdpf_xb(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Fast Decoupled Power Flow with XB ordering.

    Same decoupled structure as ``custom_fdpf_bx`` but with an
    alternative B-matrix construction:

    * **Bp** — Constructed from the full bus susceptance matrix
      (including shunt elements).
    * **Bpp** — Constructed by neglecting resistance and shunt
      elements in the branch admittance.

    The XB ordering tends to perform better on networks where
    the R/X ratio is relatively high (distribution-like systems).

    Each iteration consists of two half-steps:

    1. **P–θ half-step**: Solve ``Bp · Δθ = ΔP / |V|``
    2. **Q–|V| half-step**: Solve ``Bpp · Δ|V| = ΔQ / |V|``

    If Bp or Bpp is ill-conditioned, the method reports
    ``failure_reason`` and returns early.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of FDPF iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    # Construct constant Bp and Bpp matrices (XB variant)
    try:
        Bp, Bpp = _build_fdpf_matrices(Ybus, ref, pv, pq, variant="xb")
        Bp_csc = Bp.tocsc()
        if npq > 0:
            Bpp_csc = Bpp.tocsc()
    except Exception as exc:
        result.failure_reason = f"Failed to construct Bp/Bpp matrices: {exc}"
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.warning("custom_fdpf_xb: %s", result.failure_reason)
        return result

    # Compute initial mismatch
    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_fdpf_xb: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_fdpf_xb: %s", issue)
            return result

        # --- P-θ half-step ---
        S_calc = V * conj(Ybus @ V)
        S_mis = S_calc - Sbus
        dP = S_mis[pvpq].real
        rhs_p = dP / Vm[pvpq]

        try:
            dVa = spsolve(Bp_csc, -rhs_p)
        except LinAlgError:
            result.failure_reason = "Failed to construct Bp/Bpp matrices"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_fdpf_xb: Singular Bp matrix at iteration %d", i + 1)
            return result

        if np.isnan(dVa).any() or np.isinf(dVa).any():
            result.failure_reason = "Failed to construct Bp/Bpp matrices"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning(
                "custom_fdpf_xb: NaN/Inf in Bp solve at iteration %d", i + 1,
            )
            return result

        Va[pvpq] += dVa
        V = Vm * exp(1j * Va)

        # --- Q-|V| half-step (only for PQ buses) ---
        if npq > 0:
            S_calc = V * conj(Ybus @ V)
            S_mis = S_calc - Sbus
            dQ = S_mis[pq].imag
            rhs_q = dQ / Vm[pq]

            try:
                dVm = spsolve(Bpp_csc, -rhs_q)
            except LinAlgError:
                result.failure_reason = "Failed to construct Bp/Bpp matrices"
                result.V = V.copy()
                result.elapsed_sec = time.perf_counter() - start
                logger.warning(
                    "custom_fdpf_xb: Singular Bpp matrix at iteration %d", i + 1,
                )
                return result

            if np.isnan(dVm).any() or np.isinf(dVm).any():
                result.failure_reason = "Failed to construct Bp/Bpp matrices"
                result.V = V.copy()
                result.elapsed_sec = time.perf_counter() - start
                logger.warning(
                    "custom_fdpf_xb: NaN/Inf in Bpp solve at iteration %d", i + 1,
                )
                return result

            Vm[pq] += dVm
            V = Vm * exp(1j * Va)

        # Recompute full mismatch for convergence check
        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_fdpf_xb: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_fdpf_xb: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_fdpf_xb: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Decoupled Newton-Raphson (Alternating P-θ and Q-V Sub-iterations)
# ---------------------------------------------------------------------------

def custom_decoupled_nr(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Decoupled Newton-Raphson with alternating P–θ and Q–|V| sub-iterations.

    Unlike FDPF (which uses constant B matrices), this method
    recomputes the diagonal Jacobian sub-blocks at each iteration,
    but ignores the off-diagonal coupling blocks ``J12`` (∂P/∂|V|)
    and ``J21`` (∂Q/∂θ).

    Each iteration consists of two alternating NR sub-steps:

    1. **P–θ sub-step**: Solve ``J11 · Δθ = −ΔP`` where
       ``J11 = ∂P/∂θ`` (pvpq × pvpq).
    2. **Q–|V| sub-step**: Solve ``J22 · Δ|V| = −ΔQ`` where
       ``J22 = ∂Q/∂|V|`` (pq × pq).

    This method converges faster than FDPF (because the diagonal
    blocks are updated) but slower than full NR (because coupling
    is neglected).  It is particularly effective for networks with
    weak P–Q coupling.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of decoupled NR iterations.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    # Compute initial mismatch
    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    result.convergence_history.append(float(normF))
    initial_normF = normF

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info("custom_decoupled_nr: converged at iteration 0 (initial)")
        return result

    for i in range(max_iter):
        result.iterations = i + 1

        issue = _check_numerical_issues(V, i + 1)
        if issue is not None:
            result.failure_reason = issue
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_decoupled_nr: %s", issue)
            return result

        # --- P-θ sub-step ---
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
        J11 = dS_dVa[pvpq, :][:, pvpq].real  # ∂P/∂θ

        S_calc = V * conj(Ybus @ V)
        S_mis = S_calc - Sbus
        dP = S_mis[pvpq].real

        try:
            dVa_step = spsolve(J11.tocsc(), -dP)
        except LinAlgError:
            result.failure_reason = "Singular Jacobian matrix"
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning(
                "custom_decoupled_nr: Singular J11 at iteration %d", i + 1,
            )
            return result

        if np.isnan(dVa_step).any() or np.isinf(dVa_step).any():
            result.failure_reason = (
                f"NaN/Inf in P-θ correction at iteration {i + 1}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_decoupled_nr: %s", result.failure_reason)
            return result

        Va[pvpq] += dVa_step
        V = Vm * exp(1j * Va)

        # --- Q-|V| sub-step (only for PQ buses) ---
        if npq > 0:
            dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
            J22 = dS_dVm[pq, :][:, pq].imag  # ∂Q/∂|V|

            S_calc = V * conj(Ybus @ V)
            S_mis = S_calc - Sbus
            dQ = S_mis[pq].imag

            try:
                dVm_step = spsolve(J22.tocsc(), -dQ)
            except LinAlgError:
                result.failure_reason = "Singular Jacobian matrix"
                result.V = V.copy()
                result.elapsed_sec = time.perf_counter() - start
                logger.warning(
                    "custom_decoupled_nr: Singular J22 at iteration %d", i + 1,
                )
                return result

            if np.isnan(dVm_step).any() or np.isinf(dVm_step).any():
                result.failure_reason = (
                    f"NaN/Inf in Q-|V| correction at iteration {i + 1}"
                )
                result.V = V.copy()
                result.elapsed_sec = time.perf_counter() - start
                logger.warning("custom_decoupled_nr: %s", result.failure_reason)
                return result

            Vm[pq] += dVm_step
            V = Vm * exp(1j * Va)

        # Recompute full mismatch for convergence check
        F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
        normF = norm(F, np.inf)
        result.convergence_history.append(float(normF))

        if normF < tol:
            result.converged = True
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.info(
                "custom_decoupled_nr: converged in %d iterations (mismatch=%.2e)",
                i + 1,
                normF,
            )
            return result

        if initial_normF > 0 and normF > _DIVERGENCE_FACTOR * initial_normF:
            result.failure_reason = (
                f"Diverging mismatch at iteration {i + 1}: "
                f"{normF:.2e} > {_DIVERGENCE_FACTOR}x initial {initial_normF:.2e}"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_decoupled_nr: %s", result.failure_reason)
            return result

    result.failure_reason = (
        f"Did not converge within {max_iter} iterations "
        f"(final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_decoupled_nr: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Continuation Power Flow (Predictor-Corrector)
# ---------------------------------------------------------------------------

def custom_nr_continuation(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> ACMethodResult:
    """Continuation power flow using predictor-corrector for voltage stability.

    The continuation method traces the P–V curve (nose curve) by
    parameterizing the load with a scalar loading factor ``λ``:

    .. math::

        S(\\lambda) = S_{bus} \\cdot \\lambda

    Starting from the base case (``λ = 1.0``), the method performs
    a predictor step (tangent-vector prediction along the solution
    curve) followed by a corrector step (NR solve with an augmented
    system that fixes one variable — either ``λ`` or the most
    sensitive voltage magnitude).

    **Predictor step**: Compute the tangent vector ``[dθ; d|V|; dλ]``
    by solving the augmented Jacobian with the continuation equation.

    **Corrector step**: A standard NR solve of the augmented system
    with the parameterization variable fixed.

    The method returns the voltage solution at the base operating
    point (``λ = 1.0``).  If the method diverges or cannot maintain
    ``λ = 1.0``, the ``failure_reason`` is set accordingly.

    For this implementation, the continuation is simplified:
    we start at ``λ = 0`` (flat start / no load) and ramp up to
    ``λ = 1`` in controlled steps, solving a corrector NR at each
    step to trace the solution path.  This is particularly useful
    for heavily-loaded networks where a direct NR from flat start
    may not converge.

    Args:
        Ybus: Bus admittance matrix (sparse CSC).
        Sbus: Complex power injection vector (per-unit, at full load).
        V0: Initial complex voltage vector.
        ref: Slack bus indices (array-like).
        pv: PV bus indices (array-like).
        pq: PQ bus indices (array-like).
        max_iter: Maximum number of continuation steps.
        tol: Convergence tolerance (infinity-norm of mismatch).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()
    start = time.perf_counter()

    ref = np.asarray(ref, dtype=int)
    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = r_[pv, pq]

    npvpq = len(pvpq)
    npq = len(pq)

    if npvpq == 0:
        result.converged = True
        result.V = V0.copy()
        result.elapsed_sec = time.perf_counter() - start
        return result

    # --- Continuation parameters ---
    # Number of load ramp steps from λ=0 to λ=1
    n_steps = min(max_iter, 10)
    # Maximum corrector NR iterations per step
    max_corrector = max(max_iter // n_steps, 3)

    V = V0.copy().astype(complex)
    Va = np.angle(V)
    Vm = np.abs(V)

    total_iter = 0

    # Ramp loading factor from 0 to 1
    for step_idx in range(n_steps):
        lam = (step_idx + 1) / n_steps
        Sbus_lam = Sbus * lam

        # --- Predictor: use previous solution as starting point ---
        # (Tangent prediction is implicit — we simply reuse the last V)

        # --- Corrector: NR solve at the current loading level ---
        for j in range(max_corrector):
            total_iter += 1
            result.iterations = total_iter

            issue = _check_numerical_issues(V, total_iter)
            if issue is not None:
                result.failure_reason = issue
                result.V = V.copy()
                result.elapsed_sec = time.perf_counter() - start
                logger.warning("custom_nr_continuation: %s", issue)
                return result

            F, _ = _compute_mismatch(V, Ybus, Sbus_lam, pv, pq)
            normF = norm(F, np.inf)
            result.convergence_history.append(float(normF))

            if normF < tol:
                break  # Corrector converged for this λ step

            # Compute Jacobian and solve
            dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
            J = _build_jacobian(dS_dVa, dS_dVm, pv, pq)

            try:
                dx = spsolve(J.tocsc(), -F)
            except LinAlgError:
                result.failure_reason = (
                    f"Singular Jacobian at continuation step {step_idx + 1} "
                    f"(λ={lam:.2f})"
                )
                result.V = V.copy()
                result.elapsed_sec = time.perf_counter() - start
                logger.warning("custom_nr_continuation: %s", result.failure_reason)
                return result

            if np.isnan(dx).any() or np.isinf(dx).any():
                result.failure_reason = (
                    f"NaN/Inf in correction at continuation step {step_idx + 1} "
                    f"(λ={lam:.2f})"
                )
                result.V = V.copy()
                result.elapsed_sec = time.perf_counter() - start
                logger.warning("custom_nr_continuation: %s", result.failure_reason)
                return result

            Va[pvpq] += dx[:npvpq]
            Vm[pq] += dx[npvpq:]
            V = Vm * exp(1j * Va)

        # Check if corrector diverged at this step
        F, _ = _compute_mismatch(V, Ybus, Sbus_lam, pv, pq)
        normF_step = norm(F, np.inf)
        if normF_step > tol * 1e3:
            result.failure_reason = (
                f"Corrector did not converge at continuation step {step_idx + 1} "
                f"(λ={lam:.2f}, mismatch={normF_step:.2e})"
            )
            result.V = V.copy()
            result.elapsed_sec = time.perf_counter() - start
            logger.warning("custom_nr_continuation: %s", result.failure_reason)
            return result

    # Final check at full loading (λ = 1)
    F, _ = _compute_mismatch(V, Ybus, Sbus, pv, pq)
    normF = norm(F, np.inf)
    if normF not in [h for h in result.convergence_history[-1:]]:
        result.convergence_history.append(float(normF))

    if normF < tol:
        result.converged = True
        result.V = V.copy()
        result.elapsed_sec = time.perf_counter() - start
        logger.info(
            "custom_nr_continuation: converged in %d total iterations (mismatch=%.2e)",
            total_iter,
            normF,
        )
        return result

    result.failure_reason = (
        f"Did not converge at full loading (λ=1.0) after {total_iter} "
        f"total iterations (final mismatch={normF:.2e})"
    )
    result.V = V.copy()
    result.elapsed_sec = time.perf_counter() - start
    logger.warning("custom_nr_continuation: %s", result.failure_reason)
    return result


# ---------------------------------------------------------------------------
# Method registry for all 15 custom solvers
# ---------------------------------------------------------------------------

def get_custom_solver_methods() -> List[Dict[str, Any]]:
    """Return descriptors for all 15 custom PYPOWER-level solver methods.

    Each descriptor is a dict with:

    * ``name`` — Unique method identifier (e.g. ``"custom_nr"``).
    * ``category`` — One of ``"custom_nr"``, ``"custom_iterative"``,
      or ``"custom_decoupled"``.
    * ``description`` — Human-readable description of the algorithm.
    * ``solver_func`` — Reference to the solver function.

    Returns:
        List of 15 method descriptor dicts.
    """
    return [
        # --- Newton-Raphson variants (7) ---
        {
            "name": "custom_nr",
            "category": "custom_nr",
            "description": (
                "Standard Newton-Raphson from scratch. Polar coordinates, "
                "full Jacobian rebuild every iteration."
            ),
            "solver_func": custom_nr,
        },
        {
            "name": "custom_nr_linesearch",
            "category": "custom_nr",
            "description": (
                "NR with Armijo backtracking line search. Improves robustness "
                "for ill-conditioned or heavily-loaded networks."
            ),
            "solver_func": custom_nr_linesearch,
        },
        {
            "name": "custom_nr_iwamoto",
            "category": "custom_nr",
            "description": (
                "NR with Iwamoto optimal step multiplier. Quadratic fit "
                "determines best step size each iteration."
            ),
            "solver_func": custom_nr_iwamoto,
        },
        {
            "name": "custom_nr_rectangular",
            "category": "custom_nr",
            "description": (
                "NR in rectangular coordinates (V = e + jf). "
                "Includes voltage-magnitude constraint for PV buses."
            ),
            "solver_func": custom_nr_rectangular,
        },
        {
            "name": "custom_nr_current",
            "category": "custom_nr",
            "description": (
                "NR with current-injection mismatch formulation. "
                "Better numerical conditioning for high-impedance branches."
            ),
            "solver_func": custom_nr_current,
        },
        {
            "name": "custom_nr_dishonest",
            "category": "custom_nr",
            "description": (
                "Dishonest NR — recomputes Jacobian every 3 iterations. "
                "Reduces per-iteration cost at the expense of convergence rate."
            ),
            "solver_func": custom_nr_dishonest,
        },
        {
            "name": "custom_nr_levenberg",
            "category": "custom_nr",
            "description": (
                "Levenberg-Marquardt damped NR. Adds adaptive λI damping "
                "to Jacobian for improved robustness."
            ),
            "solver_func": custom_nr_levenberg,
        },
        # --- Iterative methods (4) ---
        {
            "name": "custom_gs",
            "category": "custom_iterative",
            "description": (
                "Standard Gauss-Seidel. Sequential bus voltage updates "
                "with linear convergence."
            ),
            "solver_func": custom_gs,
        },
        {
            "name": "custom_gs_accelerated",
            "category": "custom_iterative",
            "description": (
                "Gauss-Seidel with acceleration factor ω=1.6. "
                "Over-relaxation amplifies updates to accelerate convergence."
            ),
            "solver_func": custom_gs_accelerated,
        },
        {
            "name": "custom_jacobi",
            "category": "custom_iterative",
            "description": (
                "Jacobi iteration with simultaneous bus voltage updates. "
                "Fully parallelizable but slower convergence than GS."
            ),
            "solver_func": custom_jacobi,
        },
        {
            "name": "custom_gs_sor",
            "category": "custom_iterative",
            "description": (
                "Successive Over-Relaxation (SOR) with ω=1.5. "
                "Blends old voltage with GS update for tunable convergence."
            ),
            "solver_func": custom_gs_sor,
        },
        # --- Decoupled / fast methods (4) ---
        {
            "name": "custom_fdpf_bx",
            "category": "custom_decoupled",
            "description": (
                "Fast Decoupled PF, BX ordering. Constant Bp/Bpp matrices "
                "with shunt-free Bp and full Bpp."
            ),
            "solver_func": custom_fdpf_bx,
        },
        {
            "name": "custom_fdpf_xb",
            "category": "custom_decoupled",
            "description": (
                "Fast Decoupled PF, XB ordering. Full Bp and "
                "shunt-free Bpp. Better for high R/X networks."
            ),
            "solver_func": custom_fdpf_xb,
        },
        {
            "name": "custom_decoupled_nr",
            "category": "custom_decoupled",
            "description": (
                "Decoupled NR — alternates P-θ and Q-|V| NR sub-iterations. "
                "Recomputes diagonal Jacobian blocks each iteration."
            ),
            "solver_func": custom_decoupled_nr,
        },
        {
            "name": "custom_nr_continuation",
            "category": "custom_decoupled",
            "description": (
                "Continuation PF (predictor-corrector). Ramps loading factor "
                "from 0 to 1 for voltage stability analysis."
            ),
            "solver_func": custom_nr_continuation,
        },
    ]
