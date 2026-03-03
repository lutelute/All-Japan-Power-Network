"""Pandapower built-in algorithm wrappers for AC power flow.

Provides 5 wrapper functions around ``pp.runpp()`` with different
algorithm options.  Each wrapper returns a standardized
``ACMethodResult`` with convergence status, iteration count, elapsed
time, and failure reason on non-convergence.

Available wrappers:

* ``pp_nr`` — Newton-Raphson (quadratic convergence, default method)
* ``pp_iwamoto_nr`` — Iwamoto damped NR (step-size optimization)
* ``pp_gs`` — Gauss-Seidel (linear convergence, robust but slow)
* ``pp_fdbx`` — Fast Decoupled BX variant (decoupled P-θ / Q-V)
* ``pp_fdxb`` — Fast Decoupled XB variant (alternative ordering)

Usage::

    from src.ac_powerflow.pandapower_methods import get_pandapower_methods

    methods = get_pandapower_methods()
    for m in methods:
        print(m["name"], m["description"])
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pandapower as pp

from src.ac_powerflow.solver_interface import ACMethodResult, PandapowerSolverFunc
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Default solver parameters for all pandapower wrappers.
_DEFAULT_MAX_ITERATION = 20
_DEFAULT_TOLERANCE = 1e-8


def _run_pp_algorithm(
    net: Any,
    algorithm: str,
    max_iteration: int = _DEFAULT_MAX_ITERATION,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ACMethodResult:
    """Run ``pp.runpp()`` with a specific algorithm and wrap the result.

    This is the shared implementation for all pandapower wrappers.
    It handles convergence checking, exception catching (non-convergence,
    singular matrix), and timing.

    Args:
        net: pandapower network (modified in place with results).
        algorithm: pandapower algorithm string
            (``"nr"``, ``"iwamoto_nr"``, ``"gs"``, ``"fdbx"``, ``"fdxb"``).
        max_iteration: Maximum number of solver iterations.
        tolerance: Convergence tolerance (power mismatch in p.u.).

    Returns:
        ACMethodResult with convergence status and solver metrics.
    """
    result = ACMethodResult()

    start = time.perf_counter()
    try:
        pp.runpp(
            net,
            algorithm=algorithm,
            numba=False,
            max_iteration=max_iteration,
            tolerance_mva=tolerance,
        )
        elapsed = time.perf_counter() - start

        result.converged = bool(net.converged)
        result.elapsed_sec = elapsed

        # Extract iteration count from pandapower internals when available
        if hasattr(net, "_ppc") and net._ppc is not None:
            internal = net._ppc.get("internal", {})
            if "iterations" in internal:
                result.iterations = int(internal["iterations"])

        if not result.converged:
            result.failure_reason = (
                f"pandapower {algorithm} did not converge "
                f"within {max_iteration} iterations"
            )

        logger.info(
            "pp_%s: converged=%s, iterations=%d, elapsed=%.4fs",
            algorithm,
            result.converged,
            result.iterations,
            result.elapsed_sec,
        )

    except pp.powerflow.LoadflowNotConverged:
        elapsed = time.perf_counter() - start
        result.converged = False
        result.elapsed_sec = elapsed
        result.failure_reason = (
            f"pandapower {algorithm} did not converge "
            f"within {max_iteration} iterations"
        )
        logger.warning("pp_%s: %s", algorithm, result.failure_reason)

    except Exception as exc:
        elapsed = time.perf_counter() - start
        result.converged = False
        result.elapsed_sec = elapsed

        exc_msg = str(exc)
        exc_type = type(exc).__name__

        # Classify common failure modes
        if "singular" in exc_msg.lower():
            result.failure_reason = f"Singular matrix in {algorithm}: {exc_msg}"
        elif "nan" in exc_msg.lower() or "inf" in exc_msg.lower():
            result.failure_reason = (
                f"Numerical instability in {algorithm}: {exc_msg}"
            )
        else:
            result.failure_reason = f"{exc_type} in {algorithm}: {exc_msg}"

        logger.warning("pp_%s failed: %s", algorithm, result.failure_reason)

    return result


# ---------------------------------------------------------------------------
# Individual pandapower wrapper functions
# ---------------------------------------------------------------------------

def pp_nr(
    net: Any,
    max_iteration: int = _DEFAULT_MAX_ITERATION,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ACMethodResult:
    """Newton-Raphson via pandapower.

    Standard NR with quadratic convergence.  This is pandapower's
    default and most reliable algorithm for well-conditioned networks.

    Args:
        net: pandapower network.
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance (MVA).

    Returns:
        ACMethodResult with solver outcome.
    """
    return _run_pp_algorithm(net, "nr", max_iteration, tolerance)


def pp_iwamoto_nr(
    net: Any,
    max_iteration: int = _DEFAULT_MAX_ITERATION,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ACMethodResult:
    """Iwamoto damped Newton-Raphson via pandapower.

    Uses optimal step-size multiplier to improve convergence on
    ill-conditioned systems where standard NR may oscillate.

    Args:
        net: pandapower network.
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance (MVA).

    Returns:
        ACMethodResult with solver outcome.
    """
    return _run_pp_algorithm(net, "iwamoto_nr", max_iteration, tolerance)


def pp_gs(
    net: Any,
    max_iteration: int = _DEFAULT_MAX_ITERATION,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ACMethodResult:
    """Gauss-Seidel via pandapower.

    Linear convergence — slower than NR but more robust for some
    network topologies.  May require more iterations.

    Args:
        net: pandapower network.
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance (MVA).

    Returns:
        ACMethodResult with solver outcome.
    """
    return _run_pp_algorithm(net, "gs", max_iteration, tolerance)


def pp_fdbx(
    net: Any,
    max_iteration: int = _DEFAULT_MAX_ITERATION,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ACMethodResult:
    """Fast Decoupled BX variant via pandapower.

    Decouples P-θ and Q-V subproblems using constant B-matrix
    approximations.  BX ordering prioritises the B-matrix in the
    real-power / angle sub-problem.

    Args:
        net: pandapower network.
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance (MVA).

    Returns:
        ACMethodResult with solver outcome.
    """
    return _run_pp_algorithm(net, "fdbx", max_iteration, tolerance)


def pp_fdxb(
    net: Any,
    max_iteration: int = _DEFAULT_MAX_ITERATION,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ACMethodResult:
    """Fast Decoupled XB variant via pandapower.

    Alternative decoupling order to FDBX.  XB ordering prioritises
    the X-matrix in the real-power / angle sub-problem.

    Args:
        net: pandapower network.
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance (MVA).

    Returns:
        ACMethodResult with solver outcome.
    """
    return _run_pp_algorithm(net, "fdxb", max_iteration, tolerance)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def get_pandapower_methods() -> List[Dict[str, Any]]:
    """Return descriptors for all pandapower built-in algorithm wrappers.

    Each descriptor is a dict with:

    * ``name`` — Unique method identifier (e.g. ``"pp_nr"``).
    * ``category`` — ``"pandapower"`` for all wrappers in this module.
    * ``description`` — Human-readable description of the algorithm.
    * ``solver_func`` — Reference to the wrapper function.

    Returns:
        List of 5 method descriptor dicts.
    """
    return [
        {
            "name": "pp_nr",
            "category": "pandapower",
            "description": (
                "Newton-Raphson via pandapower. Quadratic convergence. "
                "Default and most reliable method."
            ),
            "solver_func": pp_nr,
        },
        {
            "name": "pp_iwamoto_nr",
            "category": "pandapower",
            "description": (
                "Iwamoto damped NR via pandapower. Step-size optimization "
                "for ill-conditioned systems."
            ),
            "solver_func": pp_iwamoto_nr,
        },
        {
            "name": "pp_gs",
            "category": "pandapower",
            "description": (
                "Gauss-Seidel via pandapower. Linear convergence, "
                "robust but slower than NR."
            ),
            "solver_func": pp_gs,
        },
        {
            "name": "pp_fdbx",
            "category": "pandapower",
            "description": (
                "Fast Decoupled BX variant via pandapower. "
                "Decouples P-θ and Q-V subproblems."
            ),
            "solver_func": pp_fdbx,
        },
        {
            "name": "pp_fdxb",
            "category": "pandapower",
            "description": (
                "Fast Decoupled XB variant via pandapower. "
                "Alternative decoupling order."
            ),
            "solver_func": pp_fdxb,
        },
    ]
