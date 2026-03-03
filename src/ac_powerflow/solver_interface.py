"""Unified solver interface and result dataclass for AC power flow methods.

Defines the ``ACMethodResult`` dataclass used by all AC power flow solvers
(both pandapower wrappers and custom PYPOWER-level implementations) and
the common solver function type signature.

Usage::

    from src.ac_powerflow.solver_interface import ACMethodResult

    result = ACMethodResult(converged=True, iterations=5, elapsed_sec=0.12)
    print(result.summary)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ACMethodResult:
    """Results from a single AC power flow method execution.

    Attributes:
        converged: Whether the solver reached the tolerance within max iterations.
        iterations: Number of iterations performed.
        V: Complex voltage vector at solution (one entry per bus).
        elapsed_sec: Wall-clock time for the solver execution (seconds).
        convergence_history: Mismatch norm recorded at each iteration.
        failure_reason: Human-readable reason if the solver did not converge.
    """

    converged: bool = False
    iterations: int = 0
    V: Optional[np.ndarray] = None
    elapsed_sec: float = 0.0
    convergence_history: List[float] = field(default_factory=list)
    failure_reason: Optional[str] = None

    @property
    def summary(self) -> dict:
        """Return a compact summary for logging."""
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "final_mismatch": (
                round(self.convergence_history[-1], 8)
                if self.convergence_history
                else None
            ),
            "failure_reason": self.failure_reason,
        }


# Type alias for a custom PYPOWER-level solver function.
# Signature: (Ybus, Sbus, V0, ref, pv, pq, max_iter, tol) -> ACMethodResult
CustomSolverFunc = Callable[
    [
        "scipy.sparse.csc_matrix",  # Ybus
        np.ndarray,                 # Sbus
        np.ndarray,                 # V0
        np.ndarray,                 # ref
        np.ndarray,                 # pv
        np.ndarray,                 # pq
        int,                        # max_iter
        float,                      # tol
    ],
    ACMethodResult,
]

# Type alias for a pandapower wrapper solver function.
# Signature: (net, max_iteration, tolerance) -> ACMethodResult
PandapowerSolverFunc = Callable[
    [
        "pandapower.auxiliary.pandapowerNet",  # net
        int,                                   # max_iteration
        float,                                 # tolerance
    ],
    ACMethodResult,
]
