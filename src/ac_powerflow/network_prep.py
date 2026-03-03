"""Network preparation utilities for AC power flow solvers.

Extracts PYPOWER internal matrices (Ybus, Sbus, V0) and bus classification
(ref, pv, pq) from a pandapower network for use by custom AC power flow
solvers.

Handles the bus index offset between pandapower bus DataFrame indices and
PYPOWER internal ordering (which may differ due to bus fusing).

Usage::

    from src.ac_powerflow.network_prep import prepare_network

    data = prepare_network(net)
    print(data.Ybus.shape, data.ref, data.pv, data.pq)
"""

from dataclasses import dataclass
from typing import Any, List

import numpy as np
from scipy import sparse

import pandapower as pp
from pandapower.pypower.idx_bus import VM
from pandapower.pypower.idx_gen import GEN_BUS, VG

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NetworkData:
    """Extracted PYPOWER internal matrices for custom AC power flow solvers.

    All bus indices use PYPOWER internal ordering, which may differ from
    pandapower bus DataFrame indices due to bus fusing.  Do not mix
    these index spaces.

    Attributes:
        Ybus: Bus admittance matrix (sparse CSC format).
        Sbus: Complex power injection vector (per-unit on baseMVA).
        V0: Initial complex voltage vector (flat start with VM setpoints).
        ref: Indices of slack (reference) buses in PYPOWER internal ordering.
        pv: Indices of PV buses in PYPOWER internal ordering.
        pq: Indices of PQ buses in PYPOWER internal ordering.
        baseMVA: System base power (MVA).
    """

    Ybus: sparse.csc_matrix
    Sbus: np.ndarray
    V0: np.ndarray
    ref: List[int]
    pv: List[int]
    pq: List[int]
    baseMVA: float

    @property
    def summary(self) -> dict:
        """Return a compact summary for logging."""
        return {
            "n_buses": self.Ybus.shape[0],
            "n_ref": len(self.ref),
            "n_pv": len(self.pv),
            "n_pq": len(self.pq),
            "baseMVA": self.baseMVA,
        }


def prepare_network(net: Any) -> NetworkData:
    """Extract PYPOWER internal matrices from a pandapower network.

    Calls ``pp.runpp()`` internally to populate ``net._ppc`` with
    PYPOWER internal data structures, then extracts the admittance
    matrix, power injections, initial voltage, and bus classification
    needed by custom AC power flow solvers.

    Note:
        Bus indices in the returned data use PYPOWER internal ordering,
        which may differ from pandapower bus DataFrame indices due to
        bus fusing.  Do not mix these index spaces.

    Args:
        net: A pandapower network with at least one ext_grid (slack bus),
            buses, and branches.

    Returns:
        NetworkData with all matrices and bus classifications.

    Raises:
        RuntimeError: If the internal ppc structure cannot be populated.
    """
    _populate_ppc(net)

    ppc = net._ppc
    internal = ppc["internal"]

    # Extract admittance matrix (convert to CSC for spsolve compatibility)
    Ybus = internal["Ybus"]
    if not isinstance(Ybus, sparse.csc_matrix):
        Ybus = Ybus.tocsc()

    # Extract complex power injections (per-unit on baseMVA)
    Sbus = np.array(internal["Sbus"], dtype=complex)

    # Use the internal V vector as the initial voltage.
    # This is correctly sized for the internal bus ordering (matches Ybus),
    # unlike ppc["bus"] which may include out-of-service buses.
    V_internal = np.array(internal["V"], dtype=complex)
    V0 = _build_initial_voltage_from_internal(V_internal, ppc)

    # Extract bus classification arrays
    ref = list(internal["ref"].astype(int))
    pv = list(internal["pv"].astype(int))
    pq = list(internal["pq"].astype(int))

    baseMVA = float(ppc["baseMVA"])

    data = NetworkData(
        Ybus=Ybus,
        Sbus=Sbus,
        V0=V0,
        ref=ref,
        pv=pv,
        pq=pq,
        baseMVA=baseMVA,
    )

    logger.info("Network prepared for AC solvers: %s", data.summary)

    return data


def _populate_ppc(net: Any) -> None:
    """Run power flow to populate ``net._ppc`` internal structures.

    The power flow solver may fail at the result extraction phase (e.g.,
    read-only DataFrame in some pandapower versions), but the internal
    PYPOWER matrices (Ybus, Sbus, V) are populated during the solve step
    before extraction.

    Args:
        net: pandapower network.

    Raises:
        RuntimeError: If ``_ppc`` or its internal data are not populated.
    """
    try:
        pp.runpp(net, numba=False)
    except Exception as exc:
        # Power flow may fail at result extraction but still populate _ppc.
        # Only raise if _ppc internals are truly missing.
        logger.debug(
            "pp.runpp() raised %s: %s (checking if _ppc was populated)",
            type(exc).__name__,
            exc,
        )

    # Validate that _ppc was populated
    if not hasattr(net, "_ppc") or net._ppc is None:
        raise RuntimeError(
            "Failed to populate net._ppc — "
            "ensure the network has buses, branches, and an ext_grid"
        )

    internal = net._ppc.get("internal")
    if internal is None:
        raise RuntimeError("net._ppc['internal'] is missing after runpp()")

    required_keys = ("Ybus", "Sbus", "V", "ref", "pv", "pq")
    missing = [k for k in required_keys if k not in internal or internal[k] is None]
    if missing:
        raise RuntimeError(
            f"net._ppc['internal'] missing required keys: {missing}"
        )


def _build_initial_voltage_from_internal(
    V_internal: np.ndarray,
    ppc: dict,
) -> np.ndarray:
    """Build flat-start initial voltage from the internal V vector.

    Uses pandapower's internal V vector (which is correctly sized for
    the internal bus ordering) as a base, then resets angles to zero
    for a flat start while preserving voltage magnitude setpoints.

    Args:
        V_internal: Complex voltage vector from ``ppc["internal"]["V"]``.
        ppc: The PYPOWER case dict (``net._ppc``).

    Returns:
        Complex voltage vector (flat start) for internal buses.
    """
    nb = len(V_internal)

    # Use magnitudes from the converged/setpoint solution, angles = 0
    V0 = np.abs(V_internal).astype(complex)

    # Ensure all magnitudes are positive (fix any zeros)
    zero_mask = np.abs(V0) < 1e-10
    V0[zero_mask] = 1.0 + 0j

    return V0


def _build_initial_voltage(ppc: dict) -> np.ndarray:
    """Construct flat-start initial voltage vector with VM setpoints.

    For PV and slack (ref) buses, the voltage magnitude is set to the
    generator voltage setpoint (``VG``).  For PQ buses, the magnitude
    defaults to 1.0 p.u.  All voltage angles are initialized to zero
    (flat start).

    Args:
        ppc: The PYPOWER case dict (``net._ppc``).

    Returns:
        Complex voltage vector (flat start) for all internal buses.

    Note:
        This function uses ``ppc["bus"].shape[0]`` which may include
        out-of-service buses.  Prefer ``_build_initial_voltage_from_internal``
        for networks with topology issues.
    """
    bus = ppc["bus"]
    gen = ppc["gen"]
    nb = bus.shape[0]

    # Start with flat voltage: |V| = 1.0, angle = 0 for all buses
    V0 = np.ones(nb, dtype=complex)

    # Override magnitudes for generator buses (PV and ref) with their
    # voltage setpoints from the gen data.
    for i in range(gen.shape[0]):
        bus_idx = int(gen[i, GEN_BUS])
        vm_setpoint = gen[i, VG]
        if 0 <= bus_idx < nb and np.isfinite(vm_setpoint) and vm_setpoint > 0:
            V0[bus_idx] = vm_setpoint + 0j

    return V0
