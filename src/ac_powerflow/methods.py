"""Central registry of all AC power flow methods.

Aggregates method descriptors from ``pandapower_methods`` (5 built-in
wrappers) and ``custom_solvers`` (15 custom PYPOWER-level solvers) into
a unified registry of ~20 ``MethodDescriptor`` objects.

Usage::

    from src.ac_powerflow.methods import get_all_methods

    for method in get_all_methods():
        print(f"{method.id} ({method.category}): {method.name}")

Categories:

* ``pandapower`` — Built-in pandapower algorithm wrappers (5 methods)
* ``custom_nr`` — Custom Newton-Raphson variants (7 methods)
* ``custom_iterative`` — Custom iterative methods (4 methods)
* ``custom_decoupled`` — Custom decoupled / fast methods (4 methods)
"""

from dataclasses import dataclass
from typing import Any, Callable, List

from src.ac_powerflow.custom_solvers import get_custom_solver_methods
from src.ac_powerflow.pandapower_methods import get_pandapower_methods
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MethodDescriptor:
    """Metadata for a single AC power flow method.

    Attributes:
        id: Unique identifier string (e.g. ``"pp_nr"``, ``"custom_gs"``).
        name: Human-readable method name (e.g. ``"Newton-Raphson via pandapower"``).
        category: Method category — one of ``"pandapower"``,
            ``"custom_nr"``, ``"custom_iterative"``, ``"custom_decoupled"``.
        description: Detailed description of the algorithm.
        solver_fn: Callable that executes the solver.  For pandapower
            wrappers the signature is ``(net, max_iteration, tolerance)
            -> ACMethodResult``.  For custom solvers the signature is
            ``(Ybus, Sbus, V0, ref, pv, pq, max_iter, tol)
            -> ACMethodResult``.
    """

    id: str
    name: str
    category: str
    description: str
    solver_fn: Callable[..., Any]


# Valid category labels for validation.
_VALID_CATEGORIES = frozenset([
    "pandapower",
    "custom_nr",
    "custom_iterative",
    "custom_decoupled",
])


def _build_method_descriptor(raw: dict) -> MethodDescriptor:
    """Convert a raw method descriptor dict to a ``MethodDescriptor``.

    The dicts returned by ``get_pandapower_methods()`` and
    ``get_custom_solver_methods()`` use the keys ``name``, ``category``,
    ``description``, and ``solver_func``.  This helper maps them to the
    ``MethodDescriptor`` dataclass fields.

    Args:
        raw: Dict with keys ``name``, ``category``, ``description``,
            ``solver_func``.

    Returns:
        Populated ``MethodDescriptor`` instance.

    Raises:
        ValueError: If required keys are missing or the category is
            not recognised.
    """
    required_keys = {"name", "category", "description", "solver_func"}
    missing = required_keys - set(raw.keys())
    if missing:
        raise ValueError(
            f"Method descriptor missing required keys: {sorted(missing)}"
        )

    category = raw["category"]
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"Unknown method category '{category}'. "
            f"Valid categories: {sorted(_VALID_CATEGORIES)}"
        )

    return MethodDescriptor(
        id=raw["name"],
        name=raw["name"],
        category=category,
        description=raw["description"],
        solver_fn=raw["solver_func"],
    )


def get_all_methods() -> List[MethodDescriptor]:
    """Return descriptors for all registered AC power flow methods.

    Aggregates methods from:

    * ``get_pandapower_methods()`` — 5 pandapower built-in wrappers
    * ``get_custom_solver_methods()`` — 15 custom PYPOWER-level solvers

    Returns:
        List of ~20 ``MethodDescriptor`` objects sorted by category
        then by id within each category.
    """
    raw_methods: List[dict] = []
    raw_methods.extend(get_pandapower_methods())
    raw_methods.extend(get_custom_solver_methods())

    descriptors: List[MethodDescriptor] = []
    for raw in raw_methods:
        try:
            descriptor = _build_method_descriptor(raw)
            descriptors.append(descriptor)
        except ValueError as exc:
            logger.warning("Skipping invalid method descriptor: %s", exc)

    # Sort: pandapower first, then custom_nr, custom_iterative, custom_decoupled
    category_order = {
        "pandapower": 0,
        "custom_nr": 1,
        "custom_iterative": 2,
        "custom_decoupled": 3,
    }
    descriptors.sort(key=lambda m: (category_order.get(m.category, 99), m.id))

    logger.info(
        "Registered %d AC power flow methods across %d categories",
        len(descriptors),
        len(set(m.category for m in descriptors)),
    )

    return descriptors


def get_methods_by_category(category: str) -> List[MethodDescriptor]:
    """Return method descriptors filtered by category.

    Args:
        category: Category string (e.g. ``"pandapower"``, ``"custom_nr"``).

    Returns:
        List of ``MethodDescriptor`` objects in the given category.

    Raises:
        ValueError: If the category is not recognised.
    """
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid categories: {sorted(_VALID_CATEGORIES)}"
        )

    return [m for m in get_all_methods() if m.category == category]


def get_method_by_id(method_id: str) -> MethodDescriptor:
    """Look up a single method descriptor by its unique id.

    Args:
        method_id: Method identifier (e.g. ``"pp_nr"``, ``"custom_gs"``).

    Returns:
        The matching ``MethodDescriptor``.

    Raises:
        KeyError: If no method with the given id is registered.
    """
    for method in get_all_methods():
        if method.id == method_id:
            return method

    available = [m.id for m in get_all_methods()]
    raise KeyError(
        f"No method with id '{method_id}'. "
        f"Available methods: {available}"
    )
