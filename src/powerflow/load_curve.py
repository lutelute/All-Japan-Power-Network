"""Synthetic load curve generation for time-series power flow analysis.

Generates realistic Japanese demand patterns with three components:
    - **Daily cycle (24h):** Trough at 04:00 (~55% of peak), dual peaks at
      14:00 and 18:00.
    - **Seasonal variation (365d):** Summer (Jul-Aug) and winter (Jan-Feb)
      peaks; spring (Apr-May) and autumn (Oct-Nov) troughs.
    - **Weekly pattern:** Weekend demand ~90% of weekday.

All scale factors are normalized so that the maximum equals 1.0,
matching the regional peak demand in ``regional_demand.yaml``.

Usage::

    from src.powerflow.load_curve import generate_daily_curve, generate_annual_curve

    daily = generate_daily_curve()           # shape (24,)
    annual = generate_annual_curve(8760)     # shape (8760,)
"""

import numpy as np


def generate_daily_curve() -> np.ndarray:
    """Generate a 24-element daily load scale factor array.

    Models Japan's typical demand shape with a deep overnight trough
    and two daytime peaks (afternoon + early evening).

    Returns:
        Array of shape ``(24,)`` with values in ``[0, 1]``.
        Hour 0 corresponds to midnight (00:00).
    """
    hours = np.arange(24, dtype=float)

    # Base sinusoidal: trough at 04:00 (~0.55), broad daytime plateau
    base = 0.73 + 0.18 * np.cos(2 * np.pi * (hours - 16) / 24)

    # Morning ramp-up shoulder centred at 10:00 (σ ≈ 3h)
    morning = 0.06 * np.exp(-0.5 * ((hours - 10) / 3) ** 2)

    # Afternoon peak centred at 14:00 (σ ≈ 1.5h)
    afternoon = 0.18 * np.exp(-0.5 * ((hours - 14) / 1.5) ** 2)

    # Evening peak centred at 18:00 (σ ≈ 1.2h)
    evening = 0.14 * np.exp(-0.5 * ((hours - 18) / 1.2) ** 2)

    curve = base + morning + afternoon + evening

    # Normalise so max == 1.0
    curve /= curve.max()
    return curve


def _seasonal_factor(day_of_year: np.ndarray) -> np.ndarray:
    """Compute seasonal scale factor for each day of the year.

    Japan's electricity demand peaks in summer (cooling) and winter
    (heating), with troughs in mild spring and autumn.

    Args:
        day_of_year: Array of day-of-year values (1-based).

    Returns:
        Scale factors (peak season ≈ 1.0, trough ≈ 0.80).
    """
    # Double-peak cosine: peaks around day 30 (winter) and day 210 (summer)
    winter = 0.10 * np.cos(2 * np.pi * (day_of_year - 30) / 365)
    summer = 0.10 * np.cos(2 * np.pi * (day_of_year - 210) / 365)

    factor = 0.90 + np.maximum(winter, summer)
    # Normalise so max == 1.0
    factor /= factor.max()
    return factor


def _weekend_factor(hour_index: np.ndarray) -> np.ndarray:
    """Apply weekend discount (~90% of weekday demand).

    Assumes hour 0 corresponds to Monday 00:00.

    Args:
        hour_index: Array of hour indices starting from 0.

    Returns:
        Scale factors: 1.0 for weekdays, 0.90 for weekends.
    """
    day_of_week = (hour_index // 24) % 7  # 0=Mon … 6=Sun
    factor = np.where(day_of_week >= 5, 0.90, 1.0)
    return factor


def generate_annual_curve(hours: int = 8760) -> np.ndarray:
    """Generate an annual load scale factor array.

    Combines daily, seasonal, and weekly patterns into a single
    normalised curve.

    Args:
        hours: Number of hourly time steps (default 8760 for one year).
            Values other than 8760 are allowed (e.g. 24, 168).

    Returns:
        Array of shape ``(hours,)`` with values in ``(0, 1]``.
        Index 0 corresponds to January 1 00:00 (Monday).
    """
    idx = np.arange(hours, dtype=float)

    # Daily pattern (repeating 24h cycle)
    daily_24 = generate_daily_curve()
    daily = daily_24[np.mod(idx.astype(int), 24)]

    # Seasonal pattern (day-of-year, 1-based)
    day_of_year = (idx / 24).astype(int) % 365 + 1
    seasonal = _seasonal_factor(day_of_year)

    # Weekend pattern
    weekend = _weekend_factor(idx)

    curve = daily * seasonal * weekend

    # Normalise so max == 1.0
    curve /= curve.max()
    return curve
