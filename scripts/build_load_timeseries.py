"""Generate time-series load multiplier data for MATPOWER power flow.

Reads sample daily load curves + annual trend from YAML,
generates pre-computed multiplier arrays for each scenario:
  - 1h   : single peak-hour snapshot
  - 24h  : one day at 1h and 30min intervals
  - 8760h: full year at 1h intervals

Output: output/matpower_alljapan/load_timeseries.mat
"""

import os
import sys
import datetime

import numpy as np
import yaml
from scipy.io import savemat
from scipy.interpolate import interp1d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    # ── Load profile YAML ──
    yaml_path = os.path.join(PROJECT_ROOT, "data", "reference", "load_profiles.yaml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    daily_wd = np.array(cfg["daily_curves"]["weekday"], dtype=np.float64)  # [24]
    daily_we = np.array(cfg["daily_curves"]["weekend"], dtype=np.float64)  # [24]
    monthly = np.array([cfg["monthly_factors"][m] for m in range(1, 13)], dtype=np.float64)  # [12]

    base_year = cfg.get("base_year", 2024)
    ref = cfg.get("reference", {})
    ref_month = ref.get("month", 8)
    ref_hour = ref.get("hour", 14)
    ref_day_type = ref.get("day_type", "weekday")

    # Reference multiplier (base case loads correspond to this point)
    ref_daily = daily_wd[ref_hour] if ref_day_type == "weekday" else daily_we[ref_hour]
    ref_monthly = monthly[ref_month - 1]
    ref_mult = ref_daily * ref_monthly  # should be 1.0 × 1.0 = 1.0

    print(f"Reference point: month={ref_month}, hour={ref_hour}, "
          f"daily={ref_daily:.3f}, monthly={ref_monthly:.3f}, combined={ref_mult:.3f}")

    # ── 30-min interpolation of daily curves ──
    hours_24 = np.arange(24)
    hours_48 = np.arange(48) * 0.5  # 0.0, 0.5, 1.0, ..., 23.5

    # Wrap around for smooth interpolation (add hour 24 = hour 0)
    wd_wrap = np.append(daily_wd, daily_wd[0])
    we_wrap = np.append(daily_we, daily_we[0])
    hours_wrap = np.arange(25)

    f_wd = interp1d(hours_wrap, wd_wrap, kind="cubic")
    f_we = interp1d(hours_wrap, we_wrap, kind="cubic")

    daily_wd_30min = f_wd(hours_48)
    daily_we_30min = f_we(hours_48)

    # ── 1h scenario: single peak hour ──
    ts_1h_mult = np.array([1.0])  # base case = peak
    ts_1h_hour = np.array([float(ref_hour)])
    ts_1h_month = np.array([float(ref_month)])

    # ── 24h @ 1h scenario: summer weekday ──
    ts_24h_1h_mult = (daily_wd / ref_mult).clip(0.01, 2.0)  # relative to base case
    ts_24h_1h_hour = np.arange(24, dtype=np.float64)

    # ── 24h @ 30min scenario: summer weekday ──
    ts_24h_30min_mult = (daily_wd_30min / ref_mult).clip(0.01, 2.0)
    ts_24h_30min_hour = hours_48

    # ── 8760h scenario: full year ──
    jan1 = datetime.datetime(base_year, 1, 1, 0, 0)
    ts_8760_mult = np.zeros(8760)
    ts_8760_month = np.zeros(8760)
    ts_8760_hour = np.zeros(8760)
    ts_8760_weekday = np.zeros(8760)  # 1=weekday, 0=weekend

    for h in range(8760):
        dt = jan1 + datetime.timedelta(hours=h)
        m = dt.month
        hod = dt.hour
        is_wd = dt.weekday() < 5  # Mon-Fri

        daily_val = daily_wd[hod] if is_wd else daily_we[hod]
        monthly_val = monthly[m - 1]
        raw_mult = daily_val * monthly_val / ref_mult

        ts_8760_mult[h] = max(raw_mult, 0.01)
        ts_8760_month[h] = m
        ts_8760_hour[h] = hod
        ts_8760_weekday[h] = 1.0 if is_wd else 0.0

    # ── Save to .mat ──
    out_dir = os.path.join(PROJECT_ROOT, "output", "matpower_alljapan")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "load_timeseries.mat")

    data = {
        # Raw curves
        "daily_weekday": daily_wd.reshape(-1, 1),
        "daily_weekend": daily_we.reshape(-1, 1),
        "daily_weekday_30min": daily_wd_30min.reshape(-1, 1),
        "daily_weekend_30min": daily_we_30min.reshape(-1, 1),
        "monthly_factors": monthly.reshape(-1, 1),
        "ref_mult": ref_mult,
        # 1h scenario
        "ts_1h_mult": ts_1h_mult.reshape(-1, 1),
        "ts_1h_hour": ts_1h_hour.reshape(-1, 1),
        "ts_1h_month": ts_1h_month.reshape(-1, 1),
        # 24h @ 1h
        "ts_24h_1h_mult": ts_24h_1h_mult.reshape(-1, 1),
        "ts_24h_1h_hour": ts_24h_1h_hour.reshape(-1, 1),
        # 24h @ 30min
        "ts_24h_30min_mult": ts_24h_30min_mult.reshape(-1, 1),
        "ts_24h_30min_hour": ts_24h_30min_hour.reshape(-1, 1),
        # 8760h
        "ts_8760_mult": ts_8760_mult.reshape(-1, 1),
        "ts_8760_month": ts_8760_month.reshape(-1, 1),
        "ts_8760_hour": ts_8760_hour.reshape(-1, 1),
        "ts_8760_weekday": ts_8760_weekday.reshape(-1, 1),
    }

    savemat(out_path, data, do_compression=True)
    print(f"\nSaved: {out_path}")
    print(f"  1h:         {len(ts_1h_mult)} step")
    print(f"  24h @ 1h:   {len(ts_24h_1h_mult)} steps, mult range [{ts_24h_1h_mult.min():.3f}, {ts_24h_1h_mult.max():.3f}]")
    print(f"  24h @ 30min:{len(ts_24h_30min_mult)} steps, mult range [{ts_24h_30min_mult.min():.3f}, {ts_24h_30min_mult.max():.3f}]")
    print(f"  8760h:      {len(ts_8760_mult)} steps, mult range [{ts_8760_mult.min():.3f}, {ts_8760_mult.max():.3f}]")

    # ── Stats ──
    print(f"\n  Annual load factor statistics:")
    print(f"    Mean:   {ts_8760_mult.mean():.3f}")
    print(f"    Median: {np.median(ts_8760_mult):.3f}")
    print(f"    Peak:   {ts_8760_mult.max():.3f} (hour {np.argmax(ts_8760_mult)})")
    print(f"    Valley: {ts_8760_mult.min():.3f} (hour {np.argmin(ts_8760_mult)})")


if __name__ == "__main__":
    main()
