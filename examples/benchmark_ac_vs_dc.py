#!/usr/bin/env python3
"""AC vs DC power flow benchmark comparison.

Runs DC power flow and all ~20 AC power flow methods across pandapower
standard test cases of increasing size, measuring wall-clock time and
comparing accuracy. Generates comparison figures.

Usage::

    PYTHONPATH=. python scripts/benchmark_ac_vs_dc.py
"""

import copy
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw

# Suppress pandapower/numpy warnings during benchmark
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ac_powerflow.methods import get_all_methods
from src.ac_powerflow.network_prep import prepare_network
from src.ac_powerflow.solver_interface import ACMethodResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output/benchmark"
N_WARMUP = 1
N_REPEAT = 5  # repeat each measurement for stable timing

# Test cases: (label, network_factory, bus_count)
TEST_CASES: List[Tuple[str, Any, int]] = [
    ("case9",    nw.case9,    9),
    ("case14",   nw.case14,   14),
    ("case30",   nw.case30,   30),
    ("case57",   nw.case57,   57),
    ("case118",  nw.case118,  118),
    ("case300",  nw.case300,  300),
    ("case1354", nw.case1354pegase, 1354),
]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_dc(net: Any, n_repeat: int = N_REPEAT) -> Dict[str, Any]:
    """Benchmark DC power flow."""
    times = []
    for _ in range(N_WARMUP):
        net_copy = copy.deepcopy(net)
        pp.rundcpp(net_copy)

    for _ in range(n_repeat):
        net_copy = copy.deepcopy(net)
        t0 = time.perf_counter()
        pp.rundcpp(net_copy)
        times.append(time.perf_counter() - t0)

    return {
        "method_id": "dc",
        "method_name": "DC Power Flow",
        "category": "dc",
        "converged": True,
        "iterations": 0,
        "elapsed_sec": float(np.median(times)),
        "elapsed_all": times,
        "vm_pu": net_copy.res_bus["vm_pu"].values.copy(),
        "va_degree": net_copy.res_bus["va_degree"].values.copy(),
        "p_line_mw": net_copy.res_line["p_from_mw"].values.copy() if len(net_copy.res_line) > 0 else np.array([]),
    }


def bench_ac_method(
    method: Any,
    net: Any,
    network_data: Any,
    n_repeat: int = N_REPEAT,
) -> Dict[str, Any]:
    """Benchmark a single AC power flow method."""
    times = []
    result = None

    last_net_copy = None
    for trial in range(N_WARMUP + n_repeat):
        if method.category == "pandapower":
            net_copy = copy.deepcopy(net)
            r = method.solver_fn(net_copy, max_iteration=30, tolerance=1e-8)
            if trial >= N_WARMUP:
                last_net_copy = net_copy
        else:
            if network_data is None:
                return {
                    "method_id": method.id,
                    "method_name": method.name,
                    "category": method.category,
                    "converged": False,
                    "iterations": 0,
                    "elapsed_sec": 0.0,
                    "elapsed_all": [],
                    "failure_reason": "Network prep failed",
                }
            r = method.solver_fn(
                network_data.Ybus,
                network_data.Sbus,
                np.copy(network_data.V0),
                np.array(network_data.ref),
                np.array(network_data.pv),
                np.array(network_data.pq),
                max_iter=30,
                tol=1e-8,
            )
        if trial >= N_WARMUP:
            times.append(r.elapsed_sec)
            result = r

    rec = {
        "method_id": method.id,
        "method_name": method.name,
        "category": method.category,
        "converged": result.converged if result else False,
        "iterations": result.iterations if result else 0,
        "elapsed_sec": float(np.median(times)) if times else 0.0,
        "elapsed_all": times,
    }

    # Extract voltage results
    if result and result.converged and result.V is not None:
        rec["vm_pu"] = np.abs(result.V)
        rec["va_degree"] = np.degrees(np.angle(result.V))
    elif result and result.converged and last_net_copy is not None:
        # pandapower wrappers: extract from net.res_bus
        if hasattr(last_net_copy, "res_bus") and "vm_pu" in last_net_copy.res_bus.columns:
            rec["vm_pu"] = last_net_copy.res_bus["vm_pu"].values.copy()
            rec["va_degree"] = last_net_copy.res_bus["va_degree"].values.copy()

    if result and result.failure_reason:
        rec["failure_reason"] = result.failure_reason

    if result and result.convergence_history:
        rec["convergence_history"] = result.convergence_history

    return rec


def run_benchmark_for_case(
    label: str,
    net_factory: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run full benchmark (DC + all AC) for one test case."""
    net = net_factory()
    n_buses = len(net.bus)
    n_branches = len(net.line) + len(net.trafo)
    print(f"\n{'='*60}")
    print(f"  {label}: {n_buses} buses, {n_branches} branches")
    print(f"{'='*60}")

    # DC benchmark
    print("  DC power flow ...", end=" ", flush=True)
    dc_result = bench_dc(net)
    print(f"{dc_result['elapsed_sec']*1000:.2f} ms")

    # Prepare network data for custom solvers
    network_data = None
    try:
        network_data = prepare_network(copy.deepcopy(net))
    except Exception as exc:
        print(f"  [WARN] Network prep failed: {exc}")

    # AC methods
    methods = get_all_methods()
    ac_results = []
    for method in methods:
        print(f"  {method.id:30s} ...", end=" ", flush=True)
        rec = bench_ac_method(method, net, network_data)
        status = "OK" if rec["converged"] else "FAIL"
        print(f"{rec['elapsed_sec']*1000:8.2f} ms  iter={rec['iterations']:3d}  [{status}]")
        ac_results.append(rec)

    return dc_result, ac_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color palette per category
CATEGORY_COLORS = {
    "dc": "#2196F3",
    "pandapower": "#4CAF50",
    "custom_nr": "#FF9800",
    "custom_iterative": "#9C27B0",
    "custom_decoupled": "#F44336",
}

CATEGORY_LABELS = {
    "dc": "DC",
    "pandapower": "pandapower",
    "custom_nr": "Custom NR",
    "custom_iterative": "Custom Iterative",
    "custom_decoupled": "Custom Decoupled",
}


def plot_time_comparison_bar(
    all_data: Dict[str, Tuple[Dict, List[Dict]]],
    output_path: str,
) -> None:
    """Bar chart: execution time per method per test case."""
    case_labels = list(all_data.keys())
    n_cases = len(case_labels)

    # Collect all method IDs (dc + ac) in consistent order
    first_dc, first_ac = list(all_data.values())[0]
    method_ids = ["dc"] + [r["method_id"] for r in first_ac]
    method_names = ["DC"] + [r["method_id"] for r in first_ac]
    categories = ["dc"] + [r["category"] for r in first_ac]

    # Filter to converged methods (at least one case converged)
    converged_mask = []
    for i, mid in enumerate(method_ids):
        any_conv = False
        for case_label in case_labels:
            dc_r, ac_list = all_data[case_label]
            if mid == "dc":
                any_conv = True
            else:
                for r in ac_list:
                    if r["method_id"] == mid and r["converged"]:
                        any_conv = True
        converged_mask.append(any_conv)

    method_ids_f = [m for m, ok in zip(method_ids, converged_mask) if ok]
    method_names_f = [m for m, ok in zip(method_names, converged_mask) if ok]
    categories_f = [c for c, ok in zip(categories, converged_mask) if ok]

    # Build time matrix (methods x cases)
    n_methods = len(method_ids_f)
    time_matrix = np.full((n_methods, n_cases), np.nan)

    for j, case_label in enumerate(case_labels):
        dc_r, ac_list = all_data[case_label]
        ac_dict = {r["method_id"]: r for r in ac_list}
        for i, mid in enumerate(method_ids_f):
            if mid == "dc":
                time_matrix[i, j] = dc_r["elapsed_sec"] * 1000
            elif mid in ac_dict and ac_dict[mid]["converged"]:
                time_matrix[i, j] = ac_dict[mid]["elapsed_sec"] * 1000

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(n_cases)
    width = 0.8 / n_methods

    for i in range(n_methods):
        color = CATEGORY_COLORS.get(categories_f[i], "#666666")
        cat_label = CATEGORY_LABELS.get(categories_f[i], categories_f[i])
        label = f"{method_names_f[i]} ({cat_label})"
        vals = time_matrix[i]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, color=color, alpha=0.8,
                      label=label, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Test Case", fontsize=12)
    ax.set_ylabel("Execution Time [ms]", fontsize=12)
    ax.set_title("AC vs DC Power Flow — Execution Time Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({all_data[c][0].get('n_buses', '?')} bus)"
                        if 'n_buses' in all_data[c][0] else c
                        for c in case_labels],
                       fontsize=10)
    ax.set_xticklabels(case_labels, fontsize=10)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}" if y < 10 else f"{y:.0f}"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Legend: only unique categories
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        cat = l.split("(")[-1].rstrip(")")
        if cat not in seen:
            seen[cat] = True
            unique_handles.append(h)
            unique_labels.append(cat)
    ax.legend(unique_handles, unique_labels, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scaling_lines(
    all_data: Dict[str, Tuple[Dict, List[Dict]]],
    bus_counts: Dict[str, int],
    output_path: str,
) -> None:
    """Line plot: execution time vs network size, one line per method category."""
    case_labels = list(all_data.keys())
    sizes = [bus_counts[c] for c in case_labels]

    # Group by category, take best (fastest converged) per category per case
    category_times: Dict[str, List[float]] = {}

    # DC
    dc_times = []
    for c in case_labels:
        dc_r, _ = all_data[c]
        dc_times.append(dc_r["elapsed_sec"] * 1000)
    category_times["dc"] = dc_times

    # AC categories
    for cat in ["pandapower", "custom_nr", "custom_iterative", "custom_decoupled"]:
        cat_times = []
        for c in case_labels:
            _, ac_list = all_data[c]
            conv = [r["elapsed_sec"] * 1000 for r in ac_list
                    if r["category"] == cat and r["converged"]]
            cat_times.append(min(conv) if conv else np.nan)
        category_times[cat] = cat_times

    fig, ax = plt.subplots(figsize=(10, 6))

    for cat, times in category_times.items():
        color = CATEGORY_COLORS.get(cat, "#666")
        label = CATEGORY_LABELS.get(cat, cat)
        marker = "o" if cat == "dc" else "s" if cat == "pandapower" else "^" if cat == "custom_nr" else "D" if cat == "custom_iterative" else "v"
        ax.plot(sizes, times, marker=marker, color=color, label=label,
                linewidth=2, markersize=8, alpha=0.9)

    ax.set_xlabel("Number of Buses", fontsize=12)
    ax.set_ylabel("Execution Time [ms]", fontsize=12)
    ax.set_title("Power Flow Computation Time Scaling\n(best method per category)",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}" if y < 10 else f"{y:.0f}"))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_convergence_comparison(
    all_data: Dict[str, Tuple[Dict, List[Dict]]],
    output_path: str,
) -> None:
    """Convergence history for a representative case (case118)."""
    target_case = "case118"
    if target_case not in all_data:
        target_case = list(all_data.keys())[-1]

    _, ac_list = all_data[target_case]

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in ac_list:
        if not r["converged"] or "convergence_history" not in r:
            continue
        hist = r["convergence_history"]
        if len(hist) < 2:
            continue
        color = CATEGORY_COLORS.get(r["category"], "#666")
        ax.semilogy(range(len(hist)), hist, color=color, alpha=0.6,
                    linewidth=1.5, label=r["method_id"])

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Mismatch Norm [p.u.]", fontsize=12)
    ax.set_title(f"Convergence History — {target_case}", fontsize=14, fontweight="bold")
    ax.axhline(y=1e-8, color="red", linestyle="--", alpha=0.5, label="tolerance=1e-8")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Deduplicate legend by category
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 15:
        # Too many, show category-level only
        seen = {}
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            rec = next((r for r in ac_list if r["method_id"] == l), None)
            if rec:
                cat = CATEGORY_LABELS.get(rec["category"], rec["category"])
                if cat not in seen:
                    seen[cat] = True
                    unique_h.append(h)
                    unique_l.append(cat)
            elif l == "tolerance=1e-8":
                unique_h.append(h)
                unique_l.append(l)
        ax.legend(unique_h, unique_l, fontsize=10, loc="upper right")
    else:
        ax.legend(fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_accuracy_heatmap(
    all_data: Dict[str, Tuple[Dict, List[Dict]]],
    output_path: str,
) -> None:
    """Heatmap: voltage magnitude deviation from pp_nr reference."""
    case_labels = list(all_data.keys())

    # Use pp_nr as reference for each case
    first_dc, first_ac = list(all_data.values())[0]
    method_ids = [r["method_id"] for r in first_ac if r["converged"]]

    # For each case, compute max |Vm| deviation from pp_nr
    ref_id = "pp_nr"
    deviations = {}

    for case_label in case_labels:
        dc_r, ac_list = all_data[case_label]
        ref_rec = next((r for r in ac_list if r["method_id"] == ref_id and r["converged"]), None)
        if ref_rec is None or "vm_pu" not in ref_rec:
            continue

        ref_vm = ref_rec["vm_pu"]
        deviations[case_label] = {}

        # DC vs reference
        if "vm_pu" in dc_r:
            dc_vm = dc_r["vm_pu"]
            if len(dc_vm) == len(ref_vm):
                deviations[case_label]["DC"] = float(np.max(np.abs(dc_vm - ref_vm)))

        # AC methods vs reference
        for r in ac_list:
            if r["method_id"] == ref_id:
                continue
            if r["converged"] and "vm_pu" in r:
                ac_vm = r["vm_pu"]
                if len(ac_vm) == len(ref_vm):
                    deviations[case_label][r["method_id"]] = float(np.max(np.abs(ac_vm - ref_vm)))

    if not deviations:
        print("  [SKIP] No voltage data for accuracy heatmap")
        return

    # Build DataFrame
    all_methods_set = set()
    for d in deviations.values():
        all_methods_set.update(d.keys())
    all_methods_sorted = sorted(all_methods_set)

    df = pd.DataFrame(index=all_methods_sorted, columns=list(deviations.keys()), dtype=float)
    for case_label, devs in deviations.items():
        for mid, val in devs.items():
            df.at[mid, case_label] = val

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
    data = df.values.astype(float)

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                   norm=matplotlib.colors.LogNorm(vmin=max(1e-12, np.nanmin(data[data > 0])),
                                                   vmax=max(1e-1, np.nanmax(data))))

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_title("Voltage Magnitude Deviation from pp_nr Reference\n(max |ΔVm| per bus)",
                 fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Max |ΔVm| [p.u.]", fontsize=11)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                text = f"{val:.1e}" if val > 0 else "0"
                color = "white" if val > 1e-3 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary_table(
    all_data: Dict[str, Tuple[Dict, List[Dict]]],
    output_path: str,
) -> None:
    """Summary table figure with key metrics."""
    # Use largest case for the table
    case_label = list(all_data.keys())[-1]
    dc_r, ac_list = all_data[case_label]

    rows = []
    rows.append({
        "Method": "DC Power Flow",
        "Category": "DC",
        "Converged": "Yes",
        "Iterations": "-",
        "Time [ms]": f"{dc_r['elapsed_sec']*1000:.2f}",
        "Speedup vs pp_nr": "",
    })

    # Get pp_nr time as reference
    pp_nr_time = None
    for r in ac_list:
        if r["method_id"] == "pp_nr" and r["converged"]:
            pp_nr_time = r["elapsed_sec"]

    for r in ac_list:
        speedup = ""
        if pp_nr_time and r["converged"] and r["elapsed_sec"] > 0:
            speedup = f"{pp_nr_time / r['elapsed_sec']:.2f}x"

        rows.append({
            "Method": r["method_id"],
            "Category": CATEGORY_LABELS.get(r["category"], r["category"]),
            "Converged": "Yes" if r["converged"] else "No",
            "Iterations": str(r["iterations"]),
            "Time [ms]": f"{r['elapsed_sec']*1000:.2f}" if r["converged"] else "-",
            "Speedup vs pp_nr": speedup,
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.35)))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(df.columns)):
        cell = table[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")

    # Color rows by category
    for i, row in df.iterrows():
        cat_key = next((k for k, v in CATEGORY_LABELS.items() if v == row["Category"]), "")
        color = CATEGORY_COLORS.get(cat_key, "#FFFFFF")
        for j in range(len(df.columns)):
            cell = table[i + 1, j]
            cell.set_facecolor(color + "22")  # light tint

    ax.set_title(f"Benchmark Summary — {case_label}\n(median of {N_REPEAT} runs)",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  AC vs DC Power Flow Benchmark")
    print(f"  {N_REPEAT} repetitions per measurement, {N_WARMUP} warmup")
    print("=" * 60)

    all_data: Dict[str, Tuple[Dict, List[Dict]]] = {}
    bus_counts: Dict[str, int] = {}

    for label, factory, bus_count in TEST_CASES:
        dc_result, ac_results = run_benchmark_for_case(label, factory)
        dc_result["n_buses"] = bus_count
        all_data[label] = (dc_result, ac_results)
        bus_counts[label] = bus_count

    # Generate figures
    print(f"\n{'='*60}")
    print("  Generating figures...")
    print(f"{'='*60}")

    plot_scaling_lines(all_data, bus_counts, os.path.join(OUTPUT_DIR, "scaling_time.png"))
    plot_convergence_comparison(all_data, os.path.join(OUTPUT_DIR, "convergence_history.png"))
    plot_accuracy_heatmap(all_data, os.path.join(OUTPUT_DIR, "accuracy_heatmap.png"))
    plot_summary_table(all_data, os.path.join(OUTPUT_DIR, "summary_table.png"))

    # Save raw CSV
    csv_rows = []
    for case_label, (dc_r, ac_list) in all_data.items():
        csv_rows.append({
            "case": case_label,
            "n_buses": bus_counts[case_label],
            "method": "dc",
            "category": "dc",
            "converged": True,
            "iterations": 0,
            "time_ms": dc_r["elapsed_sec"] * 1000,
        })
        for r in ac_list:
            csv_rows.append({
                "case": case_label,
                "n_buses": bus_counts[case_label],
                "method": r["method_id"],
                "category": r["category"],
                "converged": r["converged"],
                "iterations": r["iterations"],
                "time_ms": r["elapsed_sec"] * 1000,
            })

    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Print quick summary
    print(f"\n{'='*60}")
    print("  Quick Summary")
    print(f"{'='*60}")
    for case_label, (dc_r, ac_list) in all_data.items():
        n_conv = sum(1 for r in ac_list if r["converged"])
        best_ac = min((r["elapsed_sec"] for r in ac_list if r["converged"]), default=0)
        dc_t = dc_r["elapsed_sec"]
        ratio = best_ac / dc_t if dc_t > 0 else float("inf")
        print(f"  {case_label:12s}  DC={dc_t*1000:8.2f}ms  "
              f"Best_AC={best_ac*1000:8.2f}ms  "
              f"AC/DC={ratio:6.1f}x  "
              f"converged={n_conv}/{len(ac_list)}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
