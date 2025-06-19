"""
SPC (Statistical Process Control) for GO/rGO Synthesis Monitoring
Implements X-bar and R charts for process quality metrics (C/O ratio, interlayer spacing, conductivity).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Standard SPC constants for subgroup sizes 2-10
SPC_CONSTANTS = {
    2: {"A2": 1.88, "D3": 0,    "D4": 3.267},
    3: {"A2": 1.023, "D3": 0,    "D4": 2.574},
    4: {"A2": 0.729, "D3": 0,    "D4": 2.282},
    5: {"A2": 0.577, "D3": 0,    "D4": 2.114},
    6: {"A2": 0.483, "D3": 0,    "D4": 2.004},
    7: {"A2": 0.419, "D3": 0.076,"D4": 1.924},
    8: {"A2": 0.373, "D3": 0.136,"D4": 1.864},
    9: {"A2": 0.337, "D3": 0.184,"D4": 1.816},
    10:{"A2": 0.308, "D3": 0.223,"D4": 1.777},
}

def compute_subgroups(data, subgroup_size=4):
    """Divide data into subgroups of given size."""
    n = len(data)
    num_groups = n // subgroup_size
    subgroups = [data[i*subgroup_size:(i+1)*subgroup_size] for i in range(num_groups)]
    return subgroups

def calculate_spc_limits(subgroups, subgroup_size):
    """Calculate X-bar and R chart control limits."""
    A2 = SPC_CONSTANTS[subgroup_size]["A2"]
    D3 = SPC_CONSTANTS[subgroup_size]["D3"]
    D4 = SPC_CONSTANTS[subgroup_size]["D4"]
    xbars = np.array([np.mean(g) for g in subgroups])
    ranges = np.array([np.ptp(g) for g in subgroups])
    xbar_bar = np.mean(xbars)
    r_bar = np.mean(ranges)
    # X-bar chart limits
    xbar_UCL = xbar_bar + A2 * r_bar
    xbar_LCL = xbar_bar - A2 * r_bar
    # R chart limits
    r_UCL = D4 * r_bar
    r_LCL = D3 * r_bar
    return {
        "xbars": xbars, "ranges": ranges,
        "xbar_bar": xbar_bar, "r_bar": r_bar,
        "xbar_UCL": xbar_UCL, "xbar_LCL": xbar_LCL,
        "r_UCL": r_UCL, "r_LCL": r_LCL
    }

def plot_xbar_chart(xbars, xbar_bar, UCL, LCL, metric, save_path=None, highlight=None):
    plt.figure(figsize=(10,5))
    plt.plot(xbars, marker='o', label='X-bar')
    plt.axhline(xbar_bar, color='g', linestyle='--', label='Centerline')
    plt.axhline(UCL, color='r', linestyle='--', label='UCL')
    plt.axhline(LCL, color='r', linestyle='--', label='LCL')
    if highlight is not None:
        plt.scatter(highlight, xbars[highlight], color='red', zorder=5, label='Out of Control')
    plt.title(f'X-bar Chart for {metric}')
    plt.xlabel('Subgroup')
    plt.ylabel(f'{metric}')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_r_chart(ranges, r_bar, UCL, LCL, metric, save_path=None, highlight=None):
    plt.figure(figsize=(10,5))
    plt.plot(ranges, marker='o', label='Range')
    plt.axhline(r_bar, color='g', linestyle='--', label='Centerline')
    plt.axhline(UCL, color='r', linestyle='--', label='UCL')
    plt.axhline(LCL, color='r', linestyle='--', label='LCL')
    if highlight is not None:
        plt.scatter(highlight, ranges[highlight], color='red', zorder=5, label='Out of Control')
    plt.title(f'R Chart for {metric}')
    plt.xlabel('Subgroup')
    plt.ylabel(f'Range of {metric}')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def update_spc_chart(metric, batch_id, measured_values, subgroup_size=4, output_dir="outputs/quality_charts", log_file="results/spc_log.csv"):
    """
    Update SPC charts for a given metric and log violations.
    measured_values: list or np.array of measurements for the batch
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    subgroups = compute_subgroups(measured_values, subgroup_size)
    if not subgroups:
        print("Not enough data for SPC chart.")
        return
    spc = calculate_spc_limits(subgroups, subgroup_size)
    # Check for violations
    out_of_control = [i for i, x in enumerate(spc["xbars"]) if (x > spc["xbar_UCL"] or x < spc["xbar_LCL"])]
    out_of_control_r = [i for i, r in enumerate(spc["ranges"]) if (r > spc["r_UCL"] or r < spc["r_LCL"])]
    # Plot charts
    xbar_path = os.path.join(output_dir, f"{metric}_xbar_chart_batch{batch_id}.png")
    r_path = os.path.join(output_dir, f"{metric}_r_chart_batch{batch_id}.png")
    plot_xbar_chart(spc["xbars"], spc["xbar_bar"], spc["xbar_UCL"], spc["xbar_LCL"], metric, save_path=xbar_path, highlight=out_of_control)
    plot_r_chart(spc["ranges"], spc["r_bar"], spc["r_UCL"], spc["r_LCL"], metric, save_path=r_path, highlight=out_of_control_r)
    # Log violations
    if out_of_control or out_of_control_r:
        with open(log_file, "a") as f:
            for idx in out_of_control:
                f.write(f"{datetime.now()},BATCH {batch_id},{metric},XBAR,{spc['xbars'][idx]},OUT_OF_CONTROL\n")
            for idx in out_of_control_r:
                f.write(f"{datetime.now()},BATCH {batch_id},{metric},R,{spc['ranges'][idx]},OUT_OF_CONTROL\n")
    print(f"SPC charts updated for {metric} (batch {batch_id}). Out-of-control points: {out_of_control + out_of_control_r}")

# Example synthetic test (remove or comment out in production)
if __name__ == "__main__":
    # Synthetic data for C/O ratio
    np.random.seed(42)
    co_ratios = np.random.normal(2.5, 0.15, 20).tolist() + [3.9]  # Last value is an outlier
    update_spc_chart("CO_ratio", batch_id=1, measured_values=co_ratios)
