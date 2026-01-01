# ==================== Imports ====================
import argparse
import os
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
# ==================== Targets ====================
target_peak_date = pd.to_datetime("2020-07-29")
target_seroprevalence = 2_974_393

# ==================== Tolerances and CI ====================
# Seroprevalence 95% CI (for z-score normalization)
sero_L = 2_616_955
sero_U = 3_370_128
sigma_sero = (sero_U - sero_L) / (2 * 1.96)   # ≈ 1.92e5

# Peak lateness normalization and strength
PEAK_UNIT_DAYS = 7.0              # 1 unit = 7 days
PEAK_PENALTY_STRENGTH = 2.0       # >1 increases penalty if peak is late

# Active prevalence 0.2% (95% CI 0.1–0.4) of Lagos population ≈ 12.7M
active_L = 0.001 * 12_765_630 #= 12_766
active_U = 0.004 * 12_765_630 #= 51_062
target_active = 0.002 * 12_765_630 #= 25_531
sigma_active = (active_U - active_L) / (2 * 1.96)   # ≈ 9.7e3

# ==================== Output Dirs ====================
output_dir = "Result/test_run"
plot_dir = os.path.join(output_dir, "plot_result")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ==================== Search Space ====================
search_space = [
    Real(0.010, 0.015, name='BETA_3'),
    Integer(100, 1000, name='Daily_Imported_Cases')  # NEW search variable
]

# ==================== Lockdown Dates ====================
first_case_date = pd.to_datetime("2020-02-24")
initial_lockdown_start = pd.to_datetime("2020-03-30")
initial_lockdown_end = pd.to_datetime("2020-05-04")
phase1_start = pd.to_datetime("2020-05-04")
phase1_end = pd.to_datetime("2020-06-01")
phase2_start = pd.to_datetime("2020-06-02")
phase2_end = pd.to_datetime("2020-06-29")
phase3_start = pd.to_datetime("2020-06-30")
phase3_end = pd.to_datetime("2020-07-27")
# ============ Helper for run logging ============
def append_run_summary(beta3, imports, peak_date, final_sero, final_active, loss):
    summary_path = os.path.join(output_dir, "run_summary.csv")
    header = not os.path.exists(summary_path)
    df_row = pd.DataFrame([{
        "BETA_3": beta3,
        "Daily_Imported_Cases": imports,
        "Peak_Date": peak_date.strftime("%Y-%m-%d"),
        "Final_Seroprevalence": final_sero,
        "Final_Active": final_active,
        "Loss": loss
    }])
    df_row.to_csv(summary_path, mode="a", header=header, index=False)
@use_named_args(search_space)
def objective(BETA_3, Daily_Imported_Cases):
    BETA_1 = BETA_3 * 0.89
    BETA_2 = BETA_3 * 0.26

    param_dict = {
        'RHO': 0.306,
        'LAMBDA': 5.5,
        'ALPHA': 2.3,
        'GAMMA_I': 3.5,
        'GAMMA_A': 3.5,
        'average_contact_rate': 9.05,
        'average_contact_rate_home': 6.27,
        'seeding_date': "2020-02-01",
        'random_seed': 42,
        'manual_infected_node_id': "",
        'national_avg_population': 2378,
        'BETA_1': BETA_1,
        'BETA_2': BETA_2,
        'BETA_3': BETA_3,
        'Number_of_Initial_Exposed': 1,
        'Exposed_Nodes_Canidates': 1,
        'Number_of_Initial_Exposed_Nodes': 1,
        # === NEW PARAMETERS ===
        'Daily_Imported_Cases': int(Daily_Imported_Cases),  # now dynamic
        'Number_of_Imported_Nodes': 30     # number of nodes to receive imports
    }

    with open("params.json", "w") as f:
        json.dump(param_dict, f)

    suffix = f"{BETA_3:.8f}_{Daily_Imported_Cases}"
    print(f"▶ Trying: β3={BETA_3:.8f}, Imports/day={Daily_Imported_Cases}")

    try:
        subprocess.run(["python", "Epidemic_Simulation.py"], check=True)

        df = pd.read_csv("node_dynamics_geohash.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2020-10-31')]

        daily = df.groupby('date')[['exposed', 'asymptomatic_infected', 'presymptomatic_infected', 'infected', 'recovered']].sum()
        daily['Total_EAPIR'] = (
            daily['asymptomatic_infected'] +
            daily['presymptomatic_infected'] +
            daily['infected'] +
            daily['recovered']
        )
        daily['Active'] = (
            daily['asymptomatic_infected'] +
            daily['presymptomatic_infected'] +
            daily['infected']
        )

        peak_date = daily['exposed'].idxmax()
        final_sero = daily['Total_EAPIR'].iloc[-1]
        final_active = daily['Active'].iloc[-1]

        # --- normalized components ---
        days_offset = (peak_date - target_peak_date).days
        if days_offset <= 0:
            e_peak = 0.0
        else:
            r = days_offset / PEAK_UNIT_DAYS
            # quadratic boost to discourage late peaks
            e_peak = r * (1.0 + PEAK_PENALTY_STRENGTH * r)

        # seroprevalence z-score based on the reported 95% CI
        e_sero = abs(final_sero - target_seroprevalence) / sigma_sero

        # active normalized by a tolerance band
        e_active = abs(final_active - target_active) / sigma_active

        loss = e_peak + e_sero + e_active
        print(
            f"   🔹 Peak: {peak_date.date()}, "
            f"Sero: {int(final_sero):,}, Active: {int(final_active):,} | "
            f"e_peak={e_peak:.3f}, e_sero={e_sero:.3f}, e_active={e_active:.3f}, "
            f"Loss={loss:.3f}"
        )
        # ---- append summary record ----
        append_run_summary(
            beta3=BETA_3,
            imports=int(Daily_Imported_Cases),
            peak_date=peak_date,
            final_sero=final_sero,
            final_active=final_active,
            loss=loss
        )
        output_path = os.path.join(output_dir, f"node_dynamics_geohash_{suffix}.csv")
        df.to_csv(output_path, index=False)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        for col in ['exposed', 'asymptomatic_infected', 'presymptomatic_infected', 'infected']:
            axes[0].plot(daily.index, daily[col], label=col.replace('_', ' ').title())

        for (start, end, color, label) in [
            (initial_lockdown_start, initial_lockdown_end, "blue", "Initial Lockdown"),
            (phase1_start, phase1_end, "red", "Phase 1"),
            (phase2_start, phase2_end, "orange", "Phase 2"),
            (phase3_start, phase3_end, "green", "Phase 3"),
        ]:
            axes[0].axvspan(start, end, color=color, alpha=0.15, label=label)

        axes[0].axvline(first_case_date, color='black', linestyle='--', linewidth=1)
        axes[0].axvline(peak_date, color='magenta', linestyle='--', linewidth=1)
        axes[0].text(peak_date, axes[0].get_ylim()[1]*0.85, f'Peak\n({peak_date.date()})',
                     rotation=90, va='top', ha='right', fontsize=9, color='magenta')

        axes[0].set_title(f'Epidemic Dynamics ({suffix})')
        axes[0].set_ylabel('Number of Agents')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(daily.index, daily['Total_EAPIR'], color='purple', label='Seroprevalence number')
        axes[1].axhline(final_sero, color='red', linestyle='--')
        axes[1].axhline(final_active, color='orange', linestyle='--')
        axes[1].text(daily.index[-1], final_sero * 1.01,
                     f'Seroprev: {int(final_sero):,}', color='red', ha='right', va='bottom', fontsize=9)
        axes[1].text(daily.index[-1], final_active * 1.01,
                     f'Active: {int(final_active):,}', color='orange', ha='right', va='bottom', fontsize=9)

        axes[1].set_title('Seroprevalence Number Over Time')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Agents')
        axes[1].legend()
        axes[1].grid(True)

        for ax in axes:
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='y')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{suffix}.png")
        plt.savefig(plot_path)
        plt.close()

        return loss

    except Exception as e:
        print("   ❌ Error:", e)
        return 1e12

def generate_plots(res):
    """Generate convergence and diagnostic plots from a saved optimization result."""
    from skopt.plots import plot_objective

    # ==================== Convergence Plot ====================
    fig = plt.figure(figsize=(8, 5))
    plot_convergence(res)
    plt.tight_layout()
    conv_path = os.path.join(plot_dir, "convergence.png")
    plt.savefig(conv_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to: {conv_path}")

    # ==================== Objective Landscape ====================
    print("Generating objective landscape plot...")
    fig = plt.figure(figsize=(8, 8))
    plot_objective(res)
    plt.tight_layout()
    objective_path = os.path.join(plot_dir, "objective_landscape.png")
    plt.savefig(objective_path, dpi=150)
    plt.close()
    print(f"Saved objective landscape plot to: {objective_path}")

    # ==================== Parameter Coverage Scatter ====================
    print("Generating parameter exploration scatter plot...")
    results = np.array(res.x_iters)
    losses = np.array(res.func_vals)

    param_names = ["BETA_3", "Daily_Imported_Cases"]
    fig, axes = plt.subplots(1, results.shape[1], figsize=(5 * results.shape[1], 4))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.scatter(results[:, i], losses, c=losses, cmap="viridis", s=40)
        ax.set_xlabel(param_names[i])
        if i == 0:
            ax.set_ylabel("Loss")
        ax.set_title(f"Coverage: {param_names[i]}")
        ax.grid(True)
    plt.tight_layout()
    coverage_path = os.path.join(plot_dir, "parameter_coverage.png")
    plt.savefig(coverage_path, dpi=150)
    plt.close()
    print(f"Saved parameter coverage plot to: {coverage_path}")


def main():
    parser = argparse.ArgumentParser(description="Run SEIR calibration or regenerate plots from cached results.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip optimization and regenerate plots from a cached result file.",
    )
    parser.add_argument(
        "--result-file",
        default=os.path.join(output_dir, "optimization_result.pkl"),
        help="Path to store or load the cached gp_minimize result.",
    )
    args = parser.parse_args()

    result_path = os.path.abspath(args.result_file)

    if args.plot_only:
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Cached result not found at: {result_path}")
        res = load(result_path)
        print(f"Loaded optimization result from: {result_path}")
    else:
        res = gp_minimize(objective, search_space, n_calls=70, n_initial_points=15, random_state=42, verbose=True)
        dump(res, result_path)
        print(f"Saved optimization result to: {result_path}")

    generate_plots(res)

    # ==================== Output Best ====================
    print(f"BETA_3: {res.x[0]:.7f}")
    print(f"Daily Imported Cases: {res.x[1]}")


if __name__ == "__main__":
    main()

