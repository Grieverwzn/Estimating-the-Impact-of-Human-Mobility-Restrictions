import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
import time
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# CONFIGURATION
# ==========================================
REGULARIZATION_ALPHA = 1e-4
OUTPUT_DIR = "Optimization_Results"

FILENAMES = {
    'geo_lga': "assigned_geohash6_LGA_v2.csv",
    'tours': "Lagos_Raw_trip_oct_home_flags_homebound_tours_cross_lga.csv",
    'metrics': "lagos_daily_trip_spread_metrics_by_lga.csv",
    'gravity': "lp-estimated_trips_LGA.csv"
}

# Create Output Directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def set_nature_style():

    # Font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 7
    
    # Text Sizes
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['figure.titlesize'] = 8

    # Lines and Markers
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    
    # Ticks
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    
    # Aesthetics
    plt.rcParams['axes.grid'] = False  # No grid for cleaner look
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Resolution
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

def solve_dual_maxent(A_csr, b, n_tours):
    def dual_objective(lam):
        at_lam = A_csr.T.dot(lam)
        at_lam = np.clip(at_lam, -20, 20) 
        x_est = np.exp(at_lam)
        obj = np.sum(x_est) - np.dot(b, lam) + (REGULARIZATION_ALPHA * np.sum(lam**2))
        return obj

    def dual_gradient(lam):
        at_lam = A_csr.T.dot(lam)
        at_lam = np.clip(at_lam, -20, 20)
        x_est = np.exp(at_lam)
        ax = A_csr.dot(x_est)
        grad = ax - b + (2 * REGULARIZATION_ALPHA * lam)
        return grad

    lambda_0 = np.zeros(len(b))

    res = minimize(
        dual_objective,
        lambda_0,
        method='L-BFGS-B',
        jac=dual_gradient,
        options={'maxiter': 5000, 'disp': False}
    )
    
    final_lam = res.x
    at_lam_final = A_csr.T.dot(final_lam)
    at_lam_final = np.clip(at_lam_final, -50, 50)
    weights = np.exp(at_lam_final)
    return weights

def create_dummy_tours_from_leftover(missing_demand, leftover_pop, target_date, target_biweek, geo_to_lga):
    dummy_rows = []
    pop_by_lga = leftover_pop.groupby('LGA_code')
    dest_anchors = leftover_pop.sort_values('population_pixel_assigned', ascending=False).groupby('LGA_code')['node_id'].first().to_dict()
    
    remaining_demand = missing_demand.copy()
    pairs = list(remaining_demand.keys())
    
    for (origin_lga, dest_lga) in pairs:
        needed_trips = remaining_demand.get((origin_lga, dest_lga), 0)
        if needed_trips < 1.0: continue
        
        reverse_needed = remaining_demand.get((dest_lga, origin_lga), 0)
        agents_needed = needed_trips 
        
        if origin_lga not in pop_by_lga.groups: continue
        candidates = pop_by_lga.get_group(origin_lga).sort_values('leftover_capacity', ascending=False)
        dest_geo = dest_anchors.get(dest_lga)
        if not dest_geo: continue
        
        for _, row in candidates.iterrows():
            if agents_needed <= 0: break
            
            geo_id = row['node_id']
            capacity_agents = row['leftover_capacity']
            if capacity_agents <= 0.1: continue
            
            agents_to_assign = min(capacity_agents, agents_needed)
            if agents_to_assign < 0.01: continue

            row['leftover_capacity'] -= agents_to_assign
            
            tour_id = f"DUMMY_GAP_{origin_lga}_{dest_lga}_{geo_id}"
            maid_id = f"dummy_gap_{origin_lga}_{dest_lga}_{geo_id}"
            
            # Trip 1
            dummy_rows.append({
                'maid': maid_id,
                'local_datetime_1': f"{target_date} 08:00:00",
                'local_datetime_2': f"{target_date} 09:00:00",
                'custom_day': target_date,
                'Date_dep': target_date,
                'geohash6_1': geo_id,
                'geohash6_2': dest_geo,
                'trip_type': 'home_start',
                'home_geohash': geo_id,
                'if_start': True,
                'if_end': False,
                'bi_week_num': target_biweek,
                'tour_id': tour_id,
                'lga_code_1': origin_lga,
                'lga_code_2': dest_lga,
                'cross_lga_tour': True,
                'agent_count': agents_to_assign
            })
            
            # Trip 2
            dummy_rows.append({
                'maid': maid_id,
                'local_datetime_1': f"{target_date} 17:00:00",
                'local_datetime_2': f"{target_date} 18:00:00",
                'custom_day': target_date,
                'Date_dep': target_date,
                'geohash6_1': dest_geo,
                'geohash6_2': geo_id,
                'trip_type': 'original',
                'home_geohash': geo_id,
                'if_start': False,
                'if_end': True,
                'bi_week_num': target_biweek,
                'tour_id': tour_id,
                'lga_code_1': dest_lga,
                'lga_code_2': origin_lga,
                'cross_lga_tour': True,
                'agent_count': agents_to_assign
            })
            
            agents_needed -= agents_to_assign
            
            if reverse_needed > 0:
                reduction = min(reverse_needed, agents_to_assign)
                remaining_demand[(dest_lga, origin_lga)] = reverse_needed - reduction
                reverse_needed -= reduction
        
        remaining_demand[(origin_lga, dest_lga)] = agents_needed
            
    return pd.DataFrame(dummy_rows)

def process_single_day(target_date, df_geo, df_tours, df_metrics, df_gravity, geo_to_lga):
    print(f"\n=== Processing Date: {target_date} ===")
    
    daily_metrics = df_metrics[df_metrics['custom_day'] == target_date].copy()
    if daily_metrics.empty: return None
    daily_gravity = df_gravity[df_gravity['date'] == target_date].copy()
    if daily_gravity.empty: return None
    try:
        target_biweek = df_tours[df_tours['custom_day'] == target_date]['bi_week_num'].iloc[0]
    except IndexError: return None
    
    geo_caps = pd.merge(df_geo, daily_metrics[['admin2Pcod_1', 'pct_cross_lga_maids']], 
                        left_on='LGA_code', right_on='admin2Pcod_1', how='left')
    geo_caps['max_agents'] = geo_caps['population_pixel_assigned'] * (geo_caps['pct_cross_lga_maids'] / 100.0)
    geo_caps['max_agents'] = geo_caps['max_agents'].fillna(0)
    lga_max_agents_map = geo_caps.groupby('LGA_code')['max_agents'].sum().to_dict()

    flow_constraints = {}
    total_target_volume = 0
    for _, row in daily_gravity.iterrows():
        origin = row['from_node_id']
        dest = row['to_node_id']
        val = row['est_trip_num']
        flow_constraints[(origin, dest)] = val
        total_target_volume += val
        
    print(f"  - Full Gravity Target: {total_target_volume:,.0f} trips")

    real_tours = df_tours[
        (df_tours['bi_week_num'] == target_biweek) & 
        (df_tours['cross_lga_tour'] == True)
    ].copy()
    real_tours['custom_day'] = target_date 
    real_tours['Date_dep'] = target_date

    tour_groups = real_tours.groupby('tour_id')
    tour_ids = []
    tour_home_lgas = []
    tour_flows_indices = []
    tour_leg_counts = []
    
    flow_keys = list(flow_constraints.keys())
    flow_key_to_idx = {k: i for i, k in enumerate(flow_keys)}
    valid_flow_indices = set() 

    for tid, group in tour_groups:
        home_geo = group['home_geohash'].iloc[0]
        home_lga = geo_to_lga.get(home_geo)
        if not home_lga: home_lga = group['lga_code_1'].iloc[0]
            
        current_tour_flow_indices = []
        legs = 0
        for _, row in group.iterrows():
            od_pair = (row['lga_code_1'], row['lga_code_2'])
            if od_pair in flow_key_to_idx:
                idx = flow_key_to_idx[od_pair]
                current_tour_flow_indices.append(idx)
                valid_flow_indices.add(idx)
                legs += 1
        
        if legs > 0:
            tour_ids.append(tid)
            tour_home_lgas.append(home_lga)
            tour_flows_indices.append(current_tour_flow_indices)
            tour_leg_counts.append(legs)

    final_flow_targets = []
    final_flow_mapping = {} 
    for i in range(len(flow_keys)):
        if i in valid_flow_indices:
            new_idx = len(final_flow_targets)
            final_flow_mapping[i] = new_idx 
            final_flow_targets.append(flow_constraints[flow_keys[i]])

    b_raw = np.array(final_flow_targets)
    if len(b_raw) == 0: return None
    norm_factor = np.sum(b_raw)
    b = b_raw / norm_factor

    n_tours = len(tour_ids)
    A_lil = lil_matrix((len(b), n_tours))
    for t_idx, flow_idxs in enumerate(tour_flows_indices):
        for old_flow_idx in flow_idxs:
            if old_flow_idx in final_flow_mapping:
                row_idx = final_flow_mapping[old_flow_idx]
                A_lil[row_idx, t_idx] += 1       
    A_csr = A_lil.tocsr()

    raw_weights = solve_dual_maxent(A_csr, b, n_tours)
    
    covered_target_vol = np.sum(b_raw) 
    legs_array = np.array(tour_leg_counts)
    generated_trips_prob = np.sum(raw_weights * legs_array)
    scale_to_volume = covered_target_vol / generated_trips_prob if generated_trips_prob > 0 else 0
    stage1_weights = raw_weights * scale_to_volume
    
    df_weights = pd.DataFrame({'tour_id': tour_ids, 'home_lga': tour_home_lgas, 'weight': stage1_weights})
    lga_usage = df_weights.groupby('home_lga')['weight'].sum()
    
    lga_scaling = {}
    for lga, usage in lga_usage.items():
        cap = lga_max_agents_map.get(lga, 0)
        if usage > cap and cap > 0:
            lga_scaling[lga] = cap / usage
        else:
            lga_scaling[lga] = 1.0
            
    df_weights['final_weight'] = df_weights.apply(lambda x: x['weight'] * lga_scaling.get(x['home_lga'], 1.0), axis=1)
    tour_weight_map = dict(zip(df_weights['tour_id'], df_weights['final_weight']))
    real_tours['agent_count'] = real_tours['tour_id'].map(tour_weight_map).fillna(0)
    
    geo_usage = real_tours.drop_duplicates('tour_id').groupby('home_geohash')['agent_count'].sum().to_dict()
    geo_caps['used_agents'] = geo_caps['node_id'].map(geo_usage).fillna(0)
    geo_caps['leftover_capacity'] = geo_caps['max_agents'] - geo_caps['used_agents']
    leftovers = geo_caps[geo_caps['leftover_capacity'] > 0.1].copy()
    
    current_trips = real_tours.groupby(['lga_code_1', 'lga_code_2'])['agent_count'].sum().to_dict()
    missing_demand = {}
    for (orig, dest), target in flow_constraints.items():
        curr = current_trips.get((orig, dest), 0)
        gap = target - curr
        if gap > 1.0:
            missing_demand[(orig, dest)] = gap
            
    dummy_tours_df = create_dummy_tours_from_leftover(missing_demand, leftovers, target_date, target_biweek, geo_to_lga)
    
    if not dummy_tours_df.empty:
        final_day_df = pd.concat([real_tours, dummy_tours_df], ignore_index=True)
    else:
        final_day_df = real_tours
        
    final_day_df['agent_id'] = final_day_df['tour_id'].astype(str) + '_' + str(target_date)
    final_day_df['agent_count'] = np.floor(final_day_df['agent_count']).astype(int)
    final_day_df = final_day_df[final_day_df['agent_count'] > 0]
    
    print(f"  - Completed. Generated {len(final_day_df)} rows.")
    return final_day_df

def calculate_metrics_and_plot(optimized_df, gravity_df, label, ax=None):
    """
    Calculates metrics and plots to a specific axis using Nature style.
    """
    processed_dates = optimized_df['custom_day'].unique()
    grav_filtered = gravity_df[gravity_df['date'].isin(processed_dates)].copy()
    
    if grav_filtered.empty or optimized_df.empty:
        return None

    grav_agg = grav_filtered.groupby(['from_node_id', 'to_node_id'])['est_trip_num'].sum().reset_index()
    grav_agg.rename(columns={'from_node_id': 'Origin', 'to_node_id': 'Destination', 'est_trip_num': 'Gravity_Trips'}, inplace=True)

    opt_agg = optimized_df.groupby(['lga_code_1', 'lga_code_2'])['agent_count'].sum().reset_index()
    opt_agg.rename(columns={'lga_code_1': 'Origin', 'lga_code_2': 'Destination', 'agent_count': 'Optimized_Trips'}, inplace=True)

    comparison = pd.merge(grav_agg, opt_agg, on=['Origin', 'Destination'], how='left')
    comparison['Optimized_Trips'] = comparison['Optimized_Trips'].fillna(0)

    total_grav = comparison['Gravity_Trips'].sum()
    total_opt = comparison['Optimized_Trips'].sum()
    
    if len(comparison) > 1:
        r2 = r2_score(comparison['Gravity_Trips'], comparison['Optimized_Trips'])
        corr = comparison['Gravity_Trips'].corr(comparison['Optimized_Trips'])
        rmse = np.sqrt(mean_squared_error(comparison['Gravity_Trips'], comparison['Optimized_Trips']))
    else:
        r2, corr, rmse = 0, 0, 0

    ratio = total_opt / total_grav if total_grav > 0 else 0

    if ax:
        # Nature Style Scatter
        ax.scatter(comparison['Gravity_Trips'], comparison['Optimized_Trips'], alpha=0.6, s=5, c='#1f77b4', edgecolors='none')
        
        # Perfect Fit Line
        max_val = max(comparison['Gravity_Trips'].max(), comparison['Optimized_Trips'].max())
        ax.plot([0, max_val], [0, max_val], color='#d62728', linestyle='--', linewidth=1)
        
        # Title and Labels
        ax.set_title(f"{label}\n$R^2$={r2:.2f}, $r$={corr:.2f}", fontsize=7)
        
        # Adjust axis ticks to be cleaner
        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_tick_params(width=0.5)

    return {
        "Period": label,
        "Total_Gravity_Trips": total_grav,
        "Total_Optimized_Trips": total_opt,
        "Volume_Ratio": ratio,
        "R2": r2,
        "Correlation": corr,
        "RMSE": rmse
    }

def main():
    print(f"--- Starting Full Multi-Day Optimization (Nature Style) ---")
    start_time = time.time()
    
    # Set Style
    set_nature_style()

    if not all(os.path.exists(f) for f in FILENAMES.values()):
        print("ERROR: One or more files not found.")
        return

    df_geo = pd.read_csv(FILENAMES['geo_lga'])
    df_tours = pd.read_csv(FILENAMES['tours'])
    df_metrics = pd.read_csv(FILENAMES['metrics'])
    df_gravity = pd.read_csv(FILENAMES['gravity'])
    geo_to_lga = dict(zip(df_geo['node_id'], df_geo['LGA_code']))

    unique_dates = sorted(df_tours['custom_day'].unique())
    print(f"Found {len(unique_dates)} dates to process: {unique_dates}")
    
    all_results = []
    for date in unique_dates:
        res = process_single_day(date, df_geo, df_tours, df_metrics, df_gravity, geo_to_lga)
        if res is not None:
            all_results.append(res)
            
    if all_results:
        print("\nCombining and saving results...")
        final_combined_df = pd.concat(all_results, ignore_index=True)
        
        # 1. Final Cleanup & Save
        cols_to_drop = ['local_datetime_1', 'local_datetime_2', 'home_geohash']
        final_combined_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        output_csv_path = os.path.join(OUTPUT_DIR, "optimized_tours_all_days_final.csv")
        final_combined_df.to_csv(output_csv_path, index=False)
        
        print(f"--- Success! Saved Main CSV to {output_csv_path} ---")
        
        # 2. Validation & Reporting
        print("\n--- Generating Nature-Style Validation Reports ---")
        metrics_list = []
        
        # A. Global Aggregate Plot (Single Column Width ~3.5 inch)
        plt.figure(figsize=(3.5, 3.5))
        ax_global = plt.gca()
        
        agg_metrics = calculate_metrics_and_plot(final_combined_df, df_gravity, "Aggregate Calibration", ax_global)
        metrics_list.append(agg_metrics)
        
        ax_global.set_xlabel('Gravity Model Trips')
        ax_global.set_ylabel('Optimized Agent Trips')
        
        plt.tight_layout()
        plot_agg_path = os.path.join(OUTPUT_DIR, "calibration_plot_aggregate.png")
        plt.savefig(plot_agg_path)
        plt.close()
        print(f"Saved Aggregate Plot to {plot_agg_path}")
        
        # B. Month-by-Month Subplots (Double Column Width ~7.2 inch)
        final_combined_df['temp_date'] = pd.to_datetime(final_combined_df['custom_day'])
        final_combined_df['YearMonth'] = final_combined_df['temp_date'].dt.to_period('M')
        unique_months = sorted(final_combined_df['YearMonth'].unique())
        
        if len(unique_months) > 0:
            n_months = len(unique_months)
            cols = 3
            rows = math.ceil(n_months / cols)
            
            # Height depends on rows, width is fixed for double column (~7.2 inch)
            fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.2 * rows), constrained_layout=True)
            axes_flat = axes.flatten() if n_months > 1 else [axes]
            
            for i, ym in enumerate(unique_months):
                month_str = str(ym)
                month_df = final_combined_df[final_combined_df['YearMonth'] == ym].copy()
                
                m_metrics = calculate_metrics_and_plot(month_df, df_gravity, f"{month_str}", axes_flat[i])
                if m_metrics:
                    metrics_list.append(m_metrics)
            
            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].axis('off')
                
            plot_monthly_path = os.path.join(OUTPUT_DIR, "calibration_plot_monthly_combined.png")
            plt.savefig(plot_monthly_path)
            print(f"Saved Combined Monthly Plot to {plot_monthly_path}")
            plt.close()
            
        # C. Save Metrics CSV
        metrics_df = pd.DataFrame(metrics_list)
        metrics_csv_path = os.path.join(OUTPUT_DIR, "validation_metrics_summary.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved Validation Metrics to {metrics_csv_path}")
        
        # Print Summary
        print("\n--- Summary Report ---")
        print(metrics_df[['Period', 'R2', 'Correlation', 'Volume_Ratio']].to_string(index=False))
        
    else:
        print("No results generated.")

    print(f"Total Run Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()