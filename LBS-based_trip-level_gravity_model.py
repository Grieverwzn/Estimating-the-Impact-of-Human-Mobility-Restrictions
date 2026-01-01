"""
LBS-Based Model Calibration using Gravity Model and Iterative Proportional Fitting (IPF)

Author: Xin Wu
        Villanova University
        xwu03@villanova.edu

Description:
    This script calibrates a gravity model for trip distribution using Location-Based Services (LBS) data
    and applies Iterative Proportional Fitting (IPF) to generate spatially and temporally refined trip estimates.

Methodology:
    1. Gravity Model Calibration: Uses observed LBS trip data to calibrate gravity model parameters
       (alpha, beta, gamma, G) via log-linear regression. The gravity model estimates trip flows as:
       T_ij = G * (P_i^alpha) * (P_j^beta) / (d_ij^gamma)
       where T_ij is trips from zone i to j, P is population, and d is distance.

    2. Iterative Proportional Fitting (IPF): Adjusts gravity model estimates to match production and
       attraction constraints derived from:
       - Population data (spatial distribution of trip generators/attractors)
       - LBS-derived trip rates (trips per person per day by zone and time period)
       - External trip rates (proportion of trips captured in the study area)

    3. Temporal Disaggregation: Distributes period-level (weekly/monthly) trip estimates to daily
       estimates using observed temporal patterns from LBS data.

Key Features:
    - Multi-period calibration (weekly, monthly, or yearly)
    - Zone-specific trip generation rates (LGA-level heterogeneity)
    - Origin-zone-specific temporal patterns
    - Complete spatial coverage via gravity model extrapolation
    - Convergence-based IPF with configurable tolerance

Input Data:
    - node.csv: Spatial zones with population, coordinates, and attributes
    - demand.csv: Observed OD trip flows from LBS data
    - trip_rate.csv: Trip generation rates and external trip rates by zone and period

Output:
    - Calibrated gravity model parameters by period
    - Full OD matrices with IPF-adjusted trip estimates
    - Daily trip estimates for the entire study period
    - Diagnostic plots and statistical summaries

Configuration:
    - period: Temporal aggregation level ('year_week', 'month', 'year')
    - resolution_of_TAZs: Spatial resolution ('LGA' for Local Government Areas)
    - output_folder: Directory for output files

Date: 2024
Version: 2.0 - Enhanced with zone-specific rates and origin-based temporal patterns
"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import time
import geopandas as gpd
from scipy.stats import gaussian_kde
from collections import Counter
import os

# pd.options.display.float_format = '{:,.1f}'.format
pd.set_option('display.max_columns', None)
period = 'year_week'  # can be 'year_week', 'year', or 'month'
resolution_of_TAZs = 'LGA'

output_folder = 'output_calibration'


def _calculate_distance_from_geometry(lon1, lat1, lon2, lat2):
    """
    Calculate Haversine distance between two sets of lat/lon coordinates (in miles),
    input can be scalar or NumPy arrays.
    """
    radius = 6371.0  # Radius of Earth in km
    # Convert decimal degrees to radians
    lon1 = np.radians(lon1)  # degree * pi / 180
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Haversine formula
    d_latitude = lat2 - lat1
    d_longitude = lon2 - lon1

    a = np.sin(d_latitude / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_longitude / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    km = radius * c  # Radius of Earth in km
    miles = km * 1000 / 1609.34  # Convert to miles

    return miles


def _generate_links(nodes):
    node_ids = nodes['node_id'].values
    x_coords = nodes['x_coord'].values
    y_coords = nodes['y_coord'].values
    population = nodes['population'].values
    n = len(node_ids)

    # Generate full OD grid (including self-links)
    from_ids = np.repeat(node_ids, n)
    to_ids = np.tile(node_ids, n)
    from_x = np.repeat(x_coords, n)
    from_y = np.repeat(y_coords, n)
    to_x = np.tile(x_coords, n)
    to_y = np.tile(y_coords, n)
    from_pop = np.repeat(population, n)
    to_pop = np.tile(population, n)

    # Remove self-links
    mask = from_ids != to_ids

    # Compute distances only for valid OD pairs
    distances = _calculate_distance_from_geometry(from_x[mask], from_y[mask], to_x[mask], to_y[mask])

    from_x_str = from_x[mask].astype(str)
    from_y_str = from_y[mask].astype(str)
    to_x_str = to_x[mask].astype(str)
    to_y_str = to_y[mask].astype(str)

    # Construct DataFrame
    # link_id use integer
    links = pd.DataFrame({
        'link_id': np.arange(len(from_ids[mask]), dtype=int),
        'from_node_id': from_ids[mask],
        'to_node_id': to_ids[mask],
        'distance': distances,
        'from_node_population': from_pop[mask],
        'to_node_population': to_pop[mask]
    })
    print(f"Generated {len(links)} links...")
    # remove links with from_node_population or to_node_population <= 0
    links = links[(links['from_node_population'] > 0) & (links['to_node_population'] > 0)]
    # remove links with distance <= 0
    links = links[links['distance'] > 0]
    print(f"Generated {len(links)} links after filtering...")
    # take log of distance, from_node_population, and to_node_population for full gravity model
    links['log_distance'] = np.log(links['distance'])
    links['log_from_node_population'] = np.log(links['from_node_population'])
    links['log_to_node_population'] = np.log(links['to_node_population'])
    standize_cols = ['log_from_node_population', 'log_to_node_population', 'log_distance']
    stand_scaler = StandardScaler()
    links[standize_cols] = stand_scaler.fit_transform(links[standize_cols])
    links.to_csv(f'{output_folder}/links.csv', index=False)
    return links


def _gravity_model(from_population, to_population, distance, aa, bb, gg, GG):
    """
    Gravity model function to estimate trip counts.
    """
    return GG * (from_population ** aa) * (to_population ** bb) / (distance ** gg)


def _linear_gravity_model(params, log_from_population, log_to_population, log_distance, log_trip_num):
    """
    Linear gravity model function to estimate trip counts.
    """
    a, b, g, log_g = params
    log_trip_estimated = log_g + a * log_from_population + b * log_to_population + g * log_distance
    residuals = log_trip_num - log_trip_estimated
    return np.mean(residuals ** 2)  # Mean Squared Error (MSE) as the objective function


def _fix_param(p, default):
    """
    Fix the parameter p to a default value if it is less than or equal to 0.
    """
    return p if p > 0 else default


def _linear_regression_with_summary(x_var, y_var):
    x_var1 = sm.add_constant(x_var)  # add constant term for intercept
    lp_model = sm.OLS(y_var, x_var1).fit()  # Ordinary Least Squares regression
    aa = lp_model.params['log_from_node_population']
    bb = lp_model.params['log_to_node_population']
    gg = lp_model.params['log_distance']
    big_g = np.exp(lp_model.params['const'])  # intercept is log(G), so we exponentiate it
    print("Stage 1: parameters: alpha =", aa, ", beta =", bb, ", gamma =", gg, ", G =", big_g)
    print("Stage 1: R^2 =", lp_model.rsquared)
    # print("Stage 1: summary:", lp_model.summary())
    r_squared = lp_model.rsquared
    return aa, bb, gg, big_g, r_squared


def _linear_regression(x_var, y_var):
    """
    Perform linear regression and return the coefficients.
    """
    model = LinearRegression()
    model.fit(x_var, y_var)  # fit the linear regression model
    aa = model.coef_[0]
    bb = model.coef_[1]
    gg = model.coef_[2]
    big_g = np.exp(model.intercept_)
    print("Stage 1: parameters: alpha =", aa, ", beta =", bb, ", gamma =", gg, ", G =", big_g)
    print("Stage 1: R^2 =", model.score(x_var, y_var))
    r_squared = model.score(x_var, y_var)
    return aa, bb, gg, big_g, r_squared


def _linear_regression_minimization(x_var, y_var):
    alpha_init = 0.01
    beta_init = 0.01
    gamma_init = -1.5
    log_g_init = np.log(1e-6)  # G should be positive, so we use log(1e-6) as initial guess
    init_params = np.array([alpha_init, beta_init, -gamma_init, log_g_init])
    bounds = [(0, None),  # alpha
              (0, None),  # beta
              (None, -1e-6),  # gamma
              (0, None)]  # log_G

    result = minimize(_linear_gravity_model, init_params,
                      args=(group_df['log_from_node_population'].values,
                            group_df['log_to_node_population'].values,
                            group_df['log_distance'].values,
                            group_df['log_trip_num'].values),
                      bounds=bounds)
    opt_alpha, opt_beta, opt_gamma, opt_log_g = result.x  # convert back to positive gamma
    print(f"Stage 1: parameters: alpha = {opt_alpha}, "
          f"beta = {opt_beta}, gamma = {opt_gamma}, G = {np.exp(opt_log_g)}")
    # calculate R^2
    r_squared = result.fun  # this is the mean squared error, we need to calculate R^2
    r_squared = 1 - (r_squared / np.var(Y))  # R^2 = 1 - (MSE / variance)
    print("Stage 1: R^2 =", r_squared)
    return opt_alpha, opt_beta, opt_gamma, np.exp(opt_log_g), r_squared


# Step 0: Load data
start_time = time.time()
node_df = pd.read_csv('node.csv')
# only keep 'node_type' == 'geohash6'
node_df = node_df[node_df['node_type'] == resolution_of_TAZs]
# if population is 0 or NaN, do not consider this node
node_df = node_df[(node_df['population'] > 0) & (node_df['population'].notna())]
print(f"Loaded {len(node_df)} nodes...")
link_df = _generate_links(node_df)
trip_rate_df = pd.read_csv('trip_rate.csv')
end_time = time.time()
print("Data loading time: ", end_time - start_time, "seconds")

# Step 1: sum up the observed trip num for each year_week (according to the date column)
start_time = time.time()
demand_df = pd.read_csv('demand.csv')

# remove trips where from_node_id == to_node_id
demand_df = demand_df[demand_df['from_node_id'] != demand_df['to_node_id']]
demand_df['date'] = pd.to_datetime(demand_df['date'])
trip_rate_df['date'] = pd.to_datetime(trip_rate_df['date'])
demand_df = demand_df.sort_values(by='date')
demand_df = demand_df.groupby(['date', 'from_node_id', 'to_node_id']).agg({'trip_num': 'sum'}).reset_index()

# generate period column based on the period variable
if period == 'year_week':
    demand_df['year_week'] = (demand_df['date'].dt.isocalendar().year.astype(str) +
                              '-w-' + demand_df['date'].dt.isocalendar().week.astype(str).str.zfill(2))
    trip_rate_df['year_week'] = (trip_rate_df['date'].dt.isocalendar().year.astype(str) +
                                 '-w-' + trip_rate_df['date'].dt.isocalendar().week.astype(str).str.zfill(2))
elif period == 'year':
    demand_df['year'] = demand_df['date'].dt.year
    trip_rate_df['year'] = trip_rate_df['date'].dt.year
elif period == 'month':
    demand_df['month'] = (demand_df['date'].dt.isocalendar().year.astype(str) +
                          '-m-' + demand_df['date'].dt.month.astype(str).str.zfill(2))
    trip_rate_df['month'] = (trip_rate_df['date'].dt.isocalendar().year.astype(str) +
                             '-m-' + trip_rate_df['date'].dt.month.astype(str).str.zfill(2))
    # zfill(2) means to fill the week number with 0 if it is less than 10
else:
    raise ValueError(f"Invalid period: {period}")

# calculate the trip rate per day within each period FOR EACH FROM_NODE (origin zone)
# This ensures better coverage - all OD pairs from the same origin share the same temporal pattern
demand_df['period_total_trip_from_node'] = demand_df.groupby([period, 'from_node_id'])['trip_num'].transform('sum')
demand_df['date_total_trip_from_node'] = demand_df.groupby(['date', 'from_node_id'])['trip_num'].transform('sum')
demand_df['trip_rate_within_period'] = demand_df['date_total_trip_from_node'] / demand_df['period_total_trip_from_node']

start_date = demand_df['date'].min()
start_date = trip_rate_df['date'].min() if start_date < trip_rate_df['date'].min() else start_date
end_date = demand_df['date'].max()
end_date = trip_rate_df['date'].max() if end_date > trip_rate_df['date'].max() else end_date
# only keep the demand_df and trip_rate_df within the start_date and end_date
demand_df = demand_df[(demand_df['date'] >= start_date) & (demand_df['date'] <= end_date)]
trip_rate_df = trip_rate_df[(trip_rate_df['date'] >= start_date) & (trip_rate_df['date'] <= end_date)]

# create dictionaries with TUPLE KEYS: (date, from_node_id)
# All OD pairs originating from the same from_node_id will share the same temporal pattern
date_trip_rate_dict = dict(zip(
    zip(demand_df['date'], demand_df['from_node_id']),
    demand_df['trip_rate_within_period']
))
date_period_dict = dict(zip(demand_df['date'], demand_df[period]))

# calculate the number of days in the period
demand_df['period_days'] = demand_df.groupby(period)['date'].transform('nunique')
nb_day_period_dict = dict(zip(demand_df[period], demand_df['period_days']))

# drop unnecessary columns
demand_df = demand_df.drop(columns=['period_total_trip_from_node', 'date_total_trip_from_node',
                                     'trip_rate_within_period', 'period_days'])

# Process trip_rate_df to create dictionaries with TUPLE KEYS: (period, from_node_id)
trip_rate_df = trip_rate_df.groupby([period, 'node_id']).agg({
    'trip_person': 'mean',
    'external_trip_rate': 'mean'
}).reset_index()

period_expected_trip_rate_dict = dict(zip(
    zip(trip_rate_df[period], trip_rate_df['node_id']),
    trip_rate_df['trip_person']
))
period_external_trip_rate_dict = dict(zip(
    zip(trip_rate_df[period], trip_rate_df['node_id']),
    trip_rate_df['external_trip_rate']
))

# Aggregate demand_df by period and OD pair
demand_df = demand_df.groupby([period, 'from_node_id', 'to_node_id']).agg({'trip_num': 'sum'}).reset_index()

print(f"Aggregated demand data for each {period}...")

# create dictionaries to map node_id to x_coord, y_coord, and population
node_x_dict = dict(zip(node_df['node_id'], node_df['x_coord']))
node_y_dict = dict(zip(node_df['node_id'], node_df['y_coord']))
node_pop_dict = dict(zip(node_df['node_id'], node_df['population']))

# merge x_coord and y_coord into demand_df according to from_node_id and to_node_id,
demand_df['from_node_x_coord'] = demand_df['from_node_id'].map(node_x_dict)
demand_df['from_node_y_coord'] = demand_df['from_node_id'].map(node_y_dict)
demand_df['to_node_x_coord'] = demand_df['to_node_id'].map(node_x_dict)
demand_df['to_node_y_coord'] = demand_df['to_node_id'].map(node_y_dict)
demand_df['from_node_population'] = demand_df['from_node_id'].map(node_pop_dict)
demand_df['to_node_population'] = demand_df['to_node_id'].map(node_pop_dict)

# use map_reduce to calculate distance
demand_df['distance'] = _calculate_distance_from_geometry(demand_df['from_node_x_coord'],
                                                          demand_df['from_node_y_coord'],
                                                          demand_df['to_node_x_coord'],
                                                          demand_df['to_node_y_coord'])

end_time = time.time()
print("Data processing time: ", end_time - start_time, "seconds")

# Step 2: calibrate gravity model parameters for each period
param_dict = {}
# three steps in this loop:
# 1. gravity model calibration using observed trip data
# 2. generate full od estimates based on calibrated gravity model parameters
# 3. adjust trip estimates to match real world totals via iterative proportional fitting (IPF)
period_demand_dict = {}
for period_id, group_df in demand_df.groupby(period):
    print("================== Start of period", period_id, "calibration ==================")
    print(f"Stage 1: Calibrating gravity model for {period_id}...")
    start_time = time.time()
    # step 2.1: determine the number of days in the period
    nb_days = nb_day_period_dict[period_id]

    # step 2.2: data cleaning
    # drop trip_num, from_node_population, to_node_population, distance with NaN
    group_df = group_df.dropna(subset=['trip_num', 'from_node_population', 'to_node_population', 'distance'])
    # drop infinity values
    group_df = group_df[~group_df['trip_num'].isin([np.inf, -np.inf])]
    group_df = group_df[~group_df['from_node_population'].isin([np.inf, -np.inf])]
    group_df = group_df[~group_df['to_node_population'].isin([np.inf, -np.inf])]
    group_df = group_df[~group_df['distance'].isin([np.inf, -np.inf])]
    group_df = group_df[group_df['trip_num'] > 0]
    group_df = group_df[group_df['from_node_population'] > 0]
    group_df = group_df[group_df['to_node_population'] > 0]
    group_df = group_df[group_df['distance'] > 0]

    # step 2.3: calculate the log values
    group_df['log_trip_num'] = np.log(group_df['trip_num'])
    group_df['log_from_node_population'] = np.log(group_df['from_node_population'])
    group_df['log_to_node_population'] = np.log(group_df['to_node_population'])
    group_df['log_distance'] = np.log(group_df['distance'])

    # step 2.4: standardize the data
    x_cols = ['log_from_node_population', 'log_to_node_population', 'log_distance']
    scaler = StandardScaler()
    group_df[x_cols] = scaler.fit_transform(group_df[x_cols])

    # step 2.5: fit the linear regression model for the gravity model
    X = group_df[x_cols]
    Y = group_df['log_trip_num']

    # alpha, beta, gamma, G, R2 = _linear_regression_with_summary(X, Y)
    # alpha, beta, gamma, G, R2 = _linear_regression(X, Y)
    alpha, beta, gamma, G, R2 = _linear_regression_minimization(X, Y)
    group_df['est_trip_num'] = np.exp(group_df['log_from_node_population'] * alpha +
                                      group_df['log_to_node_population'] * beta +
                                      group_df['log_distance'] * gamma + np.log(G))

    # save to csv
    group_df.to_csv(f'{output_folder}/lp-{resolution_of_TAZs}-calibrated_trip_count_{period_id}.csv', index=False)

    # plot estimated trip count vs actual trip count
    plt.scatter(group_df['trip_num'], group_df['est_trip_num'], alpha=0.5)
    plt.xlabel('Actual Trip Count')
    plt.ylabel('Estimated Trip Count')
    plt.title(f'Estimated vs Actual Trip Count for Month {period_id}:\nalpha = {alpha:.2f}, '
              f'beta = {beta:.2f}, gamma = {gamma:.2f}, '
              f'G = {G:.2f},\nR2 = {R2:.2f}',
              fontsize=8)
    # plot y=x line
    plt.plot([0, max(group_df['trip_num'])], [0, max(group_df['trip_num'])], 'r--')
    # save the plot
    plt.tight_layout()
    plt.savefig(f'{output_folder}/lp-{resolution_of_TAZs}-calibrated_trip_count_{period_id}.png', dpi=300)
    plt.close()
    end_time = time.time()
    print("Stage 1: Gravity model calibration time for period", period_id, ": ", end_time - start_time, "seconds")

    print(f"Stage 2: Generating full OD estimates for {period_id}...")
    start_time = time.time()
    temp_link_df = link_df.copy()
    temp_link_df['period'] = period_id
    temp_link_df['est_trip_num'] = np.exp(
        temp_link_df['log_from_node_population'] * alpha +
        temp_link_df['log_to_node_population'] * beta +
        temp_link_df['log_distance'] * gamma + np.log(G)
    )
    end_time = time.time()
    print("Stage 2: Full OD estimates generation time for period", period_id, ": ", end_time - start_time, "seconds")

    # adjust trip estimates to match real world totals via iterative proportional fitting (IPF)
    print(f"Stage 3: Adjusting trip estimates for {period_id} using iterative proportional fitting (IPF)...")
    start_time = time.time()

    # Use LGA-specific trip rates with tuple keys: (period_id, from_node_id)
    def get_trip_rate(node_id, rate_dict, period_id):
        """Helper function to get trip rate with fallback to mean if not found"""
        key = (period_id, node_id)
        if key in rate_dict:
            return rate_dict[key]
        else:
            # Fallback to mean of all values for this period
            period_values = [v for k, v in rate_dict.items() if k[0] == period_id]
            if period_values:
                return np.mean(period_values)
            else:
                return 1.0  # Default fallback

    temp_link_df['expected_trip_rate'] = temp_link_df['from_node_id'].apply(
        lambda x: get_trip_rate(x, period_expected_trip_rate_dict, period_id)
    )
    temp_link_df['expected_external_rate'] = temp_link_df['from_node_id'].apply(
        lambda x: get_trip_rate(x, period_external_trip_rate_dict, period_id)
    )

    expected_prod = (temp_link_df['from_node_population'].astype('float64') * nb_days *
                     temp_link_df['expected_trip_rate'] * temp_link_df['expected_external_rate'])
    temp_link_df['expected_production'] = expected_prod.astype('float64')

    # For attraction, use to_node_id
    temp_link_df['expected_trip_rate_to'] = temp_link_df['to_node_id'].apply(
        lambda x: get_trip_rate(x, period_expected_trip_rate_dict, period_id)
    )
    temp_link_df['expected_external_rate_to'] = temp_link_df['to_node_id'].apply(
        lambda x: get_trip_rate(x, period_external_trip_rate_dict, period_id)
    )

    expected_attraction = (temp_link_df['to_node_population'].astype('float64') * nb_days *
                           temp_link_df['expected_trip_rate_to'] * temp_link_df['expected_external_rate_to'])
    temp_link_df['expected_attraction'] = expected_attraction.astype('float64')

    # initialize scaled_trip with estimated trip numbers
    temp_link_df['scaled_trips'] = temp_link_df['est_trip_num']

    for iteration in range(100):
        previous_scaled_trip = temp_link_df['scaled_trips'].copy()
        temp_prod = (temp_link_df.groupby('from_node_id')['scaled_trips'].transform('sum'))
        temp_link_df['temp_prod'] = temp_prod.astype('float64')

        temp_link_df['scaled_trips'] = (temp_link_df['scaled_trips'] /
                                        temp_link_df['temp_prod']) * temp_link_df['expected_production']
        temp_attr = (temp_link_df.groupby('to_node_id')['scaled_trips'].transform('sum'))
        temp_link_df['temp_attr'] = temp_attr.astype('float64')
        temp_link_df['scaled_trips'] = (temp_link_df['scaled_trips'] /
                                        temp_link_df['temp_attr']) * temp_link_df['expected_attraction']
        # check convergence
        max_diff = np.max(np.abs(temp_link_df['scaled_trips'] - previous_scaled_trip))
        if max_diff < 1e-6:
            print(f"Stage 3: IPF converges after {iteration + 1} iterations.")
            break
    # save pivot table of scaled trip counts
    temp_link_df.rename(columns={'scaled_trips': 'est_trips'}, inplace=True)
    pivot_table = temp_link_df.pivot_table(index='from_node_id', columns='to_node_id', values='est_trips',
                                           fill_value=0)
    pivot_table.to_csv(f'{output_folder}/lp-{resolution_of_TAZs}-est_od_trips_{period_id}.csv')
    # IPF loop
    # visualize scaled trip vs population
    plt.scatter(temp_link_df['from_node_population'], temp_link_df['est_trips'], alpha=0.5)
    plt.xlabel('From Node Population')
    plt.ylabel('Scaled Trip Count')
    plt.title(f'Scaled Trip Count vs From Node Population for {period_id}')
    plt.savefig(f'{output_folder}/lp-{resolution_of_TAZs}-est_trip_vs_from_population_{period_id}.png')
    plt.close()

    plt.scatter(temp_link_df['to_node_population'], temp_link_df['est_trips'], alpha=0.5)
    plt.xlabel('To Node Population')
    plt.ylabel('Scaled Trip Count')
    plt.title(f'Scaled Trip Count vs To Node Population for {period_id}')
    plt.savefig(f'{output_folder}/lp-{resolution_of_TAZs}-est_trip_vs_to_population_{period_id}.png')
    plt.close()

    plt.scatter(temp_link_df['distance'], temp_link_df['est_trips'], alpha=0.5)
    plt.xlabel('Distance')
    plt.ylabel('Scaled Trip Count')
    plt.title(f'Scaled Trip Count vs Distance for {period_id}')
    plt.savefig(f'{output_folder}/lp-{resolution_of_TAZs}-est_trip_vs_distance_{period_id}.png')
    plt.close()

    period_demand_dict[period_id] = temp_link_df[['from_node_id', 'to_node_id', 'est_trips']].copy()
    end_time = time.time()
    print("Stage 3: IPF adjustment time for period", period_id, ": ", end_time - start_time, "seconds")
    print("================== End of period", period_id, "calibration ==================")

# Step 3: generate daily trip according to the total trip per day within the period FOR EACH OD PAIR
start_time = time.time()
date_list = []
for date in pd.date_range(start=start_date, end=end_date):
    print(f"Stage 4: Estimating trips for date {date.strftime('%Y-%m-%d')}...")
    date_str = date.strftime('%Y-%m-%d')
    period_id = date_period_dict[date]
    period_df = period_demand_dict[period_id].copy()

    # Apply from_node-specific trip rates using tuple keys: (date, from_node_id)
    # This ensures all OD pairs from the same origin have the same temporal pattern
    def get_from_node_trip_rate(row):
        """Helper function to get from_node-specific trip rate with fallback"""
        key = (date, row['from_node_id'])
        if key in date_trip_rate_dict:
            return date_trip_rate_dict[key]
        else:
            # Fallback: use average trip rate for this date across all origins
            date_keys = [k for k in date_trip_rate_dict.keys() if k[0] == date]
            if date_keys:
                return np.mean([date_trip_rate_dict[k] for k in date_keys])
            else:
                # If no data for this date, assume uniform distribution within period
                return 1.0 / nb_day_period_dict[period_id]

    period_df['trip_rate_within_period'] = period_df.apply(get_from_node_trip_rate, axis=1)
    period_df['daily_est_trips'] = period_df['est_trips'] * period_df['trip_rate_within_period']
    period_df['date'] = date_str
    period_df = period_df[['from_node_id', 'to_node_id', 'date', 'daily_est_trips']]
    date_list.append(period_df)

# concatenate all period_df into a single DataFrame
final_df = pd.concat(date_list, ignore_index=True)
# rename columns
final_df.rename(columns={'daily_est_trips': 'est_trip_num'}, inplace=True)
# save final_df to csv
final_df.to_csv(f'{output_folder}/lp-estimated_trips_{resolution_of_TAZs}.csv', index=False)
end_time = time.time()
print("Stage 4: Daily trip estimation time: ", end_time - start_time, "seconds")
# Step 4: enrich the estimated trips with node attributes for Kepler.gl visualization
print("Enriching estimated trips with node attributes for visualization...")
start_time = time.time()

# Load the files
estimated_trips = pd.read_csv('output_calibration/lp-estimated_trips_LGA.csv')
node_df = pd.read_csv('node.csv')

# Filter to only LGA nodes
node_df = node_df[node_df['node_type'] == 'LGA']

# Convert date to datetime and format for Kepler.gl
estimated_trips['date'] = pd.to_datetime(estimated_trips['date'])
estimated_trips['datetime'] = estimated_trips['date'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

# Create dictionaries for node attributes
node_name_dict = dict(zip(node_df['node_id'], node_df['node_name']))
node_x_dict = dict(zip(node_df['node_id'], node_df['x_coord']))
node_y_dict = dict(zip(node_df['node_id'], node_df['y_coord']))
node_pop_dict = dict(zip(node_df['node_id'], node_df['population']))

# Attach from_node attributes
estimated_trips['from_node_name'] = estimated_trips['from_node_id'].map(node_name_dict)
estimated_trips['from_x_coord'] = estimated_trips['from_node_id'].map(node_x_dict)
estimated_trips['from_y_coord'] = estimated_trips['from_node_id'].map(node_y_dict)
estimated_trips['from_population'] = estimated_trips['from_node_id'].map(node_pop_dict)

# Attach to_node attributes
estimated_trips['to_node_name'] = estimated_trips['to_node_id'].map(node_name_dict)
estimated_trips['to_x_coord'] = estimated_trips['to_node_id'].map(node_x_dict)
estimated_trips['to_y_coord'] = estimated_trips['to_node_id'].map(node_y_dict)
estimated_trips['to_population'] = estimated_trips['to_node_id'].map(node_pop_dict)

# Calculate est_trip_per_person (trips per capita from origin)
estimated_trips['est_trip_per_person'] = estimated_trips['est_trip_num'] / estimated_trips['from_population']

# Create LineString geometry in WKT format for Kepler.gl
estimated_trips['geometry'] = estimated_trips.apply(
    lambda row: f"LINESTRING({row['from_x_coord']} {row['from_y_coord']}, {row['to_x_coord']} {row['to_y_coord']})",
    axis=1
)

# Reorder columns for better readability
column_order = [
    'datetime',
    'from_node_id', 'from_node_name', 'from_x_coord', 'from_y_coord', 'from_population',
    'to_node_id', 'to_node_name', 'to_x_coord', 'to_y_coord', 'to_population',
    'est_trip_num', 'est_trip_per_person',
    'geometry'
]
estimated_trips = estimated_trips[column_order]

# Save enriched file for Kepler.gl
estimated_trips.to_csv('output_calibration/lp-estimated_trips_LGA_enriched.csv', index=False)

print(f"Enriched dataset saved with {len(estimated_trips)} rows")
print(f"Sample datetime: {estimated_trips['datetime'].iloc[0]}")
print(f"Sample geometry: {estimated_trips['geometry'].iloc[0]}")
end_time = time.time()
print("Data enrichment time: ", end_time - start_time, "seconds")
