from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from scipy.ndimage import label


def find_evt_pot_threshold(zone_data, min_p=0.85, max_p=0.995, num_candidates=50):
    """Finds a statistically robust extreme threshold for a single zone's 3-hourly
    residual load profile using Extreme Value Theory (Peaks-Over-Threshold).

    Parameters:
    -----------
    zone_data : np.ndarray or xr.DataArray
        1D array of 3-hourly residual loads for the reference period.
    min_p : float
        The lowest percentile to start testing candidate thresholds.
    max_p : float
        The highest percentile to test.
    num_candidates : int
        The number of threshold increments to evaluate.

    Returns:
    --------
    float
        The calculated robust threshold in GW.
    """
    # Remove negative values to focus strictly on supply shortages (deficits)
    positive_rl = zone_data[zone_data > 0]

    # Generate candidate thresholds spanning from low to high percentiles
    candidates = np.percentile(
        positive_rl, np.linspace(min_p * 100, max_p * 100, num_candidates)
    )

    best_threshold = None

    # Step through candidates from highest to lowest to find where stability breaks
    # (Standard practice for finding the linear region of the GPD tail)
    for u in reversed(candidates):
        excesses = positive_rl[positive_rl > u] - u

        # EVT requires a reasonable sample size in the tail to fit a distribution
        if len(excesses) < 30:
            continue

        try:
            # Fit the Generalized Pareto Distribution (GPD) to the excesses
            # c = shape parameter, loc = location (fixed at 0), scale = scale parameter
            c, loc, scale = stats.genpareto.fit(excesses, floc=0)

            # Perform a Chi-Square Goodness-of-Fit test to see if the excesses
            # actually follow the fitted GPD.
            # Divide the excesses into 5 equal-probability bins based on the fitted CDF
            quantiles = np.linspace(0.1, 0.9, 4)
            bins = stats.genpareto.ppf(quantiles, c, loc=loc, scale=scale)
            bins = np.sort(np.concatenate(([0], bins, [np.max(excesses) + 1])))

            observed, _ = np.histogram(excesses, bins=bins)
            expected = np.ones(len(observed)) * (len(excesses) / len(observed))

            # Calculate p-value. High p-value means we CANNOT reject the hypothesis
            # that the data follows a GPD (i.e., the tail has successfully stabilized).
            _, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

            if p_value > 0.05:
                best_threshold = u
            else:
                # If it fails the test, we've fallen out of the asymptotic extreme tail
                break
        except:
            continue

    # Fallback in case the optimization fails to converge on a clean tail
    if best_threshold is None:
        best_threshold = np.percentile(positive_rl, 97.5)

    return float(best_threshold)


def compute_evt_thresholds_for_all_zones(ds_reference, country_dim="zone"):
    """Vectorized wrapper to run the EVT threshold identification across all

    6 real-time balancing zones in an xarray dataset.
    """
    threshold_dict = {}

    for zone in ds_reference[country_dim].values:
        # Extract the 1D time series for the specific zone
        zone_series = ds_reference.sel({country_dim: zone}).values.flatten()

        # Compute the threshold
        u_robust = find_evt_pot_threshold(zone_series)
        threshold_dict[zone] = u_robust

    # Convert to a clean, indexable xarray DataArray matching your pipeline geometry
    da_thresholds = xr.DataArray(
        list(threshold_dict.values()),
        coords={country_dim: list(threshold_dict.keys())},
        dims=[country_dim],
        name="evt_threshold",
    )

    return da_thresholds


def calculate_climatological_baseline(ds_reference_period, time_dim="time"):
    """Computes expected 3-hourly seasonal baselines for demand and EVERY individual

    generation asset type using a combined MultiIndex coordinate.
    """
    # 1. Extract the time components
    months = ds_reference_period[time_dim].dt.month
    hours = ds_reference_period[time_dim].dt.hour

    # 2. Create a pandas MultiIndex from the two arrays
    multi_idx = pd.MultiIndex.from_arrays(
        [months.values, hours.values], names=["month", "hour"]
    )

    # 3. Assign this index as a new coordinate matching the 'time' dimension
    ds_stacked = ds_reference_period.assign_coords(
        month_hour=(time_dim, multi_idx)
    )

    # Target variables include demand and all 4 granular generation types
    target_vars = ["total_demand", "gen_pv", "gen_wind_on", "gen_wind_off", "gen_hydro_ror"]

    # 4. Group by the combined coordinate and take the mean, then unstack it back
    # into clean 'month' and 'hour' dimensions
    baseline = (
        ds_stacked[target_vars]
        .groupby("month_hour")
        .mean(dim=time_dim)
        .unstack("month_hour")
    )

    return baseline


def compute_continuous_system_metrics(
    ds_residual_load, ds_components, ds_baseline, zone_dim="zone", time_dim="time"
):
    """Computes continuous time-series metrics (Ramping and PCIT) across the
    entire dataset before slicing into isolated extreme event blocks.
    """
    # Create a shallow copy to store new variables without modifying input data
    ds_metrics = ds_residual_load.copy()

    # --- 1. COMPUTE RAMPING RATES (GW/hour) ---
    # Shift the dataset by 1 time step to get the previous step's residual load
    rl_prev = ds_metrics["residual_load"].shift({time_dim: 1})

    # Delta over 3 hours divided by 3 gives the average hourly rate of change
    ds_metrics["ramping_rate"] = (ds_metrics["residual_load"] - rl_prev) / 3.0

    # Fill the first NaN step with 0 to keep array lengths identical
    ds_metrics["ramping_rate"] = ds_metrics["ramping_rate"].fillna(0.0)

    # --- 2. COMPUTE CONTINUOUS GLOBAL PCIT (GW) ---
    # Sum of absolute imbalances across all zones per time step
    sum_absolute_imbalances = np.abs(ds_metrics["residual_load"]).sum(dim=zone_dim)

    # Absolute value of the net continental imbalance per time step
    net_continental_imbalance = np.abs(ds_metrics["residual_load"].sum(dim=zone_dim))
    ds_metrics["net_continental_imbalance"] = net_continental_imbalance

    # Apply the 1/2 scalar to extract cross-border transport volume
    ds_metrics["pcit"] = 0.5 * (sum_absolute_imbalances - net_continental_imbalance)

    # --- 3. COMPUTE CLIMATE ANOMALIES VIA EXTERNAL BASELINE ---
    # Extract time components from the current target dataset
    current_months = ds_metrics[time_dim].dt.month
    current_hours = ds_metrics[time_dim].dt.hour

    # Vectorized broadcasting: map the external (month, hour) baseline matrix
    # directly onto the full continuous timeline of the target dataset
    # Clip negative values to 0.0 to prevent localized generation or demand
    # surpluses from mathematically masking climate drivers during crisis hours
    expected_d = ds_baseline["total_demand"].sel(month=current_months, hour=current_hours)
    ds_metrics["anomaly_demand"] = (ds_components["total_demand"] - expected_d).clip(min=0.0)

    for tech in ["gen_pv", "gen_wind_on", "gen_wind_off", "gen_hydro_ror"]:
        expected_tech = ds_baseline[tech].sel(month=current_months, hour=current_hours)
        # Positive anomaly means generation fell below the seasonal expectation
        ds_metrics[f"anomaly_{tech}"] = (expected_tech - ds_components[tech]).clip(min=0.0)

    return ds_metrics


def compute_comprehensive_event_metrics(
    ds_with_continuous_metrics, da_thresholds, zone_dim="zone"
):
    """Computes Frequency, Duration, Peak Capacity, and Energy Integral metrics,
    while contextually extracting Ramping and PCIT signatures during threshold violations.
    """
    metrics_summary = {}

    # Extract years once — shared across all zones in this period
    year_series = ds_with_continuous_metrics.time.dt.year.values
    all_years   = np.unique(year_series)

    for zone in ds_with_continuous_metrics[zone_dim].values:
        # Extract the specific threshold for this zone
        H = float(da_thresholds.sel({zone_dim: zone}).values)

        # Pull the continuous numpy arrays for the current zone
        rl_series = (
            ds_with_continuous_metrics["residual_load"].sel({zone_dim: zone}).values
        )
        ramp_series = (
            ds_with_continuous_metrics["ramping_rate"].sel({zone_dim: zone}).values
        )

        # The demand and generation anomalies, crucial for understanding the underlying climate drivers during extreme events
        d_anom_series = (
            ds_with_continuous_metrics["anomaly_demand"].sel({zone_dim: zone}).values
        )
        pv_anom_series = (
            ds_with_continuous_metrics["anomaly_gen_pv"].sel({zone_dim: zone}).values
        )
        w_anom_series = (
            ds_with_continuous_metrics["anomaly_gen_wind_on"].sel({zone_dim: zone}).values
        )
        woff_anom_series = (
            ds_with_continuous_metrics["anomaly_gen_wind_off"].sel({zone_dim: zone}).values
        )
        hydro_anom_series = (
            ds_with_continuous_metrics["anomaly_gen_hydro_ror"].sel({zone_dim: zone}).values
        )

        # PCIT is shared globally across the system, so we don't slice it by zone
        pcit_series = ds_with_continuous_metrics["pcit"].values

        # Number of years in this period (used to normalise total → annual)
        n_years = len(all_years)

        # Identify threshold violations
        is_extreme = rl_series > H

        # Label continuous blocks of threshold violations
        labeled_events, num_features = label(is_extreme)

        if num_features == 0:
            metrics_summary[zone] = {
                "Frequency": 0,
                "Avg_Duration_Hours": 0.0,
                "Max_Duration_Hours": 0.0,
                "Max_Peak_Intensity_GW": 0.0,
                "Max_Upward_Ramp_GW_per_hr": 0.0,
                "Total_Annual_Energy_Severity_GWh": 0.0,
                "Avg_Energy_Severity_GWh": 0.0,
                "Max_Single_Event_Severity_GWh": 0.0,
                "Avg_PCIT_During_Events_GW": 0.0,
                "Max_PCIT_During_Events_GW": 0.0,
                "Std_Annual_Energy_Severity_GWh": 0.0,
                "P90_Annual_Energy_Severity_GWh": 0.0,
                "CV_Annual_Energy_Severity": 0.0,
                "Demand_Driven_Share_%": 0.0,
                "Generation_Driven_Share_%": 0.0,
            }
            continue

        (
            event_durations,
            event_peaks,
            event_integrals,
            event_max_ramps,
            event_mean_pcits,
        ) = [], [], [], [], []
        per_year_severity = {}   # {year: cumulative GWh above threshold}
        tot_d_anom = 0.0
        tot_pv_anom = 0.0
        tot_w_anom = 0.0
        tot_woff_anom = 0.0
        tot_hydro_anom = 0.0

        # Process each isolated event window
        for i in range(1, num_features + 1):
            event_indices = np.where(labeled_events == i)[0]

            # 1. Duration (1 step = 3 hours)
            duration_hours = len(event_indices) * 3
            event_durations.append(duration_hours)

            # 2. Peak Intensity (GW)
            event_rl_values = rl_series[event_indices]
            peak_gw = np.max(event_rl_values)
            event_peaks.append(peak_gw)

            # 3. Integral Energy Severity (GWh) with 3-hour Riemann adjustment
            integral_gwh = np.sum(event_rl_values - H) * 3
            event_integrals.append(integral_gwh)
            # Tag this event's energy to the calendar year it started in
            event_year = year_series[event_indices[0]]
            per_year_severity[event_year] = (
                per_year_severity.get(event_year, 0.0) + integral_gwh
            )

            # 4. Contextual Ramping (GW/hour)
            # Find the steepest upward ramp rate that occurred within this event block
            event_ramp_values = ramp_series[event_indices]
            max_ramp = np.max(event_ramp_values)
            event_max_ramps.append(max_ramp)

            # 5. Contextual PCIT (GW)
            # Find the average cross-border congestion volume during this specific event block
            event_pcit_values = pcit_series[event_indices]
            mean_pcit = np.mean(event_pcit_values)
            event_mean_pcits.append(mean_pcit)

            # 6. Climate Driver Attribution
            # Sum the demand and generation anomalies during this event block to understand the underlying climate drivers
            # Accumulate the absolute GW deficits across all crisis hours for this zone
            tot_d_anom += np.sum(d_anom_series[event_indices])
            tot_pv_anom += np.sum(pv_anom_series[event_indices])
            tot_w_anom += np.sum(w_anom_series[event_indices])
            tot_woff_anom += np.sum(woff_anom_series[event_indices])
            tot_hydro_anom += np.sum(hydro_anom_series[event_indices])

        # ── Annual severity statistics (year-to-year variability) ─────────────
        # Years with no events contribute 0 GWh to the distribution.
        annual_vals = np.array([per_year_severity.get(yr, 0.0) for yr in all_years])
        std_annual  = float(np.std(annual_vals))
        p90_annual  = float(np.percentile(annual_vals, 90))
        mean_annual = float(np.sum(event_integrals)) / n_years  # consistent with Total_Annual
        cv_annual   = std_annual / mean_annual if mean_annual > 0 else 0.0

        # Compute climate driver percentages over the evaluated period
        # Total anomaly mass
        denominator = (
            tot_d_anom + tot_pv_anom + tot_w_anom + tot_woff_anom + tot_hydro_anom
        )

        if denominator > 0:
            d_share = (tot_d_anom / denominator) * 100
            pv_share = (tot_pv_anom / denominator) * 100
            w_share = (tot_w_anom / denominator) * 100
            woff_share = (tot_woff_anom / denominator) * 100
            hydro_share = (tot_hydro_anom / denominator) * 100
        else:
            d_share, pv_share, w_share, woff_share, hydro_share = (
                20.0,
                20.0,
                20.0,
                20.0,
                20.0,
            )

        # Compile the final dictionary summarizing the operational profile of the zone
        metrics_summary[zone] = {
            # --- Event Frequency & Longevity ---
            "Frequency": int(num_features),
            "Avg_Duration_Hours": float(np.mean(event_durations)),
            "Max_Duration_Hours": float(
                np.max(event_durations)
            ),  # <-- CRITICAL CAPABILITY FIXED
            # --- Capacity & Ramping Intensity (Power Constraints) ---
            "Max_Peak_Intensity_GW": float(np.max(event_peaks)),
            "Max_Upward_Ramp_GW_per_hr": float(np.max(event_max_ramps)),
            # --- Energy Volume Severity (Storage Constraints) ---
            "Total_Annual_Energy_Severity_GWh": float(np.sum(event_integrals)) / n_years,
            "Avg_Energy_Severity_GWh": float(np.mean(event_integrals)),
            "Max_Single_Event_Severity_GWh": float(
                np.max(event_integrals)
            ),  # <-- ISOLATED CRISIS BOUND
            # --- Grid Interconnection Support ---
            "Avg_PCIT_During_Events_GW": float(np.mean(event_mean_pcits)),
            "Max_PCIT_During_Events_GW": float(
                np.max(event_mean_pcits)
            ),  # <-- WORST-CASE INTERCONNECTOR STRAIN
            # --- Annual Severity Statistics (year-to-year variability) ---
            "Std_Annual_Energy_Severity_GWh": std_annual,
            "P90_Annual_Energy_Severity_GWh": p90_annual,
            "CV_Annual_Energy_Severity": cv_annual,
            # --- Climate Driver Attribution ---
            "Demand_Driven_Share_%": round(float(d_share), 1),
            "PV_Driven_Share_%": round(float(pv_share), 1),
            "Wind_Onshore_Driven_Share_%": round(float(w_share), 1),
            "Wind_Offshore_Driven_Share_%": round(float(woff_share), 1),
            "Hydro_RoR_Driven_Share_%": round(float(hydro_share), 1),
        }

    return metrics_summary


def compile_and_compare_periods(ref_dict, mid_dict, end_dict):
    """Compiles metrics from three eras into a structured DataFrame and computes

    the quantitative shifts from the reference baseline.
    """
    records = []

    # Map dictionaries to their respective period names
    periods = {
        "1_Reference": ref_dict,
        "2_Mid-Century": mid_dict,
        "3_End-Century": end_dict,
    }

    for period_name, period_data in periods.items():
        for zone, metrics in period_data.items():
            row = {"Period": period_name, "Zone": zone}
            row.update(metrics)
            records.append(row)

    # Create base DataFrame
    df = pd.DataFrame(records)

    # Pivot to make it easy to compute differences across periods
    # Columns will be Metrics, Rows will be Zone and Period
    df_pivot = df.set_index(["Zone", "Period"]).sort_index()

    # Create a comparative summary showing delta shifts for key metrics
    summary_metrics = [
        "Frequency",
        "Avg_Duration_Hours",
        "Max_Duration_Hours",
        "Total_Annual_Energy_Severity_GWh",
        "Avg_Energy_Severity_GWh",
        "Max_Single_Event_Severity_GWh",
        "Avg_PCIT_During_Events_GW",
        "Max_PCIT_During_Events_GW",
        "Std_Annual_Energy_Severity_GWh",
        "P90_Annual_Energy_Severity_GWh",
        "CV_Annual_Energy_Severity",
        "Demand_Driven_Share_%",
        "PV_Driven_Share_%",
        "Wind_Onshore_Driven_Share_%",
        "Wind_Offshore_Driven_Share_%",
        "Hydro_RoR_Driven_Share_%",
    ]

    deltas = {}
    for zone in df["Zone"].unique():
        deltas[zone] = {}
        for metric in summary_metrics:
            ref_val = df_pivot.loc[(zone, "1_Reference"), metric]
            mid_val = df_pivot.loc[(zone, "2_Mid-Century"), metric]
            end_val = df_pivot.loc[(zone, "3_End-Century"), metric]

            # Compute percentage changes (handle division by zero safely)
            mid_pct = ((mid_val - ref_val) / ref_val * 100) if ref_val > 0 else 0
            end_pct = ((end_val - ref_val) / ref_val * 100) if ref_val > 0 else 0

            deltas[zone][f"{metric}_Mid_Change_%"] = round(mid_pct, 1)
            deltas[zone][f"{metric}_End_Change_%"] = round(end_pct, 1)

    df_deltas = pd.DataFrame.from_dict(deltas, orient="index")
    df_deltas.index.name = "Zone"

    return df_pivot, df_deltas
