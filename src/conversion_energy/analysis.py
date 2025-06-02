from typing import Tuple
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import seaborn as sns

import statsmodels.api as sm

sns.set_style("whitegrid", {"grid.color": ".93"})


def compute_trend_statsmodel(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05,
        round_results: bool = True,
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Compute linear trend using statsmodel's OLS class. This method
        returns the coefficients alongside their confidence interval.

        Parameters
        ----------
        X : np.ndarray
            X array to be passed to the Ordinary Least Squares method
        y : np.ndarray
            Array of target values to compute a linear trend on
        alpha : float, optional
            The alpha level for the confidence interval, by default 0.05
        round_results : bool, optional
            Whether to round the coefficients, by default True

        Returns
        -------
        Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]
            The coefficient and the intercept of the linear regression, along
            with the predicted values of the linear regression, the
            confidence interval for the coefficients and the residuals of the regression
        """
        lr = sm.OLS(y, sm.add_constant(X)).fit()
        conf_interval = lr.conf_int(alpha)
        pred = lr.predict(sm.add_constant(X))
        coef = lr.params[1]
        intercept = lr.params[0]
        residuals = lr.resid
        p_value = lr.pvalues[1]
        if round_results:
            coef = round(coef, 3)
            intercept = round(intercept, 3)

        return coef, intercept, pred, conf_interval, residuals, p_value


def analyze_trend(ds, selected_var):
    """
    Analyze the trend of yearly means and plot the results.

    Parameters:
    - ds: xarray.Dataset containing yearly mean data
    - selected_var: str, the variable to analyze

    Returns:
    - trend: float, the slope of the trend
    """
    # Extract time and values
    time = ds.time.values
    values = ds[selected_var].values.reshape(-1)

    trend, _, pred, conf_interval, _, _ = compute_trend_statsmodel(
        time, values, alpha=0.05, round_results=True
    )
    lower_bound, upper_bound = conf_interval[1]

    # Generate predictions and confidence intervals
    X = sm.add_constant(time)
    ci_lower = pred + (lower_bound - trend) * (X[:, 1] - X[:, 1].mean())
    ci_upper = pred + (upper_bound - trend) * (X[:, 1] - X[:, 1].mean())

    # Print the results
    print(f"Trend: {trend:.4f} per year")
    print(f"95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")

    # Plot the yearly means, regression line, and confidence interval
    plt.figure(figsize=(10, 6))
    plt.plot(time, values, label="Yearly Mean", marker="o")
    plt.plot(time, pred, label="Trend Line", color="red")
    plt.fill_between(time, ci_lower, ci_upper, color="red", alpha=0.2, label="95% CI")
    plt.xlabel("Year")
    plt.ylabel("Yearly Mean of Daily Sums (Wh)")
    plt.title(f"Yearly Means of Daily Sums for {selected_var}")
    plt.grid()
    plt.legend()
    plt.show()

    return trend, (lower_bound, upper_bound)

