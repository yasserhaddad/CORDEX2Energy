from typing import Tuple

import numpy as np
import statsmodels.api as sm
import xarray as xr
from xclim import indicators


def compute_trend_statsmodel(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    round_results: bool = True,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute the linear trend using statsmodels' OLS class. This method
    returns the regression coefficients along with their confidence intervals.

    Parameters
    ----------
    X : np.ndarray
        The independent variable(s) for the Ordinary Least Squares regression.
    y : np.ndarray
        The dependent variable for the regression.
    alpha : float, optional
        The significance level for the confidence interval, by default 0.05.
    round_results : bool, optional
        Whether to round the coefficients and intercept to three decimal places, by default True.

    Returns
    -------
    Tuple[float, float, np.ndarray, np.ndarray]
        A tuple containing:
        - The slope (coefficient) of the linear regression.
        - The intercept of the linear regression.
        - The predicted values from the linear regression.
        - The confidence interval for the coefficients.
    """
    lr = sm.OLS(y, sm.add_constant(X)).fit()
    conf_interval = lr.conf_int(alpha)
    pred = lr.predict(sm.add_constant(X))
    coef = lr.params[1]
    intercept = lr.params[0]
    if round_results:
        coef = round(coef, 3)
        intercept = round(intercept, 3)

    return coef, intercept, pred, conf_interval