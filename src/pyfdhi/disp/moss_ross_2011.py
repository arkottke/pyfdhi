# This file doesn't have to go here, but I was trying to find a place and name for it separate from the other structure

import numpy as np

from pyfdhi.disp.model import DispModel, FloatParameter, StringParameter

from scipy import stats

import numpy.typing as npt
import pandas as pd


class MossRoss2011(DispModel):
    PARAMS = [
        FloatParameter("magnitude"),
        FloatParameter("percentile"),
        StringParameter("style", ["strike-slip", "normal", "reverse"]),
    ]

    @classmethod
    def _calc_disp_avg(
        cls,
        *,
        magnitude: npt.ArrayLike,
        percentile: npt.ArrayLike,
        style: npt.ArrayLike,
        complete=False,
        **kwds,
    ) -> pd.DataFrame:
        # Calculate distribution parameters
        mu, sigma = cls._calc_distrib_params_mag_ad(magnitude=magnitude)

        # Calculate natural log of displacement (vectorized approach)
        if np.any(percentile == -1):
            # Compute the mean
            log10_displ_mean = mu + (np.log(10) / 2 * np.power(sigma, 2))
        else:
            log10_displ_mean = np.nan

        # Compute the aleatory quantile
        log10_displ_normal = stats.norm.ppf(percentile, loc=mu, scale=sigma)

        # Use np.where for vectorization
        log10_displ = np.where(percentile == -1, log10_displ_mean, log10_displ_normal)

        # Calculate displacement
        avg_displ = np.power(10, log10_displ)

        d = {
            "magnitude": magnitude,
            "style": style,
            "percentile": percentile,
            "avg_displ": avg_displ,
        }

        if complete:
            d |= {
                "mu": mu,
                "sigma": sigma,
            }

        return pd.DataFrame(d)

    # FIXME: You defined the function as (*, magnitude) to force a keyword. If you only
    # have one argument, I feel this is a little unwarranted -- especially for private
    # functions
    @classmethod
    def _calc_distrib_params_mag_ad(cls, magnitude):
        """
        Calculate mu and sigma for the AD=f(M) relation in Moss & Ross (2011) Eqn 8.

        Parameters
        ----------
        magnitude : Union[float, np.ndarray]
            Earthquake moment magnitude.

        Returns
        -------
        Tuple[np.array, np.array]
            mu : Mean prediction in log10 units.
            sigma : Standard deviation in log10 units.

        Notes
        ------
        Mu and sigma are in log10 units
        """

        magnitude = np.atleast_1d(magnitude)

        a, b, sigma = -2.2192, 0.3244, 0.17
        mu = a + b * magnitude

        return mu, np.full(len(mu), sigma)

    @classmethod
    def _calc_distrib_params_mag_md(cls, magnitude):
        """
        Calculate mu and sigma for the MD=f(M) relation in Moss & Ross (2011) Eqn 9.

        Parameters
        ----------
        magnitude : Union[float, np.ndarray]
            Earthquake moment magnitude.

        Returns
        -------
        Tuple[np.array, np.array]
            mu : Mean prediction in log10 units.
            sigma : Standard deviation in log10 units.

        Notes
        ------
        Mu and sigma are in log10 units
        """
        magnitude = np.atleast_1d(magnitude)

        a, b, sigma = -3.1971, 0.5102, 0.31
        mu = a + b * magnitude

        return mu, np.full(len(mu), sigma)

    @classmethod
    def _calc_distrib_params_d_ad(cls, location):
        """
        Calculate alpha and beta per Eqn 7 based on location.

        Parameters
        ----------
        location : Union[float, np.ndarray]
            Normalized location along rupture length, range [0, 1.0].

        Returns
        -------
        Tuple[float, float]
            alpha : Shape parameter for Gamma distribution.
            beta : Scale parameter for Gamma distribution.

        """

        location = np.atleast_1d(location)

        folded_location = np.minimum(location, 1 - location)

        alpha = np.exp(
            -30.4 * folded_location**3 + 19.9 * folded_location**2 - 2.29 * folded_location + 0.574
        )
        beta = np.exp(
            50.3 * folded_location**3 - 34.6 * folded_location**2 + 6.6 * folded_location - 1.05
        )

        return alpha, beta

    @classmethod
    def _calc_distrib_params_d_md(cls, location):
        """
        Calculate alpha and beta per Eqn "7.5" based on location. (Eqn # number is missing in
        manuscript, but it is the MD formulation between Eqns 7 and 8 on page 1547.)

        Parameters
        ----------
        location : Union[float, np.ndarray]
            Normalized location along rupture length, range [0, 1.0].

        Returns
        -------
        Tuple[float, float]
            alpha : Shape parameter for Beta distribution.
            beta : Shape parameter for Beta distribution.

        """

        location = np.atleast_1d(location)

        folded_location = np.minimum(location, 1 - location)

        a1, a2 = 0.901, 0.713
        b1, b2 = -1.86, 1.74

        alpha = a1 * folded_location + a2
        beta = b1 * folded_location + b2

        return alpha, beta
