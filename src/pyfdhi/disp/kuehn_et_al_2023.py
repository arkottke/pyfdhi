import numpy as np
from pathlib import Path

from pyfdhi.disp.model import DispModel, FloatParameter, StringParameter
from typing import Dict

from scipy import stats

import numpy.typing as npt
import pandas as pd


class KuehnEtAl2023(DispModel):
    PARAMS = [
        FloatParameter("magnitude"),
        FloatParameter("percentile"),
        StringParameter("style", ["strike-slip", "normal", "reverse"]),
    ]

    POSTERIOR = None

    MAG_BREAK, DELTA = 7.0, 0.1

    @classmethod
    def _calc_disp_avg(
        cls,
        *,
        magnitude: npt.ArrayLike,
        style: npt.ArrayLike,
        complete=False,
        **kwds,
    ) -> pd.DataFrame:
        """
        Run KEA23 displacement model to calculate the average displacement that is implied by the model
        prediction. All parameters must be passed as keyword arguments. The mean model (i.e., mean
        coefficients) is used. Any number of scenarios (i.e., magnitudes, styles) are allowed.

        Parameters
        ----------
        magnitude : Union[float, list, numpy.ndarray]
            Earthquake moment magnitude.

        style : Union[str, list, numpy.ndarray]
            Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
            'normal'.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the following columns:
            - 'magnitude': Earthquake moment magnitude [from user input].
            - 'style': Style of faulting [from user input].
            - 'model_number': Model coefficient row number. Returns -1 for mean model.
            - 'avg_displ': Average displacement in meters.

        Raises (inherited from `run_displacement_model.py`)
        ------
        ValueError
            If the provided `style` is not one of the supported styles.

        Notes
        ------
        Command-line interface usage
            Run (e.g.) `python run_average_displacement.py --magnitude 5 6 7 --style reverse normal`
            Run `python run_average_displacement.py --help`

        #TODO
        ------
        Raise a UserWarning for magntiudes outside recommended ranges.
        """

        # Calculate mean slip profile
        # NOTE: `percentile=-1` is used for mean
        # NOTE: `location_step=0.01` is used to create well-descritized profile for intergration
        results = cls.calc_profile(
            magnitude=magnitude, style=style, percentile=-1, location_step=0.01
        )

        # Group by magnitude and style
        def calc_avg(profile):
            return np.trapz(profile["displ_site"], profile["location"])

        results = (
            results.groupby(["magnitude", "model_number", "style"]).apply(calc_avg).reset_index()
        )

        # # Calculate area under the mean slip profile; this is the Average Displacement (AD)
        # # NOTE: use dictionary comprehension, it is probably slightly faster than apply lambda
        # areas = {
        #     (mag, model, style): np.trapz(group["displ_site"], group["location"])
        #     for (mag, model, style), group in grouped
        # }

        # # Create output dataframe
        # magnitudes, model_numbers, styles, area_values = zip(
        #     *[(mag, model, style, area) for (mag, model, style), area in areas.items()]
        # )

        # values = (
        #     list(magnitudes),
        #     list(styles),
        #     list(model_numbers),
        #     list(area_values),
        # )

        # type_dict = {
        #     "magnitude": float,
        #     "style": str,
        #     "model_number": str,
        #     "avg_displ": float,
        # }
        # dataframe = pd.DataFrame(np.column_stack(values), columns=type_dict.keys())
        # dataframe = dataframe.astype(type_dict)

        return results

    @classmethod
    def calc_profile(
        cls,
        # FIXME: Maybe defining a specific Type for this would be cleaner
        # magnitude: Union[float, int, npt.ArrayLike],
        magnitude: npt.ArrayLike,
        style: npt.ArrayLike,
        percentile: npt.ArrayLike,
        location_step: float = 0.05,
    ) -> pd.DataFrame:
        """
        Run KEA23 displacement model to create slip profile. All parameters must be passed as keyword
        arguments. The mean model (i.e., mean coefficients) is used. Any number of scenarios (i.e.,
        magnitudes, styles, percentiles) are allowed.

        Parameters
        ----------
        magnitude : Union[float, list, numpy.ndarray]
            Earthquake moment magnitude.

        style : Union[str, list, numpy.ndarray]
            Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
            'normal'.

        percentile : Union[float, list, numpy.ndarray]
            Aleatory quantile value. Use -1 for mean.

        location_step : float, optional
            Profile step interval in percentage. Default 0.05.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the following columns:
            - 'magnitude': Earthquake moment magnitude [from user input].
            - 'location':  Normalized location along rupture length [generated from location_step].
            - 'style': Style of faulting [from user input].
            - 'percentile': Percentile value [from user input].
            - 'model_number': Model coefficient row number. Returns -1 for mean model.
            - 'lambda': Box-Cox transformation parameter.
            - 'mu_left': Mean transformed displacement for the left-peak profile.
            - 'sigma_left': Standard deviation transformed displacement for the left-peak profile.
            - 'mu_right': Mean transformed displacement for the right-peak profile.
            - 'sigma_right': Standard deviation transformed displacement for the right-peak profile.
            - 'Y_left': Transformed displacement for the left-peak profile.
            - 'Y_right': Transformed displacement for the right-peak profile.
            - 'Y_folded': Transformed displacement for the folded (symmetrical) profile.
            - 'displ_left': Displacement in meters for the left-peak profile.
            - 'displ_right': Displacement in meters for the right-peak profile.
            - 'displ_folded': Displacement in meters for the folded (symmetrical) profile.

        Raises (inherited from `run_displacement_model.py`)
        ------
        ValueError
            If the provided `style` is not one of the supported styles.

        Notes
        ------
        Command-line interface usage
            Run (e.g.) `python run_displacement_profile.py --magnitude 6 7 --style strike-slip --percentile 0.5 -step 0.01`
            Run `python run_displacement_profile.py --help`

        #TODO
        ------
        Raise a ValueError for invalid location_step size.
        Raise a ValueError for invalid percentile.
        Raise a UserWarning for magntiudes outside recommended ranges.
        """

        # NOTE: Check for appropriate style is handled in `run_model`

        # Create profile location array
        locations = np.arange(0, 1 + location_step, location_step)

        dataframe = cls._calc_disp_profile(
            magnitude=magnitude,
            location=locations,
            style=style,
            percentile=percentile,
            mean_model=True,
        )

        return dataframe.sort_values(
            by=["magnitude", "style", "percentile", "location"]
        ).reset_index(drop=True)

    @classmethod
    def _calculate_Y(cls, mu, sigma, lam, percentile):
        """
        A vectorized helper function to calculate predicted displacement in transformed units.

        Parameters
        ----------
        mu : np.array
            Mean prediction in transformed units.

        sigma : np.array
            Total standard deviation in transformed units.

        lam : np.array
            "Lambda" transformation parameter in Box-Cox transformation.

        percentile : np.array
            Aleatory quantile value. Use -1 for mean.

        Returns
        -------
        Y : np.array
            Predicted displacement in transformed units.
        """
        if np.any(percentile == -1):
            # Compute the mean
            # NOTE: Analytical solution from https://robjhyndman.com/hyndsight/backtransforming/
            D_mean = (np.power(lam * mu + 1, 1 / lam)) * (
                1 + (np.power(sigma, 2) * (1 - lam)) / (2 * np.power(lam * mu + 1, 2))
            )
            # NOTE: Analytical soluion is in meters, so convert back to Y transform for consistency
            Y_mean = (np.power(D_mean, lam) - 1) / lam
        else:
            Y_mean = np.nan

        # Compute the aleatory quantile
        Y_normal = stats.norm.ppf(percentile, loc=mu, scale=sigma)

        # Use np.where for vectorization
        Y = np.where(percentile == -1, Y_mean, Y_normal)

        return Y

    @classmethod
    def _calculate_displacement(cls, predicted_Y, lam):
        """
        A vectorized helper function to calculate predicted displacement in meters.

        Parameters
        ----------
        predicted_Y : np.array
            Predicted displacement in transformed units.

        lam : np.array
            "Lambda" transformation parameter in Box-Cox transformation.

        Returns
        -------
        D : np.array
            Predicted displacement in meters.
        """

        D = np.power(predicted_Y * lam + 1, 1 / lam)

        # Handle values that are too small to calculate
        D = np.where(np.isnan(D), 0, D)

        return D

    # FIXME: Made this into a private class all of the preparation of the vecotrs should
    # be handled separately
    @classmethod
    def _calc_disp_profile(
        cls,
        magnitude: npt.ArrayLike,
        location: npt.ArrayLike,
        style: npt.ArrayLike,
        percentile: npt.ArrayLike,
        mean_model: bool = True,
    ) -> pd.DataFrame:
        """
        Run KEA23 displacement model. All parameters must be passed as keyword arguments.
        A couple "gotchas":
            If full model is run (i.e., `mean_model=False`), then only one scenario is allowed.
            If mean model is run (i.e., `mean_model=True` or default), then any number of scenarios is allowed.
            A scenario is defined as a magnitude-location-style-percentile combination.

        Parameters
        ----------
        magnitude : Union[float, list, numpy.ndarray]
            Earthquake moment magnitude.

        location : Union[float, list, numpy.ndarray]
            Normalized location along rupture length, range [0, 1.0].

        style : Union[str, list, numpy.ndarray]
            Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
            'normal'.

        percentile : Union[float, list, numpy.ndarray]
            Aleatory quantile value. Use -1 for mean.

        mean_model : bool, optional
            If True, use mean coefficients. If False, use full coefficients. Default True.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the following columns:
            - 'magnitude': Earthquake moment magnitude [from user input].
            - 'location':  Normalized location along rupture length [from user input].
            - 'style': Style of faulting [from user input].
            - 'percentile': Aleatory quantile value [from user input].
            - 'model_number': Model coefficient row number. Returns -1 for mean model.
            - 'lambda': Box-Cox transformation parameter.
            - 'mu_site': Mean transformed displacement for the location.
            - 'sigma_site': Standard deviation transformed displacement for the location.
            - 'mu_complement': Mean transformed displacement for the complementary location.
            - 'sigma_complement': Standard deviation transformed displacement for the complementary location.
            - 'Y_site': Transformed displacement for the location.
            - 'Y_complement': Transformed displacement for the complementary location.
            - 'Y_folded': Transformed displacement for the folded location.
            - 'displ_site': Displacement in meters for the location.
            - 'displ_complement': Displacement in meters for the complementary location.
            - 'displ_folded': Displacement in meters for the folded location.

        Raises
        ------
        ValueError
            If the provided `style` is not one of the supported styles.

        TypeError
            If more than one value is provided for `magnitude`, `location`, `style`, or `percentile` when `mean_model=False`.

        Notes
        ------
        Command-line interface usage
            Run (e.g.) `python run_displacement_model.py --magnitude 7 --location 0.5 --style strike-slip --percentile 0.5 0.84`
            Run `python run_displacement_model.py --help`

        #TODO
        ------
        Raise a ValueError for invalid location
        Raise a ValueError for invalid percentile.
        Raise a UserWarning for magntiudes outside recommended ranges.
        """

        # Get distribution parameters for site and complement
        mu_site, sigma_site, lam, model_number = cls._calculate_distribution_parameters(
            magnitude=magnitude, location=location, style=style, mean_model=mean_model
        )
        mu_complement, sigma_complement, _, _ = cls._calculate_distribution_parameters(
            magnitude=magnitude,
            location=1 - location,
            style=style,
            mean_model=mean_model,
        )

        # Calculate Y (transformed displacement)
        Y_site = cls._calculate_Y(mu=mu_site, sigma=sigma_site, lam=lam, percentile=percentile)
        Y_complement = cls._calculate_Y(
            mu=mu_complement, sigma=sigma_complement, lam=lam, percentile=percentile
        )
        Y_folded = np.mean([Y_site, Y_complement], axis=0)

        # Calculate displacement in meters
        displ_site = cls._calculate_displacement(predicted_Y=Y_site, lam=lam)
        displ_complement = cls._calculate_displacement(predicted_Y=Y_complement, lam=lam)
        displ_folded = cls._calculate_displacement(predicted_Y=Y_folded, lam=lam)

        # Create a DataFrame
        n = max(len(magnitude), len(mu_site))
        df = pd.DataFrame(
            {
                "magnitude": np.full(n, magnitude),
                "location": np.full(n, location),
                "style": np.full(n, style),
                "percentile": np.full(n, percentile),
                "model_number": np.full(n, model_number).astype(int),
                "mu_site": mu_site,
                "sigma_site": sigma_site,
                "mu_complement": mu_complement,
                "sigma_complement": sigma_complement,
                "Y_site": Y_site,
                "Y_complement": Y_complement,
                "Y_folded": Y_folded,
                "displ_site": displ_site,
                "displ_complement": displ_complement,
                "displ_folded": displ_folded,
            }
        )
        # FIXME: Alternative way
        # .astype({"model_number": int})
        return df

    @classmethod
    def _load_posterior(cls) -> Dict[str, Dict[str, pd.DataFrame]]:
        # FIXME: this was previously split across 80 lines of code and none of the funcions were reusable

        # Filepath for model coefficients
        dir_data = Path(__file__).parent / "data" / "KuehnEtAl2023"

        # Filenames for model coefficients
        filenames = {
            "strike-slip": "coefficients_posterior_SS_powtr.csv",
            "reverse": "coefficients_posterior_REV_powtr.csv",
            "normal": "coefficients_posterior_NM_powtr.csv",
        }

        def load(fname) -> Dict[str, pd.DataFrame]:
            samples = pd.read_csv(dir_data / fname).rename(columns={"Unnamed: 0": "model_number"})

            mean = samples.mean(axis=0).to_frame().transpose()
            mean.loc[0, "model_number"] = -1  # Define model id as -1 for mean coeffs
            mean["model_number"] = mean["model_number"].astype(int)
            return {"mean": mean, "samples": samples}

        return {key: load(fname) for key, fname in filenames.items()}

    @classmethod
    def _calculate_distribution_parameters(cls, magnitude, location, style, mean_model):
        """
        A vectorized helper function to calculate predicted mean and standard deviation in transformed
        units and the Box-Cox transformation parameter.

        Parameters
        ----------
        magnitude : np.array
            Earthquake moment magnitude.

        location : np.array
            Normalized location along rupture length, range [0, 1.0].

        style : np.array
            Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
            'normal'.

        mean_model : bool
            If True, use mean coefficients. If False, use full coefficients.

        Returns
        -------
        Tuple[np.array, np.array, np.array, np.array]
            mu : Mean prediction in transformed units.
            sigma : Total standard deviation in transformed units.
            lam : Box-Cox transformation parameter.
            model_num : Model coefficient row number. Returns -1 for mean model.
        """
        if cls.POSTERIOR is None:
            cls.POSTERIOR = cls._load_posterior()

        if mean_model:
            # Calculate for all submodels
            # NOTE: it is actually faster to just do this instead of if/else, loops, etc.

            # Define coefficients (loaded with module imports)
            # NOTE: Coefficients are pandas dataframes; convert here to recarray for faster computations
            # NOTE: Check for appropriate style is handled in `run_model`

            styles = list(cls.POSTERIOR)

            coeffs = {
                style: cls.POSTERIOR[style]["mean"].to_records(index=False) for style in styles
            }

            # Conditions for np.select
            conditions = [style == s for s in styles]

            # Choices for mu and sigma
            choices_mu = [cls._calc_mu(coeffs[style], magnitude, location) for style in styles]
            choices_sd = [
                cls._calc_sd(coeffs[style], magnitude, location, style) for style in styles
            ]

            # Use np.select to get the final mu, sigma, and lambda
            mu = np.select(conditions, choices_mu, default=np.nan)
            sd = np.select(conditions, choices_sd, default=np.nan)

            lam = np.select(conditions, [coeffs[s]["lambda"] for s in styles], default=np.nan)
            model_num = np.select(
                conditions, [coeffs[s]["model_number"] for s in styles], default=np.nan
            )
        else:
            coeffs = cls.POSTERIOR[style]["samples"]

            mu = cls._calc_mu(coeffs, magnitude, location)
            sd = cls._calc_sd(coeffs, magnitude, location, style)

            lam = coeffs["lambda"]
            model_num = coeffs["model_number"]

        return mu, sd, lam, model_num

    @classmethod
    def _calc_mu(cls, coef, mag, loc):
        # Model constants

        fm = (
            coef["c1"]
            + coef["c2"] * (mag - cls.MAG_BREAK)
            + (coef["c3"] - coef["c2"])
            * cls.DELTA
            * np.log(1 + np.exp((mag - cls.MAG_BREAK) / cls.DELTA))
        )

        a = fm - coef["gamma"] * np.power(
            coef["alpha"] / (coef["alpha"] + coef["beta"]), coef["alpha"]
        ) * np.power(coef["beta"] / (coef["alpha"] + coef["beta"]), coef["beta"])

        # FIXME: Had np.asarray(...), why?
        return a + coef["gamma"] * np.power(loc, coef["alpha"]) * np.power(1 - loc, coef["beta"])

    @classmethod
    def _calc_sd(cls, coef, mag, loc, style):
        if style == "strike-slip":
            sd_loc = coef["s_s1"] + coef["s_s2"] * np.power(
                loc - coef["alpha"] / (coef["alpha"] + coef["beta"]), 2
            )
            # Bilinear model
            sd_mode = (
                coef["s_m,s1"]
                + coef["s_m,s2"] * (mag - coef["s_m,s3"])
                - coef["s_m,s2"]
                * cls.DELTA
                * np.log(1 + np.exp((mag - coef["s_m,s3"]) / cls.DELTA))
            )
        elif style == "normal":
            sd_loc = coef["sigma"]
            # Sigmoidal
            sd_mode = coef["s_m,n1"] - coef["s_m,n2"] / (
                1 + np.exp(-1 * coef["s_m,n3"] * (mag - cls.MAG_BREAK))
            )
        elif style == "reverse":
            sd_loc = coef["s_r1"] + coef["s_r2"] * np.power(
                loc - coef["alpha"] / (coef["alpha"] + coef["beta"]), 2
            )
            sd_mode = coef["s_m,r"]
        else:
            raise NotImplementedError

        return np.sqrt(sd_mode**2 + sd_loc**2)
