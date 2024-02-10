from itertools import repeat

import matplotlib
from astropy.table import Table, Column, vstack

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
from scipy.constants import c

from pcigale.data import SimpleDatabase as Database
from pcigale.utils.console import ERROR, INFO, WARNING, console
from pcigale.utils.counter import Counter
from pcigale.utils.io import read_table
from pcigale_plots.plot_types import Plotter

# Name of the file containing the best models information
BEST_RESULTS = "results.fits"
MOCK_RESULTS = "results_mock.fits"
OBSERVATIONS = "observations.fits"

AVAILABLE_SERIES = [
    "stellar_attenuated",
    "stellar_unattenuated",
    "nebular",
    "dust",
    "agn",
    "radio",
    "model",
]


class SED(Plotter):
    def __init__(
        self, config, sed_type, nologo, xrange, yrange, series, format, outdir
    ):
        """Plot the best SED with associated observed and modelled fluxes."""
        self.configuration = config.config
        if self.configuration["data_file"]:
            obs = Table.read(outdir / OBSERVATIONS)
            mod = Table.read(outdir / BEST_RESULTS)

            # Replace masked values by NaN to suppress warnings
            obs["id"] = Column(obs["id"])
            obs = obs.filled(np.nan)
            mod = mod.filled(np.nan)
            observed = True
        else:
            obs = repeat([])
            modfiles = outdir.glob("models-block-*.fits")
            mod = vstack([Table.read(modfile) for modfile in modfiles])
            observed = False

        with Database("filters") as db:
            filters = {
                name: db.get(name=name)
                for name in self.configuration["bands"]
                if not (name.endswith("_err") or name.startswith("line"))
            }

        if nologo:
            logo = False
        else:
            logo = plt.imread(
                pkg_resources.resource_filename(
                    __name__, "../../resources/CIGALE.png"
                )
            )

        counter = Counter(len(mod), 1, "Object")
        items = zip(
            obs,
            mod,
            repeat(filters),
            repeat(sed_type),
            repeat(logo),
            repeat(xrange),
            repeat(yrange),
            repeat(series),
            repeat(format),
            repeat(outdir),
            repeat(observed),
        )
        self._parallel_job(items, counter)

        # Print the final value as it may not otherwise be printed
        counter.global_counter.value = len(mod)
        counter.progress.join()
        console.print(f"{INFO} Done.")

    @staticmethod
    def initializer(counter):
        """Initializer of the pool of processes to share variables between workers.
        Parameters
        ----------
        :param counter: Counter class object for the number of models plotted
        """
        global gbl_counter

        gbl_counter = counter

    @staticmethod
    def worker(
        obs,
        mod,
        filters,
        sed_type,
        logo,
        xrange,
        yrange,
        series,
        format,
        outdir,
        observed,
    ):
        """Plot the best SED with the associated fluxes in bands

        Parameters
        ----------
        obs: Table row
            Data from the input file regarding one object.
        mod: Table row
            Data from the best model of one object.
        filters: dict
            The observed fluxes in each filter.
        sed_type: string
            Type of SED to plot. It can either be "mJy" (flux in mJy and observed
            frame) or "lum" (luminosity in W and rest frame)
        logo: numpy.array | boolean
            The readed logo image data. Has shape
            (M, N) for grayscale images.
            (M, N, 3) for RGB images.
            (M, N, 4) for RGBA images.
            Do not add the logo when set to False.
        xrange: tuple(float|boolean, float|boolean)
        yrange: tuple(float|boolean, float|boolean)
        series: list
        format: string
            One of png, pdf, ps, eps or svg.
        outdir: Path
            Path to outdir
        observed: Boolean
            True is the model is fitted to observations

        """
        np.seterr(invalid="ignore")
        gbl_counter.inc()

        id_best_model_file = outdir / f"{mod['id']}_best_model.fits"
        if id_best_model_file.is_file():
            sed = Table.read(id_best_model_file)

            filters_wl = (
                np.array([filt.pivot for filt in filters.values()]) * 1e-3
            )
            wavelength_spec = sed["wavelength"] * 1e-3
            if observed:
                obs_fluxes = np.array(
                    [
                        obs[filt] if filt in obs.colnames else np.nan
                        for filt in filters.keys()
                    ]
                )
                obs_fluxes_err = np.array(
                    [
                        obs[f"{filt}_err"] if f"{filt}_err" in obs.colnames else np.nan
                        for filt in filters.keys()
                    ]
                )
                mod_fluxes = np.array(
                    [
                        mod["best." + filt]
                        if "best." + filt in mod.colnames
                        else np.nan
                        for filt in filters.keys()
                    ]
                )
            else:
                mod_fluxes = np.array([mod[filt] for filt in filters])
                obs_fluxes = mod_fluxes
                obs_fluxes_err = mod_fluxes

            if observed:
                z = float(mod["best.universe.redshift"])
                surf = (
                    4.0 * np.pi * mod["best.universe.luminosity_distance"] ** 2
                )
            else:
                z = float(mod["universe.redshift"])
                surf = 4.0 * np.pi * mod["universe.luminosity_distance"] ** 2
            zp1 = 1.0 + z

            xmin = 0.9 * np.min(filters_wl) if xrange[0] is False else xrange[0]
            xmax = 1.1 * np.max(filters_wl) if xrange[1] is False else xrange[1]

            if sed_type == "lum":
                k_corr_SED = 1e-29 * surf * c / (filters_wl * 1e-6)
                if observed:
                    obs_fluxes *= k_corr_SED
                    obs_fluxes_err *= k_corr_SED
                mod_fluxes *= k_corr_SED

                for cname in sed.colnames[1:]:
                    sed[cname] *= wavelength_spec * 1e3

                filters_wl /= zp1
                wavelength_spec /= zp1
                xmin /= zp1
                xmax /= zp1
            elif sed_type == "mJy":
                k_corr_SED = 1.0

                fact = 1e29 * 1e-3 * wavelength_spec**2 / c / surf
                for cname in sed.colnames[1:]:
                    sed[cname] *= fact
            else:
                console.print(f"{ERROR} Unknown plot type.")

            mask_ok = np.logical_and(obs_fluxes > 0.0, obs_fluxes_err > 0.0)
            mask_uplim = np.logical_and(
                np.logical_and(obs_fluxes > 0.0, obs_fluxes_err < 0.0),
                obs_fluxes_err > -9990.0 * k_corr_SED,
            )

            wsed = np.where((wavelength_spec > xmin) & (wavelength_spec < xmax))
            figure = plt.figure()
            if observed:
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            else:
                gs = gridspec.GridSpec(1, 1)
            if (sed.columns[1][wsed] > 0.0).any():
                ax1 = plt.subplot(gs[0])
                if observed:
                    ax2 = plt.subplot(gs[1])

                # Stellar emission
                if (
                    "stellar_attenuated" in series
                    and "stellar.young" in sed.columns
                ):
                    spectrum = (
                        sed["stellar.young"][wsed] + sed["stellar.old"][wsed]
                    )

                    if "nebular.absorption_young" in sed.columns:
                        spectrum += sed["nebular.absorption_young"][wsed]
                        spectrum += sed["nebular.absorption_old"][wsed]

                    if "attenuation.stellar.young" in sed.columns:
                        spectrum += sed["attenuation.stellar.young"][wsed]
                        spectrum += sed["attenuation.stellar.old"][wsed]

                    ax1.loglog(
                        wavelength_spec[wsed],
                        spectrum,
                        label="Stellar attenuated",
                        color="gold",
                        marker=None,
                        nonpositive="clip",
                        linestyle="-",
                        linewidth=1.0,
                    )

                if (
                    "stellar_unattenuated" in series
                    and "stellar.young" in sed.columns
                ):
                    ax1.loglog(
                        wavelength_spec[wsed],
                        (sed["stellar.old"][wsed] + sed["stellar.young"][wsed]),
                        label="Stellar unattenuated",
                        color="xkcd:deep sky blue",
                        marker=None,
                        nonpositive="clip",
                        linestyle="--",
                        linewidth=1.0,
                    )

                # Nebular emission
                if "nebular" in series and "nebular.emission_young" in sed.columns:
                    spectrum = (
                        sed["nebular.emission_young"][wsed]
                        + sed["nebular.emission_old"][wsed]
                    )

                    if "attenuation.nebular.emission_young" in sed.columns:
                        spectrum += sed["attenuation.nebular.emission_young"][wsed]
                        spectrum += sed["attenuation.nebular.emission_old"][wsed]

                    ax1.loglog(
                        wavelength_spec[wsed],
                        spectrum,
                        label="Nebular emission",
                        color="xkcd:true green",
                        marker=None,
                        nonpositive="clip",
                        linewidth=1.0,
                    )

                # Dust emission Draine & Li
                if "dust" in series and "dust.Umin_Umin" in sed.columns:
                    ax1.loglog(
                        wavelength_spec[wsed],
                        (
                            sed["dust.Umin_Umin"][wsed]
                            + sed["dust.Umin_Umax"][wsed]
                        ),
                        label="Dust emission",
                        color="xkcd:bright red",
                        marker=None,
                        nonpositive="clip",
                        linestyle="-",
                        linewidth=1.0,
                    )

                # Dust emission Dale
                if "dust" in series and "dust" in sed.columns:
                    ax1.loglog(
                        wavelength_spec[wsed],
                        sed["dust"][wsed],
                        label="Dust emission",
                        color="xkcd:bright red",
                        marker=None,
                        nonpositive="clip",
                        linestyle="-",
                        linewidth=1.0,
                    )

                # AGN emission
                if "agn" in series and (
                    "agn.fritz2006_torus" in sed.columns
                    or "agn.SKIRTOR2016_torus" in sed.columns
                ):
                    if "agn.fritz2006_torus" in sed.columns:
                        agn_sed = (
                            sed["agn.fritz2006_polar_dust"]
                            + sed["agn.fritz2006_torus"]
                            + sed["agn.fritz2006_disk"]
                        )
                    elif "agn.SKIRTOR2016_torus" in sed.columns:
                        agn_sed = (
                            sed["agn.SKIRTOR2016_polar_dust"]
                            + sed["agn.SKIRTOR2016_torus"]
                            + sed["agn.SKIRTOR2016_disk"]
                        )
                    if "xray.agn" in sed.columns:
                        agn_sed += sed["xray.agn"]
                    if "radio.agn" in sed.columns:
                        agn_sed += sed["radio.agn"]
                    ax1.loglog(
                        wavelength_spec[wsed],
                        agn_sed[wsed],
                        label="AGN emission",
                        color="xkcd:apricot",
                        marker=None,
                        nonpositive="clip",
                        linestyle="-",
                        linewidth=1.0,
                    )

                # Radio emission
                if "radio" in series and "radio.sf_nonthermal" in sed.columns:
                    ax1.loglog(
                        wavelength_spec[wsed],
                        sed["radio.sf_nonthermal"][wsed],
                        label="Radio SF nonthermal",
                        color="brown",
                        marker=None,
                        nonpositive="clip",
                        linestyle="-",
                        linewidth=1.0,
                    )

                if "model" in series:
                    ax1.loglog(
                        wavelength_spec[wsed],
                        sed["L_lambda_total"][wsed],
                        label="Model spectrum",
                        color="k",
                        nonpositive="clip",
                        linestyle="-",
                        linewidth=1.5,
                    )

                ax1.set_autoscale_on(False)
                ax1.scatter(
                    filters_wl,
                    mod_fluxes,
                    marker="o",
                    color="xkcd:strawberry",
                    s=8,
                    zorder=3,
                    label="Model fluxes",
                )
                if observed:
                    ax1.errorbar(
                        filters_wl[mask_ok],
                        obs_fluxes[mask_ok],
                        yerr=obs_fluxes_err[mask_ok],
                        ls="",
                        marker="o",
                        label="Observed fluxes",
                        markerfacecolor="None",
                        markersize=5,
                        markeredgecolor="xkcd:pastel purple",
                        color="xkcd:light indigo",
                        capsize=0.0,
                        zorder=3,
                        lw=1,
                    )
                    if mask_uplim.any():
                        ax1.errorbar(
                            filters_wl[mask_uplim],
                            obs_fluxes[mask_uplim],
                            yerr=-obs_fluxes_err[mask_uplim],
                            ls="",
                            marker="v",
                            label="Observed upper limits",
                            markerfacecolor="None",
                            markersize=6,
                            markeredgecolor="g",
                            capsize=0.0,
                        )
                    mask_noerr = np.logical_and(
                        obs_fluxes > 0.0, obs_fluxes_err < -9990.0 * k_corr_SED
                    )
                    if mask_noerr.any():
                        ax1.errorbar(
                            filters_wl[mask_noerr],
                            obs_fluxes[mask_noerr],
                            ls="",
                            marker="p",
                            markerfacecolor="None",
                            markersize=5,
                            markeredgecolor="r",
                            label="Observed fluxes, no errors",
                            capsize=0.0,
                        )
                    mask = np.where((obs_fluxes > 0.0) & (obs_fluxes_err > 0.0))
                    ax2.errorbar(
                        filters_wl[mask],
                        (obs_fluxes[mask] - mod_fluxes[mask])
                        / obs_fluxes[mask],
                        yerr=obs_fluxes_err[mask] / obs_fluxes[mask],
                        marker="_",
                        label="(Obs-Mod)/Obs",
                        color="k",
                        capsize=0.0,
                        ls="None",
                        lw=1,
                    )
                    ax2.plot([xmin, xmax], [0.0, 0.0], ls="--", color="k")
                    ax2.set_xscale("log")
                    ax2.minorticks_on()

                ax1.tick_params(
                    direction="in",
                    axis="both",
                    which="both",
                    top=True,
                    left=True,
                    right=True,
                    bottom=False,
                )
                if observed:
                    ax2.tick_params(
                        direction="in", axis="both", which="both", right=True
                    )

                figure.subplots_adjust(hspace=0.0, wspace=0.0)

                ax1.set_xlim(xmin, xmax)

                if yrange[0]:
                    ymin = yrange[0]
                else:
                    if mask_uplim.any():
                        ymin = min(
                            min(
                                np.min(obs_fluxes[mask_ok]),
                                np.min(obs_fluxes[mask_uplim]),
                            ),
                            min(
                                np.min(mod_fluxes[mask_ok]),
                                np.min(mod_fluxes[mask_uplim]),
                            ),
                        )
                    elif mask_ok.any():
                        ymin = min(
                            np.min(obs_fluxes[mask_ok]),
                            np.min(mod_fluxes[mask_ok]),
                        )
                    else:  # No valid flux (e.g., fitting only properties)
                        ymin = ax1.get_ylim()[0]
                    ymin *= 1e-1

                if yrange[1]:
                    ymax = yrange[1]
                else:
                    if mask_uplim.any():
                        ymax = max(
                            max(
                                np.max(obs_fluxes[mask_ok]),
                                np.max(obs_fluxes[mask_uplim]),
                            ),
                            max(
                                np.max(mod_fluxes[mask_ok]),
                                np.max(mod_fluxes[mask_uplim]),
                            ),
                        )
                    elif mask_ok.any():
                        ymax = max(
                            np.max(obs_fluxes[mask_ok]),
                            np.max(mod_fluxes[mask_ok]),
                        )
                    else:  # No valid flux (e.g., fitting only properties)
                        ymax = ax1.get_ylim()[1]
                    ymax *= 1e1
                xmin = xmin if xmin < xmax else xmax - 1e1
                ymin = ymin if ymin < ymax else ymax - 1e1

                ax1.set_xlim(xmin, xmax)
                ax1.set_ylim(ymin, ymax)
                if observed:
                    ax2.set_xlim(xmin, xmax)
                    ax2.set_ylim(-1.0, 1.0)
                    ax = ax2
                else:
                    ax = ax1
                if sed_type == "lum":
                    ax.set_xlabel(r"Rest-frame wavelength [$\mu$m]")
                    ax1.set_ylabel("Luminosity [W]")
                else:
                    ax.set_xlabel(r"Observed $\lambda$ ($\mu$m)")
                    ax1.set_ylabel(r"S$_\nu$ (mJy)")
                ax1.legend(fontsize=6, loc="best", frameon=False)
                if observed:
                    ax2.set_ylabel("Relative\nresidual")
                    ax2.legend(fontsize=6, loc="best", frameon=False)
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.setp(ax1.get_yticklabels()[1], visible=False)
                if observed:
                    figure.suptitle(
                        f"Best model for {obs['id']}\n (z={z:.3}, "
                        f"reduced χ²={mod['best.reduced_chi_square']:.2})"
                    )
                else:
                    figure.suptitle(f"Model {mod['id']}\n (z={z:.3})")
                if logo is not False:
                    # Multiplying the dpi by 2 is a hack so the figure is small
                    # and not too pixelated
                    figwidth = figure.get_figwidth() * figure.dpi * 2.0
                    figure.figimage(
                        logo,
                        figwidth - logo.shape[0],
                        0,
                        origin="upper",
                        zorder=0,
                        alpha=1,
                    )

                figure.savefig(
                    outdir / f"{mod['id']}_best_model.{format}",
                    dpi=figure.dpi * 2.0,
                )
                plt.close(figure)
            else:
                gbl_counter.message.put(
                    f"{WARNING} No valid best SED found for {obs['id']}. No plot created."
                )
        else:
            gbl_counter.message.put(
                f"{WARNING} No SED found for {obs['id']}. No plot created."
            )
