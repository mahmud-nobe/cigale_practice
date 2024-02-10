from itertools import product
from pathlib import Path

from astropy.table import Table
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcigale.utils.console import INFO, console
from pcigale.utils.counter import Counter
from pcigale.utils.io import read_table
from pcigale_plots.plot_types import Plotter

OBSERVATIONS = "observations.fits"

class Chi2(Plotter):
    def __init__(self, config, format, outdir):
        """Plot the χ² values of analysed variables."""
        self.configuration = config.config
        input_data = read_table(outdir / OBSERVATIONS)
        save_chi2 = self.configuration["analysis_params"]["save_chi2"]

        chi2_vars = []
        if "all" in save_chi2 or "properties" in save_chi2:
            chi2_vars += self.configuration["analysis_params"]["variables"]
        if "all" in save_chi2 or "fluxes" in save_chi2:
            chi2_vars += self.configuration["analysis_params"]["bands"]

        items = list(product(input_data["id"], chi2_vars, [format], [outdir]))
        counter = Counter(len(items), 1, "Item")

        self._parallel_job(items, counter)

        # Print the final value as it may not otherwise be printed
        counter.global_counter.value = len(items)
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
    def worker(obj_name, var_name, format, outdir):
        """Plot the reduced χ² associated with a given analysed variable

        Parameters
        ----------
        obj_name: string
            Name of the object.
        var_name: string
            Name of the analysed variable..
        outdir: Path
            Path to outdir

        """
        gbl_counter.inc()
        figure = plt.figure()
        ax = figure.add_subplot(111)

        var_name = var_name.replace("/", "_")
        fnames = sorted(outdir.glob(f"{obj_name}_{var_name}-block-*.fits"))
        fchi2s = sorted(outdir.glob(f"{obj_name}_chi2-block-*.fits"))
        for fname, fchi2 in zip(fnames, fchi2s):
            values = Table.read(fname)
            chi2 = Table.read(fchi2)
            data_df = pd.DataFrame([ values[var_name], chi2["chi2"]], index = [var_name, 'chi2']).T
            data_df['reduced_chi2'] = data_df.chi2 / min(data_df.chi2)
            filtered_df = data_df[data_df.reduced_chi2 <= 2]
            ax.scatter(filtered_df[var_name], filtered_df["chi2"], color="k", s=0.1)
        ax.set_xlabel(var_name)
        ax.set_ylabel(r"Relative $\chi^2$ ($\chi^2 / \chi^2_{min}$)")
        ax.set_ylim(
            0.0,
        )
        ax.minorticks_on()
        figure.suptitle(
            f"Relative $\chi^2$ distribution of {var_name} for {obj_name}."
        )
        figure.savefig(outdir / f"{obj_name}_{var_name}_chi2.{format}")
        plt.close(figure)
