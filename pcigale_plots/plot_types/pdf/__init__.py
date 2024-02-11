from itertools import product

from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcigale.utils.console import INFO, WARNING, console
from pcigale.utils.counter import Counter
from pcigale.utils.io import read_table
from pcigale_plots.plot_types import Plotter

OBSERVATIONS = "observations.fits"


class PDF(Plotter):
    def __init__(self, config, format, outdir):
        """Plot the PDF of analysed variables."""
        self.configuration = config.config
        save_chi2 = self.configuration["analysis_params"]["save_chi2"]

        pdf_vars = []
        if "all" in save_chi2 or "properties" in save_chi2:
            pdf_vars += self.configuration["analysis_params"]["variables"]
        if "all" in save_chi2 or "fluxes" in save_chi2:
            pdf_vars += self.configuration["analysis_params"]["bands"]

        input_data = read_table(outdir / OBSERVATIONS)
        items = list(product(input_data["id"], pdf_vars, [format], [outdir]))
        counter = Counter(len(items), 1, "PDF")

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
        """Plot the PDF associated with a given analysed variable

        Parameters
        ----------
        obj_name: string
            Name of the object.
        var_name: string
            Name of the analysed variable..
        outdir: Path
            The absolute path to outdir

        """
        gbl_counter.inc()
        var_name = var_name.replace("/", "_")
        fnames = sorted(outdir.glob(f"{obj_name}_{var_name}-block-*.fits"))
        fchi2 = sorted(outdir.glob(f"{obj_name}_chi2-block-*.fits"))
        likelihood = []
        model_variable = []
        for fname, fchi2 in zip(fnames, fchi2):
            values = Table.read(fname)
            chi2 = Table.read(fchi2)

            data_df = pd.DataFrame([ np.ma.filled(values[var_name]), np.ma.filled(chi2)["chi2"]], index = [var_name, 'chi2']).T
            data_df['relative_chi2'] = data_df.chi2 / min(data_df.chi2)
            filtered_df = data_df[data_df.relative_chi2 <= 2]

            model_variable.append(filtered_df[var_name].to_numpy())
            likelihood.append(np.exp(-0.5 * filtered_df["chi2"].to_numpy()))

        if len(likelihood) > 0:
            likelihood = np.concatenate(likelihood)
            model_variable = np.concatenate(model_variable)
            w = np.where(np.isfinite(likelihood) & np.isfinite(model_variable))
            likelihood = likelihood[w]
            model_variable = model_variable[w]

            Npdf = 100
            min_hist = np.min(model_variable)
            max_hist = np.max(model_variable)
            Nhist = min(Npdf, len(np.unique(model_variable)))

            if min_hist == max_hist:
                pdf_grid = np.array([min_hist, max_hist])
                pdf_prob = np.array([1.0, 1.0])
            else:
                pdf_prob, pdf_grid = np.histogram(
                    model_variable,
                    Nhist,
                    (min_hist, max_hist),
                    weights=likelihood,
                    density=True,
                )
                pdf_x = (pdf_grid[1:] + pdf_grid[:-1]) / 2.0

                pdf_grid = np.linspace(min_hist, max_hist, Npdf)
                pdf_prob = np.interp(pdf_grid, pdf_x, pdf_prob)

            figure = plt.figure()
            ax = figure.add_subplot(111)
            ax.plot(pdf_grid, pdf_prob, color="k")
            ax.set_xlabel(var_name)
            ax.set_ylabel("Probability density")
            ax.minorticks_on()
            figure.suptitle(
                f"Probability distribution function of {var_name} for "
                f"{obj_name}"
            )
            figure.savefig(outdir / f"{obj_name}_{var_name}_pdf.{format}")
            plt.close(figure)
        else:
            console.print(
                f"{WARNING} Cannot build the PDF of {var_name} for "
                f"{obj_name}."
            )
