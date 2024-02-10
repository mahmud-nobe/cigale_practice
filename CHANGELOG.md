# Change Log

## Unreleased
### Added
- The `pcigale-filters list` command now also provides the list of nebular emission lines. (Médéric Boquien)
- It is now possible to plot models computed with the `savefluxes` module with `pcigale-plot sed`. (Médéric Boquien)
- HST WFC3 filters have been added. For now they correspond to the chip 2 of the instrument. (Médéric Boquien)
- Before building the database from a git repository, we check whether git-lfs worked as expected. If not we raise en exception, with instructions how to fix the issue. (Médéric Boquien)
### Changed
- Any known line can now be used for fitting or for estimating the flux probability distribution without modifying the nebular module source. (Yannick Roehlly & Médéric Boquien)
- To combine several lines it was before necessary to prefix each line with `line`, for instance `line.NII-654.8+line.NII-658.3`. Now a single `line` prefix is needed, for instance `line.NII-654.8+NII-658.3`. (Médéric Boquien, idea from Yannick Roehlly & Véronique Buat)
- The format of the `chi2` files has been changed from memory-mapped numpy arrays to FITS tables so that they are easier to handle. At the same time the χ² are stored in a separate file, rather than being repeated in each file, leading to a reduction of up to 50% of the total disk space usage. (Médéric)
### Fixed
- The masked values in the observation and results are replaced by NaN when plotting in order to suppress a Matplotlib warning. (Médéric Boquien)
- Numpy floating-point errors have been disabled as they do not provide useful information to the end user while generating output noise. (Médéric Boquien)
- FITS verification warnings have been disabled as they do not indicate any real issue with the generated FITS tables. (Médéric Boquien)
- It is now once again possible to compute models using a parameters file. (Médéric Boquien)
- The `--outdir` option was broken due to a workaround that was applied to preserve the compatibility with older ConfigObj versions and the way the input data file was handled. It is now more robust. (Médéric Boquien, reported by Marjorie Decleir)
- The absorption of the Lyman continuum stellar emission is now correctly included in the plots. (Médéric Boquien)
- Use a more standard version name for the development branch to avoid an exception with recent versions of the python packaging module. (Médéric Boquien)
- Make sure `pcigale-plots mock` works when `variables` is left empty in the `pdf_analysis` configuration module. (Médéric Boquien)
- The effective UV bump strength in the `dustatt_modified_starburst` was dependent on the power law slope δ. As this is undesirable, it has been fixed so that the bump strength only depends on the value of the `uv_bump_amplitude` parameter. (Médéric Boquien)
### Optimised
- The line and continuum spectral components have been merged so that less work is subsequently required, for instance for the attenuation. This leads to a speedup in the generation of the models. The amplitude depends on the exact parameters but up to 20% has been observed for a typical run on z=0 galaxies. (Médéric Boquien)
- The initial processing of the input catalog, which can be quite significant for large catalogs, has been accelerated by a factor ~2.5. (Médéric Boquien)

## 2022.1 (2022-07-04)
### Added
- By default CIGALE adds 10% of the flux in quadrature to the uncertainties to represent sources of uncertainty that are not included by default in the photometric error bars. Changing this value required manually changing a source file. Now this setting is directly available in the `pcigale.ini` file. (Médéric Boquien)
- Additional detail has been added to the description of the `data\_file` entry in the `pcigale.ini` file to describe how to indicate upper limits as well as regarding the use NaN for missing values. (Médéric Boquien)
### Changed
- New CLOUDY nebular models have been computed with an extended range in metallicities. (Patrice Theulé, Véronique Buat, and Médéric Boquien)
- The `m2005` module has been made compatible with the `nebular` module. (Médéric Boquien)
### Fixed
- The list of emission lines contained a handful of incorrect entries. It has been entirely reworked in conjunction of the computation of new nebular models. (Patrice Theulé, Véronique Buat, and Médéric Boquien)

## 2022.0 (2022-01-10)
### Added
- The polar dust model of X-CIGALE (Yang et al., 2020) for the `skirtor2016` module has been integrated into the regular version. (Médéric Boquien, based on the initial work of Guang Yang)
- An additional check is done when constructing the `pcigale.ini` and `pcigale.ini.spec` files to avoid the generation of an incorrect `pcigale.ini.spec` when `pcigale.ini` exists but `pcigale.ini.spec` does not, which is not supposed to happen under regular circumstances. (Médéric Boquien)
- Now `pcigale check` also displays the number of models per redshift. (Médéric Boquien)
- Implementation of the AGN radio emission (Guang Yang, adapted by Médéric Boquien)
- It is now possible to choose the electron density of the nebular models (10, 100, or 1000 cm¯³). (Médéric Boquien)
- The X-ray modelling from X-CIGALE has been integrated into the regular version. This mainly comprises the new `xray` module and modifications to the fitting procedure to take into account the constraints based on ΔαOX. (Guang Yang, adapted by Médéric Boquien)
- The ugrizy LSST filters (version 1.7) have been added to the default list of filters. (Médéric Boquien)
- The intrinsic UV slope β₀ is now computed at the same time as the observed UV slope β when `beta_calz94` is set to `True` in the `restframe_parameters` module. (Médéric Boquien)
- When starting `pcigale`, tables are now displayed with 1) information about cigale and the machine (cigale version, python version, and platform), 2) summary information about the run (input file names, number of objects, redshift range, number of models, band and properties to, fit number of cores, analysis module), and 3) the module used for each SED component. (Médéric Boquien & Miguel Figueira)
- The start time, end time, and run duration are now displayed (Médéric Boquien & Miguel Figueira)
- New CLOUDY nebular models. There are now 25 different metallicities and the metallicity of the stars and of the gas can differ. These models are described in detail in Villa-Vélez et al. (2021). (Jorge Villa-Vélez, Patrice Theulé, Véronique Buat, and Médéric Boquien)
### Changed
- The IGM absorption from Meiksin has been modified to include the dependence of the hydrogen absorption cross section (~λ^2.75) so that the universe becomes transparent again at very short wavelengths. (Guang Yang, adapted by Médéric Boquien)
- The `fluxes` module has been removed as it has been superseded by the `bands` parameter of the `pdf\_analysis` module. (Médéric Boquien)
- The `del` command of `pcigale-filters` has been eliminated in preparation to the change of database. (Médéric Boquien)
- The wavelengths of the plots generated by `pcigale-filters plot` are now in nm, μm, mm, or cm, whichever unit is the most appropriate for the filter in question. (Médéric Boquien)
- The filters are now normalised to a maximum of 1 in `pcigale-filters plot`. (Médéric Boquien)
- The internal database has been changed. Previously we relied on the SqlAlchemy as an interface to an SQLite database. This came with various drawbacks. First, the addition of new models was not as straightforward as it should have been. It required the creation of multiple files (e.g., for the database schema) scattered across different places and a lot of code duplication, which created a barrier of entry for newcomers. Then SQLite comes with some restriction on the size of the tables, which is a problem for some of the heavier models such as BPASS. The emphasis for the new database is simplicity. There is no need to declare the database schema. It is automatically inferred when building the database. Each model is stored in a file named after the primary keys (physical parameters). There is no code that is model-dependent. Now adding a new model reduces to 1) writing an import function 2) writing the corresponding SED module. As a side benefit, it appears faster too. (Médéric Boquien)
- How upper limits are taken into account is now more flexible. In addition to the exact computation, it is now possible to waive the adjustment of the scaling of the models, an extremely slow operation that generally does not affect the results in any substantial manner. Given the important speedup, this option is now selected by default in `pdf\_analysis`. (Médéric Boquien)
- The terminal output has been completely reworked. The level of importance of the messages are now stated (INFO, WARNING, ERROR). Progress bars have been added, indicating the total number of items processed, the number of items processed per second, the elapsed time, the remaining time. Colours are used to improve the legibility of the information. (Médéric Boquien)
### Fixed
- Ensure that `pcigale-plots` correctly detects the `skirtor2016` AGN models. (Médéric Boquien, reported by Guang Yang)
- Correct a typo in the `themis` module. (Médéric Boquien, reported by Alexandros Maragkoudakis)
- Make sure that the best fit is also given for the bands for which `pdf\_analysis` provides a Bayesian estimate. (Médéric Boquien)
- The `save_chi2` parameter was not correctly acknowledged in `pcigale-plots pdf`, leading to a crash as it tried to build the PDF of parameters for which the corresponding `chi2` files were not saved. (Médéric Boquien, reported by Laia Barrufet de Soto)
- The help of `pcigale-plots sed` incorrectly reported that the values given to the `--xrange` option were in nm, rather than μm. (Médéric Boquien, reported by Guang Yang)
- A run with an already existing `out/` directory led to a crash under Microsoft Windows. This was due to renaming `out/` to add the time with hours, minutes, and second separated by a colon. This is not allowed on this platform. The renamed directory has not been made compact and does not use colons anymore. (Médéric Boquien, reported by Samir Salim)
- The units of line fluxes were not always correct for Bayesian estimates. (Médéric Boquien, reported by Katarzyna Małek)
- An input redshift of integer type caused `pcigale-plots sed` to crash because it could not be formatted as a float. It is now explicitly converted to a float for safety. (Médéric Boquien)
- SED plots could not be generated for runs without a dust attenuation module as `pcigale-plots sed` assumed there would be one. This case is now handled. (Médéric Boquien)
- Indicate the correct possible values for the optical depth at 9.7 μm in the `skirtor2016` module. (Médéric Boquien, reported by Katarzyna Małek)
- Prevent a crash in the spectral resampling procedure. That was due to the array containing the spectra of the individual components to be of dimension 1 when not having stellar populations and the first two modules not generating spectra on the same wavelength grid. (Médéric Boquien)
- Give the correct unit for `param.restframe\_Lnu`. (Médéric Boquien, reported by Miguel Figueira)
- Fix incorrect filename handling that prevented the import of the `skirtor2016` models under Microsoft Windows. (Médéric Boquien)
- Following a past change in the way `chi2` files are saved, adapt `pcigale\_plot` to correctly produce PDF plots for physical properties that are in log rather than crashing. (Médéric Boquien)
- Improve path handling so that all the filters are correctly imported under Microsoft Windows. (Médéric Boquien)
- When an unknown variable is listed in `pdf\_analysis`, an exception is now raised before starting the run. (Médéric Boquien, reported by Guang Yang)
- The `m2005` module is not compatible with the `nebular` and `xray` modules. An exception is not emitted when trying to build the `pcigale.ini` file. (Médéric Boquien, reported by Miguel Figueira)
- When running `pcigale init` several times with an existing `pcigale.ini`, the top comment was repeatedly included due to the default behaviour of configobj. (Médéric Boquien, reported by Miguel Figueira)
- Import `Iterable` from `collections.abc` rather than from `collections` in order to prevent `ImportError`/`AttributeError` exceptions with Python 3.10+. (Médéric Boquien)
### Optimised
- The estimation of the physical properties and the related uncertainties has been made up to 50% faster. The final gain in the analysis speed accounting for all the steps depends on the number of properties to be evaluated and the number of models but can be over 25% when estimating many properties over a large parameter space. (Médéric Boquien)
- Invalid models (e.g., when the stellar populations are older than the universe) are now ignored when handling upper limits. The speedup is very variable but can be substantial in case there are many invalid models. (Médéric Boquien)
- Now that Python dictionaries are officially ordered, use regular dictionaries rather than collections.OrderedDict. It should translate into a very slight speedup and a slightly more readable code. (Médéric Boquien)
- The storage of individual spectral physical components has been improved in order to require fewer memory allocations and copies. The exact gain depends somewhat on the parameter space but speedups of about 50% are typically seen. (Médéric Boquien)
- The test to determine if we need to reinterpolate all the components when adding a new one has been made more efficient, yielding a speedup of 1.5-2.0%. (Médéric Boquien)
- The `sfhfromfile` module is much faster now, thanks to caching the file containing the SFH. (Médéric Boquien)
- The computation of the UV slope β in the `resftframe_parameters` module has been made faster. (Médéric Boquien)

## 2020.0 (2020-06-29)
### Added
- The (1+z1)/(1+z2) factor between observed and grid flux densities caused by the differential redshifting is now taken into account. With a default grid redshift rounding of two decimals this yields a difference of at most 0.5% in the estimated physical properties at z=0 and even less at higher z. (Médéric Boquien)
- The list of bands for which to carry out a Bayesian flux estimate is now configurable. By default this corresponds to the fitted bands but it also supports bands that are not included in the fit. This can be set in the `fluxes` parameter of the `pdf\_analysis` module. (Médéric Boquien)
- Implementation of the auto-detection of lines in the input flux file so they are automatically added to the list of bands in `pcigale.ini`. (Médéric Boquien)
- The `dl2007` and `dl2014` modules now provide \<U\>. (Médéric Boquien)
- Saving the χ² is now a bit more fine-grained. It is possible to save none, all, only the properties, or only the fluxes. (Médéric Boquien)
- The database now includes the JWST MIRI and NIRCam filters. (Médéric Boquien)
- The SKIRTOR AGN models (Stalevski et al. 2016) have been implemented as the `skirtor2016` module. (Médéric Boquien, with the support of Marko Stalevski)
- The units are now given alongside the physical properties in the output FITS tables. (Médéric Boquien)
### Changed
- Python 3.6 is now the minimum required version. (Médéric Boquien)
- The `pcigale-plots` executable has been largely rewritten for better modularity and to enable the various improvements indicated below. (Rodrigo González Castillo & Médéric Boquien)
- The logo has now been moved to the lower-right corner of the figure so that it does not overlap with any information and it has been updated for a less pixelated version. (Médéric Boquien & Rodrigo González Castillo)
- The wavelength range in SED plots is now dynamically adapted to cover the observed wavelengths. (Médéric Boquien)
- The uncertainties on the SED plots now correspond only to 1σ rather than 3σ so they do not appear exceedingly large. (Médéric Boquien)
- The lines linking the different bands in the residual SED plot have been eliminated to improve the readability. (Médéric Boquien)
- Some lines have been made slightly thicker in SED plots so the different components are more visible. (Médéric Boquien)
- The colours in the SED plots have been tweaked for aesthetic reasons. (Médéric Boquien)
- The markers for the observed fluxes in the SED plots have been tweaked to improve readability. (Médéric Boquien)
- The computation of all the cosmology-dependent quantities has been consolidated in pcigale/utils/cosmology.py and optimised. This leads to a slightly faster startup, in particular when there are many objects to fit, and it makes it easier to change the cosmology. (Médéric Boquien)
- The time spent computing is now displayed in hours, minutes, and seconds rather than just seconds to improve legibility. (Médéric Boquien)
- Some expected warnings have been silenced in `pcigale-plots` to avoid flooding the terminal. (Médéric Boquien)
- The id given in the input files are now checked against duplications. In case there are duplications, an exception is emitted. This prevents possible crashes later in the run. (Médéric Boquien & Yannick Roehlly)
### Fixed
- Make sure we can plot the PDF of equivalent widths. (Médéric Boquien)
- Fix a crash when generating a mock catalogue containing intensive properties. (Médéric Boquien)
- In the `sfhdelayed` and `sfhdelayedbq` modules, provide the correct description for the sfr_A parameter (Médéric Boquien & Laure Ciesla)
- Internally the luminosity distance was erroneously stored in Mpc rather than in m for non-zero redshifts. This has now been standardised to m. (Médéric Boquien)
- As the best-fit properties are computed at the exact observed redshift, correct the scaling factor as it is computed at the grid redshift. This corrects for slight offsets on the best-fit properties when the input redshift has more decimals than the grid redshift. (Médéric Boquien)
- Fix the pip install by making pcigale.managers discoverable. (Yannick Roehlly)
- When using a parameters file, Boolean values were not correctly interpreted. (Médéric Boquien, reported by Eric Martínez, INAOE)
- Make sure that the best-fit models are stored with the correct scaling factor when the distance is given explicitly (Médéric Boquien)
- Some labels and the title for the SED plots has been improved to avoid overlaps and overflows. (Médéric Boquien)
- Ensure that best models are properly computed when models are computed by blocks and that no fit could be made in one or more blocks. This can be case if all the models in the block are older than the age of the universe. (Médéric)
- Make sure that the parameters are saved with the proper scale (linear or logarithmic) in the χ² files. (Médéric Boquien)
- Some math libraries such as MKL or OpenBLAS sometimes try to be (too) smart, starting computation threads on their own. As cigale is already parallel, this just oversubscribes the CPU and can lead to important slowdowns. An environment variable could be set by the user to disable this, but this is cumbersome. Rather, we set these variables directly in the code at the startup of cigale. (Yannick Roehlly & Médéric Boquien)
- Fix a crash in `pcigale-plots` when plotting the SED of models computed without stellar populations. (Médéric Boquien)
- Make sure that upper limits on physical properties are correctly taken into account. (Médéric Boquien)
- Improve the sanitation of input data so that upper limits of extensive properties are not eliminated from the quantities to be fitted when upper limits are activated. (Médéric Boquien)
- Ensure that the radio module still works with recent versions of numpy. (Médéric Boquien & Laure Ciesla, reported by Wenjia Zhou)
- The unphysical wavy structure of the nebular continuum in the mid-infrared has been eliminated. (Médéric Boquien)
- The percentage of models with a low reduced χ² was too small by a factor 100. (Médéric Boquien, reported by Guang Yang)
### Optimised
- Slight speedup of the computation of the likelihood from the χ² using a multiplication rather than a division. (Médéric Boquien)
- Speedup of the computation of the χ² by ~10% taking the opposite of a scalar rather than of an array. (Médéric Boquien)
- Thanks to a change in the layout of the models storage in RAM, the computation of the χ² is now massively faster when the run contains multiple redshifts. (Médéric Boquien)
- The computation of the weighted means and standard deviations has been made ~50% faster by normalising the likelihood. (Médéric Boquien)
- The `fritz2006` module should now run faster thanks to an optimisation of the computation of the luminosity of the various AGN components (Médéric Boquien & Guang Yang)
- Various optimisations have been made regarding shared arrays to make their access faster. The overall effect is a speedup of 3-4% for the computation of the models. (Médéric Boquien)
- All the cores should now be used over the entire duration of the computation of the Bayesian and best-fit estimates. Before the number of active cores could drop towards the end of the computation. (Médéric Boquien)

## 2018.0 (2018-11-06)
### Added
- It is now possible to optionally indicate the distance in Mpc in the input file. If present it will be used in lieu of the distance computed from the redshift. This is especially useful in the nearby universe where the redshift is a very poor indicator of the actual distance. (Médéric Boquien)
- It is now possible to fit any physical property indicated by the code (e.g. equivalent width, dust luminosity, etc.). For this the physical property needs to be given in the input file and the properties to be fitted must be given in the properties filed in pcigale.ini. (Héctor Salas & Médéric Boquien)
- It is now possible to fit emission lines. For this the line has to be indicated in the same way as any other band both in the input flux file (in units of W/m²) and in the list of bands in `pcigale.ini`. Lines are prefixed with `line.` followed by the name of the line, for instance `line.H-alpha` for Hɑ. The following lines are supported at the moment: `Ly-alpha`, `CII-133.5`, `SiIV-139.7`, `CIV-154.9`, `HeII-164.0`, `OIII-166.5`, `CIII-190.9`, `CII-232.6`, `MgII-279.8`, `OII-372.7`, `H-10`, `H-9`, `NeIII-386.9` `HeI-388.9`, `H-epsilon`, `SII-407.0`, `H-delta`, `H-gamma`, `H-beta`, `OIII-495.9`, `OIII-500.7`, `OI-630.0`, `NII-654.8`, `H-alpha`, `NII-658.4`, `SII-671.6`, `SII-673.1`. (Médéric Boquien)
- When emission lines are not corrected for absorption lines (e.g., in the case of very low resolution spectroscopy) the previous method, which computes the theoretical line fluxes, is not optimal. Rather we offer the possibility to measure the fluxes through special filters that are used like regular filters. The idea is to define filters with a positive part on the line, a negative part on the continuum, and a zero-valued integral. In such case the integration over the spectrum directly gives the flux of the integral. So this works at all redshifts, the filter is automatically redshifted at runtime. (Médéric Boquien & David Corre)
- Two new dust attenuation modules have been added: `dustatt\_modified\_CF00` and `dustatt\_modified\_starburst`. The former implements a modified 2-component Charlot & Fall (2000) model whereas the latter implements a modified starburst law with the continuum attenuated with a Calzetti (2000) curve and the lines extincted with a Milky Way or a Magellanic Cloud law. The previous models `dustatt\_powerlaw`, `dustatt\_2powerlaws`, and `dustatt\_calzleit` are still available but are deprecated. (Médéric Boquien & David Corre)
- In addition to the physical properties, the fluxes are now also estimated through a Bayesian analysis. (Médéric Boquien)
- The module `sfhdelayedbq` has been added. It implements a delayed SFH with a burst/quench. It is fully described in Ciesla et al. (2017).

### Changed
- The `sfhdelayed` module has been extended to optionally include an exponential burst to model the latest episode of star formation. (Médéric Boquien & Barbara Lo Faro)
- On Linux platforms the method to start the parallel processes has been changed from "spawn" to "fork". This allows for a much faster startup. On other platforms is remains unchanged as Windows does not support "fork" and MacOS is bugged when using "fork", resulting in a free of cigale. (Médéric Boquien)
- The list of modules has been made more explicit in the `pcigale.ini` file. (Médéric Boquien)

### Fixed
- The histogram bin width was not computed optimally when some models were invalid. (David Corre & Médéric Boquien)
- Missing import in the `m2005` module. (Médéric Boquien, reported by Dominika Wylezalek)
- The plot of the PDF could not be generated for physical properties estimated in log (Médéric Boquien)
- We do not attempt anymore to estimate the physical properties of galaxies with insanely large χ² that lead to an underflow in the computation of the likelihood. (Médéric Boquien)
- The best fit is now plotted at the exact redshift rather than at the rounded redshift. (Médéric Boquien)
- It is now possible to plot the best fit obtained in redshifting mode. (Médéric Boquien)

### Optimised
- The estimation of the physical properties is made a bit faster when all the models are valid. (Médéric Boquien)
- The access to the module cache has been made faster and the model cache has been made much simpler, avoiding plenty of complex computations. This results in a speedup of at least ~6% in the computation of the models. The speedup can be higher when using few photometric bands. At the same time it considerably reduces the number of page faults seen in some rare circumstances. (Médéric Boquien)
- The models counter was a bottleneck when using many cores as updating it could stall other parallel processes. Now the internal counter is updated much less frequently. The speedup goes from between negligible (few cores) up to a factor of a few (many cores). The downside is the the updates on the screen may be a bit irregular. (Médéric Boquien)
- It turns out that elevating an array to some power is an especially slow operation in python. The `dustatt_calzleit` module has been optimised leading to a massive speed improvement. This speedup is especially large for models that do not include dust emission. (Médéric Boquien)
- Making copies of partially computed SED when storing them to the cache can be slow. Now we avoid making copies of the redshifted SED. The speedup should be especially noticeable when computing a set of models with numerous redshifts. (Médéric Boquien)

## 0.12.1 (2018-02-27)
### Fixed
- The best fit could not be computed in photo-z mode because the redshift was negative. (Médéric Boquien)
- The bayesian estimates could not be computed when some models were older than the age of the universe. (Médéric Boquien)
- The usage of `dustatt\_cazleit` causes some confusion regarding the reddening of the stars and of the gas. We have clarified that they are both attenuated with the same law and switched the differential reddening to 1 by default. (Médéric Boquien & Véronique Buat)
- When some models were invalid, it was not possible to plot the PDF. (Médéric Boquien & Denis Burgarella)


## 0.12.0 (2018-02-19)
### Added
- Provide the possibility not to store a given module in cache. This can be useful on computers with a limited amount of memory. The downside is that when not caching the model generation will be slower. (Médéric Boquien)
- An option `redshift\_decimals` is now provided in `pdf\_analysis` to indicate the number of decimals to round the observed redshifts to compute the grid of models. By default the model redshifts are rounded to two decimals but this can be insufficient at low z and/or when using narrow-band filters for instance. This only applies to the grid. The physical properties are still computed for the redshift at full precision. (Médéric Boquien)
- Bands with negative fluxes are now considered valid and are fitted as any other band. (Médéric Boquien)
- Allow the models to be computed by blocks in `savefluxes`. This can be useful when computing a very large grid and/or to split the results file into various smaller files as large files can be difficult to handle. The number of blocks is set with the `blocks` parameters in the pcigale.ini. (Médéric Boquien)
- Allow the observations to be analysed by blocks of models in `pdf\_analysis`. This is useful when computing a very large grid of models that would not fit in memory. The number of blocks is set with the `blocks` parameters in the pcigale.ini. (Médéric Boquien)
- The integrated stellar luminosity is now provided as `stellar.lum`. (Médéric Boquien)
- The high resolution BC03 models have been added. They can be activated when building the database by adding `--bc03res=hr` to the build command. In that case the low resolution models are not built. (Médéric Boquien)
- Dust templates generated with THEMIS (Jones et al. 2017) have been contributed by the DustPedia team (Davies et al. 2017). Special acknowledgement to Angelos Nersesian and Frédéric Galliano for creating the dust templates and writing the code. (Dustpedia team)
- The Herschel SPIRE filters for extended sources have been added. (Médéric Boquien)

### Changed
- Make the timestamp more readable when moving the out/ directory. (Médéric Boquien)
- To accommodate the analysis of the observations by blocks, all models are now included in the estimation of the physical properties and no cut in chi² is done anymore. (Médéric Boquien)
- To accommodate the analysis of the observations by blocks, the `save_pdf` option has been eliminated. To plot PDF one needs to set `save_chi2` to True and then run `pcigale-plots pdf`. (Médéric Boquien)
- In order to capture rapid evolutionary phases, we assume that in a given period of 1 Myr, 10 small episodes of star formation occurred every 0.1 Myr, rather than one episode every 1 Myr.
- When computing the attenuation curve, the bump is now added so that its relative strength does not depend on δ. (Médéric Boquien, issue reported by Samir Salim)
- Fν was computed by calculating Fλ and then converting to Fν, which led to typical differences in fluxes of typically 1-2% and a bit more for a handful of pathological filters. Now Fν is computed directly and a bit faster. (Médéric Boquien, issue reported by Yannick Roehlly and Wouter Dobbels)

### Fixed
- Corrected a typo that prevented `restframe\_parameters` from being listed among the available modules. (Médéric Boquien)
- The filters in the residual plot of `pcigale-plots sed` are now drawn in order of increasing wavelength so that the line joining all the filters does not make loops. (Médéric Boquien)
- In the absence of a nebular component `restframe\_parameters` would crash when attempting to compute the equivalent widths of the lines listed in `EW_lines`. Now they are simply ignored. (Médéric Boquien)
- The luminosity spectrum of the best fit was saved assuming the distance corresponding to the redshift rounded to two decimals. This was an issue in particular at very low redshift as a difference of 0.005 in redshift can translate to a large difference on the luminosity distance. Now the exact luminosity distance of the object is used to compute the spectrum luminosity. (Médéric Boquien, reported by Jorge Melnick)
- When using the `parameters\_file` option, the indices of the models now correspond to the line number of the input file. (Médéric Boquien)
- When using the `parameters\_file` option, the list of modules is read from `sed\_modules` rather than being inferred from the input file. (Médéric Boquien)
- The computation of the upper limits would only work for the first few models, reverting back to regular fits for the others. (Médéric Boquien)
- A more explicit message is now given when the flux table cannot be read properly. (Médéric Boquien)
- Make sure that we do not try to fit data that have an error bar of 0 mJy. (Médéric Boquien)
- An erroneous warning was displayed when using the `restframe\_parameters` module. (Médéric Boquien)
- The formula from Sawicki et al. (2012) used to compute the χ² in the presence of upper limits was not correct. This led the χ² to depend directly on the absolute value of the upper limit. The formula has been rederived and corrected. (Médéric Boquien & Denis Burgarella)
- For some reason the wavelengths of the SCUBA 450 μm filter were a factor 10 too small. (Médéric Boquien)
- Ensure that the computation of the continuum level is correct when determining the equivalent width, in particular when the line width is very narrow. (Médéric Boquien)
- When using different line widths during a single run, ensure that the fluxes and other quantities are always computed correctly. (Médéric Boquien, special thanks to Genoveva Micheva)
- Compute Dn4000 more rigorously by integrating properly over Fν. (Médéric Boquien)

### Optimised
- The cache architecture has been simplified, making it somewhat faster. It speeds up the model generation by ~1%. (Médéric Boquien)

## 0.11.0 (2017-02-10)
### Added
- The stellar mass-weighted age is now provided. This is a much more usual measure of the age than the age of the oldest star. This is accessible with the `stellar.age_m_star` keyword in the `bc03` module with with the `stellar.age_mass` keyword in the `m2005` module. (Médéric Boquien)
- The nebular models have been expanded from log U=-3 to log U=-4. (Médéric Boquien & Akio Inoue)
- The nebular models are now sampled in steps of 0.1 dex in log U rather than 1.0 dex steps. (Médéric Boquien & Akio Inoue)
- A new set of filters from GAZPAR has been added. The pattern of the filter name is "telescope.instrument.filter", e.g. "hst.wfc3.F160W". If the telescope has one instrument, it is skipped, e.g. "galex.FUV". For now the original set of filters is still provided. (Médéric Boquien & Olivier Ilbert)
- A brand new module `restframe\_parameters` has been added to compute rest frame parameters: UV slope β (Calzetti et al. 1994), Dn4000 (Balogh et al. 1999), IRX, emission lines equivalent widths at any wavelength, luminosity in any filter, colour in any pair of filters. This module has to be inserted right before the redshifting module. (Médéric Boquien)

### Changed
- We do not output the break strength from the `bc03` module anymore as these were not computed properly. (Médéric Boquien)
- The new `restframe\_parameters` module replaces the unofficial `param` module, which has now been trimmed to only compute fluxes in the observed frame as the rest of its functionalities have been transferred to the much more efficient `restframe\_parameters` module. To reflect this, it has been renamed `fluxes`. (Médéric Boquien)
- We now make use of new features available in Python 3.5. Previous versions are henceforth unsupported. However Python 3.6 or later is recommended for better performance. (Médéric Boquien)

### Fixed
- When the pcigale.ini file was missing, pcigale would crash and display a fairly cryptic backtrace. Now it explicitly states that the file could not be found. (Médéric Boquien)
- The nebular emission now takes into account deviations from the 10000K case B assumption. In practice this yields fluxes about 10% fainter. (Médéric Boquien & Akio Inoue)
- Some filters were incorrectly assumed to be defined in units of energy when they were actually defined in units of photons, yielding slightly incorrect fluxes. Now all the filters are converted into units of energy when imported. (Médéric Boquien)
- Remove the PSEUDO_D4000 filter which was incorrect. (Médéric Boquien)
- Indicate the correct transmission type for the PACS green and red filters. (Médéric Boquien)
- IRAS filters are defined in energy rather than photons. (Médéric Boquien)
- Remove a Numpy warning in the computation of the IGM absorption (Médéric Boquien)
- The Herschel passbands only included the filter. Now they include the full throughput of the instrument. Flux differences are expected to be no more than 1%. (Médéric Boquien)
- The `dustatt_powerlaw` module indicated that the Milky Way bump has an amplitude of 3. This is only valid for the `dustatt_calzleit` module. As `dustatt_powerlaw` is normalised to A(V) rather than E(B-V) for `dustatt_calzleit`, the bump is a factor Rv larger. A more reasonable value is now given. (Médéric Boquien)
- The correction of the χ² for the upper limits now properly takes into account earlier changes intended to reduce memory usage and speed up the analysis. This prevents a crash. (Médéric Boquien)
- Fix a crash in the computation of the rectangular periodic SFH with Numpy 1.12. (Médéric Boquien)

### Optimised
- By default the MKL library created many threads for each for the parallel processes. Not only was this not necessary as a high-level parallelisation already exists, but it generated a strong oversubscription on the CPU and on the RAM. The slowdown was over a factor of ~2 in some cases. Now we mandate KML to use only 1 thread fo each process. (Médéric Boquien & Yannick Roehlly)
- The generic numpy interpolation function was used. As we are in a well-controlled environment, this generated unnecessary verifications on the type and shape of the arrays. The compiled numpy interpolation function is now used, bypassing those checks. This generates a gain of 5-10% in computing speed for the generation of the models. (Médéric Boquien)
- The interpolation of the spectra of the different physical components on a new wavelength grid was not optimal as the interpolation was done separately for each component. Now a specific function has been implemented caching repetitive computations and vectorising the interpolation to compute it for all the components in a single step. This generates a gain of 10% in computing speed for the generation of the models. (Médéric Boquien)
- An optimisation in the sorting of a 2D array led to a gain of 10% in computing speed for the generation of the models. (Médéric Boquien)

## 0.10.0 (2016-09-15)
### Added
- Enable the possibility of having different normalisation factors for the star formation history. (Médéric Boquien)

### Changed
- Various descriptions have been improved and clarified. (Médéric Boquien)
- The `output\_file` and `output\_format` parameters have been removed from the `savefluxes` module. They served little purpose and made the code more complex. The same strategy as for the `pdf\_analysis` modules is now adopted, saving the output both as FITS and ASCII tables. Old configuration file still work, with these two parameters simply ignored. (Médéric Boquien)

### Fixed
- With the new sanity check of the input parameters, cigale did not handle the fact that the redshift could be given in the parameters file. Now this is handled properly. (Médéric Boquien)
- When giving the list of parameters through a file, cigale did not compute properly what parameter changed between to successive models. (Médéric Boquien)
- Using the m2005 module led to a crash. This is now fixed. (Yannick Roehlly)
- When computing models on a grid, the order could change from one run to the next, which is an issue when comparing the outputs of `savefluxes` for instance. Now models are always computed in the same order. The last parameter of the last module being in innermost loop and the first parameter of the first module being the outermost loop. (Médéric Boquien)

### Optimised
- A significant fraction of the total run time is spent computing integrals (e.g. fluxes in passbands). We can make the integration faster by rewriting the trapezoidal rule in terms of np.dot(). This allows to offload the computation to optimised libraries. The end result is that the integration is twice as fast, with a gain of ~10-15% on the total run time. (Médéric Boquien)
- The conversion from luminosity to flux is now a bit faster. (Médéric Boquien)
- The order the models are stored in memory has been changed to make the computation of the χ² faster. (Médéric Boquien)

## 0.9.0 (2016-04-04)
### Added
- When using the `savefluxes` module, all the output parameters were saved. This is not efficient when the user is only interested in some of the output parameters but not all. We introduce the "variables" configuration parameter for `savefluxes` to list the output parameters the user wants to save. If the list is left empty, all parameters are saved, preserving the current behaviour. This should increase the speed substantially when saving memory. (Médéric Boquien)
- Similarly to the `savefluxes` module, in the `pdf_analysis` module if the list of physical properties is left empty, all physical parameters are now analysed. (Médéric Boquien)
- It is now possible to pass the parameters of the models to be computed from a file rather than having to indicate them in pcigale.ini. This means that the models do not necessarily need to be computed on a systematic grid of parameters. The name of the file is passed as an argument to the parameters\_file keyword in pcigale.ini. If this is done, the creation\_modules argument is ignored. Finally, the file must be formatted as following: each row is a different model and each column a different parameter. They must follow the naming scheme: module\_name.parameter\_name, that is "bc03.imf" for instance. (Médéric Boquien)
- Addition of the `schreiber2016` SED creation module implementing the Schreiber et al. (2016) dust models. (Laure Ciesla)
- The physical parameters provided in pcigale.ini were not checked at startup against what the modules could accept. This could lead to a runtime crash if an unexpected value was passed to the module. Now the parameters are checked at startup. If an issue is found, it is indicated and the user is asked to fix it before launching cigale again. The validation file is build at the same time as pcigale.ini. (Médéric Boquien)
- There is a new attenuation module (`dustatt_2powerlaws`) based on a double power law: a birth cloud power law applied only to the young star population and an ISM power law applied to both the young and the old star population. (Yannick Roehlly & Véronique Buat)

### Changed
- The estimates of the physical parameters from the analysis of the PDF and from the best fit were recorded in separate files. This can be bothersome when trying to compare quantities from different files. Rather, we generate a single file containing all quantities. The ones estimated from the analysis of the PDF are prefixed with "bayes" and the ones from the best fit with "best". (Médéric Boquien)
- To homogenize input and output files, the "observation\_id" has been changed to "id" in the output files. (Médéric Boquien)
- The output files providing estimates of the physical properties are now generated both in the form of text and FITS files. (Médéric Boquien)
- When using the `dustatt_calzleit module`, choosing ẟ≠0 leads to an effective E(B-V) different from the one set by the user. Now the E(B-V) will always correspond to the one specified by the user. This means that at fixed E(B-V), A(V) depends on ẟ. (Médéric Boquien)
- The pcigale-mock tool has been merged into pcigale-plots; the mock plots can be obtained with the "mock" command. (Médéric Boquien)
- The `sfhdelayed` module is now initialised with \_init_code() to be consistent with the way things are done in other modules. This should give a slight speedup under some sircumstances too. (Médéric Boquien)
- In `sfhfromfile`, the specification of the time grid was vague and therefore could lead to incorrect results if it was not properly formatted by the end user. The description has been clarified and we now check that the time starts from 0 and that the time step is always 1 Myr. If it is not the case we raise an exception. (Médéric Boquien)
- When the redshift is not indicated in pcigale.ini, the analysis module fills the list of redshifts from the redshifts indicated in the input flux file. This is inefficient as analysis modules should not have to modify the configuration. Now this is done when interpreting pcigale.ini before calling the relevant analysis module. As a side effect, "pigale check" now returns the total number of models that cigale will compute rather than the number of models per redshift bin. (Médéric Boquien)
- The optionally saved spectra in the `pdf_analysis` and `savefluxes` modules were saved in the VO-table format. The most important downside is that it is very slow to write to, which proves to be a major bottleneck in the computing speed. To overcome this issue, we rather save the spectra using the FITS formation. Instead of having one file containing the spectra (including the various components) and the SFH in a single file, now we have one file for the spectra and one file for the SFH.

### Fixed
- To estimate parameters in log, pcigale determines which variables end with the "\_log" string and removed it to compute the models. However in some circumstances, it was overzealous. This has been fixed. (Médéric Boquien)
- When estimating a parameter in log, these were not scaled appropriately and taken in log when saving the related χ² and PDF. (Médéric Boquien)
- In the presence of upper limits, correct the scaling factor of the models to the observations before computing the χ², not after. (Médéric Boquien)
- When called without arguments, pcigale-plots would crash and display the backtrace. Now it displays the a short help showing how to use it. (Médéric Boquien)
- For `sfh2exp`, when setting the scale of the SFH with sfr0, the normalisation was incorrect by a factor exp(-1/tau\_main). (Médéric Boquien)
- The mass-dependent physical properties are computed assuming the redshift of the model. However because we round the observed redshifts to two decimals, there can be a difference of 0.005 in redshift between the models and the actual observation if CIGALE computes the list of redshifts itself. At low redshift, this can cause a discrepancy in the mass-dependent physical properties: ~0.35 dex at z=0.010 vs 0.015 for instance. Therefore we now evaluate these physical quantities at the observed redshift at full precision. (Médéric Boquien, issue reported by Samir Salim)
- In the `sfhfromfile` module, an extraneous offset in the column index made that it took the previous column as the SFR rather than the selected column. (Médéric Boquien)
- In `sfhfromfile`, if the SFR is made of integers cigale crashed. Now we systematically convert it to float. (Médéric Boquien)
- The order of the parameters for the analysis modules would change each time a new pcigale.ini was generated. Now the order is fixed. (Médéric Boquien)
- In the output the sfh.age parameter would correspond to the input value minus 1. Now both values are consistent with one another. (Laure Ciesla & Médéric Boquien)
- In rare circumstances requiring a specific distribution of redshifts the integration of the spectrum in some filters was not done correctly, inducing relative errors of ~10¯⁵-10¯⁶. (Médéric Boquien)
- The absorption of the Lyman continuum from old stars tended to be overestimated leading to some “negative fluxes” for the Lyman continuum. (Médéric Boquien)

### Optimised
- Prior to version 0.7.0, we needed to maintain the list of redshifts for all the computed models. Past 0.7.0 we just infer the redshift from a list unique redshifts. This means that we can now discard the list of redshifts for all the models and only keep the list of unique redshifts. This saves ~8 MB of memory for every 10⁶ models. the models should be computed slightly faster but it is in the measurement noise. (Médéric Boquien)
- The `sfhfromfile` module is now fully initialised when it is instantiated rather than doing so when processing the SED. This should be especially sensitive when processing different SED. (Médéric Boquien)
- We do not store the time grid in the SED anymore given that we assume it starts at 0 Myr with steps of 1 Myr, we can easily reconstruct to save it if needed. It should save a little bit of memory and it should go a little bit faster. (Médéric Boquien)
- To compute the stellar spectrum of the young component, do not pass the full SFH with the old part set to 0. Rather, only pass the corresponding part of the SFH. This nearly doubles the computing speed of the stellar spectrum (Médéric Boquien)
- Computers are much better at multiplying than at dividing. Therefore to correct the spectral emission when redshifting we multiply by 1/(1+z) rather than dividing by 1+z. (Médéric Boquien)

---

## 0.8.1 (2015-12-07)
### Fixed
-  To estimate parameters in log, pcigale determines which variables end with the "\_log" string and removed it to compute the models. However in some circumstances, it was overzealous. This has been fixed. (Médéric Boquien)

## 0.8.0 (2015-12-01)
### Added
- The evaluation of the parameters is always done linearly. This can be a problem when estimating the SFR or the stellar mass for instance as it is usual to estimate their log rather. Because the log is non-linear, the likelihood-weighted mean of the log is not the log of the likelihood-weighted mean. Therefore the estimation of the log of these parameters has to be done during the analysis step. This is now possible. The variables to be analysed in log just need to be indicated with the suffix "\_log", for instance "stellar.m\_star\_log". (Médéric Boquien, idea suggested by Samir Salim)

### Fixed
- Running the scripts in parallel trigger a deadlock on OS X with python 3.5. A workaround has been implemented. (Médéric Boquien)
- When no dust emission module is used, pcigale genconf complains that no dust attenuation module is used. Correctly specify dust emission and not attenuation. (Médéric Boquien and Laure Ciesla)
- Allowing more flexibility to read ASCII files broke the handling of FITS files. It is now fixed. (Yannick Roehlly)

### Changed
- The attenuation.ebvs\_main and attenuation.ebvs\_old parameters are no longer present as they were duplicates of attenuation.E\_BVs.stellar.old and attenuation.E\_BVs.stellar.young (that are still available).

---

## 0.7.0 (2015-11-19)
### Added
- The pcigale-mock utility has been added to generate plots comparing the exact and pcigale-estimated parameters. This requires pcigale to be run beforehand with the `pdf_analysis` module and the mock\_flag option set to True. (Denis Burgarella and Médéric Boquien)
- The pcigale-filter utility has been added to easily list, plot, add, and remove filters without having the rebuild the database entirely. (Médéric Boquien)
- It is now possible to analyse the flux in a band as a regular parameter. It can be useful for flux predictions. (Yannick Roehlly)
- The redshift can be a now used as a free parameter, enabling pcigale to estimate the photometric redshift. (Médéric Boquien)
- When running "pcigale genconf", the list of modules is automatically checked against the list of official modules. If modules are missing, information is printed on the screen indicating the level of severity (information, warning, or error) and the list of modules that can be used. (Médéric Boquien)

### Changed
- The galaxy\_mass parameter was very ambiguous. In reality it corresponds to the integral of the SFH. Consequently it has been renamed sfh.integrated. (Médéric Boquien)
- In the Calzetti attenuation module, add a warning saying that is the power law slope is different than 0, E(B-V) will no longer be the real one. (Yannick Roehlly)
- Add "B\_B90" to the list of computed attenuation so that users can calculate the effective E(B-V). (Yannick Roehlly)
- Computing the parameters and their uncertainties through the histogram of the PDF is slow and can introduce biases in some cases. Rather, now the estimated values of the parameters and the corresponding uncertainties are simply computed from the weighted mean and standard deviation of the models that are at least 0.1% as likely as the best model to reproduce the observations. The differences in the estimates are very small except when very few models are used. (Médéric Boquien)
- Magic values to indicate invalid values (e.g. values lower than -99) are difficult to handle safely. They have been replaced with NaN wherever appropriate. The logic of the input flux file stays the same for the time being but the magic values are converted internally after reading it. Users are advised to replace magic values with NaN. The output files now use NaN instead of magic number to indicate invalid values. (Médéric Boquien)
- Rename the the AGN faction added by dale2014 module from agn.fracAGN to agn.fracAGN\_dale2014 to avoid conflict with fritz2006 module. (Yannick Roehlly)
- Remove the radio component from the dale2014 model so that it can be used with the more flexible radio module, courtesy Daniel Dale. (Laure Ciesla and Médéric Boquien)

### Fixed
- The SFH is modelled using discrete star formation episodes every 1 Myr. This means that as the SFH is not really continuous (the input single stellar population do not allow us to compute that properly), we should not integrate SFH(t), but simply sum SFH(t) as t is discrete. In most cases the difference is very small. The only case where it makes a difference is for a very rapidly varying SFH, for instance taking τ=1 Myr. (Médéric Boquien)
- Ensure that the flux can be computed even if the redshifting module has not been applied. By default in that case we assume a distance of 10 parsecs. While in practice it should never happen as the redshifting module is mandatory, this can be more important when using pcigale as a library. (Médéric Boquien and Yannick Roehlly)
- When called without arguments, pcigale would crash. Now it displays a brief message to remind how it should be invoked. (Médéric Boquien)
- Raise an exception instead of crash when an unknown IMF is requested. (Médéric Boquien)
- When the error column for a flux was present in the observation table but not in the used column list (when the user prefers to use a default error instead of the real one), the error on the flux was set to 0. (Yannick Roehlly)
- For some reason a point in the GALEX FUV filter has a negative transmission. That should not happen. After comparison with the filter on the GALEX website it has been set to 0. (Médéric Boquien)
- Shorten the left and right 0 parts of the pseudo D4000 filter so that it can be applied on smaller spectra. (Yannick Roehlly)
- To compute the reduced χ², we need to divide by the number of bands-1 (and not the number of bands). We do that because we consider that the models depend on one meta-parameter. (Médéric Boquien)
- The test to determine when to take into account upper limits did not work according the specifications. Now upper limits are always taken into account when they should. (Médéric Boquien)
- The nebular emission could be counted in excess in the dust luminosity as both lines and the Lyman continuum could be attenuated. Now we do not extend the attenuation under 91 nm. Also, a new component as been added, taking specifically the Lyman continuum absorption by gas, allowing to conserve the information about the intrinsic stellar Lyman continuum if need be. (Yannick Roehlly and Médéric Boquien)
- The Flambda table in the VO-table export does not reflect the fact that it stores luminosity densities. Accordingly, it has been renamed Llambda. (Yannick Roehlly and Médéric Boquien)
- When the flux file contains a mix of spaces and tabulations as column separators, pcigale discards the header and takes the first data line as the header. Now pcigale properly handles such a combination. Bug reported by Paola Santini. (Médéric Boquien)

### Optimised
- Major speedup to build the database by inserting multiple models in the database at once rather one model at a time. On an SSD, the total run time of "python setup.py build" goes from 5m20s to 2m42s. The speedup should be even more spectacular on a rotating hard drive. (Médéric Boquien)
- Memory usage reduction using in-place operations (e.g. a/=2 rather than a=a/2, saving the creation of a temporary array the size of a) where possible. (Médéric Boquien)
- The creation and handling of mock catalogues has been streamlined. (Médéric Boquien)
- Slight speedup using np.full() where possible to create an array with all elements set to the same value. (Médéric Boquien)
- Computing the scaling factors and the χ² in one step over the entire grid of models is very memory-intensive, leading to out-of-memory issues when using multiple cores. Rather, let's compute them band by band, as this avoids the creation of large temporary arrays, while keeping the computation fast. (Médéric Boquien).
- Each core copied the subset of models corresponding to the redshift of the object to be analysed. This is a problem as it can strongly increase memory usage with the number of cores, especially when there are many models and just one redshift. Rather than making a copy, we use a view, which not only saves a considerable amount of memory but is also faster as there is no need to allocate new, large arrays. This is made possible as models are regularly ordered with redshift. (Médéric Boquien)
- Various minor optimisations. (Médéric Boquien)

---

## 0.6.0 (2015-09-07)
### Added
- New module to compute a star formation history as described in Buat et al. 2008. (Yannick Roehlly)
- New module to compute a periodic SFH. Each star formation episode can be exponential, "delayed", or rectangular. (Médéric Boquien and Denis Burgarella)
- New module performing a quench on the star formation history. (Yannick Roehlly)
- New module to compute the emission of a modified black body. (Denis Burgarella)
- New module to compute the physical properties measured on the emission spectrum (e.g. spectral indices, ultraviolet slope, etc.). (Denis Burgarella)
- New pseudo filters to compute line fluxes and spectral indices. (Yannick Roehlly)
- The dust masses are now computed for the draine2007 and draine2014 modules. (Médéric Boquien)

### Changed
- The nebular\_lines\_width parameter of the nebular module is now called lines\_width as the nebular prefix was redundant. (Médéric Boquien)
- Prefix the ouput variables of SFH-related modules with "sfh" to facilitate their identification in the output files. (Médéric Boquien)
- Prefix the output variables of the fritz2006 AGN module with "agn" to facilitate their identification in the output files. (Médéric Boquien)
- Prefix the redshift with "universe". (Médéric Boquien)
- With pcigale-plot, draw the spectra only to λ=50 cm as the models do not extend much further and there is very rarely any observation beyond λ=21 cm. (Médéric Boquien)
- As pcigale is getting much faster, display the number of computed models every 250 models rather than every 100 models. (Médéric Boquien)
- Give default values for the dl2014, sfh\_buat, and sfhdelayed modules to allow for quick test runs. (Médéric Boquien)
- Now pcigale-plots plots errors on upper limits. (Denis Burgarella)

### Fixed
- When plotting, round the redshift to two decimals to match the redshift of the model. (Médéric Boquien)
- Ensure that all the input parameters of the nebular module are also output so it is possible to analyse them. (Médéric Boquien)
- Properly take the Lyman continuum photons absorbed by dust into account to compute the dust emission. (Médéric Boquien)
- Improve the readability of the pcigale-plots generated spectra by drawing the observed fluxes on top of other lines. (Médéric Boquien)
- The nebular components are now plotted with pcigale-plots. (Médéric Boquien)
- When a filter that was not in the database was called, pcigale would crash ungracefully as the exception invoked does not exist in Python 3.x. Now use a Python 3.x exception. (Médéric Boquien)
- The dustatt\_powerlaw module could not identify which physical components to attenuate when the nebular module was called. (Médéric Boquien)
- The displayed counter for the number of objects already analysed could be slightly offset from the number of models actually computed. (Médéric Boquien)
- Change the method to compute the χ² in the presence of upper limits as the previous method did not always converge. (Denis Burgarella and Médéric Boquien)

### Optimised
- When plotting, do not recompute the luminosity distance, which is very slow, but rather get the one computed during the analysis and that is given in the output files. (Médéric Boquien)
- Adding new physical components with a wavelength sampling different than that of the pre-existing grid is slow as a common grid has to be computed and all components interpolated over it. The nebular lines are especially bad for that, owing to the fact that each line is sampled with 19 points. This is excessive as sampling over 9 points barely changes the fluxes while speeding up the computation of the models by ~20%. (Médéric Boquien)
- Rather than resampling the filter transmission on the wavelength grid every time the flux is computed in a given filter, put the resampled filter into a cache. (Médéric Boquien)
- Rather than recomputing every time the merged wavelength grid from two different wavelength grids (for instance when adding a new physical component or when integrating the spectrum in a filter), put the results in a cache. (Médéric Boquien)
- Before adding a new component to a SED, we first copy the original SED without that component from the cache. This copy can be very slow when done automatically by python. We rather do this copy manually, which is much faster. (Médéric Boquien)
- When adding a new physical component with a different wavelength sampling, rather than reinterpolating all the components over the new grid, compute the interpolation only for new wavelengths. (Médéric Boquien)
- Various minor optimisations. (Médéric Boquien)
- When computing the flux in filters, np.trapz() becomes a bottleneck of the code. A large part of the time is actually spent on safeguards and on operations for nD arrays. However here we only have 1D arrays and some variables can be cached, which allows some optimisations to compute fluxes faster. (Médéric Boquien)
- The output parameters of a model were stored in an ordered dictionary. While convenient to keep the order of insertion it is very slow as it is implemented in pure Python for versions up to 3.4. Rather we use a regular dictionary and we reorder the parameters alphabetically. (Médéric Boquien)
- To store the SED in memory and retrieve them later, we index them with the list of parameters used to compute them. We serialise those using JSON. However JSON is slow. As these data are purely internal, rather use marshal, which is much faster than JSON. (Médéric Boquien)

---

## 0.5.1 (2015-04-28)
### Changed
- Set the default dale2014 AGN fraction to 0 to avoid the accidental inclusion of AGN. (Denis Burgarella)
- Modify the name of the averaged SFR: two averaged SFRs over 10 (sfh.sfr10Myrs) and 100Myrs (sfh.sfr100Myrs). (Denis Burgarella)
- Improve the documentation of the `savefluxes` module. (Denis Burgarella)

### Fixed
- Correction of the x-axis limits. (Denis Burgarella)
- Fix the detection of the presence of the agn.fritz2006\_therm in pcigale-plots. (Denis Burgarella)
- Correct the wavelength in the SCUBA 450 μm filter. (Denis Burgarella)
- Install the ancillary data required to make plots. (Yannick Roehlly)

## 0.5.0 (2015-04-02)

---

## 0.4.0 (2014-10-09)

---

## 0.3.0 (2014-07-06)

---

## 0.2.0 (2014-06-10)

---

## 0.1.0 (2014-05-26)
