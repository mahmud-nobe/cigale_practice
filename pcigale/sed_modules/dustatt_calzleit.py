"""
Calzetti et al. (2000) and Leitherer et al. (2002) attenuation module
=====================================================================

This module implements the Calzetti et al. (2000) and  Leitherer et al. (2002)
attenuation formulae, adding an UV-bump and a power law.

"""

import numpy as np

from pcigale.sed_modules import SedModule

__category__ = "dust attenuation"


def k_calzetti2000(wavelength):
    """Compute the Calzetti et al. (2000) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Calzetti at al. (2000). This formula
    is given for wavelengths between 120 nm and 2200 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
    result = np.zeros(len(wavelength))

    # Attenuation between 120 nm and 630 nm
    mask = (wavelength < 630)
    result[mask] = 2.659 * (-2.156 + 1.509e3 / wavelength[mask] -
                            0.198e6 / wavelength[mask] ** 2 +
                            0.011e9 / wavelength[mask] ** 3) + 4.05

    # Attenuation between 630 nm and 2200 nm
    mask = (wavelength >= 630)
    result[mask] = 2.659 * (-1.857 + 1.040e3 / wavelength[mask]) + 4.05

    return result


def k_leitherer2002(wavelength):
    """Compute the Leitherer et al. (2002) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Leitherer at al. (2002). This formula
    is given for wavelengths between 91.2 nm and 180 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
    result = (5.472 + 0.671e3 / wavelength -
              9.218e3 / wavelength ** 2 +
              2.620e6 / wavelength ** 3)

    return result


def uv_bump(wavelength, central_wave, gamma, ebump):
    """Compute the Lorentzian-like Drude profile.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.
    central_wave: float
        Central wavelength of the bump in nm.
    gamma: float
        Width (FWHM) of the bump in nm.
    ebump: float
        Amplitude of the bump.

    Returns
    -------
    a numpy array of floats

    """
    return (ebump * wavelength ** 2 * gamma ** 2 /
            ((wavelength ** 2 - central_wave ** 2) ** 2 +
             wavelength ** 2 * gamma ** 2))


def power_law(wavelength, delta):
    """Power law 'centered' on 550 nm..

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid in nm.
    delta: float
        The slope of the power law.

    Returns
    -------
    array of floats

    """
    return (wavelength / 550) ** delta


def a_vs_ebv(wavelength, bump_wave, bump_width, bump_ampl, power_slope):
    """Compute the complete attenuation curve A(λ)/E(B-V)*

    The Leitherer et al. (2002) formula is used between 91.2 nm and 150 nm, and
    the Calzetti et al. (2000) formula is used after 150 (we do an
    extrapolation after 2200 nm). When the attenuation becomes negative, it is
    kept to 0. This continuum is multiplied by the power law and then the UV
    bump is added.

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid (in nm) to compute the attenuation curve on.
    bump_wave: float
        Central wavelength (in nm) of the UV bump.
    bump_width: float
        Width (FWHM, in nm) of the UV bump.
    bump_ampl: float
        Amplitude of the UV bump.
    power_slope: float
        Slope of the power law.

    Returns
    -------
    attenuation: array of floats
        The A(λ)/E(B-V)* attenuation at each wavelength of the grid.

    """
    attenuation = np.zeros(len(wavelength))

    # Leitherer et al.
    mask = (wavelength > 91.2) & (wavelength < 150)
    attenuation[mask] = k_leitherer2002(wavelength[mask])
    # Calzetti et al.
    mask = (wavelength >= 150)
    attenuation[mask] = k_calzetti2000(wavelength[mask])
    # We set attenuation to 0 where it becomes negative
    mask = (attenuation < 0)
    attenuation[mask] = 0
    # Power law
    attenuation *= power_law(wavelength, power_slope)

    # As the powerlaw slope changes E(B-V), we correct this so that the curve
    # always has the same E(B-V) as the starburst curve. This ensures that the
    # E(B-V) requested by the user is the actual E(B-V) of the curve.
    wl_BV = np.array([440., 550.])
    EBV_calz = ((k_calzetti2000(wl_BV) * power_law(wl_BV, 0.)) +
                uv_bump(wl_BV, bump_wave, bump_width, bump_ampl))
    EBV = ((k_calzetti2000(wl_BV) * power_law(wl_BV, power_slope)) +
           uv_bump(wl_BV, bump_wave, bump_width, bump_ampl))
    attenuation *= (EBV_calz[1] - EBV_calz[0]) / (EBV[1] - EBV[0])

    # UV bump. It is added after the renormalization as the bump strength
    # should correspond to the requested E(B-V) and should therefore not be
    # changed by the renormalization.
    attenuation += uv_bump(wavelength, bump_wave, bump_width, bump_ampl)

    return attenuation


class CalzLeit(SedModule):
    """Calzetti + Leitherer attenuation module

    This module computes the dust attenuation using the formulae from
    Calzetti et al. (2000) and Leitherer et al. (2002). Note that both the
    stars and the gas are attenuated with the same curve as opposed to Calzetti
    et al. (2000) where the gas is attenuated with a Milky Way curve.

    The attenuation can be computed on the whole spectrum or on a specific
    contribution and is added to the SED as a negative contribution.

    """

    parameters = {
        "E_BVs_young": (
            "cigale_list(minvalue=0.)",
            "E(B-V)*, the colour excess of the stellar continuum light for "
            "the young population.",
            0.3
        ),
        "E_BVs_old_factor": (
            "cigale_list(minvalue=0., maxvalue=1.)",
            "Reduction factor for the E(B-V)* of the old population compared "
            "to the young one (<1).",
            1.0
        ),
        "uv_bump_wavelength": (
            "cigale_list(minvalue=0.)",
            "Central wavelength of the UV bump in nm.",
            217.5
        ),
        "uv_bump_width": (
            "cigale_list()",
            "Width (FWHM) of the UV bump in nm.",
            35.
        ),
        "uv_bump_amplitude": (
            "cigale_list(minvalue=0.)",
            "Amplitude of the UV bump. For the Milky Way: 3.",
            0.
        ),
        "powerlaw_slope": (
            "cigale_list()",
            "Slope delta of the power law modifying the attenuation curve.",
            0.
        ),
        "filters": (
            "string()",
            "Filters for which the attenuation will be computed and added to "
            "the SED information dictionary. You can give several filter "
            "names separated by a & (don't use commas).",
            "B_B90 & V_B90 & FUV"
        )
    }

    def _init_code(self):
        """Get the filters from the database"""
        self.ebvs = {}
        self.ebvs['young'] = float(self.parameters["E_BVs_young"])
        self.ebvs_old_factor = float(self.parameters["E_BVs_old_factor"])
        self.ebvs['old'] = self.ebvs_old_factor * self.ebvs['young']
        self.uv_bump_wavelength = float(self.parameters["uv_bump_wavelength"])
        self.uv_bump_width = float(self.parameters["uv_bump_width"])
        self.uv_bump_amplitude = float(self.parameters["uv_bump_amplitude"])
        self.powerlaw_slope = float(self.parameters["powerlaw_slope"])

        self.filter_list = [item.strip() for item in
                            self.parameters["filters"].split("&")]
        # We cannot compute the attenuation until we know the wavelengths. Yet,
        # we reserve the object.
        self.contatt = {}
        self.lineatt = {}

    def process(self, sed):
        """Add the CCM dust attenuation to the SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        wavelength = sed.wavelength_grid

        # Fλ fluxes (only from continuum) in each filter before attenuation.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Compute attenuation curve
        if len(self.contatt) == 0:
            sel_att = a_vs_ebv(wavelength, self.uv_bump_wavelength,
                               self.uv_bump_width, self.uv_bump_amplitude,
                               self.powerlaw_slope)
            for age in ['young', 'old']:
                self.contatt[age] = 10 ** (-.4 * self.ebvs[age] * sel_att)

        # Compute the attenuation curves on the line wavelength grid
        if len(self.lineatt) == 0:
            names = [k for k in sed.lines]
            linewl = np.array([sed.lines[k][0] for k in names])
            sel_att = a_vs_ebv(linewl, self.uv_bump_wavelength,
                               self.uv_bump_width, self.uv_bump_amplitude,
                               self.powerlaw_slope)
            old_curve = 10 ** (-.4 * self.ebvs['old'] * sel_att)
            young_curve = 10 ** (-.4 * self.ebvs['young'] * sel_att)

            for name, old, young in zip(names, old_curve, young_curve):
                self.lineatt[name] = (old, young)

        attenuation_total = 0.
        contribs = [contrib for contrib in sed.luminosities if
                    'absorption' not in contrib]
        for contrib in contribs:
            age = contrib.split('.')[-1].split('_')[-1]
            luminosity = sed.luminosities[contrib]
            attenuation_spectrum = luminosity * (self.contatt[age] - 1.)
            # We integrate the amount of luminosity attenuated (-1 because the
            # spectrum is negative).
            attenuation = -.5 * np.dot(np.diff(wavelength),
                                       attenuation_spectrum[1:] +
                                       attenuation_spectrum[:-1])
            attenuation_total += attenuation

            sed.add_module(self.name, self.parameters)
            sed.add_info("attenuation.E_BVs." + contrib, self.ebvs[age],
                         unit='mag')
            sed.add_info("attenuation." + contrib, attenuation, True,
                         unit='mag')
            sed.add_contribution("attenuation." + contrib, wavelength,
                                 attenuation_spectrum)

        for name, (linewl, old, young) in sed.lines.items():
            sed.lines[name] = (linewl, old * self.lineatt[name][0],
                               young * self.lineatt[name][1])

        # Total attenuation
        if 'dust.luminosity' in sed.info:
            sed.add_info("dust.luminosity",
                         sed.info["dust.luminosity"] + attenuation_total, True,
                         True, unit='W')
        else:
            sed.add_info("dust.luminosity", attenuation_total, True, unit='W')

        # Fλ fluxes (only from continuum) in each filter after attenuation.
        flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Attenuation in each filter
        for filt in self.filter_list:
            sed.add_info("attenuation." + filt,
                         -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]),
                         unit='mag')

        sed.add_info('attenuation.ebvs_old_factor', self.ebvs_old_factor)
        sed.add_info('attenuation.uv_bump_wavelength', self.uv_bump_wavelength,
                     unit='nm')
        sed.add_info('attenuation.uv_bump_width', self.uv_bump_width, unit='nm')
        sed.add_info('attenuation.uv_bump_amplitude', self.uv_bump_amplitude)
        sed.add_info('attenuation.powerlaw_slope', self.powerlaw_slope)


# SedModule to be returned by get_module
Module = CalzLeit
