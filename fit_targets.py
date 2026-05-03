from astropy import units as u
import numpy as np


class FitObject(object):

    def __init__(self, name, fluxes, flux_err, wavelengths,
                 distance=100, janskys=False):
        self.name = name
        self.wavelengths = np.array(wavelengths)
        self.distance = distance

        # Flux scaling
        fluxes, flux_err = np.array(fluxes), np.array(flux_err)
        if janskys:
            flux_u = self.convert_to_jy(fluxes)
            flux_err_u = self.convert_to_jy(flux_err)
        else:
            flux_u = fluxes
            flux_err_u = flux_err
        self.fluxes = self.distance_scale(flux_u)
        self.flux_err = self.distance_scale(flux_err_u)

        self.flux_err += self.fluxes / 20

    def convert_to_jy(self, fluxes):
        flux_u = (fluxes * u.Jy).to(u.erg / u.cm ** 2 / u.s / u.Hz)
        return flux_u.value * 3e8 / (self.wavelengths * 1e-6)

    def distance_scale(self, fluxes):
        return fluxes * float(((100 * u.pc) / (self.distance * u.pc)) ** 2)


IMLupi = FitObject("IM Lupi",
                   [0.0655, 0.120, 0.216,
                    0.483, 0.591, 0.511,
                    0.324, 0.220, 0.313,
                    0.370, 0.765, 1.42,
                    1.581, 1.480, 1.260],
                   [0.0007, 0.00012, 0.0022,
                    0.0048, 0.0059, 0.0051,
                    0.0184, 0.0178, 0.0156,
                    0.0223, 0.0708, 0.220,
                    0.127, 0.37, 0.51],
                   [0.545, 0.638, 0.797,
                    1.22, 1.63, 2.2,
                    3.6, 4.5, 5.8,
                    8.0, 24, 61.1,
                    70, 74.8, 89.3,
                    ],
                   190, True)

DRTau = FitObject("DR Tau",
                  [0.061, 0.095, 0.149,
                   0.260, 0.460, 0.780, 1.220,
                   1.860, 1.890, 2.340, 1.920,
                   3.160, 4.300, 5.510, 5.730,
                   4.100, 2.380, 0.610,
                   0.533, 0.230, 0.109],
                  [0.035, 0.047, 0.076,
                   0.150, 0.010, 0.040, 0.020,
                   0.200, 0.150, 0.020, 0.010,
                   0.030, 0.050, 0.040, 0.630,
                   0.850, 0.170, 0.050,
                   0.007, 0.020, 0.011],
                  [0.44, 0.55, 0.64,
                   0.78, 1.25, 1.65, 2.17,
                   3.60, 4.50, 5.80, 8.00,
                   12.0, 25.0, 60.0, 100.,
                   200., 450., 600.,
                   850., 1056, 1300],
                  140, True)

# https://www.aanda.org/articles/aa/pdf/2014/07/aa21987-13.pdf
FTTau = FitObject("FT Tau",
                  [6.69e-12, 1.8e-11, 1.08e-11, 2.64e-11, 3.47e-11,
                   1.07e-10, 6.39e-11, 2.56e-10, 1.22e-10, 3.15e-10,
                   4.58e-10, 3.32e-10, 2.06e-10, 1.72e-10,
                   1.15e-10, 1.00e-10, 1.15e-10, 6.59e-11, 4.9e-11,
                   4.3e-11, 2.56e-11, 9.48e-12,
                   2.91e-12, 1.25e-12, 9.75e-13,
                   3.89e-13, 3.00e-13],
                  [0.06e-12, 0.02e-11, 0.01e-11, 0.01e-11, 0.02e-11,
                   0.01e-10, 0.06e-11, 0.02e-10, .01e-10, .07e-10,
                   .11e-10, .06e-10, .04e-10, .03e-10,
                   .02e-10, .03e-10, .16e-10, .12e-11, .17e-11,
                   .43e-11, .47e-11, .7e-12,
                   .37e-12, .48e-12, 1.95e-13,
                   1.13e-13, .32e-13],
                  [0.36, 0.44, 0.48, 0.55, 0.62,
                   0.68, 0.76, 0.80, 0.90, 1.25,
                   1.65, 2.16, 3.6, 4.5,
                   5.8, 8.0, 12, 22.19, 24,
                   60, 70, 350,
                   450, 624, 769,
                   1056, 1300],
                  140, False)
