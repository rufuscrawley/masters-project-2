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

    def convert_to_jy(self, fluxes):
        flux_u = (fluxes * u.Jy).to(u.erg / u.cm ** 2 / u.s / u.Hz)
        return flux_u.value * 3e8 / (self.wavelengths * 1e-6)

    def distance_scale(self, fluxes):
        return fluxes * float(((190 * u.pc) / (self.distance * u.pc)) ** 2)


IMLupi = FitObject("IM Lupi",
                   [0.0655, 0.120, 0.216,
                    0.483, 0.591, 0.511,
                    0.324, 0.220, 0.313,
                    0.370, 0.765, 1.42,
                    1.581, 1.480, 1.260,
                    ],
                   [0.0007, 0.00012, 0.0022,
                    0.0048, 0.0059, 0.0051,
                    0.0184, 0.0178, 0.0156,
                    0.0223, 0.0708, 0.0220,
                    0.127, 0.37, 0.51,
                    ],
                   [0.545, 0.638, 0.797,
                    1.22, 1.63, 2.2,
                    3.6, 4.5, 5.8,
                    8.0, 24, 61.1,
                    70, 74.8, 89.3,
                    ],
                   190, True)

hd_142666 = (np.array([0.44,
                       0.55, 0.71, 0.7625,
                       1.235, 1.662,
                       2.159, 3.368, 4.618,
                       9.0, 12.0,
                       18.0, 25.0, 60.0,
                       65.0, 70.0, 90.0,
                       100.0, 140.0, 160.0,
                       450.0, 800.0, 850.0,
                       1100.0, 1200.0, 1300.0]),
             np.array([0.53,
                       0.86, 0.96, 1.46,
                       1.83, 2.06,
                       2.47, 3.04, 3.78,
                       5.15, 8.57,
                       6.58, 11.20, 7.47,
                       5.26, 6.56, 5.73,
                       5.91, 5.97, 4.33,
                       1.09, 0.35, 0.26,
                       0.18, 0.079, 0.127]),
             np.array([0.024,
                       0.03, 0.05, 1.34,
                       0.04, 0.05,
                       0.04, 0.20, 0.17,
                       0.36, 4.00,
                       0.01, 6.00, 5.00,
                       0.37, 0.33, 0.32,
                       0.30, 0.83, 0.22,
                       0.06, 0.02, 0.08,
                       0.01, 0.004, 0.009]))

ry_lupi = (np.array([3.40, 5.03, 11.5, 23.8, 61.8, 102, 890]),
           np.array([1.33e-12, 7.39e-13, 3.87e-13, 3.55e-13, 2.72e-13, 1.62e-13, 9.26e-16]),
           np.array([2.6e-13, 1.5e-13, 2.3e-14, 2.5e-14, 2.4e-14, 1.9e-14, 3.3e-18]))

dr_tau = (np.array([0.36, 0.44, 0.55, 0.64,
                    0.78, 1.25, 1.65, 2.17,
                    3.60, 4.50, 5.80, 8.00,
                    12.0, 25.0, 60.0, 100.,
                    200., 450., 600., 729.,
                    850., 1056, 1300]),
          np.array([0.050, 0.061, 0.095, 0.149,
                    0.260, 0.460, 0.780, 1.220,
                    1.860, 1.890, 2.340, 1.920,
                    3.160, 4.300, 5.510, 5.730,
                    4.100, 2.380, 0.610, 0.400,
                    0.533, 0.230, 0.109]),
          np.array([0.033, 0.035, 0.047, 0.076,
                    0.150, 0.010, 0.040, 0.020,
                    0.200, 0.150, 0.020, 0.010,
                    0.030, 0.050, 0.040, 0.630,
                    0.850, 0.170, 0.050, 0.080,
                    0.007, 0.020, 0.011]))
