from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, interp1d, PchipInterpolator, CubicSpline
import numpy as np
import variables as v

janskys = [0.0655, 0.120, 0.216,
           0.483, 0.591, 0.511,
           0.324, 0.220, 0.313,
           0.370, 0.765, 1.42,
           1.581, 1.480, 1.260, 0.1758]
lupi_wavelengths = [0.545, 0.638, 0.797,
                    1.22, 1.63, 2.2,
                    3.6, 4.5, 5.8,
                    8.0, 24, 61.1,
                    70, 74.8, 89.3, 1300]

janskys_to_interpolate = np.linspace(0, 1.6, 100)
spline = CubicSpline(v.wavelengths, janskys_to_interpolate, extrapolate=False)
true_spline = spline(lupi_wavelengths)

plt.plot(lupi_wavelengths, janskys, label="non-interpolated")
plt.plot(lupi_wavelengths, true_spline, label="interpolated")

plt.title("Observed SED")
plt.xlabel("Wavelength (Microns)")
plt.ylabel("Flux (erg / s / cm^2)")
plt.xscale("log")
plt.legend()
plt.show()
