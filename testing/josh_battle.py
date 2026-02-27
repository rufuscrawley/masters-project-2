from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_smoothing_spline

import conversion
import variables

josh_janskys = [0.06907, 0.1348, 0.276,
                0.7187, 0.97, 0.7909,
                0.5641, 0.4537, 0.4334,
                0.5358, 1.445, 3.103,
                3.344, 3.392, 3.048,
                0.01512]
josh_wavelengths = [0.545, 0.638, 0.797,
                    1.22, 1.63, 2.2,
                    3.6, 4.5, 5.8,
                    8, 24, 61.1,
                    70, 74.8, 89.3,
                    1.3e+03]
jys = conversion.JanskyWavelengths(josh_janskys, josh_wavelengths)



cs = CubicSpline(josh_wavelengths, jys.convert_to_si(), extrapolate=True)
interpolated_flux = cs(variables.wavelengths)


plt.scatter(jys.wavelengths, jys.convert_to_si())
plt.plot(variables.wavelengths, interpolated_flux)

plt.title("Observed SED")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Wavelength (Microns)")
plt.ylabel("Flux (erg / s / cm^2)")

plt.show()