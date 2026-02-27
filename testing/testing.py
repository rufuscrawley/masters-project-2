import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u

from conversion import JanskyWavelengths

wavelengths = [0.545, 0.638, 0.797,
               1.22, 1.63, 2.2,
               3.6, 4.5, 5.8,
               8.0, 24, 61.1,
               70, 74.8, 89.3, 1300]

janskys = [0.0655, 0.120, 0.216,
           0.483, 0.591, 0.511,
           0.324, 0.220, 0.313,
           0.370, 0.765, 1.42,
           1.581, 1.480, 1.260, 0.1758]

jw = JanskyWavelengths(janskys, wavelengths)
janskys = np.array([janskys])
wavelengths = np.array(wavelengths)
astro_janskys = (janskys * u.Jy).to(u.erg / u.cm ** 2 / u.s / u.Hz)
# v = fl
# f = v/l
speed_of_light = 3e14

f_lambda = astro_janskys * speed_of_light / wavelengths

print(jw.convert_to_si())
print(wavelengths)

plt.scatter(wavelengths, f_lambda, label="astropy")
plt.plot(wavelengths, jw.convert_to_si(), label="rufus")


plt.title("Observed SED")
plt.xlabel("Wavelength (Microns)")
plt.ylabel("Flux (erg / s / cm^2)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()
