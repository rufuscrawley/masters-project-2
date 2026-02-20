import os

from matplotlib import pyplot as plt

import conversion

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lupi_wavelengths = [0.545, 0.638, 0.797,
                    1.22, 1.63, 2.2,
                    3.6, 4.5, 5.8,
                    8.0, 24, 61.1,
                    70, 74.8, 89.3
                    ]

janskys = [0.0655, 0.120, 0.216,
           0.483, 0.591, 0.511,
           0.324, 0.220, 0.313,
           0.370, 0.765, 1.42,
           1.581, 1.480, 1.260
           ]

vals = conversion.JanksyWavelengths(janskys, lupi_wavelengths)

plt.scatter(vals.wavelengths, vals.convert_to_si(), label="Observed SED")
plt.title("Observed SED")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Wavelength (Microns)")
plt.ylabel("Flux (erg / s / cm^2)")

plt.legend()
plt.show()
