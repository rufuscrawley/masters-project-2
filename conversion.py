class JanskyWavelengths:
    def __init__(self, janskys, wavelengths):
        """
        :param janskys: Janskys in Jy
        :param wavelengths: Wavelengths in microns (um)
        """
        self.janskys = janskys
        self.wavelengths = wavelengths

    def convert_to_si(self):
        si_units = []
        for n, jansky in enumerate(self.janskys):
            si_units.append(jansky * 10e-23 * (3e7 / (self.wavelengths[n] * 1e-6)))
        return si_units


class SIWavelengths:
    def __init__(self, fluxes, wavelengths):
        self.fluxes = fluxes
        self.wavelengths = wavelengths

    def convert_to_jy(self):
        janskys = []
        for n, flux in enumerate(self.fluxes):
            janskys.append((flux * self.wavelengths[n] * 1e-6) / (10e-23 * 3e7))
        return janskys
