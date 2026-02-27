import genetic_algorithm as ga
import variables as v
import conversion

outputs_for_josh = [7.58E-14, 8.78E-13, 1.93E-12, 2.20E-09, 8.18E-09, 2.47E-08, 4.26E-08, 5.31E-08, 5.66E-08, 4.67E-08,
                    6.01E-08, 8.52E-08, 1.07E-07, 1.19E-07, 2.59E-07, 4.59E-07, 4.38E-07, 4.06E-07, 3.75E-07, 3.36E-07,
                    3.17E-07, 2.84E-07, 2.58E-07, 2.19E-07, 2.38E-07, 1.82E-07, 1.91E-07, 1.66E-07, 1.53E-07, 1.32E-07,
                    1.27E-07, 1.09E-07, 1.05E-07, 9.50E-08, 8.33E-08, 9.07E-08, 8.54E-08, 8.03E-08, 7.37E-08, 7.14E-08,
                    6.53E-08, 6.81E-08, 7.08E-08, 7.05E-08, 8.03E-08, 7.94E-08, 1.09E-07, 1.60E-07, 2.11E-07, 2.19E-07,
                    2.19E-07, 2.46E-07, 2.67E-07, 3.29E-07, 4.20E-07, 4.98E-07, 5.72E-07, 6.94E-07, 8.00E-07, 8.47E-07,
                    9.09E-07, 1.01E-06, 1.16E-06, 1.23E-06, 1.35E-06, 1.44E-06, 1.49E-06, 1.51E-06, 1.51E-06, 1.49E-06,
                    1.43E-06, 1.40E-06, 1.20E-06, 1.14E-06, 1.02E-06, 8.94E-07, 8.17E-07, 6.94E-07, 5.94E-07, 5.06E-07,
                    4.17E-07, 3.47E-07, 2.83E-07, 2.37E-07, 1.83E-07, 1.48E-07, 1.16E-07, 9.29E-08, 7.33E-08, 5.71E-08,
                    4.35E-08, 3.33E-08, 2.46E-08, 1.87E-08, 1.40E-08, 1.05E-08, 8.24E-09, 5.96E-09, 4.15E-09, 3.03E-09]

for n, output in enumerate(outputs_for_josh):
    new_output = output / (v.wavelengths[n] * 10)
    outputs_for_josh[n] = new_output
jansky = conversion.SIWavelengths(outputs_for_josh, v.wavelengths).convert_to_jy()

test = conversion.JanskyWavelengths(jansky, v.wavelengths).convert_to_si()
print(outputs_for_josh)
print(test)

print(ga.interpolate_fluxes(jansky, [0.545, 0.638, 0.797,
                                     1.22, 1.63, 2.2,
                                     3.6, 4.5, 5.8,
                                     8, 24, 61.1,
                                     70, 74.8, 89.3,
                                     1.3e+03]))
