import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import corner
import emcee
from emcee.moves import WalkMove, StretchMove
from scipy import stats

warnings.filterwarnings("ignore",
                        category=FutureWarning,
                        module="arviz")
import numpy as np
from matplotlib import pyplot as plt

import genetic_algorithm as ga
import utilities
import variables as v
import network_variables as nv

gene_spaces = ga.get_gene_spaces()


def model(theta):
    """
    Idealistic fit to compare against.
    :param theta: Parameters being changed by the MCMC fit to find uncertainties.
    :return:
    """
    return nv.predict(theta)


def log_likelihood(theta, x, _y, _y_err):
    """
    Calculate how well our parameters fit the data.
    :return:
    """
    # Reads in 100 NORMALISED flux values.
    y_model = model(theta)
    chi_squared = stats.chisquare(y_model, x,
                                  sum_check=False, ddof=v.split).statistic
    return chi_squared


def log_prior(theta):
    for n, value in enumerate(theta):
        if gene_spaces[n]["low"] < value < gene_spaces[n]["high"]:
            # Continue iterating
            continue
        else:
            return -np.inf
    # If all checks pass:
    return 0.0


def log_probability(theta, x, y, y_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, y_err)


# region variables
wavelengths = [0.545, 0.638, 0.797,
               1.220, 1.630, 2.200,
               3.600, 4.500, 5.800,
               8.000, 24.00, 61.10,
               70.00, 74.80, 89.30,
               1300]

x_solutions = [np.float64(16.54132807784993), np.float64(1.1660064775871395),
               np.float64(2.439135833842663), np.float64(0.0003330905078751758)]

x_solutions = np.array(x_solutions)

expected_model = model(x_solutions)

y_fluxes = [0.0655, 0.12, 0.216,
            0.483, 0.591, 0.511,
            0.324, 0.220, 0.313,
            0.370, 0.765, 1.420,
            1.581, 1.480, 1.260,
            0.176]
y_fluxes = utilities.JanskyWavelengths(y_fluxes, wavelengths).convert_to_si()
y_fluxes = np.array(y_fluxes)
y_errs = y_fluxes * .1

n_walkers = 75
n_dim = v.split

pos = np.array(x_solutions) + (.001 * np.random.randn(n_walkers, n_dim))
# endregion

# Run MCMC
print("Running MCMC...")

print("Multithreading...")
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability,
                                args=(expected_model, y_fluxes, y_errs),
                                moves=[(StretchMove(), 0.8), (WalkMove(), 0.2)])
print("Running sampler...")
sampler.run_mcmc(pos, 10_00,
                 progress=True)

# burnout and thinning :)
t_autocorrelation = sampler.get_autocorr_time(quiet=True)
discard = int(2.0 * np.max(t_autocorrelation))
thin = int(0.5 * np.min(t_autocorrelation))

# Get chain ready
samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

labels = []
for name in v.names:
    if name not in v.included:
        continue
    else:
        labels.append(name)

fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels)
plt.show()
