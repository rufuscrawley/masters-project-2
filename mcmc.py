import warnings

import corner
import emcee
import keras
from emcee.moves import WalkMove, StretchMove
from scipy import stats

warnings.filterwarnings("ignore",
                        category=FutureWarning,
                        module="arviz")
import numpy as np
from matplotlib import pyplot as plt

import genetic_algorithm as ga
import normalisation as norm
import utilities
import variables as v

gene_spaces = ga.get_gene_spaces()

nn = keras.models.load_model(f"models/{v.filename}_model.keras")


def model(theta):
    """
    Idealistic fit to compare against.
    :param theta: Parameters being changed by the MCMC fit to find uncertainties.
    :return:
    """
    solution = nn.predict(np.array([theta]), verbose=0)[0]
    solution = norm.interpolate_fluxes(solution, wavelengths)
    return solution


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
wavelengths = [0.545, 0.638, 0.797, 1.22,
               1.63, 2.2, 3.6, 4.5,
               5.8, 8.0, 24, 61.1,
               70, 74.8, 89.3, 1300]

x_solutions = [0.05059, 838.2612, 87.487,
               185.33, 19.577, 1.2216,
               9.49196e-5, 4.6972, 4357.21]
x_solutions = norm.normalise_inputs(x_solutions)
x_solutions = np.array(x_solutions)

expected_model = model(x_solutions)

y_fluxes = [0.06907, 0.1348, 0.276, 0.7187,
            0.97, 0.7909, 0.5641, 0.4537,
            0.4334, 0.5358, 1.445, 3.103,
            3.344, 3.392, 3.048, 0.01512]
y_fluxes = utilities.JanskyWavelengths(y_fluxes, wavelengths).convert_to_si()
y_fluxes = norm.normalise_uneven_fluxes(y_fluxes, wavelengths)
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
sampler.run_mcmc(pos, 1_000,
                 progress=True)

# burnout and thinning :)
t_autocorrelation = sampler.get_autocorr_time(quiet=True)
discard = int(2.0 * np.max(t_autocorrelation))
thin = int(0.5 * np.min(t_autocorrelation))

# Get chain ready
samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

labels = []
for name in v.names:
    if name in v.excluded:
        continue
    else:
        labels.append(name)

fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels)
plt.show()
