import warnings

import corner
import emcee

warnings.filterwarnings("ignore",
                        category=FutureWarning,
                        module="arviz")
import numpy as np
from matplotlib import pyplot as plt

import genetic_algorithm as ga
import utilities
import variables as v
import network_variables as nv

# Set up initial variables
gene_spaces = ga.get_gene_spaces()


def model(solution_guess, wavelengths):
    """
    Idealistic fit to compare against.
    :param solution_guess: Parameters being changed by the MCMC fit to find uncertainties.
    :return:
    """
    flux_guess = nv.predict_fluxes(solution_guess)
    flux_guess = utilities.interpolate_fluxes(flux_guess, wavelengths)
    # Returns 100 fluxes.
    return flux_guess


def log_likelihood(solution_guess, fluxes, y_err, wavelengths):
    """
    Calculate how well our parameters fit the data.
    :return:
    """
    y_model = model(solution_guess, wavelengths)
    if not np.all(np.isfinite(y_model)):
        return -np.inf
    return -0.5 * np.sum(((fluxes - y_model) / y_err) ** 2)


def log_prior(solution_guess):
    for n, parameter in enumerate(solution_guess):
        if gene_spaces[n]["low"] < parameter < gene_spaces[n]["high"]:
            # Continue iterating
            continue
        else:
            return -np.inf
    # If all checks pass:
    return 0.0


def log_probability(solution_guess, fluxes, _y, y_err, wavelengths):
    lp = log_prior(solution_guess)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(solution_guess, fluxes, y_err, wavelengths)


def run(parameters, initial_guess, n_steps, n_walkers):
    # First - take in the solutions from GA, predict a model from them
    wavelengths, y_fluxes = parameters
    expected_model = model(initial_guess, wavelengths)
    y_fluxes = np.array(y_fluxes)
    y_errs = y_fluxes * .1

    # Third - set up the walker
    print("Running sampler...")
    n_walkers *= v.split
    n_dim = v.split
    pos = initial_guess + (1e-4 * np.random.randn(n_walkers, n_dim))
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability,
                                    args=(expected_model, y_fluxes,
                                          y_errs, wavelengths))
    sampler.run_mcmc(pos,
                     n_steps,
                     progress=True)

    return sampler


def analyse_run(sampler):
    t_autocorrelation = sampler.get_autocorr_time(quiet=True)
    discard = int(2.0 * np.max(t_autocorrelation))
    thin = int(0.5 * np.min(t_autocorrelation))
    samples = sampler.get_chain(discard=discard,
                                thin=thin,
                                flat=True)
    samples = samples.transpose()
    final_samples = []
    for n, sample in enumerate(samples):
        final_samples.append(nv.denormalise(samples, nv.x_consts[n], nv.logs[n]))
    final_samples = np.array(final_samples).transpose()
    # Fifth - do corner plots.
    labels = []
    for name in v.names:
        if name not in v.included:
            continue
        else:
            labels.append(name)
    fig = corner.corner(final_samples,
                        quantiles=[0.16, 0.5, 0.84],
                        labels=labels,
                        show_titles=True,
                        title_fmt=".2e")
    plt.show()
