import warnings

import corner
import emcee
import pandas as pd

from fit_targets import FitObject

warnings.filterwarnings("ignore",
                        category=FutureWarning,
                        module="arviz")
import numpy as np

import variables_early as ve
import variables_late as vl


def get_gene_spaces():
    # Set up the gene space for our parameters.
    n_csv = pd.read_csv(ve.n_file)
    arr_gs = []
    for variable in ve.included:
        if ve.included[variable] is not None:
            continue
        arr_gs.append({"low": n_csv[variable].min() * 1.05,
                       "high": n_csv[variable].max() * 0.95})
    # Now apply gene space for extinction (our expected final value)
    arr_gs.append({"low": 0.0,
                   "high": 3.0})
    return arr_gs


# Set up initial variables
gene_spaces = get_gene_spaces()


def model(solution_guess, sol_con, wavelengths):
    """
    Idealistic fit to compare against.
    :param sol_con:
    :param wavelengths:
    :param solution_guess: Parameters being changed by the MCMC fit to find uncertainties.
    :return:
    """

    sol_con[vl.unconstrained_indices] = solution_guess[:-1]
    sol_con[-1] = solution_guess[-1]
    flux_guess = vl.predict_fluxes(sol_con[:-1], True)
    vl.apply_extinction(flux_guess, solution_guess[-1])
    return vl.interpolate_fluxes(flux_guess, wavelengths)


def log_likelihood(solution_guess, fluxes, y_err, wavelengths, sol_con):
    """
    Calculate how well our parameters fit the data.
    :return:
    """
    y_model = model(solution_guess, sol_con, wavelengths)
    if not np.all(np.isfinite(y_model)):
        return -np.inf
    return -0.5 * np.sum(((fluxes - y_model) / y_err) ** 2)


def log_prior(solution_guess):
    for n, parameter in enumerate(solution_guess):
        if gene_spaces[n]["low"] < parameter < gene_spaces[n]["high"]:
            continue
        else:
            return -np.inf
    # If all checks pass:
    return 0.0


def log_probability(solution_guess, fluxes, _y, y_err, wavelengths, sol_con):
    lp = log_prior(solution_guess)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(solution_guess, fluxes, y_err, wavelengths, sol_con)


def run(target: FitObject, initial_guess, n_steps, n_walkers):
    # Set up 0 array with length equivalent to guess
    constrain_arr = np.zeros(len(initial_guess))
    # Fill this 0 array with constrained values from initial guess
    constrain_arr[vl.constraint_index] = initial_guess[vl.constraint_index]

    # Construct "guess" array of guess + extinction
    initial_guess = np.concatenate((initial_guess[vl.unconstrained_indices],
                                    initial_guess[-1:]))

    expected_model = model(initial_guess, constrain_arr, target.wavelengths)

    # Set up the walker
    n_dim = ve.split + 1 - len(vl.constraint_arr)
    print(f"Running sampler over {n_dim} dimensions...")

    pos = initial_guess[initial_guess != 0.0] + (1e-3 * np.random.randn(n_walkers, n_dim))

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability,
                                    args=(expected_model, target.fluxes,
                                          target.flux_err, target.wavelengths,
                                          constrain_arr))
    sampler.run_mcmc(pos, n_steps, progress=True)
    return sampler


def analyse_run(sampler):
    t_autocorrelation = sampler.get_autocorr_time(quiet=True)
    discard = int(2.0 * np.max(t_autocorrelation))
    thin = int(0.5 * np.min(t_autocorrelation))
    samples = sampler.get_chain(discard=discard,
                                thin=thin,
                                flat=True)

    samples, extinctions = samples[:, :-1], samples[:, -1]
    samples = samples.transpose()

    n_samples = []
    for n, inputs in enumerate(samples):
        s = vl.denormalise(inputs, vl.mean_x[n], vl.std_x[n], vl.logs[n])
        n_samples.append(s)

    n_samples = np.array(n_samples)

    n_samples = list(n_samples)
    n_samples.append(extinctions)
    n_samples = np.array(n_samples)
    n_samples = n_samples.transpose()

    labels = []
    for name in ve.names:
        if name not in ve.included:
            continue
        if ve.included[name] is not None:
            continue
        else:
            labels.append(name)
    labels.append("jevans")
    fig = corner.corner(n_samples,
                        quantiles=[0.16, 0.5, 0.84],
                        labels=labels,
                        show_titles=True,
                        title_fmt=".2e",
                        smooth=1)
    ##TODO - dynamic axis scaling
    fig.show()
