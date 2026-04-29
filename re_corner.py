import emcee
import numpy as np
import variables_early as ve
import variables_late as vl
import corner

sampler = emcee.backends.HDFBackend("MCMC.h5")

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
    idx = vl.unconstrained_indices[n]
    s = vl.denormalise(inputs, vl.mean_x[idx], vl.std_x[idx], vl.logs[idx])
    n_samples.append(s)
n_samples.append(extinctions)
n_samples = np.array(n_samples)
# Pop the poor fits
n_samples = n_samples[4:]

n_samples = n_samples.transpose()

labels, axes_scale = [], []
for name in ve.names:
    if name not in ve.included:
        continue
    if ve.included[name] is not None:
        continue
    else:
        labels.append(name)
        if ve.names[name]:
            axes_scale.append("log")
        else:
            axes_scale.append("linear")
labels.append("extinction")
axes_scale.append("linear")

# Pop poor labels
axes_scale = axes_scale[4:]
labels = labels[4:]

fig = corner.corner(n_samples,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_fmt=".2e",
                    labels=labels,
                    axes_scale=axes_scale,
                    smooth=1)
fig.show()
