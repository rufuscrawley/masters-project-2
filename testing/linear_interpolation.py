from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, interp1d
import numpy as np

x = np.linspace(0, 9, 10)
y = np.random.rand(10)

x_newer = np.linspace(0, 9, 100)

# Option 1: Linear interpolation
f = interp1d(x, y, kind='linear')
y_new = f(x_newer)

# Option 2: Cubic spline (smoother curve)
cs = make_interp_spline(x, y, k=3)
y_newer = cs(x_newer)
