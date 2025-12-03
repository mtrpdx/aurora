import numpy as np


# Impedance calculation
def delany_bazley_layer(f, rho0, c, sigma, d):
    X = rho0 * f / sigma
    Zc = rho0 * c * (1 + 0.0571 * X**-0.754 - 1j * 0.087 * X**-0.732)
    kc = 2 * np.pi * f / c * (1 + 0.0978 * (X**-0.700) - 1j * 0.189 * (X**-0.595))
    Z_s = -1j * Zc * (1 / np.tan(kc * d))
    return Z_s
