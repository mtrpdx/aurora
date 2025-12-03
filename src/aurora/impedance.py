"""
This file contains the impedance base class and related impedance models.


References:
    - https://jsdokken.com/dolfinx-tutorial/chapter2/helmholtz_code.html
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
import numpy.typing as npt


class Impedance(ABC):
    """Impedance base class."""

    def __init__(
        self,
        domain: Mesh,
        function: Any,
        args: dict
    ):
        """
        Initialize  impedance model.
        """
        self._domain = domain,
        self._function = function,
        self._args = args

    # def assign_function(
    #     self,
    #     function
    # ):
    #     self._function = function

    @abstractmethod
    def compute_impedance(
        self,
        function,
        args
    ):
        pass

class DelanyBazley(Impedance):

    def __init__(
        self,
        domain: Mesh,
        function: Any,
        args: dict
    ):
        super().__init__(domain, function, args)


    sigma = 1.5e4
    d = 0.01

    def __delany_bazley_layer(f, rho0, c, sigma, d):
        """Impedance calculation."""
        X = rho0 * f / sigma
        Zc = rho0 * c * (1 + 0.0571 * X**-0.754 - 1j * 0.087 * X**-0.732)
        kc = 2 * np.pi * f / c * (1 + 0.0978 * (X**-0.700) - 1j * 0.189 * (X**-0.595))
        Z_s = -1j * Zc * (1 / np.tan(kc * d))
        return Z_s

    def compute_impedance(
        self,
        function,
        args
    ):
        function(args[0],)


# Z_s = delany_bazley_layer(freq, rho0, c, sigma, d)
# Z = Constant(domain, default_scalar_type(0))
# Z.value = Z_s[nf]
