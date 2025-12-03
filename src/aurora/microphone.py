"""
This file contains the Microphone class.

Computes the sound pressure at specified location.

Adapted from the MicrophonePressure class written by Antonio Baiano Svizzero
and Jørgen S. Dokken

References:
    - https://jsdokken.com/dolfinx-tutorial/chapter2/helmholtz_code.html
"""

# from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from dolfinx import default_scalar_type, geometry
from dolfinx.fem import Function
from dolfinx.mesh import Mesh


class Microphone:
    """
    Microphone class to compute sound pressure at specified location.

    Attributes
    ----------
    domain : geometry.Mesh
        The domain on which to insert microphones
    microphone_position : npt.ArrayLike
        Position of the microphone(s)
    pressure_function : dolfinx.fem.Function
        FEM function governing sound pressure

    Methods
    -------
    compute_local_microphones() :
        Compute the local microphone positions within a mesh
    listen(recompute_collisions)
        Compute sound pressure at microphone locations
    """

    def __init__(
        self,
        domain: Mesh,
        microphone_position: npt.ArrayLike,
    ):
        """
        Initialize microphone(s).

        Args
        ----
        domain :
            The domain to insert microphones on
        microphone_position :
            Position of the microphone(s). Assumed to be ordered as
            ``(mic0_x, mic1_x, ..., mic0_y, mic1_y, ..., mic0_z, mic1_z, ...)``
        pressure_function: FEM function governing sound pressure. This will be
            evaluated at mic locations
        """
        self._domain = domain
        self._position = np.asarray(
            microphone_position, dtype=self._domain.geometry.x.dtype
        ).reshape(3, -1)
        self._local_cells, self._local_position = self.compute_local_microphones()

    def compute_local_microphones(
        self,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.floating]]:
        """
        Compute the local microphone positions for a distributed mesh.

        Returns
        -------
        (local_cells, local_points) :
            Two lists containing the local cell indices and the local points
        """
        points = self._position.T
        bb_tree = geometry.bb_tree(self._domain, self._domain.topology.dim)

        cells = []
        points_on_proc = []

        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(
            self._domain, cell_candidates, points
        )

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        return np.asarray(cells, dtype=np.int32), np.asarray(
            points_on_proc, dtype=self._domain.geometry.x.dtype
        )

    def listen(
        self,
        pressure_function: Function,
        recompute_collisions: bool = False
    ) -> npt.NDArray[np.complexfloating]:
        """
        Compute sound pressure using pressure function at specified mic locations.

        Args
        ----
        recompute_collisions :
            Bool to determine whether or not to recompute mic collisions

        Returns
        -------
        Array of sound pressure measurements

        """
        if recompute_collisions:
            self._local_cells, self._local_position = self.compute_local_microphones()
        if len(self._local_cells) > 0:
            return pressure_function.eval(self._local_position, self._local_cells)
        else:
            return np.zeros(0, dtype=default_scalar_type)
