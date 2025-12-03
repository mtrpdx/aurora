import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import numpy.typing as npt
from pathlib import Path
import os
import sys
import triangle as tr
from typing import Optional
import gmsh  # type: ignore
from dolfinx import (
    fem,
    default_scalar_type,
    geometry,
    __version__ as dolfinx_version,
)
from dolfinx.io import gmsh as gmshio


def displayImage(
        img: npt.ArrayLike,
        axes: Optional[str] = 'on',
        grid: Optional[str] = 'on',
):
    """ Displays image using matplotlib.pyplot.imshow.

    Inputs:
    - img : image array, can be grayscale or color
    - axes : 'on' for axes on, 'off' to turn axes off
    - grid : 'on' for grid on, 'off' to turn grid off

    """

    # h, w, chan = img.shape
    # print(f"Height: {h}, Width: {w}, Channels: {chan}")
    if img.ndim == 3:
        h, w, c = img.shape
    elif img.ndim <= 2:
        h, w = img.shape
    print(f"Shape of image array: {np.shape(img)}")
    fig = plt.figure()
    ax = plt.gca()

    if axes == 'on':
        majorLocator = FixedLocator(np.arange(0, h, 50))
        minorLocator = FixedLocator(np.arange(0, h, 10))

        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.set_xlim([0, w])
        ax.set_ylim([h, 0])
        ax.set_ylabel('pixels')

        ax.tick_params(axis='x', which='both', labelrotation=0, bottom=False, labelbottom=False, top=True, labeltop=True)
    else:
        ax.set_axis_off()

    midx = w/2
    midy = h/2
    ax.plot([midx], [midy], 'r.')

    if grid == 'on':
        ax.grid(which='both')

    if img.ndim == 3:
        ax.imshow(
            img,
            interpolation='none',
            vmin=0, vmax=1,
            aspect='equal'
        )
    elif img.ndim <= 2:
        ax.imshow(
            img,
            cmap='gray',
            interpolation='none',
            vmin=0, vmax=1,
            aspect='equal'
        )
    plt.tight_layout()


def segmentsFromVertices(v: npt.ArrayLike):
    """
    Input:
    - v : array of vertices
    Returns:
    - seg : array of segments
    """
    N = np.shape(v)[0]
    i = np.arange(N)
    seg = np.stack([i, i + 1], axis=1) % N
    return seg


def trianglesFromVertices(v: npt.ArrayLike):
    """ Uses triangle to triangulate data given an array of vertices.
    Input:
    - v : array of vertices
    Returns:
    - tri: dict of triangulated data
    """
    seg = segmentsFromVertices(v)
    pslg = dict(vertices=v, segments=seg)
    tri = tr.triangulate(pslg, 'p')
    return tri


def gmsh_rect(model: gmsh.model, name: str, tag: int,
              L: float, W: float, H: float) -> gmsh.model:
    """Create a Gmsh model of a rectangle and tag sub entitites
    from all co-dimensions (peaks, ridges, facets and cells).

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a rectangle mesh added.

    """
    model.add(name)
    model.setCurrent(name)
    # sphere = model.occ.addSphere(0, 0, 0, 1, tag=1)
    rectangle = gmsh.model.occ.addRectangle(
        x=0, y=0, z=H, dx=L, dy=W, tag=tag
    )

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical tag for sphere
    model.add_physical_group(dim=3, tags=[rectangle], tag=tag)

    # Embed all sub-entities from the GMSH model into the sphere and tag
    # them
    for dim in [0, 1]:
        entities = model.getEntities(dim)
        entity_ids = [entity[1] for entity in entities]
        model.mesh.embed(dim, entity_ids, 2, rectangle)
        model.add_physical_group(dim=dim, tags=entity_ids, tag=dim)

    # Generate the mesh
    model.mesh.generate(dim=3)
    return model


def gmsh_box(
        model: gmsh.model, name: str,
        x: float, y: float, z: float,
        L: float, W: float, H: float
) -> gmsh.model:
    """Create a Gmsh model of a box and tag sub entitites
    from all co-dimensions (peaks, ridges, facets and cells).

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a box mesh added.

    """
    model.add(name)
    model.setCurrent(name)
    # sphere = model.occ.addSphere(0, 0, 0, 1, tag=1)
    rectangle = gmsh.model.occ.addBox(
        x=x, y=y, z=z, dx=L, dy=W, dz=H, tag=1
    )

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical tag for box
    model.add_physical_group(dim=3, tags=[box], tag=1)

    # Embed all sub-entities from the GMSH model into the sphere and tag
    # them
    for dim in [0, 1, 2]:
        entities = model.getEntities(dim)
        entity_ids = [entity[1] for entity in entities]
        model.mesh.embed(dim, entity_ids, 3, box)
        model.add_physical_group(dim=dim, tags=entity_ids, tag=dim)

    # Generate the mesh
    model.mesh.generate(dim=3)
    return model


