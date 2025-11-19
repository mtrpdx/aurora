import sys
import os
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import (
    fem,
    default_scalar_type,
    geometry,
    __version__ as dolfinx_version,
)
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx.fem import Function, Constant # FunctionSpace replaced by fem.functionspace
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl
from ufl import dx, grad, inner, Measure
from tqdm import tqdm
from rich.console import Console

from microphone import Microphone

console = Console()

ROOT_DIR = Path("/Users/teen/Projects/project_aurora/")
# Geometry directory. Can be 2d files (binary png, etc) or
# 3d files (step, vtk, etc)
GEO_DIR = os.path.join(ROOT_DIR, "geo/3d")
MESH_DIR = os.path.join(ROOT_DIR, "meshes/3d")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

console.print(f"project root: {ROOT_DIR}", style="blue")
console.print(f"geometry dir: {GEO_DIR}", style="blue")
console.print(f"mesh dir: {MESH_DIR}", style="blue")

# import the gmsh generated mesh
msh, cell, facet_tags = gmshio.read_from_msh(
    os.path.join(MESH_DIR, "air_mesh.msh", MPI.COMM_WORLD, 0, gdim=3)
)

airv_tag = 1
v_bc_tag = 2
Z_bc_tag = 3

# Create discrete frequency domain
freq = np.arange(50, 1000, 5) # Hz

# define trial and test functions
V = fem.functionspace(msh, ("CG", 2))
p = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# microphone location
mic = np.array([0.06, 0.06, 0.03])

# define air characteristics
c = 340 # sound speed in m/s
rho0 = 1.225 # air density in kg/m^3
#
omega = Constant(msh, default_scalar_type(0))
k = Constant(msh, default_scalar_type(0))

# normal velocity boundary condition
v_n = 0.01

# surface impedance
Z_s = rho0 * c

# specify integration measure
ds = Measure("ds", domain=msh, subdomain_data=facet_tags)

# define bilinear and linear forms
a = (
    inner(grad(p), grad(v)) * dx
    - k**2 * inner(p, v) * dx
)
L = inner(1j * omega * rho0 * v_n, v) * ds(v_bc_tag)

p_a = Function(V)
p_a.name = "pressure"

solver = LinearProblem(
    a,
    L,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="helmholtz",
)

def frequency_loop(nf):
    k.value = 2 * np.pi * freq[nf] / c
    omega.value = 2 * np.pi * freq[nf]

    solver.solve()

    # if freq[nf] % 100 == 0:
    #     with XDMFFile(msh.comm, os.path.join(OUTPUT_DIR, "air_solution" + str(freq[nf]) + ".xdmf"), "w") as xdmf:
    #                       xdmf.write_mesh(msh)
    #                       xdmf.write_function(p_a)
    return p_value


from multiprocessing import Pool

nf = range(0, len(freq))

if __name__== '__main__':
    print("Computing...")
    pool = Pool(3)
    p_mic = pool.map(frequency_loop.nf)
    pool.close()
    pool.join()
