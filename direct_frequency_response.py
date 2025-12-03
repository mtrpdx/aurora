"""This file runs the direct frequency response and plots the output.
- Define (import) geometry as mesh data
- Define frequency steps
- Define air characteristics
- Define point sources
- Define microphones to measure sound pressure
"""

import dolfinx
from dolfinx import (
    fem,
    default_scalar_type,
    geometry,
    __version__ as dolfinx_version,
)
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx.fem import Function, Constant
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from rich.console import Console
import scifem
from tqdm import tqdm
import ufl
from ufl import dx, grad, inner, Measure, SpatialCoordinate

import sys
import os
from pathlib import Path

from aurora.paths import PROJECT_DIR, GEO_DIR, MESH_DIR, OUTPUT_DIR
from aurora.microphone import Microphone

# Start rich console for colorful output
console = Console()

console.print(f"project root: {PROJECT_DIR}", style="blue")
console.print(f"geometry dir: {GEO_DIR}", style="blue")
console.print(f"mesh dir: {MESH_DIR}", style="blue")
console.print(f"mesh dir: {OUTPUT_DIR}", style="blue")

# =========================================================
# PART 1: SETUP (Static)
# =========================================================

# Import the gmsh generated mesh

mesh_data = gmshio.read_from_msh(
    os.path.join(MESH_DIR, "air_mesh.msh"), MPI.COMM_WORLD, 0, gdim=3
)
domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

airv_tag = 1
v_bc_tag = 2
Z_bc_tag = 3

# Create discrete frequency domain
freq = np.arange(10, 1000, 5)  # Hz

# define physical parameters
c = 340  # sound speed in m/s
rho0 = 1.225  # air density in kg/m^3

# Initialize constants
omega = Constant(domain, default_scalar_type(0))
k = Constant(domain, default_scalar_type(0))

# normal velocity boundary condition
# v_n = 1e-5  # this gets replaced by point source

# define function space (complex, degree 1)
V = fem.functionspace(domain, ("Lagrange", 1))

# create standard variational form
p = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = inner(grad(p), grad(v)) * dx - k**2 * inner(p, v) * dx
L = -1j * omega * rho0 * inner(Constant(domain, default_scalar_type(0)), v) * dx

# Compile forms once outside loop
bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Assemble the standard part (L=0)
b = fem.petsc.create_vector(fem.extract_function_spaces(linear_form))

# Create function to solve
p_a = Function(V)
p_a.name = "pressure"

# define point source
gamma = 1.0  # amplitude
# If using MPI, only one process gets populated array, rest get empty array
if domain.comm.rank == 0:
    points = np.array([0.85, 0.05, 0.05], dtype=domain.geometry.x.dtype)
else:
    points = np.zeros((0, 3), dtype=domain.geometry.x.dtype)

ps = scifem.PointSource(V, points, magnitude=gamma)

# Initialize microphone
p_mic = np.zeros((len(freq), 1), dtype=complex)
mic = np.array([0.15, 0.05, 0.05])  # microphone location
microphone = Microphone(domain, mic)

if domain.comm.rank == 0:
    coords = domain.geometry.x
    print(f"Mesh X range: {np.min(coords[:, 0])} to {np.max(coords[:, 0])}")
    print(f"Mesh Y range: {np.min(coords[:, 1])} to {np.max(coords[:, 1])}")
    print(f"Mesh Z range: {np.min(coords[:, 2])} to {np.max(coords[:, 2])}")

# Solver setup
ksp = PETSc.KSP().create(domain.comm)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)
ksp.getPC().setFactorSolverType("mumps")


# =========================================================
# PART 2: FREQUENCY LOOP
# =========================================================

# Frequency loop
pbar = tqdm(range(0, len(freq)))
for nf in pbar:
    pbar.set_description("Computing frequency %s Hz..." % str(freq[nf]))
    k.value = 2 * np.pi * freq[nf] / c
    omega.value = 2 * np.pi * freq[nf]
    # Omitting impedance implies 'rigid walls'

    # Assemble A within loop because k, omega change
    A = fem.petsc.assemble_matrix(bilinear_form)
    A.assemble()

    # Reassemble b (Reset and apply source)
    with b.localForm() as loc_b:
        loc_b.set(0)

    b = fem.petsc.assemble_vector(b, linear_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # Apply point source
    ps.apply_to_vector(b)
    # Finalize b
    # b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignores
    b.assemble()

    ksp.setOperators(A)
    ksp.solve(b, p_a.x.petsc_vec)
    # solver.u.x.scatter_forward()
    p_a.x.scatter_forward()

    p_f_local = microphone.listen(p_a)
    p_f = domain.comm.gather(p_f_local, root=0)

    if domain.comm.rank == 0:
        assert p_f is not None
        p_mic[nf] = np.hstack(p_f)

if domain.comm.rank == 0:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(25, 8))
    plt.plot(freq, 20 * np.log10(np.abs(p_mic) / np.sqrt(2) / 2e-5), linewidth=2)
    plt.grid(True)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPL [dB]")
    plt.xlim([freq[0], freq[-1]])
    plt.ylim([0, 180])
    plt.legend()
    plt.show()

# from multiprocessing import Pool

# nf = range(0, len(freq))

# if __name__== '__main__':
#     print("Computing...")
#     pool = Pool(3)
#     p_mic = pool.map(frequency_loop.nf)
#     pool.close()
#     pool.join()
