#!/usr/bin/env python3

import gmsh
import os
import sys
from pathlib import Path

ROOT_DIR = Path("/Users/teen/Projects/aurora/")
# Geometry directory. Can be 2d files (binary png, etc) or 3d files (step, vtk, etc)
GEO_DIR = os.path.join(ROOT_DIR, "geo/3d")
MESH_DIR = os.path.join(ROOT_DIR, "mesh/3d")

print(f"project root: {ROOT_DIR}")
print(f"geometry dir: {GEO_DIR}")
print(f"mesh dir: {MESH_DIR}")

gmsh.initialize(sys.argv)

# gmsh.option.setNumber("General.NumThreads", 1)
# Set units to meters, default is mm
# gmsh.option.setString("Geometry.OCCTargetUnit", "M")
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)

gmsh.model.add("air_mesh")

# define geometry
L = 1
W = 0.1
H = W

gmsh.model.occ.addBox(0, 0, 0, L, W, H)

# gmsh.merge(os.path.join(GEO_DIR, "air_volume.step"))
# v = gmsh.model.occ.importShapes(os.path.join(GEO_DIR, "air_volume.step"))

gmsh.model.occ.synchronize()

# set up physical groups
gmsh.model.addPhysicalGroup(3, [1], 1, "air_volume")
gmsh.model.addPhysicalGroup(2, [1], 2, "velocity_BC")
gmsh.model.addPhysicalGroup(2, [2], 3, "impedance_BC")

# gmsh.model.occ.synchronize()

# breakpoint()

gmsh.model.mesh.generate(3)

gmsh.write(os.path.join(MESH_DIR, "air_mesh.msh"))

gmsh.fltk.run()

gmsh.finalize()
