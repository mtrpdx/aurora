"""This file's members can be imported to set up project paths.

Available to import
-------------------
PROJECT_DIR : Main project directory. Scripts and setup files are found here.
GEO_DIR : Geometry file directory. Can be 2d files (binary png, etc) or
3d files (step, vtk, etc).
MESH_DIR : Mesh file directory.
OUTPUT_DIR : Output directory for graphs and images.
"""

import os

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), os.pardir)
)
GEO_DIR = os.path.join(PROJECT_DIR, "geo/3d")
MESH_DIR = os.path.join(PROJECT_DIR, "mesh/3d")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
