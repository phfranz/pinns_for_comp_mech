from numpy import linalg as la

from scipy.interpolate import griddata
from scipy.integrate import dblquad

import numpy as np
import pandas as pd
from pathlib import Path

"""
Compare L2 domain integral norm with l2 vector norm of points
for Hertzian normal contact example inluding simulation data from BACI.

@author: danwitz based on Hertzian_*_data.py
"""


# The difference is quantified using the relative error.
def relativeError(L2, l2):
    return abs(L2 - l2) / L2


### Check 1
# As a dummy check that scipy.integrate works as expected, let's compute the
# area of a quarter circel.
area = np.pi / 4.0


def unit_integrand(x, y):
    return 1


areaInt, errorbound = dblquad(
    unit_integrand,
    -1.0,
    0.0,
    lambda x: -np.sqrt(1.0 - x**2),
    0.0,
    epsabs=1e-3,
)
print("Area computed with scipy.integrate dblquad:", areaInt)
print("Relative error:", relativeError(area, areaInt))

### Check 2
# We look at the displacement in y-direction computed with BACI,
# represented by the values at the FEM nodes.
# load external data
fem_path = (
    str(Path(__file__).parent.parent.parent)
    + "/Hertzian_fem/Hertzian_fem_fine_mesh.csv"
)

df = pd.read_csv(fem_path)
fem_results = df[
    [
        "Points_0",
        "Points_1",
        "displacement_0",
        "displacement_1",
        "nodal_cauchy_stresses_xyz_0",
        "nodal_cauchy_stresses_xyz_1",
        "nodal_cauchy_stresses_xyz_3",
    ]
]
fem_results = fem_results.to_numpy()
node_coords_xy = fem_results[:, 0:2]
displacement_fem = fem_results[:, 2:4]
stress_fem = fem_results[:, 4:7]

# What we actualy need:
points = node_coords_xy
disp_y = fem_results[:, 3:4]


# Here, we integrate the plain value, result can be easily verified with
# Paraview Filters Integrate.
def test_integrand(x, y):
    return griddata(points, disp_y, (x, y), method="linear")[0]


testInteg, errorbound = dblquad(
    test_integrand,
    -1.0,
    0.0,
    lambda x: -np.sqrt(1.0 - x**2),
    0.0,
    epsabs=1e-3,
)

print("disp_y integrated:", testInteg)


### Check 3
# For the L2-norm, we square the quantity.
disp_y_Squared = disp_y * disp_y


def integrand(x, y):
    grid_z1 = griddata(points, disp_y_Squared, (x, y), method="linear")[0]
    return grid_z1


disp_y_Squared_int, errorbound = dblquad(
    integrand,
    -1.0,
    0.0,
    lambda x: -np.sqrt(1.0 - x**2),
    0.0,
    epsabs=1e-3,
)


print("disp_y_Squared integrated:", disp_y_Squared_int)

L2 = np.sqrt(disp_y_Squared_int)
print("L2-norm: (taking the sqrt)", L2)
# print("Relative error:", relativeError(L2, l2))
print("Upper bound for integration error reported by nquad()", errorbound, ".")

# To compare with the domain integral, we scale with
# sqrt(area/N)
l2 = la.norm(np.array(disp_y, dtype=float)) * np.sqrt(area / disp_y.size)
print("l2-vector norm:", l2)


print("Relative error:", relativeError(L2, l2))
