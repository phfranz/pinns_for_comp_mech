import deepxde as dde
import numpy as np
import os
from deepxde.backend import torch
from pyevtk.hl import unstructuredGridToVTK
import time
from pathlib import Path
import matplotlib.pyplot as plt
from deepxde.icbc.boundary_conditions import npfunc_range_autocache
from deepxde import utils as deepxde_utils
from deepxde import backend as bkd

from utils.geometry.gmsh_models import Block_2D
from utils.geometry.custom_geometry import GmshGeometryElement

from utils.vpinns.quad_rule import GaussQuadratureRule
from utils.vpinns.quad_rule import get_test_function_properties

from utils.vpinns.v_pde import VariationalPDE
import gmsh

'''
Solving 1D advection-diffusion equation via VPINNS

author: @phfranz, Jan' 24

'''

def u_exact(x):
    a = 1
    k = 0.1

    return -np.sin(np.pi*(x[:,0:1]-a*x[:,1:2]))*np.exp(-k*np.pi**2*x[:,1:2])

def boundary_condition(x):

    """
    Evaluates the boundary condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the time and space coordinates x_t and x_s.
    """
    k = 0.1
    a = 1

    return -np.sin(np.pi*(x[:,0:1]-a*x[:,1:2]))*np.exp(-k*np.pi**2*x[:,1:2])

def initial_condition(x):
    """
    Evaluates the initial condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the spatial coordinate x_s.
        
    """   
    return -np.sin(np.pi*x[:,0:1])


# Define GMSH and geometry parameters (quasi-rectangular mesh)
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
coord_left_corner=[-1,0]
coord_right_corner=[1,2]

# create a block
block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.1, gmsh_options=gmsh_options)

quad_rule = GaussQuadratureRule(rule_name="gauss_labotto", dimension=2, ngp=3) # gauss_legendre gauss_labotto, 3D not possible yet
coord_quadrature, weight_quadrature = quad_rule.generate() # in parameter space 

n_test_func = 10
test_function, test_function_derivative = get_test_function_properties(n_test_func, coord_quadrature, approach="2") #modified legendre polynoms, zero on boundary (weak form!), reinforce Neumann boundary conditions
# approach: 2: 

# generate gmsh model
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]

geom = GmshGeometryElement(gmsh_model, 
                           dimension=2,
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature=weight_quadrature, 
                           test_function=test_function, 
                           test_function_derivative=test_function_derivative, 
                           n_test_func=n_test_func,
                           revert_curve_list=revert_curve_list, 
                           revert_normal_dir_list=revert_normal_dir_list)


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    # Use tf.sin for backend tensorflow.compat.v1 or tensorflow
    x = deepxde_utils.to_numpy(x) #convert to numpy
    
    return -dy_xx  #- np.pi ** 2 * torch.sin(np.pi * x)

def weak_form(inputs, outputs, beg, n_e, n_gp, g_jacobian, g_weights, g_test_function, g_test_function_derivative):
    du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)[beg:] #where points of domain start(no boundary points considered)
    du_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)[beg:]
    
    vx_x = g_test_function_derivative[:,0:1]
    vy_y = g_test_function_derivative[:,1:2]
    
    vx = g_test_function[:,0:1]
    vy = g_test_function[:,1:2]

    k = 0.1
    a = 1
    
    du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)[beg:]
    #du_xx = dde.grad.jacobian(du_x, inputs, i=0, j=0)
    du_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)[beg:]
    du_xx = dde.grad.hessian(outputs, inputs, i=0, j=0)[beg:] #specify component for vector fields
    
    # considers Jacobian for space transformation and weights for numerical integration
    weighted_residual = -g_weights[:,0:1]*g_weights[:,1:2]*(du_t+a*du_x-k*du_xx)*vx_x*vy_y*g_jacobian # what grid do we have??
    return bkd.reshape(weighted_residual, (n_e, n_gp))

def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], -1) 

def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1) 

def boundary_initial(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0)

bc_l = dde.icbc.DirichletBC(geom, boundary_condition, boundary_l, component=0)
bc_r = dde.icbc.DirichletBC(geom, boundary_condition, boundary_r, component=0)
ic = dde.icbc.DirichletBC(geom, initial_condition, boundary_initial, component=0)

n_dummy = 1
weak = True

if weak:
    data = VariationalPDE(geom, 
                        weak_form, 
                        [bc_l,bc_r, ic], 
                        num_domain=n_dummy, 
                        num_boundary=n_dummy, 
                        solution=u_exact
                        )
else:
    data = dde.data.PDE(geom, 
                        pde, 
                        bc,
                        num_domain=n_dummy, 
                        num_boundary=n_dummy, 
                        solution=u_exact, 
                        num_test=n_dummy)

layer_size = [2] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

def mean_squared_error(y_true, y_pred):
    return bkd.mean(bkd.square(y_true - y_pred), dim=0)

model.compile("adam", lr=0.001, loss=mean_squared_error, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

model.compile("L-BFGS", loss=mean_squared_error)
losshistory, train_state = model.train(display_every=200)

################ Post-processing ################
gmsh.clear()
gmsh.finalize()

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
coord_left_corner=[-1,0]
coord_right_corner=[1,2]

# create a block
block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)

# generate gmsh model
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)
geom = GmshGeometryElement(gmsh_model, dimension=2, only_get_mesh=True)

X, offset, cell_types, elements = geom.get_mesh()

u_pred = model.predict(X)
u_act = u_exact(X)
error = np.abs(u_pred - u_act)

combined_disp_pred = tuple(np.vstack((u_pred.flatten(), np.zeros(u_pred.shape[0]), np.zeros(u_pred.shape[0]))))
combined_disp_act = tuple(np.vstack((u_act.flatten(), np.zeros(u_act.shape[0]), np.zeros(u_act.shape[0]))))
combined_error = tuple(np.vstack((error.flatten(), np.zeros(error.shape[0]), np.zeros(error.shape[0]))))


file_path = os.path.join(os.getcwd(), "1D_advection_diffusion")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros_like(y)

unstructuredGridToVTK(file_path, x, y, z, elements.flatten(), offset, 
                      cell_types, pointData = {"disp_pred" : combined_disp_pred, 
                                               "disp_act" : combined_disp_act,
                                               "error": combined_error})