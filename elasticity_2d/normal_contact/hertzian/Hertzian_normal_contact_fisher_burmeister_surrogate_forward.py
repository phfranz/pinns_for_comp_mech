import deepxde as dde
import numpy as np
import os
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK
import time
from pathlib import Path
import pandas as pd
import matplotlib.tri as tri

from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import QuarterDisc
from utils.geometry.geometry_utils import polar_transformation_2d, calculate_boundary_normals

from utils.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain, calculate_traction_mixed_formulation
from utils.elasticity.elasticity_utils import zero_neumann_x_mixed_formulation, zero_neumann_y_mixed_formulation
from utils.contact_mech.contact_utils import zero_tangential_traction
from utils.elasticity import elasticity_utils
from utils.contact_mech import contact_utils

'''
Creates a surrogate model to solve Hertzian normal contact example taking external pressure as neural network input.

@author: tsahin
'''
#dde.config.real.set_float64()

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
radius = 1
center = [0,0]

Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.025, angle=263, refine_times=100, gmsh_options=gmsh_options)

gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)
batch_size = gmsh_model.mesh.getNodes(2, -1, includeBoundary=True)[2].shape[0]//2

revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]
external_dim_size = 5
borders = [-0.2,-1.0]
geom = GmshGeometry2D(gmsh_model,external_dim_size=external_dim_size, borders=borders, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

# # change global variables in elasticity_utils, they are used for getting the material properties for analytical model
lame = 115.38461538461539
shear = 76.92307692307692
elasticity_utils.lame = lame
elasticity_utils.shear = shear
nu,lame,shear,e_modul = problem_parameters() # with dimensions, will be used for analytical solution
# This will lead to e_modul=200 and nu=0.3

# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom
contact_utils.geom = geom

# how far above the block from ground
distance = 0

# assign local parameters from the current file in contact_utils and elasticity_utils
contact_utils.distance = distance

def calculate_gap_in_normal_direction(x,y,X):
    '''
    Calculates the gap in normal direction
    '''
    # calculate the gap in y direction    
    gap_y = x[:,1:2] + y[:,1:2] + radius + distance

    # calculate the boundary normals
    normals, cond = calculate_boundary_normals(X,geom)

    # Here is the idea to calculate gap_n:
    # gap_n/|n| = gap_y/|ny| --> since n is unit vector |n|=1
    gap_n = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))
    
    return gap_n

def zero_fischer_burmeister(x,y,X):
    '''
    Enforces KKT conditions using Fisher-Burmeister equation
    '''
    # ref https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf
    Tx, Ty, Pn, Tt = calculate_traction_mixed_formulation(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)
    
    a = gn
    b = -Pn
    
    return a + b - tf.sqrt(tf.maximum(a**2+b**2, 1e-9))

def boundary_circle_not_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x[:2] - center, axis=-1), radius) and (x[0]<x_loc_partition)

def boundary_circle_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x[:2] - center, axis=-1), radius) and (x[0]>=x_loc_partition)

# Neumann BC
bc_zero_traction_x = dde.OperatorBC(geom, zero_neumann_x_mixed_formulation, boundary_circle_not_contact)
bc_zero_traction_y = dde.OperatorBC(geom, zero_neumann_y_mixed_formulation, boundary_circle_not_contact)

# Contact BC
bc_zero_fischer_burmeister = dde.OperatorBC(geom, zero_fischer_burmeister, boundary_circle_contact)
bc_zero_tangential_traction = dde.OperatorBC(geom, zero_tangential_traction, boundary_circle_contact)

bcs = [bc_zero_traction_x,bc_zero_traction_y,bc_zero_tangential_traction,bc_zero_fischer_burmeister]


n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain,
    bcs,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    '''
    Hard BCs:
        Dirichlet terms
            u(x=0)=0
        
        Neumann terms:
            sigma_yy(y=0) = ext_traction
            sigma_xy(x=0) = 0
            sigma_xy(y=0) = 0
    
    General formulation to enforce BC hardly:
        N'(x) = g(x) + l(x)*N(x)
    
        where N'(x) is network output before transformation, N(x) is network output after transformation, g(x) Non-homogenous part of the BC and 
            if x is on the boundary
                l(x) = 0 
            else
                l(x) < 0
    
    For instance sigma_yy(y=0) = -ext_traction
        N'(x) = N(x) = sigma_yy
        g(x) = ext_traction
        l(x) = -y
    so
        u' = g(x) + l(x)*N(x)
        sigma_yy = ext_traction + -y*sigma_yy
    '''
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    ext_traction = x[:, 2:3]
    
    #return tf.concat([u*(-x_loc), ext_dips + v*(-y_loc), sigma_xx, sigma_yy, sigma_xy*(x_loc)*(y_loc)], axis=1)
    return tf.concat([u*(-x_loc)/e_modul, v/e_modul, sigma_xx, ext_traction + sigma_yy*(-y_loc),sigma_xy*(x_loc)*(y_loc)], axis=1)

# 2 inputs: x and y, 5 outputs: ux, uy, sigma_xx, sigma_yy and sigma_xy
layer_size = [3] + [75] * 8 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

# weights due to PDE
w_pde_1,w_pde_2,w_pde_3,w_pde_4,w_pde_5 = 1e0,1e0,1e0,1e0,1e0
# weights due to Neumann BC
w_zero_traction_x, w_zero_traction_y = 1e0,1e0
# weights due to Contact BC
w_zero_tangential_traction = 1e0
w_zero_fischer_burmeister = 1e4

loss_weights = [w_pde_1,w_pde_2,w_pde_3,w_pde_4,w_pde_5,w_zero_traction_x,w_zero_traction_y,w_zero_tangential_traction,w_zero_fischer_burmeister]

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=loss_weights)
losshistory, train_state = model.train(epochs=2000, display_every=100, batch_size=batch_size) 

model.compile("L-BFGS-B", loss_weights=loss_weights)
losshistory, train_state = model.train(display_every=200, batch_size=batch_size)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

fem_path = str(Path(__file__).parent.parent.parent)+"/Hertzian_fem/Hertzian_fem_fine_mesh.csv"
df = pd.read_csv(fem_path)
fem_results = df[["Points_0","Points_1","displacement_0","displacement_1","nodal_cauchy_stresses_xyz_0","nodal_cauchy_stresses_xyz_1","nodal_cauchy_stresses_xyz_3"]]
fem_results = fem_results.to_numpy()
node_coords_xy = fem_results[:,0:2]

X = node_coords_xy
x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)
triangles = tri.Triangulation(x, y)

borders=[-0.15,-1.1]
external_dim = np.linspace(borders[0],borders[1],external_dim_size).reshape(-1,1)
node_coords_xy_rp = np.tile(node_coords_xy,(external_dim_size,1))
external_dim_rp = np.repeat(external_dim,node_coords_xy.shape[0],axis=0)
X = np.hstack((node_coords_xy_rp,external_dim_rp))

# predictions
start_time_calc = time.time()
output = model.predict(X)
end_time_calc = time.time()
final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
print(final_time)

u_pred, v_pred = output[:,0], output[:,1]
sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2:3], output[:,3:4], output[:,4:5]
sigma_rr_pred, sigma_theta_pred, sigma_rtheta_pred = polar_transformation_2d(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, X)

file_name = os.path.basename(__file__).split(".")[0]
file_path = os.path.join(os.getcwd(), file_name)

dol_triangles = triangles.triangles
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1]).astype(dol_triangles.dtype)
cell_types = np.ones(dol_triangles.shape[0])*5

### store results for each chunck

# we will store the results for each load case since the mesh is 2D (we don't have the connectivity for 3D) 
chunk_size = int(X.shape[0]/external_dim_size)

for i in range(external_dim_size):
    start = i*chunk_size
    end = (i+1)*chunk_size
    
    u_pred_c = u_pred[start:end]
    v_pred_c = v_pred[start:end]
    
    sigma_xx_pred_c = sigma_xx_pred[start:end]
    sigma_yy_pred_c = sigma_yy_pred[start:end]
    sigma_xy_pred_c = sigma_xy_pred[start:end]
    
    sigma_rr_pred_c = sigma_rr_pred[start:end]
    sigma_theta_pred_c = sigma_theta_pred[start:end]
    sigma_rtheta_pred_c = sigma_rtheta_pred[start:end]
    
    combined_disp_pred_c = tuple(np.vstack((np.array(u_pred_c.tolist()),np.array(v_pred_c.tolist()),np.zeros(u_pred_c.shape[0]))))
    combined_stress_pred_c = tuple(np.vstack((np.array(sigma_xx_pred_c.flatten().tolist()),np.array(sigma_yy_pred_c.flatten().tolist()),np.array(sigma_xy_pred_c.flatten().tolist()))))
    combined_stress_polar_pred_c = tuple(np.vstack((np.array(sigma_rr_pred_c.tolist()),np.array(sigma_theta_pred_c.tolist()),np.array(sigma_rtheta_pred_c.tolist()))))

    file_path_c = file_path + "_" + str(i)
    
    unstructuredGridToVTK(file_path_c, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = {"displacement" : combined_disp_pred_c,
                                               "stress" : combined_stress_pred_c, 
                                               "polar_stress" : combined_stress_polar_pred_c
                                            })