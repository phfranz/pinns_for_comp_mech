#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solves a space-time model problem based on the heat equation.
Follows Section 3.3.1 in Compressible Flow Simulation with Space-Time FE

Created on Wed Nov 17 13:27:23 2021

@author: maxvondanwitz
"""

import matplotlib.pyplot as plt
import numpy as np

import deepxde as dde
from deepxde.backend import tf
import time
import pandas

def pde(x, y):
    """
    Expresses the PDE residual of the heat equation.
    """
    # Diffusion coefficient 
    k = 0.1
    
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - k * dy_xx

# Initial condition
def initial_condition(x):
    """
    Evaluates the initial condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the space coordinate.
    """
    x_s = x[:,0:1]
    return tf.cos(np.pi*x_s)

# Boundary condition
def boundary_condition(x):
    """
    Evaluates the boundary condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the time coordinate.
    """
    x_t = x[:,1:2]
    k = 0.1
    
    return -tf.exp(-k*(np.pi)**2*x_t)


# Analytical solution
def analytical_solution(x, t, k):
    """
    Returns the exact solution of the model problem at a point identified by
    its x- and t-coordinates for given k.

    Parameters
    ----------
    x : x-coordinate
    t : time-coordinate
    k : diffusion coefficient
    """

    return np.exp(-k*np.pi**2*t) * np.cos(np.pi*x)

def postProcess(model):
    '''
    Performs heat equation specific post-processing of a trained model.

    Parameters
    ----------
    X : trained deepxde model

    '''
    import os, sys
    from pathlib import Path
    path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
    sys.path.append(path_utils)
    from exportVtk import meshGeometry, solutionFieldOnMeshToVtk

    geom = model.data.geom

    X, triangles = meshGeometry(geom, numberOfPointsOnBoundary=20)

    temperature = model.predict(X)

    pointData = { "temperature" : temperature.flatten()}

    file_path = os.path.join(os.getcwd(),"heatEquation2D")

    solutionFieldOnMeshToVtk(X, triangles, pointData, file_path)


# Computational domain
xmin = -1.0
xmax = 1.0
tmin = 0.0
tmax = 2.0

spaceDomain = dde.geometry.Interval(xmin, xmax)
timeDomain = dde.geometry.TimeDomain(tmin, tmax)
spaceTimeDomain = dde.geometry.GeometryXTime(spaceDomain, timeDomain)

# Why do we define these functions. TimePDE seems to provide alreaddy a
# boolean that indicates whether a point is on the boundary.
def boundary_space(x, on_boundary):
    return on_boundary

def boundary_initial(x, on_initial):
    return on_initial

# Boundary and initial conditions
bc = dde.DirichletBC(spaceTimeDomain, boundary_condition, boundary_space)
ic = dde.IC(spaceTimeDomain, initial_condition , boundary_initial)

# First guess on some scaling of the individual terms in the loss function
# ToDo: Can we derive a physics-informed scaling of these terms?
lw = [1.0, 100.0, 100.0]

# Define the PDE problem and configurations of the network:
data = dde.data.TimePDE(spaceTimeDomain, pde, [bc, ic], num_domain=250,
                        num_boundary=32, num_initial=16, num_test=254,
                        # auxiliary_var_function=diffusionCoeff
                        )
# # 1. Loop function
# counter = 1
# while counter < 10:
#     # 2. Looping the Layers
#     layer = 10
#     while layer < 11:
#         #building table
        
        
#     # 3. Looping the Neurons    
#         neurons = 10
#         while neurons < 11: 
            
layer = NUMBERLAYER
neurons = NUMBERNEURONS
actifunc = 'swish'

print('Activationfucntion: '+str(actifunc))
print('Replaced File: Layer: '+str(layer))
print('Replaced File: Neuron: '+str(neurons))

net = dde.nn.FNN([2] + [neurons] * layer + [1], actifunc , "Glorot normal")
model = dde.Model(data, net)

# Build and train the model:
model.compile("adam", lr=1e-3, loss_weights=lw)
st=time.time()
losshistory, train_state = model.train(epochs=5000)
et=time.time()
adam_time= et - st
print('Adam Training took', adam_time, 'seconds')
print()
dde.utils.external.dat_to_csv('train.dat','adamtrain.csv',"x")
dde.utils.external.dat_to_csv('test.dat','adamtest.csv',"xy")
dde.utils.external.dat_to_csv('loss.dat','adamloss.csv',"s123456")
# # Plot/print the results
# dde.saveplot(    , train_state, issave=True, isplot=True)
# note time taken and steps needed

model.compile("L-BFGS")
st=time.time()
losshistory, train_state = model.train()
et=time.time()
lbfgs_time= et - st
print('Lbfgs Training took', lbfgs_time, 'seconds')
print()

dde.utils.external.dat_to_csv('train.dat','lbfgstrain.csv',"x")
dde.utils.external.dat_to_csv('test.dat','lbfgstest.csv',"xy")
dde.utils.external.dat_to_csv('loss.dat','lbfgsloss.csv',"s123456")

# Plot/print the results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# postProcess(model)

# Define some query points on our compuational domain.
# Number of points in each dimension:
x_dim, t_dim = (21, 26)

# Bounds of 'x' and 't':
x_min, t_min = (xmin, tmin)
x_max, t_max = (xmax, tmax)

# Create tensors:
t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)

xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T

# Compute and plot the exact solution for these query points
k = 0.1
usol = analytical_solution(xx, tt, k)
plt.scatter(xx, tt, c=usol)
plt.show()

# Plot model prediction.
y_pred = model.predict(X).reshape(t_dim, x_dim)
plt.scatter(xx, tt, c=y_pred)
plt.xlabel('x')
plt.ylabel('t')
ax = plt.gca()
ax.set_aspect('equal', 'box')
#plt.colorbar(cax=ax)
plt.savefig('heatEqPred.pdf')
plt.show()

adamtest=pandas.read_csv('adamtest.csv')
adamtrain=pandas.read_csv('adamtrain.csv')
lbfgsloss=pandas.read_csv('lbfgsloss.csv')
lbfgstest=pandas.read_csv('lbfgstest.csv')
lbfgstrain=pandas.read_csv('lbfgstrain.csv')



steps = lbfgsloss['s'][7]
lossrow = 7
if steps == 6000:
        lossrow = 8
        steps = lbfgsloss['s'][8]
if steps == 7000:
        lossrow = 9
        steps =lbfgsloss['s'][9]
adamtrainloss1 = lbfgsloss['1'][5]
adamtrainloss2 = lbfgsloss['2'][5]
adamtrainloss3 = lbfgsloss['3'][5]
adamtestloss1 = lbfgsloss['4'][5]
adamtestloss2 = lbfgsloss['5'][5]
adamtestloss3 = lbfgsloss['6'][5]
lbfgstrainloss1 = lbfgsloss['1'][lossrow]
lbfgstrainloss2 = lbfgsloss['2'][lossrow]
lbfgstrainloss3 = lbfgsloss['3'][lossrow]
lbfgstestloss1 = lbfgsloss['4'][lossrow]
lbfgstestloss2 = lbfgsloss['5'][lossrow]
lbfgstestloss3 = lbfgsloss['6'][lossrow]

#maybe problem if lbfgs takes more than 1000 steps

with open('documentation.csv','a') as fd:
    fd.write(
        str(actifunc)+','+str(layer)+','+str(neurons)+','
        +str(adam_time)+','+str(lbfgs_time)+','
        +str(steps)+','
        +str(adamtrainloss1)+','+str(adamtrainloss2)+','+str(adamtrainloss3)+','
        +str(adamtestloss1)+','+str(adamtestloss2)+','+str(adamtestloss3)+','
        +str(lbfgstrainloss1)+','+str(lbfgstrainloss2)+','+str(lbfgstrainloss3)+','
        +str(lbfgstestloss1)+','+str(lbfgstestloss2)+','+str(lbfgstestloss3)
        +'\n'
        )

#add specific points of the heat equation to compare with exact solution

print('Layer: '+str(layer))
print('Neuron: '+str(neurons))
