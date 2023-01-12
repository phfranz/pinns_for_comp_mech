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

import sys, os
import subprocess as sp


# 1. Loop function
counter = 1
while counter < 10:
    # 2. Looping the Layers
    layer = 3
    while layer < 4:
        #building table
        
        
    # 3. Looping the Neurons    
        neurons = 10
        while neurons < 11: 
            
             
    #         net = dde.nn.FNN([2] + [layer] * neurons + [1], "tanh", "Glorot normal")
    #         model = dde.Model(data, net)
    
    # # Build and train the model:
    #         model.compile("adam", lr=1e-3, loss_weights=lw)
    #         losshistory, train_state = model.train(epochs=5000)
        
    #     # # Plot/print the results
    #         dde.saveplot(      , train_state, issave=True, isplot=True)
    #      # note time taken and steps needed
        
    #         model.compile("L-BFGS")
    #         losshistory, train_state = model.train()
    #     # Plot/print the results
    #         dde.saveplot(losshistory, train_state, issave=True, isplot=True)
            
    #         postProcess(model)
            
            
    #         # Define some query points on our compuational domain.
    #         # Number of points in each dimension:
    #         x_dim, t_dim = (21, 26)
            
    #         # Bounds of 'x' and 't':
    #         x_min, t_min = (xmin, tmin)
    #         x_max, t_max = (xmax, tmax)
            
    #         # Create tensors:
    #         t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    #         x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
            
    #         xx, tt = np.meshgrid(x, t)
    #         X = np.vstack((np.ravel(xx), np.ravel(tt))).T
            
    #         # Compute and plot the exact solution for these query points
    #         k = 0.1
    #         usol = analytical_solution(xx, tt, k)
    #         plt.scatter(xx,tt,c=usol)
    #         plt.show()
            
    #         # Plot model prediction.
    #         y_pred = model.predict(X).reshape(t_dim, x_dim)
    #         plt.scatter(xx,tt,c=y_pred)
    #         plt.xlabel('x')
    #         plt.ylabel('t')
    #         ax = plt.gca()
    #         ax.set_aspect('equal','box')
    #         #plt.colorbar(cax=ax)
    #         plt.savefig('heatEqPred.pdf')
    #         plt.show()
            os.system('cp heatEq.py ./inputfile.py' )
            
            with open('heatEq.py') as f:
                newText=f.read().replace('NUMBERLAYER', str(layer))
                
            with open('inputfile.py', "w") as f:
                    f.write(newText)
            
            with open('inputfile.py') as f:
                newText=f.read().replace('NUMBERNEURONS', str(neurons))
                
            with open('inputfileNew.py', "w") as f:
                f.write(newText)
            
            sp.run([sys.executable,'inputfileNew.py'])
    
            print('Counter: '+str(counter))
            print('Layer: '+str(layer))
            print('Neuron: '+str(neurons))
    # writing down data for time, loss, other values
    # change column in datasheet
            
            #end of neuron loop
            neurons += 1
        
        #end of layer loop
        layer += 1
    counter += 1