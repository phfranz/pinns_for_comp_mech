# -*- coding: utf-8 -*-
import sys, os
import numpy as np

import gmsh

"""
Generate 3D mesh including domain deformation.
"""

def generateSpaceMesh(t, Tmax, lx):
   
    # Parameters
    xc = 0
    yc = 0
    zc = 0
    
    r1 = 1
    r2 = 2

    # Mesh size.
    lcar = 0.15 * r1
    
    model = gmsh.model
    factory = model.occ
    
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal",1)
    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar);
    #gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1);

    model.add("QuarterLame")
    
    # Actually allows for a filled ellipse.
    # small disk
    s1 = factory.addDisk(xc, yc, zc, r1, r1)
    
    # large dist
    s2 = factory.addDisk(xc, yc, zc, r2, r2)
    
    s3 = factory.addRectangle(xc, yc-r2, zc, r2, r2)
    
    s4, ss4 = factory.cut([(2, s2)], [(2, s1)])
    
    factory.intersect(s4, [(2, s3)])
    
    gmsh.model.occ.synchronize()
    
    model.mesh.generate(2)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()
  
    # # Extract coordinates
    # nodes = model.mesh.getNodes(3,-1,True)
    # tags = nodes[0]
    # nn = tags.shape[0]
    # nsd = 3
    # mxyz = nodes[1].reshape(nn,nsd)
    # # Reorder according to tags
    # mxyz = mxyz[tags.argsort()]
    
    # # Extract connectivity
    # elements = model.mesh.getElements(3,-1)
    # ne = elements[1][0].shape[0]
    # nen = 4
    # mien = elements[2][0].reshape(ne,nen) - 1
   
    # #gmsh.write('mesh'+str(t)+'.msh2')
    # gmsh.write('mesh.msh2')
    # gmsh.finalize()
    
    # return mxyz, mien


############################# MESHES PREPARATION ##############################
lx = 6
TMax = 6    

dirname = 'MeshTFiner'
if not os.path.isdir(dirname): os.mkdir(dirname)
os.chdir(dirname)

# Construct slice mesh
# mxyz, mien = generateSpaceMesh(0,TMax,lx)

generateSpaceMesh(0,TMax,lx)

# sp.call([hypermesh2mixd, 'mesh.msh2'])

# ## Shift the mesh.
# mxyz = np.fromfile('mxyz.space', np.dtype('>d')).reshape(-1,3)
# mxyz[:,2] = mxyz[:,2] - 1.0
# mxyz.flatten().tofile('mxyz.space.shifted')

# ## ...and save it as a VTK in the coresponding folder
# sp.call([mixd2vtk, '../mixd2vtk.in.mesh'])
# os.chdir('..')
