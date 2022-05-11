import gmsh
import sys

class QuarterCirclewithHole(object):
    def __init__(self, center, inner_radius, outer_radius, mesh_size=0.15, gmsh_options=None):
        self.center = center
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options

    def generateGmshModel(self, visualize_mesh=False):
        '''
        Generates the quarter of a circle including a hole.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns 
        -------
        gmsh_model: Object
            gmsh model 
        '''
        # Parameters
        xc = self.center[0]
        yc = self.center[1]
        zc = self.center[2]
        r1 = self.inner_radius
        r2 = self.outer_radius

        # Mesh size.
        lcar = self.mesh_size * r1

        # create gmsh model instance
        gmsh_model = gmsh.model
        factory = gmsh_model.occ

        # initialize gmsh
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == 'str':
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)
        
        #gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1);

        gmsh_model.add("QuarterCirclewithHole")

        # Actually allows for a filled ellipse.
        # create the small disk
        s1 = factory.addDisk(xc, yc, zc, r1, r1)
        # create the large disk
        s2 = factory.addDisk(xc, yc, zc, r2, r2)
        # create the rectangle
        s3 = factory.addRectangle(xc, yc, zc, r2, r2)
        # substract the small disk from the large one
        s4, ss4 = factory.cut([(2, s2)], [(2, s1)])
        # intersect it with the rectangle
        factory.intersect(s4, [(2, s3)])

        gmsh_model.occ.synchronize()

        # generate mesh
        gmsh_model.mesh.generate(2)

        if visualize_mesh:
            if '-nopopup' not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model

class CirclewithHole(object):
    def __init__(self, center, inner_radius, outer_radius, mesh_size=0.15, gmsh_options=None):
        self.center = center
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options

    def generateGmshModel(self, visualize_mesh=False):
        '''
        Generates a circle including a hole.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns 
        -------
        gmsh_model: Object
            gmsh model 
        '''

        # Parameters
        xc = self.center[0]
        yc = self.center[1]
        zc = self.center[2]
        r1 = self.inner_radius
        r2 = self.outer_radius

        # Mesh size.
        lcar = self.mesh_size * r1

        # create gmsh model instance
        gmsh_model = gmsh.model
        factory = gmsh_model.occ

        # initialize gmsh
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == 'str':
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)
        
        #gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1);

        gmsh_model.add("CirclewithHole")

        # Actually allows for a filled ellipse.
        # create the small disk
        s1 = factory.addDisk(xc, yc, zc, r1, r1)
        # create the large disk
        s2 = factory.addDisk(xc, yc, zc, r2, r2)
        # substract the small disk from the large one
        factory.cut([(2, s2)], [(2, s1)])
        # intersect it with the rectangle

        gmsh_model.occ.synchronize()

        # generate mesh
        gmsh_model.mesh.generate(2)

        if visualize_mesh:
            if '-nopopup' not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model

class Block_2D(object):
    def __init__(self, coord_left_corner, coord_right_corner, mesh_size=0.15, gmsh_options=None):
        self.coord_left_corner = coord_left_corner
        self.coord_right_corner = coord_right_corner
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options

    def generateGmshModel(self, visualize_mesh=False):
        '''
        Generates a circle including a hole.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns 
        -------
        gmsh_model: Object
            gmsh model 
        '''

        # Parameters
        x0 = self.coord_left_corner[0]
        y0 = self.coord_left_corner[1]
        x1 = self.coord_right_corner[0]
        y1 = self.coord_right_corner[1]
        assert(x1>x0)
        assert(y1>y0)
        l = x1 - x0
        h = y1 - y0 
        # Mesh size.
        lcar = self.mesh_size * min(h,l)

        # create gmsh model instance
        gmsh_model = gmsh.model
        factory = gmsh_model.occ

        # initialize gmsh
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == 'str':
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)
        
        #gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1);

        gmsh_model.add("Rectangle")

        factory.addRectangle(x0, y0, 0, l, h)

        gmsh_model.occ.synchronize()

        # generate mesh
        gmsh_model.mesh.generate(2)

        if visualize_mesh:
            if '-nopopup' not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model

class Block_3D(object):
    def __init__(self, coord_left_corner, coord_right_corner, mesh_size=0.15, gmsh_options=None):
        self.coord_left_corner = coord_left_corner
        self.coord_right_corner = coord_right_corner
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options

    def generateGmshModel(self, visualize_mesh=False):
        '''
        Generates a circle including a hole.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns 
        -------
        gmsh_model: Object
            gmsh model 
        '''

        # Parameters
        x0 = self.coord_left_corner[0]
        y0 = self.coord_left_corner[1]
        z0 = self.coord_left_corner[2]
        x1 = self.coord_right_corner[0]
        y1 = self.coord_right_corner[1]
        z1 = self.coord_right_corner[2]
        assert(x1>x0)
        assert(y1>y0)
        assert(z1>z0)
        l = x1 - x0
        h = y1 - y0
        w = z1 - z0 
        # Mesh size.
        lcar = self.mesh_size * min(h,l,w)

        # create gmsh model instance
        gmsh_model = gmsh.model
        factory = gmsh_model.occ

        # initialize gmsh
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == 'str':
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)
        
        #gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1);

        gmsh_model.add("Box")

        factory.addBox(x0, y0, z0, l, h, w)

        gmsh_model.occ.synchronize()

        # generate mesh
        gmsh_model.mesh.generate(3)

        if visualize_mesh:
            if '-nopopup' not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model
        