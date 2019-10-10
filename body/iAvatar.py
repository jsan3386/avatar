#
#
# Class avatar
#
import bpy
import shape_utils
import numpy as np
from mathutils import Vector

class Avatar:
    """
        Here we store everything needed to run our avatar
    """
    
    def __init__ (self, addon_path= "/"):
        
        # root_path
        self.addon_path = addon_path

        self.list_bones = ["Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LowerBack",
                           "Spine", "Spine1", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LThumb", 
                           "LeftFingerBase", "LeftHandFinger1", "Neck", "Neck1", "Head", "RightShoulder", "RightArm",
                           "RightForeArm", "RightHand", "RThumb", "RightFingerBase", "RightHandFinger1", "RHipJoint",
                           "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]


        self.body = None
        self.skel = None
        self.armature = None
        self.skel_ref = []  # Initial position of skeleton: will contain pose bone matrices for rest position
        self.frame = 1
        self.hips_pos = Vector((0,0,0))
        self.write_timeline = False
        self.start_origin = False
        self.trans = [0, 0, 0] # translation vector to start avatar in origin

        # Parameters needed to deform clothes 
        self.body_kdtree = None
        self.np_mesh = None
        self.np_mesh_diff = None
        self.np_mesh_prev = None
        # 
        self.use_one_vertex = False 
        self.do_once_per_vertex = False
        self.mesh_chosen_vertices = []
        self.number_increments = 20
        self.increment_radius = 0.2

        # weights and means to control shape
        self.val_breast = self.val_torso = self.val_hips = 0.0
        self.val_armslegs = self.val_weight = self.val_muscle = self.val_strength = 0.0

        self.vertices_breast, self.vertices_torso = [], []
        self.vertices_armslegs, self.vertices_hips, self.vertices_weight = [], [], []
        self.vertices_muscle, self.vertices_strength = [], []

        self.vertices_model = []
        
    def load_shape_model (self):

        # mean model
        file_eigen_mean = "%s/body/PCA/Eigenbodies/StandardModel.txt" % (self.addon_path)
        mean_model = shape_utils.read_eigenbody(file_eigen_mean)
        self.vertices_model = shape_utils.compose_vertices_eigenmat(mean_model)

        # load weights/mean breast
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/breast/eigenbody0.txt" % (self.addon_path)
        weights = shape_utils.read_eigenbody(file_eigen_weights)
        pca_vec = weights - mean_model
        self.vertices_breast = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean torso
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/torso/eigenbody0.txt" % (self.addon_path)
        weights = shape_utils.read_eigenbody(file_eigen_weights)
        pca_vec = weights - mean_model
        self.vertices_torso = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean armslegs
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/armslegs/eigenbody0.txt" % (self.addon_path)
        weights = shape_utils.read_eigenbody(file_eigen_weights)
        pca_vec = weights - mean_model
        self.vertices_armslegs = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean hip
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/hip/eigenbody0.txt" % (self.addon_path)
        weights = shape_utils.read_eigenbody(file_eigen_weights)
        pca_vec = weights - mean_model
        self.vertices_hips = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean weight
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/weight/eigenbody0.txt" % (self.addon_path)
        weights = shape_utils.read_eigenbody(file_eigen_weights)
        pca_vec = weights - mean_model
        self.vertices_weight = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean muscle
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/muscle/eigenbody0.txt" % (self.addon_path)
        weights = shape_utils.read_eigenbody(file_eigen_weights)
        pca_vec = weights - mean_model
        self.vertices_muscle = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean strenght
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/strength/eigenbody0.txt" % (self.addon_path)
        weights = shape_utils.read_eigenbody(file_eigen_weights)
        pca_vec = weights - mean_model
        self.vertices_strength = shape_utils.compose_vertices_eigenmat(pca_vec)


    def refresh_shape(self):
        verts = self.body.data.vertices
        for i in range(0,len(verts)):
            verts[i].co = Vector((# X
                                  self.vertices_weight[i][0] * self.val_weight + 
                                  self.vertices_breast[i][0] * self.val_breast + 
                                  self.vertices_armslegs[i][0] * self.val_armslegs + 
                                  self.vertices_hips[i][0] * self.val_hips + 
                                  self.vertices_muscle[i][0] * self.val_muscle + 
                                  self.vertices_strength[i][0] * self.val_strength + 
                                  self.vertices_torso[i][0] * self.val_torso + 
                                  self.vertices_model[i][0],
                                  # Y 
                                  self.vertices_weight[i][1] * self.val_weight + 
                                  self.vertices_breast[i][1] * self.val_breast +
                                  self.vertices_armslegs[i][1] * self.val_armslegs + 
                                  self.vertices_hips[i][1] * self.val_hips + 
                                  self.vertices_muscle[i][1] * self.val_muscle + 
                                  self.vertices_strength[i][1] * self.val_strength + 
                                  self.vertices_torso[i][1] * self.val_torso + 
                                  self.vertices_model[i][1],
                                  # Z 
                                  self.vertices_weight[i][2] * self.val_weight + 
                                  self.vertices_breast[i][2] * self.val_breast + 
                                  self.vertices_armslegs[i][2] * self.val_armslegs + 
                                  self.vertices_hips[i][2] * self.val_hips + 
                                  self.vertices_muscle[i][2] * self.val_muscle + 
                                  self.vertices_strength[i][2] * self.val_strength + 
                                  self.vertices_torso[i][2] * self.val_torso + 
                                  self.vertices_model[i][2]
                                  ))

    def read_verts(self, mesh):
        mverts_co = np.zeros((len(mesh.vertices)*3), dtype=np.float)
        mesh.vertices.foreach_get("co", mverts_co)
        return np.reshape(mverts_co, (len(mesh.vertices), 3))      

    def read_edges(self, mesh):
        fastedges = np.zeros((len(mesh.edges)*2), dtype=np.int) # [0.0, 0.0] * len(mesh.edges)
        mesh.edges.foreach_get("vertices", fastedges)
        return np.reshape(fastedges, (len(mesh.edges), 2))

    def read_norms(self, mesh):
        mverts_no = np.zeros((len(mesh.vertices)*3), dtype=np.float)
        mesh.vertices.foreach_get("normal", mverts_no)
        return np.reshape(mverts_no, (len(mesh.vertices), 3))

        
    def deform_cloth(self, cloth_name):
        
        cloth_verts = None
        
        print("DEFORMING CLOTH")
        print(cloth_name)
        cloth_mesh = bpy.data.objects[cloth_name]
        cloth_verts = cloth_mesh.data.vertices
        cloth_mat_world = bpy.data.objects[cloth_name].matrix_world
        cloth_mat_world_inv  = bpy.data.objects[cloth_name].matrix_world.inverted()
        
        ##···················· SEGUIR AQUIIIIIIIIIIII
            
        total_vertices = len(cloth_verts)
            
        # all vertices in destination mesh
        for cloth_vertex_index in range(0,total_vertices):
#		for cloth_vertex_index in range(0,1):
            #self.update_vertex() 
            
            # set vertices to empty first
            self.mesh_chosen_vertices = []  

            # Need to pre-compute most of the values to make reshaping cloths faster
            current_vertex2 = cloth_verts[cloth_vertex_index].co @ cloth_mat_world_inv 
            current_vertex = cloth_mat_world @ cloth_verts[cloth_vertex_index].co    
            #self.mesh_chosen_vertices = self.select_required_verts(current_vertex,0)
#			print("Vertices found 1")
#			print(self.select_required_verts(current_vertex,0)) 

            # 2 possible versions - radius or n-neighbours
            # kd.find_range() or kd.find_n()
            for (co, index, dist) in self.body_kdtree.find_n(current_vertex2, 3):
            #for (co, index, dist) in self.body_kdtree.find_range(current_vertex, 0.2):
                #print("    ", co, index, dist)
                self.mesh_chosen_vertices.append(index)

#			print("Vertices found 2")
#			print(self.mesh_chosen_vertices)

#			for idx in range(0,len(self.mesh_chosen_vertices)):
#				self.mesh.data.vertices[self.mesh_chosen_vertices[idx]].select = True

#			cloth_verts[0].select = True
            
            # check we find some vertices
            if(len(self.mesh_chosen_vertices) == 0):
                print("Failed to find surrounding vertices")
                return False

#			# update cloth vertex position
#			result_position = Vector()    
#			for v in self.mesh_chosen_vertices:
#				result_position +=  self.mesh_prev[v].co    
#			result_position /= len(self.mesh_chosen_vertices)

#			result_position2 = Vector()
#			for v in self.mesh_chosen_vertices:
#				result_position2 += self.mesh.data.vertices[v].co        
#			result_position2 /= len(self.mesh_chosen_vertices)    
#			result = result_position2 - result_position + current_vertex        

            vals = self.np_mesh_diff[self.mesh_chosen_vertices,:]
#			print("VALUES TEST")
#			print(vals)
            disp = np.mean(vals, axis=0)
#			print(disp)
            result = Vector((disp[0], disp[1], disp[2])) + current_vertex
            #print("Result")
            #print(current_vertex)
            #print(result)

            # set vertex position
            cloth_verts[cloth_vertex_index].co = cloth_mesh.matrix_world.inverted() @ result
    
#			current_vertex = cloth_mat_world * cloth_verts[cloth_vertex_index].co    
#			cloth_verts[cloth_vertex_index].co = cloth_mesh.matrix_world.inverted() * result

    # select required vertices within a radius and return array of indices
    def select_vertices(self, center, radius):            
        src_chosen_vertices = []
        closest_vertex_index = -1
        radius_vec = center + Vector((0, 0, radius))        
        # put selection sphere in local coords.
        lco = self.mesh_mwi * center
        r   = self.mesh_mwi * (radius_vec) - lco
        closest_length = r.length        

        # select verts within radius
        for index, v in enumerate(self.mesh.data.vertices):
            is_selected = (v.co - lco).length <= r.length     
            if(is_selected):
                src_chosen_vertices.append(index)
                if(self.use_one_vertex):
                    if((v.co - lco).length <= closest_length):
                        closest_length = (v.co - lco).length
                        closest_vertex_index = index            

        # update closest vertex
        if(self.use_one_vertex):                
            src_chosen_vertices = []
            if(closest_vertex_index > - 1):
                src_chosen_vertices.append(closest_vertex_index)            

        return src_chosen_vertices

    # this select function initially starts (if level=0) by matching a point in same space as the source mesh and if it cant find similar positioned point we increment search radius   
    def select_required_verts(self, vert, rad, level=0):    
        verts = []
        if(level > self.number_increments):
            return verts 
        verts = self.select_vertices(vert, rad) 
        if(len(verts) == 0):
            return self.select_required_verts(vert, rad + self.increment_radius, level + 1)
        else:
            return verts


