#
#
# Class avatar
#

import shape_utils

class Avatar:
    """
        Here we store everything needed to run our avatar
    """
    
    def __init__ (self, addon_path= "/"):
        
        # root_path
        self.addon_path = addon_path

        self.body = None

        # 
#        self.use_one_vertex = False 
#        self.do_once_per_vertex = False
#        self.mesh_chosen_vertices = []
#        self.number_increments = 20
#        self.increment_radius = 0.2

        # weights and means to control shape
        self.weights_belly, self.weights_height, self.weights_breast, self.weights_torso = [], [], [], []
        self.weights_armslegs, self.weights_hips, self.weights_gender, self.weights_weight = [], [], [], []
        self.weights_muscle, self.weights_strength = [], []

        self.mean_belly, self.mean_height, self.mean_breast, self.mean_torso = [], [], [], []
        self.mean_armslegs, self.mean_hips, self.mean_gender, self.mean_weight = [], [], [], []
        self.mean_muscle, self.mean_strength = [], []

        self.vertices_belly, self.vertices_height, self.vertices_breast, self.vertices_torso = [], [], [], []
        self.vertices_armslegs, self.vertices_hips, self.vertices_gender, self.vertices_weight = [], [], [], []
        self.vertices_muscle, self.vertices_strength = [], []

        self.mean_model = []
        self.vertices_model = []
        
    def load_shape_model (self):

        # load weights/mean belly
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/belly/eigenbody0.txt" % (self.addon_path)
        self.weights_belly = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/belly/StandardModel.txt" % (self.addon_path)
        self.mean_belly = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_belly - self.mean_belly
        self.vertices_belly = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean height
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/height/eigenbody0.txt" % (self.addon_path)
        self.weights_height = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/height/StandardModel.txt" % (self.addon_path)
        self.mean_height = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_height - self.mean_height
        self.vertices_height = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean breast
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/breast/eigenbody0.txt" % (self.addon_path)
        self.weights_breast = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/breast/StandardModel.txt" % (self.addon_path)
        self.mean_breast = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_breast - self.mean_breast
        self.vertices_breast = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean torso
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/torso/eigenbody0.txt" % (self.addon_path)
        self.weights_torso = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/torso/StandardModel.txt" % (self.addon_path)
        self.mean_torso = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_torso - self.mean_torso
        self.vertices_torso = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean armslegs
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/armslegs/eigenbody0.txt" % (self.addon_path)
        self.weights_armslegs = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/armslegs/StandardModel.txt" % (self.addon_path)
        self.mean_armslegs = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_armslegs - self.mean_armslegs
        self.vertices_armslegs = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean hip
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/hip/eigenbody0.txt" % (self.addon_path)
        self.weights_hip = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/hip/StandardModel.txt" % (self.addon_path)
        self.mean_hip = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_hip - self.mean_hip
        self.vertices_hip = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean gender
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/gender/eigenbody0.txt" % (self.addon_path)
        self.weights_gender = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/gender/StandardModel.txt" % (self.addon_path)
        self.mean_gender = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_gender - self.mean_gender
        self.vertices_gender = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean weight
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/weight/eigenbody0.txt" % (self.addon_path)
        self.weights_weight = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/weight/StandardModel.txt" % (self.addon_path)
        self.mean_weight = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_weight - self.mean_weight
        self.vertices_weight = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean muscle
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/muscle/eigenbody0.txt" % (self.addon_path)
        self.weights_muscle = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/muscle/StandardModel.txt" % (self.addon_path)
        self.mean_muscle = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_muscle - self.mean_muscle
        self.vertices_muscle = shape_utils.compose_vertices_eigenmat(pca_vec)

        # load weights/mean strenght
        file_eigen_weights = "%s/body/PCA/Eigenbodies/parts/strength/eigenbody0.txt" % (self.addon_path)
        self.weights_strength = shape_utils.read_eigenbody(file_eigen_weights)
        file_eigen_mean = "%s/body/PCA/Eigenbodies/parts/strength/StandardModel.txt" % (self.addon_path)
        self.mean_strength = shape_utils.read_eigenbody(file_eigen_mean)
        pca_vec = self.weights_strength - self.mean_strength
        self.vertices_strength = shape_utils.compose_vertices_eigenmat(pca_vec)

        # mean model
        file_eigen_mean = "%s/body/PCA/Eigenbodies/StandardModel.txt" % (self.addon_path)
        self.mean_model = shape_utils.read_eigenbody(file_eigen_mean)
        self.vertices_model = shape_utils.compose_vertices_eigenmat(self.mean_model)

    def config_shape (self):

        verts = self.body.data.vertices
        for i in range(0,len(verts)):
            verts[i].co = Vector((vertexeigen2[i][0]*w3 + vertexeigen3[i][0]*w4 + vertexeigen4[i][0]*w5 +
                                  vertexeigen5[i][0]*w6  + vertexeigen7[i][0]*w8 + vertexeigen8[i][0]*w9 + 
                                  vertexeigen12[i][0]*w13+ vertexmean[i][0], 
                                  vertexeigen2[i][1]*w3 + vertexeigen3[i][1]*w4 + vertexeigen4[i][1]*w5 + 
                                  vertexeigen5[i][1]*w6 + vertexeigen7[i][1]*w8 + vertexeigen8[i][1]*w9 + 
                                  vertexeigen12[i][1]*w13 + vertexmean[i][1], 
                                  vertexeigen2[i][2]*w3 + vertexeigen3[i][2]*w4 + vertexeigen4[i][2]*w5 + 
                                  vertexeigen5[i][2]*w6 + vertexeigen7[i][2]*w8 + vertexeigen8[i][2]*w9 + 
                                  vertexeigen12[i][1]*w13 + vertexmean[i][2]))



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


