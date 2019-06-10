#bl_info = {
#	'name': "Avatar",
#	'author': "Jordi Sanchez-Riera",
#	'version': (0, 1, 0),
#	"blender": (2, 7, 9),
#	'location': "View3D",
#	'description': "Create and move a simple avatar",
#	'warning': '',
#	'wiki_url': '',
#	'category': 'Avatar'}

import os
import sys
import bpy
import numpy as np

import mathutils
from bpy.props import *
from mathutils import Vector

# Set a file 'config.py' with variable avt_path that contains the
# path of the script
# need to add the root path in the blender preferences panel
for p in bpy.utils.script_paths():
	sys.path.append(p)
	
from avatar.config import avt_path
#avt_path = '/home/jsanchez/Software/github-projects/avatar'

print(avt_path)

def get_vertices (obj):
	return [(obj.matrix_world * v.co) for v in obj.data.vertices]

def get_faces (obj):
	faces = []
	for f in obj.data.polygons:
		for idx in f.vertices:
			faces.append(obj.data.vertices[idx].co)
	return faces

def update_weights (self, context):
	#obj = context.active_object
	
	global mAvt

	obj = mAvt.mesh
	
	# set previous mesh vertices values
	cp_vals = obj.data.copy()
	# store as np data
	#mAvt.mesh_prev = cp_vals.vertices
	mAvt.np_mesh_prev = mAvt.read_verts(cp_vals)

	# calculate new shape with PCA shapes
	w1 = self.weight_k1/30
	w2 = self.weight_k2/11
	w3 = self.weight_k3/5
	w4 = self.weight_k4/5
	w5 = self.weight_k5/5
	w6 = self.weight_k6/5
	w7 = self.weight_k7/8
	w8 = self.weight_k8/5
	w9 = self.weight_k9/5
	w10 = self.weight_k10/5
	w11 = self.weight_k11/10
	verts = obj.data.vertices
	for i in range(0,len(verts)):
		verts[i].co = Vector((vertexeigen0[i][0]*w1 + vertexeigen1[i][0]*w2 + vertexeigen2[i][0]*w3 + vertexeigen3[i][0]*w4 + vertexeigen4[i][0]*w5+ vertexeigen5[i][0]*w6+ vertexeigen6[i][0]*w7 + vertexeigen7[i][0]*w8 + vertexeigen8[i][0]*w9+ vertexeigen9[i][0]*w10 +vertexeigen10[i][0]*w11  +vertexmean[i][0], vertexeigen0[i][1]*w1+ vertexeigen1[i][1]*w2+ vertexeigen2[i][1]*w3 + vertexeigen3[i][1]*w4 +vertexeigen4[i][1]*w5+ vertexeigen5[i][1]*w6+ vertexeigen6[i][1]*w7 + vertexeigen7[i][1]*w8 + vertexeigen7[i][1]*w9 + vertexeigen9[i][1]*w10 + vertexeigen10[i][1]*w11 +vertexmean[i][1], vertexeigen0[i][2]*w1+ vertexeigen1[i][2]*w2+ vertexeigen2[i][2]*w3 + vertexeigen3[i][2]*w4 + vertexeigen4[i][2]*w5 + vertexeigen5[i][2]*w6+ vertexeigen6[i][2]*w7 + vertexeigen7[i][2]*w8 + vertexeigen8[i][2]*w9+ vertexeigen9[i][2]*w10 + vertexeigen10[i][2]*w11 + vertexmean[i][2]))

	mAvt.np_mesh = mAvt.read_verts(obj.data)
	mAvt.np_mesh_diff = mAvt.np_mesh - mAvt.np_mesh_prev

	# move also collision mesh

	# find which vertices are modified

	# calculate position of clothes if any
	if (mAvt.has_tshirt):
		mAvt.deform_cloth(cloth_name='tshirt')
	if (mAvt.has_pants):
		mAvt.deform_cloth(cloth_name='pants')
	if (mAvt.has_dress):
		mAvt.deform_cloth(cloth_name='dress')
	

class Avatar:
	"""
		Here we store everything needed to run our avatar
	"""
	
	def __init__ (self):
		
		# root_path
		self.root_path = avt_path

		# avt mesh
		self.mesh = None # this points to mesh object
		self.np_mesh = None
		self.np_mesh_prev = None
		self.np_mesh_diff = None
		self.skel = None
		self.mesh_mwi = None
		self.collision_mesh = None
		self.body_kdtree = None
		
		# clohing
		self.has_tshirt = False
		self.has_pants = False
		self.has_dress = False
		
		self.tshirt_mesh = None
		self.pants_mesh = None
		self.dress_mesh = None
		
		# 
		self.use_one_vertex = False
		self.do_once_per_vertex = False
		self.mesh_chosen_vertices = []
		self.number_increments = 20
		self.increment_radius = 0.2

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
		if cloth_name == 'tshirt':
			cloth_mesh = self.tshirt_mesh
			cloth_verts = self.tshirt_mesh.data.vertices
			cloth_mat_world = self.tshirt_mesh.matrix_world
			cloth_mat_world_inv = self.tshirt_mesh.matrix_world.inverted()
		elif cloth_name == 'pants':
			cloth_mesh = self.pants_mesh
			cloth_verts = self.pants_mesh.data.vertices
			cloth_mat_world = self.pants_mesh.matrix_world
			cloth_mat_world_inv = self.pants_mesh.matrix_world.inverted()
		elif cloth_name == 'dress':
			cloth_mesh = self.dress_mesh
			cloth_verts = self.dress_mesh.data.vertices
			cloth_mat_world = self.dress_mesh.matrix_world
			cloth_mat_world_inv = self.dress_mesh.matrix_world.inverted()
		else:
			print("ERROR")
			
		total_vertices = len(cloth_verts)
			
		# all vertices in destination mesh
		for cloth_vertex_index in range(0,total_vertices):
#		for cloth_vertex_index in range(0,1):
			#self.update_vertex() 
			
			# set vertices to empty first
			self.mesh_chosen_vertices = []  

			# Need to pre-compute most of the values to make reshaping cloths faster
			current_vertex2 = cloth_verts[cloth_vertex_index].co * cloth_mat_world_inv 
			current_vertex = cloth_mat_world * cloth_verts[cloth_vertex_index].co    
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
			print("Result")
			print(current_vertex)
			print(result)

			# set vertex position
			cloth_verts[cloth_vertex_index].co = cloth_mesh.matrix_world.inverted() * result
	
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
		
	


mAvt = Avatar()
	

class Avatar_OT_LoadModel(bpy.types.Operator):

	bl_idname = "avt.load_model"
	bl_label = "Load human model"
	bl_description = "Loads a parametric naked human model"

	path_input = "%s/models/Eigenbodies" % avt_path

	global eigenbody0
	eigenbody0 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody0.append(float(line))
	eigenbody0 = np.array(eigenbody0)
	
	global eigenbody1
	eigenbody1 = []
	model_name = "eigenbody1"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody1.append(float(line))
	eigenbody1 = np.array(eigenbody1)
	
	global eigenbody2
	eigenbody2 = []
	model_name = "eigenbody2"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody2.append(float(line))
	eigenbody2 = np.array(eigenbody2)
	
	global eigenbody3
	eigenbody3 = []
	model_name = "eigenbody3"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody3.append(float(line))
	eigenbody3 = np.array(eigenbody3)
	
	global eigenbody4
	eigenbody4 = []
	model_name = "eigenbody4"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody4.append(float(line))
	eigenbody4 = np.array(eigenbody4)

	global eigenbody5
	eigenbody5 = []
	model_name = "eigenbody5"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody5.append(float(line))
	eigenbody5 = np.array(eigenbody5)

	global eigenbody6
	eigenbody6 = []
	model_name = "eigenbody6"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody6.append(float(line))
	eigenbody6 = np.array(eigenbody6)
	
	global eigenbody7
	eigenbody7 = []
	model_name = "eigenbody7"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody7.append(float(line))
	eigenbody7 = np.array(eigenbody7)
	
	global eigenbody8
	eigenbody8 = []
	model_name = "eigenbody8"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody8.append(float(line))
	eigenbody8 = np.array(eigenbody8)
	
	global eigenbody9
	eigenbody9 = []
	model_name = "eigenbody9"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody9.append(float(line))
	eigenbody9 = np.array(eigenbody9)
	
	global eigenbody10
	eigenbody10 = []
	model_name = "eigenbody10"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	
	for line in eigenbody:
		eigenbody10.append(float(line))
	eigenbody10 = np.array(eigenbody10)
	
	

	  
	global mean 
	mean = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean.append(float(line))
	mean = np.array(mean)
	
	bodyeigen0 = eigenbody0-mean
	global vertexeigen0
	vertexeigen0 = []
	for i in range(0,len(bodyeigen0),3):
		vertexeigen0.append([bodyeigen0[i],-bodyeigen0[i+2],bodyeigen0[i+1]])
	
	bodyeigen1 = eigenbody1-mean
	global vertexeigen1
	vertexeigen1 = []
	for i in range(0,len(bodyeigen1),3):
		vertexeigen1.append([bodyeigen1[i],-bodyeigen1[i+2],bodyeigen1[i+1]])

	
	bodyeigen2 = eigenbody2-mean
	global vertexeigen2
	vertexeigen2 = []
	for i in range(0,len(bodyeigen2),3):
		vertexeigen2.append([bodyeigen2[i],-bodyeigen2[i+2],bodyeigen2[i+1]])

	
	bodyeigen3 = eigenbody3-mean
	global vertexeigen3
	vertexeigen3 = []
	for i in range(0,len(bodyeigen3),3):
		vertexeigen3.append([bodyeigen3[i],-bodyeigen3[i+2],bodyeigen3[i+1]])

	
	bodyeigen4 = eigenbody4-mean
	global vertexeigen4
	vertexeigen4 = []
	for i in range(0,len(bodyeigen4),3):
		vertexeigen4.append([bodyeigen4[i],-bodyeigen4[i+2],bodyeigen4[i+1]])
	
	bodyeigen5 = eigenbody5-mean
	global vertexeigen5
	vertexeigen5 = []
	for i in range(0,len(bodyeigen5),3):
		vertexeigen5.append([bodyeigen5[i],-bodyeigen5[i+2],bodyeigen5[i+1]])

	bodyeigen6 = eigenbody6-mean
	global vertexeigen6
	vertexeigen6 = []
	for i in range(0,len(bodyeigen6),3):
		vertexeigen6.append([bodyeigen6[i],-bodyeigen6[i+2],bodyeigen6[i+1]])
		
	bodyeigen7 = eigenbody7-mean
	global vertexeigen7
	vertexeigen7 = []
	for i in range(0,len(bodyeigen7),3):
		vertexeigen7.append([bodyeigen7[i],-bodyeigen7[i+2],bodyeigen7[i+1]])
		
	bodyeigen8 = eigenbody8-mean
	global vertexeigen8
	vertexeigen8 = []
	for i in range(0,len(bodyeigen8),3):
		vertexeigen8.append([bodyeigen8[i],-bodyeigen8[i+2],bodyeigen8[i+1]])
		
	bodyeigen9 = eigenbody9-mean
	global vertexeigen9
	vertexeigen9 = []
	for i in range(0,len(bodyeigen9),3):
		vertexeigen9.append([bodyeigen9[i],-bodyeigen9[i+2],bodyeigen9[i+1]])
		
	bodyeigen10 = eigenbody10-mean
	global vertexeigen10
	vertexeigen10 = []
	for i in range(0,len(bodyeigen10),3):
		vertexeigen10.append([bodyeigen10[i],-bodyeigen10[i+2],bodyeigen10[i+1]])

    
	bodymean = mean

	global vertexmean
	vertexmean = []
	for i in range(0,len(bodymean),3):
		vertexmean.append([bodymean[i],-bodymean[i+2],bodymean[i+1]])


	def execute(self, context):
		global mAvt
		scn = context.scene
		obj = context.active_object
		
		# load makehuman model
		if bpy.data.objects.get("Naked_body") is None:
			#model_file = "%s/models/base_model.mhx2" % avt_path
			model_file = "%s/models/cmu_naked.mhx2" % avt_path
			#model_file = "%s/models/Base.mhx2" % avt_path
			#	bpy.ops.import_scene.fbx(filepath=model_file, axis_forward='Y', axis_up='Z')
			bpy.ops.import_scene.makehuman_mhx2(filepath=model_file)
			
			#mAvt.mesh = bpy.data.objects["Base:Body"]
			mAvt.mesh = bpy.data.objects["Cmu_naked:Body"]
			#mAvt.skel = bpy.data.objects["Cmu_naked"]
			mAvt.mesh_mwi = mAvt.mesh.matrix_world.inverted()

			# save it as kd tree data
			size = len(mAvt.mesh.data.vertices)
			mAvt.body_kdtree = mathutils.kdtree.KDTree(size)
		
			for i, v in enumerate (mAvt.mesh.data.vertices):
				mAvt.body_kdtree.insert(v.co, i)
			
			mAvt.body_kdtree.balance()
			
			
		# set mean according PCA vertices
		

		return {'FINISHED'}


class Avatar_OT_PutTshirt (bpy.types.Operator):
	
	bl_idname = "avt.tshirt"
	bl_label = "Put T-shirt"
	bl_description = "Put T-shirt on human"
	
	def execute(self, context):
		global mAvt
		scn = context.scene
		obj = context.active_object
		
		#
		tshirt_file = "%s/models/tshirt.obj" % avt_path
		bpy.ops.import_scene.obj(filepath=tshirt_file)
		
		# change name to object
		bpy.context.selected_objects[0].name = 'tshirt'
		bpy.context.selected_objects[0].data.name = 'tshirt'
		
		mAvt.tshirt_mesh = bpy.data.objects["tshirt"]
		mAvt.has_tshirt = True
				
		return {'FINISHED'}

class Avatar_OT_PutPants (bpy.types.Operator):
	
	bl_idname = "avt.pants"
	bl_label = "Put Pants"
	bl_description = "Put Pants on human"
	
	def execute(self, context):
		global mAvt
		scn = context.scene
		obj = context.active_object
		
		#
		pants_file = "%s/models/pants.obj" % avt_path
		bpy.ops.import_scene.obj(filepath=pants_file)
		
		mAvt.pants_mesh = bpy.data.objects["Supermanshirt.001"]
		mAvt.has_pants = True

		# save it as kd tree data
		size = len(mAvt.pants_mesh.data.vertices)
		mAvt.kd_pants = mathutils.kdtree.KDTree(size)
		
		for i, v in enumerate (mAvt.pants_mesh.data.vertices):
			mAvt.kd_pants.insert(v.co, i)
			
		mAvt.kd_pants.balance()
					
		return {'FINISHED'}
	
class Avatar_OT_PutDress (bpy.types.Operator):
	
	bl_idname = "avt.dress"
	bl_label = "Put Dress"
	bl_description = "Put Dress on human"
	
	def execute(self, context):
		global mAvt
		scn = context.scene
		obj = context.active_object
		
		#
		dress_file = "%s/models/dress.obj" % avt_path
		bpy.ops.import_scene.obj(filepath=dress_file)
		
		mAvt.dress_mesh = bpy.data.objects["Supermanshirt.001"]
		mAvt.has_dress = True

		# save it as kd tree data
		size = len(mAvt.dress_mesh.data.vertices)
		mAvt.kd_dress = mathutils.kdtree.KDTree(size)
		
		for i, v in enumerate (mAvt.dress_mesh.data.vertices):
			mAvt.kd_dress.insert(v.co, i)

		mAvt.kd_dress.balance()
						
		return {'FINISHED'}


class Avatar_PT_LoadPanel(bpy.types.Panel):

	bl_idname = "Avatar_PT_LoadPanel"
	bl_label = "Load model"
	bl_category = "Avatar"
	bl_space_type = "VIEW_3D"
	bl_region_type = "TOOLS"

	bpy.types.Object.weight_k1 = FloatProperty(name="K1", description="Weight 1", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k2 = FloatProperty(name="K2", description="Weight 2", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k3 = FloatProperty(name="K3", description="Weight 3", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k4 = FloatProperty(name="K4", description="Weight 4", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k5 = FloatProperty(name="K5", description="Weight 5", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k6 = FloatProperty(name="K6", description="Weight 6", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k7 = FloatProperty(name="K7", description="Weight 7", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k8 = FloatProperty(name="K8", description="Weight 8", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k9 = FloatProperty(name="K9", description="Weight 9", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k10 = FloatProperty(name="K10", description="Weight 10", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k11 = FloatProperty(name="K11", description="Weight 11", default=0.0, min=0.0, max=1.0, precision=2, update=update_weights)

	def draw(self, context):
		layout = self.layout
		obj = context.object
		scn = context.scene

		row = layout.row()
		row.operator('avt.load_model', text="Load human")
		layout.separator()
		#layout.prop(obj, "weight_k1", slider=True)		
		layout.prop(obj, "weight_k1")
		layout.prop(obj, "weight_k2")
		layout.prop(obj, "weight_k3")
		layout.prop(obj, "weight_k4")
		layout.prop(obj, "weight_k5")
		layout.prop(obj, "weight_k6")
		layout.prop(obj, "weight_k7")
		layout.prop(obj, "weight_k8")
		layout.prop(obj, "weight_k9")
		layout.prop(obj, "weight_k10")
		layout.prop(obj, "weight_k11")


class Avatar_PT_DressingPanel(bpy.types.Panel):
	
	bl_idname = "Avatar_PT_DressingPanel"
	bl_label = "Dress Human"
	bl_category = "Avatar"
	bl_space_type = "VIEW_3D"
	bl_region_type = "TOOLS"
	
	def draw(self, context):
		layout = self.layout
		obj = context.object
		scn = context.scene
		
		row = layout.row()
		row.operator('avt.tshirt', text="Load T-shirt")
		layout.separator()
		row.operator('avt.pants', text="Load Pants")
		layout.separator()
		row.operator('avt.dress', text="Load Dress")
		layout.separator()

classes  = (
			Avatar_PT_LoadPanel, 
			Avatar_OT_LoadModel, 
			Avatar_PT_DressingPanel,
			Avatar_OT_PutTshirt,
			Avatar_OT_PutPants,
			Avatar_OT_PutDress
)

def register():
	for clas in classes:
		bpy.utils.register_class(clas)

def unregister():
	for clas in classes:
		bpy.utils.unregister_class(clas)


if __name__ == '__main__':
	register()