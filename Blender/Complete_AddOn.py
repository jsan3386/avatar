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
import bpy
import numpy as np

from bpy.props import *
from mathutils import Vector

def get_vertices (obj):
	return [(obj.matrix_world * v.co) for v in obj.data.vertices]

def get_faces (obj):
	faces = []
	for f in obj.data.polygons:
		for idx in f.vertices:
			faces.append(obj.data.vertices[idx].co)
	return faces

def update_weights (self, context):
	obj = context.active_object
#	faces = get_faces(obj)
#	verts = get_vertices(obj)

	#w1 = self.weight_k1 
	#w2 = self.weight_k2
	w3 = self.weight_k3
	w4 = self.weight_k4
	w5 = self.weight_k5
	w6 = - self.weight_k6
	#w7 = self.weight_k7
	w8 = - self.weight_k8
	w9 = self.weight_k9
	w11 = - self.weight_k11
	w13 = self.weight_k13

	verts = obj.data.vertices
	for i in range(0,len(verts)):
		verts[i].co = Vector((vertexeigen2[i][0]*w3 + vertexeigen3[i][0]*w4 + vertexeigen4[i][0]*w5 + vertexeigen5[i][0]*w6  + vertexeigen7[i][0]*w8 + vertexeigen8[i][0]*w9 + vertexeigen12[i][0]*w13+ vertexmean[i][0], vertexeigen2[i][1]*w3 + vertexeigen3[i][1]*w4 + vertexeigen4[i][1]*w5 + vertexeigen5[i][1]*w6 + vertexeigen7[i][1]*w8 + vertexeigen8[i][1]*w9 + vertexeigen12[i][1]*w13 + vertexmean[i][1], vertexeigen2[i][2]*w3 + vertexeigen3[i][2]*w4 + vertexeigen4[i][2]*w5 + vertexeigen5[i][2]*w6 + vertexeigen7[i][2]*w8 + vertexeigen8[i][2]*w9 + vertexeigen12[i][1]*w13 + vertexmean[i][2]))
	# calculate new shape with PCA shapes
#	print(faces)
#	print(verts)

def update_scale (self,context):
	a = bpy.data.objects['Standard']
	w10 = self.weight_k10
	
	vector_scale = Vector((w10,w10,w10))
	a.scale = vector_scale
	w11 = self.weight_k11
	vector_ext = Vector((w11,w11,w11))
	a = bpy.data.objects['Standard']
	a.pose.bones['RightArm'].scale = vector_ext
	a.pose.bones['LeftArm'].scale = vector_ext
	a.pose.bones['RightUpLeg'].scale = vector_ext
	a.pose.bones['LeftUpLeg'].scale = vector_ext
	w12 = self.weight_k12
	vector_tors = Vector((w12,w12,w12))
	a.pose.bones['Neck'].scale = vector_tors
	
	# sino self.scale = vector_scale pero no crec



class Avatar_OT_LoadModel(bpy.types.Operator):

	bl_idname = "avt.load_model"
	bl_label = "Load human model"
	bl_description = "Loads a parametric naked human model"
	path_input = "/home/aniol/IRI/DadesMarta/models"
	path_input_belly = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/belly"
	path_input_height = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/height"
	path_input_breast = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/breast"
	path_input_torso = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/torso"
	path_input_armslegs = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/armslegs"
	path_input_hip = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/hip"
	path_input_gender = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/gender"
	path_input_weight = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/weight"
	path_input_muscle = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/muscle"
	#path_input_length = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/length"
	path_input_strength = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/strength"
	
	
	#### COMBINACIO LINEAL PES ASCENDENT I MUSCUL DESCENDENT!!!! TOT ESTÀ PERF ER I TOT ÉS POSSIBLE
	
	###
	
	global eigenbody0
	eigenbody0 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_belly, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody0.append(float(line))
	eigenbody0 = np.array(eigenbody0)
	
	###
	
	global eigenbody1
	eigenbody1 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_height, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody1.append(float(line))
	eigenbody1 = np.array(eigenbody1)
	
	###
	
	global eigenbody2
	eigenbody2 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_breast, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody2.append(float(line))
	eigenbody2 = np.array(eigenbody2)
	
	###
	
	global eigenbody3
	eigenbody3 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_torso, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody3.append(float(line))
	eigenbody3 = np.array(eigenbody3)
	
	###
	
	global eigenbody4
	eigenbody4 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_armslegs, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody4.append(float(line))
	eigenbody4 = np.array(eigenbody4)
	
	###
	
	global eigenbody5
	eigenbody5 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_hip, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody5.append(float(line))
	eigenbody5 = np.array(eigenbody5)
	
	###
	
	global eigenbody6
	eigenbody6 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_gender, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody6.append(float(line))
	eigenbody6 = np.array(eigenbody6)
	
	###
	
	global eigenbody7
	eigenbody7 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_weight, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody7.append(float(line))
	eigenbody7 = np.array(eigenbody7)
	
	###
	
	global eigenbody8
	eigenbody8 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_muscle, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody8.append(float(line))
	eigenbody8 = np.array(eigenbody8)
	
	###
	
#	global eigenbody10
#	eigenbody10 = []
#	model_name = "eigenbody0"
#	model2 = "%s/%s.txt" % (path_input_length, model_name)
#	eigenbody = open(model2,'r')

#	for line in eigenbody:
#		eigenbody10.append(float(line))
#	eigenbody10 = np.array(eigenbody10)
	
	###
	
	global eigenbody12
	eigenbody12 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_strength, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody12.append(float(line))
	eigenbody12 = np.array(eigenbody12)
	
	###
	global mean_belly
	mean_belly = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_belly, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_belly.append(float(line))
	mean_belly = np.array(mean_belly)
	
	###
	
	global mean_height
	mean_height = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_height, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_height.append(float(line))
	mean_height = np.array(mean_height)
	
	###
	
	global mean_breast
	mean_breast = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_breast, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_breast.append(float(line))
	mean_breast = np.array(mean_breast)
	
	###
	
	global mean_torso
	mean_torso = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_breast, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_torso.append(float(line))
	mean_torso = np.array(mean_torso)
	
	###
	
	global mean_armslegs
	mean_armslegs = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_armslegs, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_armslegs.append(float(line))
	mean_armslegs = np.array(mean_armslegs)
	
	###
	
	global mean_hip
	mean_hip = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_hip, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_hip.append(float(line))
	mean_hip = np.array(mean_hip)
	
	###
	
	global mean_gender
	mean_gender = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_gender, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_gender.append(float(line))
	mean_gender = np.array(mean_gender)
	
	###
	
	global mean_weight
	mean_weight = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_weight, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_weight.append(float(line))
	mean_weight= np.array(mean_weight)
	
	###
	
	global mean_muscle
	mean_muscle = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_muscle, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_muscle.append(float(line))
	mean_muscle= np.array(mean_muscle)
	
	###
	
#	global mean_length
#	mean_length = []
#	model_name = "StandardModel"
#	model2 = "%s/%s.txt" % (path_input_length, model_name)
#	eigenbody = open(model2,'r')
#	for line in eigenbody:
#		mean_length.append(float(line))
#	mean_length= np.array(mean_length)
	
	###
	
	global mean_strength
	mean_strength = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_strength, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_strength.append(float(line))
	mean_strength= np.array(mean_strength)
	
	###

	global mean
	mean = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean.append(float(line))
	mean = np.array(mean)
	
	###

	bodyeigen0 = eigenbody0-mean_belly
	global vertexeigen0
	vertexeigen0 = []
	for i in range(0,len(bodyeigen0),3):
		vertexeigen0.append([bodyeigen0[i],-bodyeigen0[i+2],bodyeigen0[i+1]])

	bodyeigen1 = eigenbody1-mean_height
	global vertexeigen1
	vertexeigen1 = []
	for i in range(0,len(bodyeigen1),3):
		vertexeigen1.append([bodyeigen1[i],-bodyeigen1[i+2],bodyeigen1[i+1]])
		
	bodyeigen2 = eigenbody2-mean_breast
	global vertexeigen2
	vertexeigen2 = []
	for i in range(0,len(bodyeigen2),3):
		vertexeigen2.append([bodyeigen2[i],-bodyeigen2[i+2],bodyeigen2[i+1]])
		
	bodyeigen3 = eigenbody3-mean_torso
	global vertexeigen3
	vertexeigen3 = []
	for i in range(0,len(bodyeigen3),3):
		vertexeigen3.append([bodyeigen3[i],-bodyeigen3[i+2],bodyeigen3[i+1]])
		
	bodyeigen4 = eigenbody4-mean_armslegs
	global vertexeigen4
	vertexeigen4 = []
	for i in range(0,len(bodyeigen4),3):
		vertexeigen4.append([bodyeigen4[i],-bodyeigen4[i+2],bodyeigen4[i+1]])
		
	bodyeigen5 = eigenbody5-mean_hip
	global vertexeigen5
	vertexeigen5 = []
	for i in range(0,len(bodyeigen5),3):
		vertexeigen5.append([bodyeigen5[i],-bodyeigen5[i+2],bodyeigen5[i+1]])
		
	bodyeigen6 = eigenbody6-mean_gender
	global vertexeigen6
	vertexeigen6 = []
	for i in range(0,len(bodyeigen6),3):
		vertexeigen6.append([bodyeigen6[i],-bodyeigen6[i+2],bodyeigen6[i+1]])
		
	bodyeigen7 = eigenbody7-mean_weight
	global vertexeigen7
	vertexeigen7 = []
	for i in range(0,len(bodyeigen7),3):
		vertexeigen7.append([bodyeigen7[i],-bodyeigen7[i+2],bodyeigen7[i+1]])
		
	bodyeigen8 = eigenbody8-mean_muscle
	global vertexeigen8
	vertexeigen8 = []
	for i in range(0,len(bodyeigen8),3):
		vertexeigen8.append([bodyeigen8[i],-bodyeigen8[i+2],bodyeigen8[i+1]])
		
#	bodyeigen10 = eigenbody10-mean_length
#	global vertexeigen10
#	vertexeigen10 = []
#	for i in range(0,len(bodyeigen10),3):
#		vertexeigen10.append([bodyeigen10[i],-bodyeigen10[i+2],bodyeigen10[i+1]])


	bodyeigen12 = eigenbody12-mean_strength
	global vertexeigen12
	vertexeigen12 = []
	for i in range(0,len(bodyeigen12),3):
		vertexeigen12.append([bodyeigen12[i],-bodyeigen12[i+2],bodyeigen12[i+1]])


	bodymean = mean

	global vertexmean
	vertexmean = []
	for i in range(0,len(bodymean),3):
		vertexmean.append([bodymean[i],-bodymean[i+2],bodymean[i+1]])


	def execute(self, context):

		scn = context.scene
		obj = context.active_object

		# load makehuman model
		if bpy.data.objects.get("Naked_body") is None:
			model_file = "/home/aniol/IRI/DadesMarta/models/standard.mhx2"
			#	bpy.ops.import_scene.fbx(filepath=model_file, axis_forward='Y', axis_up='Z')
			bpy.ops.import_scene.makehuman_mhx2(filepath=model_file)


		# set mean according PCA vertices


		return {'FINISHED'}



class Avatar_PT_LoadPanel(bpy.types.Panel):

	bl_idname = "Avatar_PT_LoadPanel"
	bl_label = "Load model"
	bl_category = "Avatar"
	bl_space_type = "VIEW_3D"
	bl_region_type = "TOOLS"

	#bpy.types.Object.weight_k1 = FloatProperty(name="stomach", description="Weight 1", default=0, min=0.0, max=1.0, precision=2, update=update_weights)
	#bpy.types.Object.weight_k2 = FloatProperty(name="height", description="Weight 2", default=0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k3 = FloatProperty(name="Breast Size", description="Weight 3", default=0, min=-0.2, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k4 = FloatProperty(name="Shoulders **", description="Weight 4", default=0, min=-0.3, max=0.3, precision=2, update=update_weights)
	bpy.types.Object.weight_k5 = FloatProperty(name="Limbs Fat", description="Weight 5", default=0, min=-0.8, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k6 = FloatProperty(name="Hip Fat", description="Weight 6", default=0, min=-0.5, max=1.0, precision=2, update=update_weights)
	#bpy.types.Object.weight_k7 = FloatProperty(name="gender", description="Weight 7", default=0, min=0.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k8 = FloatProperty(name="Weight", description="Weight 8", default=0, min=-1.0, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k9 = FloatProperty(name="Musculature", description="Weight 9", default=0, min=-1.0, max=0.3, precision=2, update=update_weights)
	bpy.types.Object.weight_k10 = FloatProperty(name="Scale", description="Weight 10", default=1, min=0, max=2.0, precision=2, update=update_scale)
	bpy.types.Object.weight_k11 = FloatProperty(name="Limbs Length", description="Weight 11", default=1, min=0.8, max=1.2, precision=2, update=update_scale)
	bpy.types.Object.weight_k12 = FloatProperty(name="Head Size", description="Weight 12", default=1, min=0.9, max=1.1, precision=2, update=update_scale)
	bpy.types.Object.weight_k13 = FloatProperty(name="Strength", description="Weight 13", default=0, min=-0.5, max=0.5, precision=2, update=update_weights)
	

	def draw(self, context):
		layout = self.layout
		obj = context.object
		scn = context.scene

		row = layout.row()
		row.operator('avt.load_model', text="Load human")
		layout.separator()
		#layout.prop(obj, "weight_k1", slider=True)
		#layout.prop(obj, "weight_k1")
		#layout.prop(obj, "weight_k2")
		layout.prop(obj, "weight_k3")
		layout.prop(obj, "weight_k4")
		layout.prop(obj, "weight_k5")
		layout.prop(obj, "weight_k6")
		#layout.prop(obj, "weight_k7")
		layout.prop(obj, "weight_k8")
		layout.prop(obj, "weight_k9")
		layout.prop(obj, "weight_k10")
		layout.prop(obj, "weight_k11")
		layout.prop(obj, "weight_k12")
		layout.prop(obj, "weight_k13")


classes  = (
			Avatar_PT_LoadPanel,
			Avatar_OT_LoadModel
)

def register():
	for clas in classes:
		bpy.utils.register_class(clas)

def unregister():
	for clas in classes:
		bpy.utils.unregister_class(clas)


if __name__ == '__main__':
	register()
