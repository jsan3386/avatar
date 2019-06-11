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

	w3 = self.weight_k3
	w9 = self.weight_k9
	w13 = self.weight_k13

	verts = obj.data.vertices
	for i in range(0,len(verts)):
		verts[i].co = Vector((vertexeigen2[i][0]*w3 + vertexeigen8[i][0]*w9 + vertexeigen12[i][0]*w13+ vertexmean[i][0], vertexeigen2[i][1]*w3 + vertexeigen8[i][1]*w9 + vertexeigen12[i][1]*w13 + vertexmean[i][1], vertexeigen2[i][2]*w3 + vertexeigen8[i][2]*w9 + vertexeigen12[i][1]*w13 + vertexmean[i][2]))


def update_scale (self,context):
	a = bpy.data.objects['Standard']
	w10 = self.weight_k10
	
	vector_scale = Vector((w10,w10,w10))
	a.scale = vector_scale
	



class Avatar_OT_LoadModel(bpy.types.Operator):

	bl_idname = "avt.load_model"
	bl_label = "Load human model"
	bl_description = "Loads a parametric naked human model"
	path_input = "/home/aniol/IRI/DadesMarta/models"
	path_input_breast = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/breast"
	path_input_muscle = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/muscle"
	path_input_strength = "/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/strength"
	
	
	#### COMBINACIO LINEAL PES ASCENDENT I MUSCUL DESCENDENT!!!! TOT ESTÀ PERF ER I TOT ÉS POSSIBLE
	

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
	
	global eigenbody8
	eigenbody8 = []
	model_name = "eigenbody0"
	model2 = "%s/%s.txt" % (path_input_muscle, model_name)
	eigenbody = open(model2,'r')

	for line in eigenbody:
		eigenbody8.append(float(line))
	eigenbody8 = np.array(eigenbody8)
	
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
	
	global mean_breast
	mean_breast = []
	model_name = "StandardModel"
	model2 = "%s/%s.txt" % (path_input_breast, model_name)
	eigenbody = open(model2,'r')
	for line in eigenbody:
		mean_breast.append(float(line))
	mean_breast = np.array(mean_breast)
	
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

		
	bodyeigen2 = eigenbody2-mean_breast
	global vertexeigen2
	vertexeigen2 = []
	for i in range(0,len(bodyeigen2),3):
		vertexeigen2.append([bodyeigen2[i],-bodyeigen2[i+2],bodyeigen2[i+1]])
		
		
	bodyeigen8 = eigenbody8-mean_muscle
	global vertexeigen8
	vertexeigen8 = []
	for i in range(0,len(bodyeigen8),3):
		vertexeigen8.append([bodyeigen8[i],-bodyeigen8[i+2],bodyeigen8[i+1]])
		
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


	bpy.types.Object.weight_k3 = FloatProperty(name="Breast Size", description="Weight 3", default=0, min=-0.2, max=1.0, precision=2, update=update_weights)
	bpy.types.Object.weight_k9 = FloatProperty(name="Musculature", description="Weight 9", default=0, min=-1.0, max=0.3, precision=2, update=update_weights)
	bpy.types.Object.weight_k10 = FloatProperty(name="Scale", description="Weight 10", default=1, min=0, max=2.0, precision=2, update=update_scale)
	bpy.types.Object.weight_k13 = FloatProperty(name="Strength", description="Weight 13", default=0, min=-0.5, max=0.5, precision=2, update=update_weights)
	

	def draw(self, context):
		layout = self.layout
		obj = context.object
		scn = context.scene

		row = layout.row()
		row.operator('avt.load_model', text="Load human")
		layout.separator()
		layout.prop(obj, "weight_k3")
		layout.prop(obj, "weight_k9")
		layout.prop(obj, "weight_k10")
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
