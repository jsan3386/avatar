#coding=utf-8

#bl_info = {
#	'name': "Avatar",
#	'author': "Jordi Sanchez-Riera",
#	'version': (0, 1, 0),
#	"blender": (2, 8, 0),
#	'location': "View3D",
#	'description': "Create and move a simple avatar",
#	'warning': '',
#	'wiki_url': '',
#	'category': 'Avatar'}



#HOLA, S'HA D'INSTAL·LAR EL MHX2 :) 

import os
import sys
import bpy

import math
from mathutils import Vector, Quaternion, Matrix

import numpy as np

#from numpy import *

#from bpy.props import (BoolProperty,
#                       IntProperty,
#                      )

import mathutils 
from bpy.props import * 


for p in bpy.utils.script_paths():
    sys.path.append(p)

# Set a file 'config.py' with variable avt_path that contains the
# path of the script
# need to add the root path in the blender preferences panel
from config import avt_path

# add extra paths
sys.path.append(avt_path + "/body")


#import load as load

from imp import reload

from iAvatar import Avatar

import iAvatar
reload(iAvatar)

import shape_utils
reload(shape_utils)

mAvt = iAvatar.Avatar(addon_path=avt_path)


def update_weights (self, context):
    #obj = context.active_object
    global mAvt

    obj = mAvt.body
    
    # calculate new shape with PCA shapes
    w3 = self.weights_belly
    w4 = self.weights_breast
    w5 = self.weights_torso
    w6 = - self.weights_hips
    w8 = - self.weights_gender
    w9 = self.weights_weight
    w11 = - self.weights_muscle
    w13 = self.weights_strength

    verts = obj.data.vertices
    for i in range(0,len(verts)):
        verts[i].co = Vector((vertexeigen2[i][0]*w3 + vertexeigen3[i][0]*w4 + vertexeigen4[i][0]*w5 + vertexeigen5[i][0]*w6  + vertexeigen7[i][0]*w8 + vertexeigen8[i][0]*w9 + vertexeigen12[i][0]*w13+ vertexmean[i][0], vertexeigen2[i][1]*w3 + vertexeigen3[i][1]*w4 + vertexeigen4[i][1]*w5 + vertexeigen5[i][1]*w6 + vertexeigen7[i][1]*w8 + vertexeigen8[i][1]*w9 + vertexeigen12[i][1]*w13 + vertexmean[i][1], vertexeigen2[i][2]*w3 + vertexeigen3[i][2]*w4 + vertexeigen4[i][2]*w5 + vertexeigen5[i][2]*w6 + vertexeigen7[i][2]*w8 + vertexeigen8[i][2]*w9 + vertexeigen12[i][1]*w13 + vertexmean[i][2]))



class Avatar_OT_LoadModel(bpy.types.Operator):

    bl_idname = "avt.load_model"
    bl_label = "Load human model"
    bl_description = "Loads a parametric naked human model"
    global avt_path


    def execute(self, context):
        global mAvt
        scn = context.scene
        obj = context.active_object
        
        # load makehuman model
        model_file = "%s/body/models/standard.mhx2" % avt_path
        bpy.ops.import_scene.makehuman_mhx2(filepath=model_file)

        mAvt.load_shape_model()
        mAvt.body = bpy.data.objects["Standard:Body"]


        return {'FINISHED'}



class Avatar_PT_LoadPanel(bpy.types.Panel):

    bl_idname = "Avatar_PT_LoadPanel"
    bl_label = "Load model"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"

    bpy.types.Object.weight_k3 = FloatProperty(name="Breast Size", description="Weight 3", default=0, min=-0.2, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.weight_k4 = FloatProperty(name="**Shoulders", description="Weight 4", default=0, min=-0.3, max=0.3, precision=2, update=update_weights)
    bpy.types.Object.weight_k5 = FloatProperty(name="Limbs Fat", description="Weight 5", default=0, min=-0.8, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.weight_k6 = FloatProperty(name="Hip Fat", description="Weight 6", default=0, min=-0.5, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.weight_k8 = FloatProperty(name="Weight", description="Weight 8", default=0, min=-1.0, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.weight_k9 = FloatProperty(name="Musculature", description="Weight 9", default=0, min=-1.0, max=0.3, precision=2, update=update_weights)
    #bpy.types.Object.weight_k10 = FloatProperty(name="Scale", description="Weight 10", default=1, min=0, max=2.0, precision=2, update=update_scale)
    #bpy.types.Object.weight_k11 = FloatProperty(name="**Limbs Length", description="Weight 11", default=1, min=0.8, max=1.2, precision=2, update=update_scale)
    #bpy.types.Object.weight_k12 = FloatProperty(name="Head Size", description="Weight 12", default=1, min=0.9, max=1.1, precision=2, update=update_scale)
    bpy.types.Object.weight_k13 = FloatProperty(name="Strength", description="Weight 13", default=0, min=-0.5, max=0.5, precision=2, update=update_weights)
    

    def draw(self, context):
        layout = self.layout
        obj = context.object 
        scene = context.scene

        row = layout.row()
        row.operator('avt.load_model', text="Load human")
        layout.separator()
        #layout.prop(obj, "weight_k1", slider=True)		
        layout.prop(obj, "weight_k3")
        layout.prop(obj, "weight_k4")
        layout.prop(obj, "weight_k5")
        layout.prop(obj, "weight_k6")
        layout.prop(obj, "weight_k8")
        layout.prop(obj, "weight_k9")
        #layout.prop(obj, "weight_k10")
        #layout.prop(obj, "weight_k11")
        #layout.prop(obj, "weight_k12")
        layout.prop(obj, "weight_k13")
        layout.separator()
#        row = layout.row()
#        row.operator('avt.reset_params', text="Reset parameters")		




classes  = (
            Avatar_PT_LoadPanel, 
            Avatar_OT_LoadModel
)

def register():
    
    from bpy.utils import register_class  
    for clas in classes:
        register_class(clas)

def unregister():
    from bpy.utils import unregister_class  
    for clas in classes:
        unregister_class(clas)



if __name__ == '__main__':
    register()