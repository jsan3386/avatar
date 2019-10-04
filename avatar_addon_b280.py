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

from numpy import *

from bpy.props import (BoolProperty,
                       IntProperty,
                      )

import mathutils 
from bpy.props import * 

import bpy.utils.previews

# Set a file 'config.py' with variable avt_path that contains the
# path of the script
# need to add the root path in the blender preferences panel
for p in bpy.utils.script_paths():
    sys.path.append(p)
    

#filename = "/home/acivit/avatar/config.py"
#exec(compile(open(filename).read(),filename,'exec'))
from config import avt_path
#from config import version   ## per la versió 280, version = '280', si la versió és diferent version = ''


#avt_path = '/home/jsanchez/Software/github-projects/avatar'
#avt_path = "/home/aniol/avatar"



vers = 0
if '2.8' in bpy.app.version_string:
    vers = 1

if vers == 1:
    import movement_280 as movement   # For the movement from frames
else:
    import movement

#import read_bvh_custom # For the movement from BVH
if vers == 1:
    sys.path.append(avt_path+"/my_makewalk_280")
else:
    sys.path.append(avt_path+"/my_makewalk")

import load as load


import retarget as retarget # funciona.

from imp import reload

import material_utils
reload(material_utils)

preview_collections = {}

import zmq

##########################################################################################################

##########################################################################################################

###################################### MODEL DEFORMATION FUNCTIONS #######################################

##########################################################################################################

##########################################################################################################

def update_offset(self, context):
    global mAvt
    
    mAvt.offset = self.start_offset
    
def update_origin(self, context):
    global mAvt
    
    mAvt.origin = self.start_origin
    print(mAvt.origin)


def get_vertices (obj):
    return [(obj.matrix_world * v.co) for v in obj.data.vertices]

def get_faces (obj):
    faces = []
    for f in obj.data.polygons:
        for idx in f.vertices:
            faces.append(obj.data.vertices[idx].co)
    return faces

def update_streaming_pose (self, context):
    if self.streaming_pose:
        bpy.ops.avt.streaming_pose('INVOKE_DEFAULT')
    return



def generate_previews():
    # We are accessing all of the information that we generated in the register function below
    pcoll = preview_collections["thumbnail_previews"]
    image_location = pcoll.images_location

    enum_items = []

    gallery = ['dress01.png', 'dress02.png', 'dress03.png', 'dress04.png', 'dress05.png', 'dress06.png', 'dress07.png',
                'glasses01.png', 'glasses02.png',
               'hat01.png', 'hat02.png', 'hat03.png', 'hat04.png',
               'jacket01.png', 'jacket02.png',
               'pants01.png', 'pants02.png', 'pants03.png', 'pants04.png', 'pants05.png', 'pants06.png',
               'shirt01.png', 'shirt02.png', 'shirt03.png', 'shirt04.png', 'shirt05.png', 'shirt06.png', 'shirt07.png',
               'shoes01.png', 'shoes02.png', 'shoes03.png', 'shoes04.png',
               'skirt01.png', 'skirt02.png',
               'suit01.png',
               'swimming01.png', 'swimming02.png', 'swimming03.png', 'swimming04.png']

    a = 0
    for i in gallery:
        a = a + 1
        #print(i)
        imagename = i.split(".")[0]
        #print(imagename)
        filepath = image_location + '/' + i
        #print(filepath)
        thumb = pcoll.load(filepath, filepath, 'IMAGE')
        enum_items.append((i, i, imagename, thumb.icon_id, a))

    return enum_items

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

    

        
    


mAvt = Avatar()

class Avatar_OT_ResetParams(bpy.types.Operator):
    
    bl_idname = "avt.reset_params"
    bl_label = "Reset Parameters"
    bl_description = "Reset original parameters of body shape"
    
    def execute(self, context):	
        global mAvt

        obj = mAvt.mesh
    
        # set previous mesh vertices values
        cp_vals = obj.data.copy()
        # store as np data
        #mAvt.mesh_prev = cp_vals.vertices
        mAvt.np_mesh_prev = mAvt.read_verts(cp_vals)

        # calculate new shape with PCA shapes		
        obj.weight_k3 = w3 = 0.0
        obj.weight_k4 = w4 = 0.0
        obj.weight_k5 = w5 = 0.0
        obj.weight_k6 = w6 = 0.0
        obj.weight_k8 = w8 = 0.0
        obj.weight_k9 = w9 = 0.0
        obj.weight_k11 = w11 = 1.0
        obj.weight_k13 = w13 = 0.0

        verts = obj.data.vertices
        for i in range(0,len(verts)):
            verts[i].co = Vector((vertexeigen2[i][0]*w3 + vertexeigen3[i][0]*w4 + vertexeigen4[i][0]*w5 + vertexeigen5[i][0]*w6  + vertexeigen7[i][0]*w8 + vertexeigen8[i][0]*w9 + vertexeigen12[i][0]*w13+ vertexmean[i][0], vertexeigen2[i][1]*w3 + vertexeigen3[i][1]*w4 + vertexeigen4[i][1]*w5 + vertexeigen5[i][1]*w6 + vertexeigen7[i][1]*w8 + vertexeigen8[i][1]*w9 + vertexeigen12[i][1]*w13 + vertexmean[i][1], vertexeigen2[i][2]*w3 + vertexeigen3[i][2]*w4 + vertexeigen4[i][2]*w5 + vertexeigen5[i][2]*w6 + vertexeigen7[i][2]*w8 + vertexeigen8[i][2]*w9 + vertexeigen12[i][1]*w13 + vertexmean[i][2]))

        mAvt.np_mesh = mAvt.read_verts(obj.data)
        mAvt.np_mesh_diff = mAvt.np_mesh - mAvt.np_mesh_prev

        # move also collision mesh

        # find which vertices are modified

        # calculate position of clothes if any
        
        for object in bpy.data.objects:
            if object.name != "Standard" and object.name != "Standard:Body" and object.name != "verylowpoly" and object.name != "Camera" and object.name != "Light":
            
                mAvt.deform_cloth(cloth_name=str(object.name))
                print("deformant això: " + object.name)

        return {'FINISHED'}
        

class Avatar_OT_LoadModel(bpy.types.Operator):

    bl_idname = "avt.load_model"
    bl_label = "Load human model"
    bl_description = "Loads a parametric naked human model"

    path_input = "%s/PCA/Eigenbodies" % avt_path
    path_input_belly = "%s/PCA/Eigenbodies/parts/belly" % avt_path
    path_input_height = "%s/PCA/Eigenbodies/parts/height" % avt_path
    path_input_breast = "%s/PCA/Eigenbodies/parts/breast" % avt_path
    path_input_torso = "%s/PCA/Eigenbodies/parts/torso" % avt_path
    path_input_armslegs = "%s/PCA/Eigenbodies/parts/armslegs" % avt_path
    path_input_hip = "%s/PCA/Eigenbodies/parts/hip" % avt_path
    path_input_gender = "%s/PCA/Eigenbodies/parts/gender" % avt_path
    path_input_weight = "%s/PCA/Eigenbodies/parts/weight" % avt_path
    path_input_muscle = "%s/PCA/Eigenbodies/parts/muscle" % avt_path
    path_input_strength = "%s/PCA/Eigenbodies/parts/strength" % avt_path

    
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

    global eigenbody8
    eigenbody8 = []
    model_name = "eigenbody0"
    model2 = "%s/%s.txt" % (path_input_muscle, model_name)
    eigenbody = open(model2,'r')

    for line in eigenbody:
        eigenbody8.append(float(line))
    eigenbody8 = np.array(eigenbody8)

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

    global mean_strength
    mean_strength = []
    model_name = "StandardModel"
    model2 = "%s/%s.txt" % (path_input_strength, model_name)
    eigenbody = open(model2,'r')
    for line in eigenbody:
        mean_strength.append(float(line))
    mean_strength= np.array(mean_strength)
    
    ###

    global mean_model
    mean_model = []
    model_name = "StandardModel"
    model2 = "%s/%s.txt" % (path_input, model_name)
    eigenbody = open(model2,'r')
    for line in eigenbody:
        mean_model.append(float(line))
    mean_model = np.array(mean_model)


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
        
    bodyeigen12 = eigenbody12-mean_strength
    global vertexeigen12
    vertexeigen12 = []
    for i in range(0,len(bodyeigen12),3):
        vertexeigen12.append([bodyeigen12[i],-bodyeigen12[i+2],bodyeigen12[i+1]])


    bodymean = mean_model

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

            model_file = "%s/models/standard.mhx2" % avt_path
            bpy.ops.import_scene.makehuman_mhx2(filepath=model_file)

            mAvt.mesh = bpy.data.objects["Standard:Body"]
            mAvt.skel = bpy.data.objects["Standard"]
            mAvt.mesh_mwi = mAvt.mesh.matrix_world.inverted()

            # save it as kd tree data
            size = len(mAvt.mesh.data.vertices)
            mAvt.body_kdtree = mathutils.kdtree.KDTree(size)
        
            for i, v in enumerate (mAvt.mesh.data.vertices):
                mAvt.body_kdtree.insert(v.co, i)
            
            mAvt.body_kdtree.balance()
            
            #low_poly_model_file = "%s/models/low_poly.obj" % avt_path
            #bpy.ops.import_scene.obj(filepath=low_poly_model_file)
            
            #bpy.context.selected_objects[0].name = 'verylowpoly'
            #bpy.context.selected_objects[0].data.name = 'verylowpoly'
            #mAvt.collision_mesh = bpy.data.objects["verylowpoly"]
            mAvt.collision_mesh = mAvt.mesh

            
            #for obj in bpy.data.objects:
            #	obj.select_set(False)
        
            #if bpy.data.objects.get("Standard") is not False:
        
            #	a = bpy.data.objects["Standard"]
            #	b = bpy.data.objects["verylowpoly"]
            #	a.select_set(True)
            #	b.select_set(True)
            #	bpy.context.view_layer.objects.active = a
            #	bpy.ops.object.parent_set(type='ARMATURE_AUTO')
                
            # importar low poly mavt.collision_mesh 
            #vp = bpy.data.objects['verylowpoly']
            #vp.select_set(True)
            bpy.context.view_layer.objects.active = mAvt.mesh
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.modifier_add(type='COLLISION')
            #
            #bpy.ops.rigidbody.objects_add(type='ACTIVE')
            #vp.hide_set(True)
            #stb = bpy.data.objects['Standard']

            # Create skin material
            mAvt.skin_mat = material_utils.create_skin_material(0)
            path_tex_img = "%s/textures/skin/tex00.png" % (avt_path)
            path_tex_norm = "%s/textures/skin/norm00.jpg" % (avt_path)
            path_tex_spec = "%s/textures/skin/spec00.jpg" % (avt_path)
            material_utils.assign_texture_skin(mAvt.mesh, mAvt.skin_mat, path_tex_img, path_tex_norm, path_tex_spec)
                        
            #st = bpy.data.objects['Standard:Body']
            #global match_list 
            #match_list = []
            #global match_list_lp 
            #match_list_lp= []
            #mesh = st.data
            #mesh_low_poly = vp.data
            
            #for vert_lp in mesh_low_poly.vertices:
            #	w_v_lp = vp.matrix_world @ vert_lp.co
            #	d = 10000000
            #	# now we have world position vertex, try to find match in other mesh
            #	for vert in mesh.vertices:
            #		w_v = st.matrix_world @ vert.co
            #		# we have a match
            #		if ((w_v_lp - w_v).length < d):
            #			d = (w_v_lp - w_v).length
            #			min_vert = vert.index
            #			min_lp_vert = vert_lp.index
                
            #	match_list.append(min_vert)
            #	match_list_lp.append(min_lp_vert)
                
            #print(len(match_list))
            #print(len(match_list_lp))
            #print(match_list_lp) IS AN ARRAY FROM 0 TO 411 LOGICALLY
            #print(match_list)
            
            #update_verts()

            
            
        # set mean according PCA vertices
        

        return {'FINISHED'}


class Avatar_OT_CreateStudio (bpy.types.Operator):

    bl_idname = "avt.create_studio"
    bl_label = "Create Studio"
    bl_description = "Set up a lighting studio for high quality renderings"

    def execute(self, context):

        # Load studio plane
        s_file = "%s/models/studio_plane.obj" % avt_path
        bpy.ops.import_scene.obj(filepath=s_file)
        
        # change name to object
        bpy.context.selected_objects[0].name = "studio_plane"
        bpy.context.selected_objects[0].data.name = "studio_plane"

        # remove lights and cameras
        for o in bpy.context.scene.objects:
            if o.type == 'CAMERA':
                o.select_set(True)
            elif o.type == 'LIGHT':
                o.select_set(True)
            else:
                o.select_set(False)

        # Call the operator only once
        bpy.ops.object.delete()

        # create camera and lights
        cam_data = bpy.data.cameras.new("CameraData")
        cam_object = bpy.data.objects.new(name="Camera", object_data=cam_data)
        bpy.context.collection.objects.link(cam_object)
        #bpy.data.cameras["Camera"].clip_end = 1000
        cam_object.location = (0, -66.2, 9.28)
        cam_object.rotation_euler = (math.radians(90), 0, 0)

        fill_data = bpy.data.lights.new(name="FillData", type='SUN')
        fill_data.energy = 1
        fill_object = bpy.data.objects.new(name="fill", object_data=fill_data)
        bpy.context.collection.objects.link(fill_object)
        bpy.context.view_layer.objects.active = fill_object
        fill_object.location = (32.29, -25.6, 48.17)
        fill_object.rotation_euler = (math.radians(-15), math.radians(30), math.radians(-14))

        back_data = bpy.data.lights.new(name="BackData", type='SUN')
        back_data.energy = 1
        back_object = bpy.data.objects.new(name="back", object_data=back_data)
        bpy.context.collection.objects.link(back_object)
        bpy.context.view_layer.objects.active = back_object
        back_object.location = (33.46, 46.93, 41.5)
        back_object.rotation_euler = (math.radians(45), math.radians(-23), math.radians(31))

        key_data = bpy.data.lights.new(name="KeyData", type='SUN')
        key_data.energy = 1
        key_object = bpy.data.objects.new(name="key", object_data=key_data)
        bpy.context.collection.objects.link(key_object)
        bpy.context.view_layer.objects.active = key_object
        key_object.location = (-36.88, -30.55, 49.1)
        key_object.rotation_euler = (math.radians(14), math.radians(-54), math.radians(11))

        # update scene, if needed
        dg = bpy.context.evaluated_depsgraph_get() 
        dg.update()
        
        return {'FINISHED'}

class Avatar_OT_WearCloth (bpy.types.Operator):
    
    bl_idname = "avt.wear_cloth"
    bl_label = "Wear Cloth"
    bl_description = "Dress human with selected cloth"
    
    def execute(self, context):
        global mAvt
        global iconname
        scn = context.scene
        obj = context.active_object
        
        #
        c_file = "%s/models/clothes/%s.obj" % (avt_path, iconname)
        cloth = load_cloth(c_file, iconname)
            
#			# save it as kd tree data: why this???
#			size = len(mAvt.dress_mesh.data.vertices)
#			mAvt.kd_dress = mathutils.kdtree.KDTree(size)
#		
#			for i, v in enumerate (mAvt.dress_mesh.data.vertices):
#				mAvt.kd_dress.insert(v.co, i)

#			mAvt.kd_dress.balance()

                
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
        dress_file = "%s/models/dress1.obj" % avt_path
        bpy.ops.import_scene.obj(filepath=dress_file)
        #print("dress file") 
        #print(dress_file)

        # change name to object
        bpy.context.selected_objects[0].name = 'dress'
        bpy.context.selected_objects[0].data.name = 'dress'
        
        b = bpy.data.objects["dress"]
        b.select_set(True)
        bpy.context.view_layer.objects.active = b
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.modifier_add(type='CLOTH')
        
        mAvt.dress_mesh = bpy.data.objects["dress"]
        mAvt.has_dress = True

        # save it as kd tree data
        size = len(mAvt.dress_mesh.data.vertices)
        mAvt.kd_dress = mathutils.kdtree.KDTree(size)
        
        for i, v in enumerate (mAvt.dress_mesh.data.vertices):
            mAvt.kd_dress.insert(v.co, i)

        mAvt.kd_dress.balance()
#		for obj in bpy.data.objects:
#			obj.select = False
            
#		if bpy.data.objects.get("Standard") is not False:
#		
#			a = bpy.data.objects["Standard"]
#			b = bpy.data.objects["dress"]
#			a.select = True
#			b.select = True
#			bpy.context.view_layer.objects.active = a
#			bpy.ops.object.parent_set(type='ARMATURE_AUTO')
                        
        return {'FINISHED'}


class Avatar_PT_LoadPanel(bpy.types.Panel):

    bl_idname = "Avatar_PT_LoadPanel"
    bl_label = "Load model"
    #bl_category = "Avatar"
    bl_space_type = "VIEW_3D"
    #bl_region_type = "TOOLS"
    bl_region_type = "UI"
    bl_category = "Avatar"

    bpy.types.Object.weight_k3 = FloatProperty(name="Breast Size", description="Weight 3", default=0, min=-0.2, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.weight_k4 = FloatProperty(name="**Shoulders", description="Weight 4", default=0, min=-0.3, max=0.3, precision=2, update=update_weights)
    bpy.types.Object.weight_k5 = FloatProperty(name="Limbs Fat", description="Weight 5", default=0, min=-0.8, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.weight_k6 = FloatProperty(name="Hip Fat", description="Weight 6", default=0, min=-0.5, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.weight_k8 = FloatProperty(name="Weight", description="Weight 8", default=0, min=-1.0, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.weight_k9 = FloatProperty(name="Musculature", description="Weight 9", default=0, min=-1.0, max=0.3, precision=2, update=update_weights)
    bpy.types.Object.weight_k10 = FloatProperty(name="Scale", description="Weight 10", default=1, min=0, max=2.0, precision=2, update=update_scale)
    bpy.types.Object.weight_k11 = FloatProperty(name="**Limbs Length", description="Weight 11", default=1, min=0.8, max=1.2, precision=2, update=update_scale)
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
        layout.prop(obj, "weight_k3")
        layout.prop(obj, "weight_k4")
        layout.prop(obj, "weight_k5")
        layout.prop(obj, "weight_k6")
        layout.prop(obj, "weight_k8")
        layout.prop(obj, "weight_k9")
        layout.prop(obj, "weight_k10")
        layout.prop(obj, "weight_k11")
        layout.prop(obj, "weight_k12")
        layout.prop(obj, "weight_k13")
        layout.separator()
        row = layout.row()
        row.operator('avt.reset_params', text="Reset parameters")		



class Avatar_PT_DressingPanel(bpy.types.Panel):
    
    bl_idname = "Avatar_PT_DressingPanel"
    bl_label = "Dress Human"
    #bl_category = "Avatar"
    bl_space_type = "VIEW_3D"
    #bl_region_type = "TOOLS"
    bl_region_type = "UI"
    bl_category = "Avatar"
    
    def draw(self, context):
        layout = self.layout
        obj = context.object
        scn = context.scene
        global iconname
        
#		row = layout.row()
#		row.operator('avt.tshirt', text="Load T-shirt")
#		#layout.separator()
#		row = layout.row()
#		row.operator('avt.pants', text="Load Pants")
#		#layout.separator()
#		row = layout.row()
#		row.operator('avt.dress', text="Load Dress")
        #layout.separator()
        # ---- example to pass properties to operator
        ### Operator call
        ##bpy.ops.wm.context_set_value(data_path="object.location", value="(1,2,3)")
        ### Add to layout
        ##props = row.operator("wm.context_set_value", text="Orgin")
        ##props.data_path = "object.location"
        ##props.value = "(0,0,0)"
        row = layout.row()
        #Presets
        row.template_icon_view(context.scene, "my_thumbnails")
        row = layout.row()

        # Just a way to access which one is selected
        iconname = bpy.context.scene.my_thumbnails
        iconname = iconname.split(".")[0]
        #print(iconname)
        col = row.column()
        cols = col.row()  #True
        # Activate item icons
        row = layout.row()
        row.operator('avt.wear_cloth', text="Load selected cloth")	
        layout.separator()
        row = layout.row()
        row.operator('avt.create_studio', text="Create studio")		
        
        
class Avatar_PT_MotionPanel(bpy.types.Panel):
    
    bl_idname = "Avatar_PT_MotionPanel"
    bl_label = "Motion"
    #bl_category = "Avatar"
    bl_space_type = "VIEW_3D"
    #bl_region_type = "TOOLS"
    bl_region_type = "UI"
    bl_category = "Avatar"

    bpy.types.Object.start_offset = IntProperty(name = "Offset", description="Start motion offset", default = 0, min = 0, max = 250, update=update_offset)
    bpy.types.Object.start_origin = BoolProperty(name = "Origin", description="Start at origin", default = False, update=update_origin)
    bpy.types.WindowManager.streaming_pose = BoolProperty(name = "Streaming", description = "Streaming button", default = False, update = update_streaming_pose)

    def draw(self, context):
        layout = self.layout
        obj = context.object
        wm = context.window_manager

        #scn = context.scene
        
        row = layout.row()
        row.operator('avt.motion_3d_points', text="Motion from 3D points")
        row = layout.row()
        label = "Streaming ON" if wm.streaming_pose else "Streaming OFF"
        layout.prop(wm, 'streaming_pose', text=label, toggle=True)
        row = layout.row()
        row.operator('avt.motion_bvh', text="Motion from BVH file")
        row = layout.separator()
        layout.prop(obj, "start_offset", text="Motion offset")
        layout.prop(obj, "start_origin", text="Start at origin")
        
        
class Avatar_OT_MotionBVH (bpy.types.Operator):
    
    bl_idname = "avt.motion_bvh"
    bl_label = "Motion BVH"
    bl_description = "Motion from BVH"
    
    filepath = bpy.props.StringProperty(subtype="FILE_PATH") 

    def invoke(self, context, event):
        bpy.context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        global mAvt
        scn = bpy.context.scene
        arm2 = mAvt.skel
        mesh_arm2 = mAvt.mesh
        original_position = []
        rest_pose_3D = []
        
        bones = ["Hips","LHipJoint","LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase","LowerBack","Spine","Spine1","LeftShoulder","LeftArm","LeftForeArm","LeftHand","LThumb","LeftFingerBase","LeftHandFinger1","Neck","Neck1","Head","RightShoulder","RightArm","RightForeArm","RightHand","RThumb","RightFingerBase","RightHandFinger1","RHipJoint","RightUpLeg","RightLeg","RightFoot","RightToeBase"]
        
            
        for i in range(len(bones)):
            bone = bones[i]
            matrix = arm2.pose.bones[bone].matrix
            original_position.append(matrix)
            

        ### Sequence  "breakdance.bvh", "Destroy.bvh" 
        
        #file_path = avt_path + "/sequences/" + "breakdance.bvh"
        file_path = self.filepath 
        
        #for bone in bone_bvh:
            #poseBone = arm2.pose.bones[bone]
            #rest_pose_3D.append(poseBone.head)
        initial_quaternion = Quaternion((1,0,0,0))
        
        extra = 0 #This variable is the angle to change the orientation of the motion.

        retarget.loadRetargetSimplify(context,file_path,original_position,mAvt.offset,extra,mAvt.origin) #Segurament original_position no es fa servir
        #retarget.loadRetargetSimplify(context,file_path)
        
        
        scn.frame_set(mAvt.offset) ## AIXÒ ÉS EL FRAME ON ES DEFINEIX L'INICI DEL MOVIMENT, ES POT CANVIAR AL FITXER LOAD.PY A LA FUNCIÓ readBvhFile (FRAME = 20)
        
        
        ref = original_position.copy()
        for i in range(len(bones)):
            bone = bones[i]
            poseBone = arm2.pose.bones[bone]
            poseBone.rotation_mode = "QUATERNION"
            initial_quaternion = Quaternion((1,0,0,0))
            poseBone.rotation_quaternion = initial_quaternion###ref[i]
            bpy.context.view_layer.update()
            
            arm2.pose.bones[bone].keyframe_insert(data_path = "rotation_quaternion", frame = 0)
        bone = arm2.pose.bones["Hips"]
        
        
        scn.frame_set(mAvt.offset)  
        quat = bone.matrix.to_quaternion()
        quat_n = Quaternion((round(quat.w),round(quat.x),round(quat.y),round(quat.z)))
        pts_skel = movement.get_skeleton_joints(arm2)
        
        
        
        scn.frame_set(0)
        quat2 = bone.matrix.to_quaternion()
        quat2_n = Quaternion((round(quat2.w),round(quat2.x),round(quat2.y),round(quat2.z)))
#		
    #Checking where the animation starts to select a correct orientation 
        
        if quat2_n != quat_n:
            bone.rotation_mode = "QUATERNION"
            initial_quaternion = Quaternion((0,0,0.87,0.42))
            bone.rotation_quaternion = initial_quaternion #Quaternion((0,0,0.91,0.41)), gir de 180 graus.
            bone.keyframe_insert(data_path = "rotation_quaternion", frame = 0)
            bone.location = Vector((0,0,0))
            bone.keyframe_insert(data_path = "location", frame = 0)
        
        
        correction_params = np.zeros((14,3),dtype=np.float32)
        q_list, initial_rotation = movement.get_skeleton_parameters_correction_BVH(arm2,pts_skel,correction_params,extra)
        
        correction_iteration = 0
        correction_iterations = 0
        initial_quaternion = Quaternion((1,0,0,0))#mAvt.skel.matrix_local.to_quaternion()#
        correction_iterations = movement.transition_to_desired_motion_BVH(q_list,initial_rotation,arm2,correction_iteration,mesh_arm2,initial_quaternion,mAvt.offset)
        
        
        print("Correction params")
        print(correction_params)
        print("Resultats: Qlist")
        print(q_list)
        print("Resultats: initial_rotation")
        print(initial_rotation)


        bpy.context.view_layer.update()
        
                
        return {'FINISHED'}

class Avatar_OT_StreamingPose(bpy.types.Operator):
    bl_idname = "avt.streaming_pose"
    bl_label = "Streaming pose"

    _timer = None

    def modal(self, context, event):
        global mAvt
        print("running")
        #points3d = recv_array(mAvt.socket)
        #print(points3d)
        if not context.window_manager.streaming_pose:
            context.window_manager.event_timer_remove(self._timer)
            return {'FINISHED'}
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        self._timer = context.window_manager.event_timer_add(time_step=0.01, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class Avatar_OT_UpdateStreaming(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"

    _timer = None

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'FINISHED'}

        if event.type == 'TIMER':
            pass
            # Here comes the code to execute every time timer interruption is executed
            # Basically should be the new rotations of the body
            # Someone needs to call the operator with
            # bpy.ops.wm.modal_timer_operator()

        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(time_step=0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)



class Avatar_OT_Motion3DPoints (bpy.types.Operator):
    
    bl_idname = "avt.motion_3d_points"
    bl_label = "Motion 3D points"
    bl_description = "Motion from 3D points"

    filepath = bpy.props.StringProperty(subtype="FILE_PATH") 

    def invoke(self, context, event):
        bpy.context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        global mAvt
        obj = bpy.context.active_object
        
        scn = bpy.context.scene
        scn.frame_start = 1
        #scn.frame_end = 400
        
        # read file points and transfer motion: TODO
        context = bpy.context
        #path_input = "%s/frames" % avt_path
        path_input = self.filepath
        print("hoilaksdflaskdjfoaisjdflaksdjf")
        print(self.filepath)
        
        # Set rotation matrix to transform from Matlab coordinates to Blender coordinates
        #
        rx = math.radians(90)
        ry = math.radians(0)
        rz = math.radians(0)

        M_rx = np.array([[1,0,0],[0,math.cos(rx),math.sin(rx)],[0,-math.sin(rx),math.cos(rx)]])
        M_ry = np.array([[math.cos(ry),0,-math.sin(ry)],[0,1,0],[math.sin(ry),0,math.cos(ry)]])
        M_rz = np.array([[math.cos(rz),math.sin(rz),0],[-math.sin(rz),math.cos(rz),0],[0,0,1]])


        ## Rotation Matrix from rotations rx ry rz
        M_mb1 = np.matmul(M_rx, M_ry)
        M_mb = np.matmul(M_mb1, M_rz)
        model = "Standard"


        correction_params = np.zeros((14,3),dtype=np.float32)
        f = 1
        arm2 = mAvt.skel
        mesh_arm2 = mAvt.mesh
        w10 = 1 + (mAvt.weight_k10-1)/5.0
        #arm2 = bpy.data.objects[model]
        trans_correction = arm2.pose.bones['Hips'].head
        original_position = []
        print("**** INITIAL MATRIX DISTRIBUTION ****")
        bones = ["Hips","LHipJoint","LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase","LowerBack","Spine","Spine1","LeftShoulder","LeftArm","LeftForeArm","LeftHand","LThumb","LeftFingerBase","LeftHandFinger1","Neck","Neck1","Head","RightShoulder","RightArm","RightForeArm","RightHand","RThumb","RightFingerBase","RightHandFinger1","RHipJoint","RightUpLeg","RightLeg","RightFoot","RightToeBase"]
        for i in range(len(bones)):
            #print(bones[i])
            bone = bones[i]
            matrix = arm2.pose.bones[bone].matrix.copy()
            original_position.append(matrix)
            #print(matrix)
        
        
            
        correction_iteration = 0
        correction_iterations = 0
        
        while f<50:

            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            if f == 1:
                ref = original_position.copy()
                for i in range(len(bones)):
                    #print(bones[i])
                    bone = bones[i]
                    poseBone = arm2.pose.bones[bone]
                    poseBone.matrix = ref[i]
                    bpy.context.view_layer.update()
                    print("RUNNING EXPERIMENT NUMBER: " + str(f))
                bpy.context.view_layer.update()
                fname = "frame_SA%02d_%05d.txt" % (2, f)
                print(fname)
                fpname = "%s/%s" % (path_input,fname)
                pts_skel = loadtxt(fpname)
                print(pts_skel)
                # adjust 3D points axis to Blender axis
                pts_skel = np.matmul(pts_skel, M_mb)
                pts_skel = movement.correct_pose(pts_skel,trans_correction,w10)
        #        print("############### ORIGINAL skeleton params ################")
        ####        reference_skel_coords = get_skeleton_parameters(arm2,pts_skel,correction_params)
        #        for x in range(0,15):
        #            print([reference_skel_coords[x].x,reference_skel_coords[x].y,reference_skel_coords[x].z])
        #        print("%%%%%%%%%% Coords at step 40 %%%%%%%%%%%%%%%%")
                skel_coords = movement.get_skeleton_joints(arm2)
                #update_verts()
        #        for x in range(0,15):
        #            print([skel_coords[x].x,skel_coords[x].y,skel_coords[x].z])
        
            elif f == 2:
                ref = original_position.copy()
                for i in range(len(bones)):
                    #print(bones[i])
                    bone = bones[i]
                    poseBone = arm2.pose.bones[bone]
                    poseBone.matrix = ref[i]
                    bpy.context.view_layer.update()
                print("RUNNING EXPERIMENT NUMBER: " + str(f))
                bpy.context.view_layer.update()
                fname = "frame_SA%02d_%05d.txt" % (2, f)
                print(fname)
                fpname = "%s/%s" % (path_input,fname)
                pts_skel = loadtxt(fpname)
                # adjust 3D points axis to Blender axis
                pts_skel = np.matmul(pts_skel, M_mb)
                pts_skel = movement.correct_pose(pts_skel,trans_correction,w10)
                print(pts_skel)
                #print("############### ORIGINAL skeleton params ################")
                print(arm2)
                q_list, initial_rotation = movement.get_skeleton_parameters_correction(arm2,pts_skel,correction_params)
                #update_verts()
                #print(q_list)
                for i in range(len(bones)):
                    #print(bones[i])
                    bone = bones[i]
                    poseBone = arm2.pose.bones[bone]
                    poseBone.matrix = ref[i]
                    bpy.context.view_layer.update()
                #update_verts()
                correction_iterations = movement.transition_to_desired_motion(q_list,initial_rotation,arm2,correction_iteration,mesh_arm2)
#				for i in range(len(bones)):
#					#print(bones[i])
#					bone = bones[i]
#					poseBone = arm2.pose.bones[bone]
#					poseBone.matrix = ref[i]
#					bpy.context.scene.update()
#				q_list = get_skeleton_parameters_correction(arm2,pts_skel,correction_params)
                #correction_iterations+=1
        


            else:
                ref = original_position.copy()
                for i in range(len(bones)):
                    #print(bones[i])
                    bone = bones[i]
                    poseBone = arm2.pose.bones[bone]
                    poseBone.matrix = ref[i]
                    bpy.context.view_layer.update()
                print("RUNNING EXPERIMENT NUMBER: " + str(f))
                bpy.context.view_layer.update()
                fname = "frame_SA%02d_%05d.txt" % (2, f)
                print(fname)
                fpname = "%s/%s" % (path_input,fname)
                pts_skel = loadtxt(fpname)
                print("initial pts skel")
                print(pts_skel)
                # adjust 3D points axis to Blender axis
                pts_skel = np.matmul(pts_skel, M_mb)
                pts_skel = movement.correct_pose(pts_skel,trans_correction,w10)
                print("final pts skel")
                print(pts_skel)
                #print("############### ORIGINAL skeleton params ################")
                #print(arm2)
                params = movement.get_skeleton_parameters(arm2,pts_skel,correction_params)
                #update_verts()
            
            #scn.frame_set(f+correction_iterations)
            
            ## EN ALGUN MOMENT S'HAURÀ D'INSERTAR BÉ ELS LOCATION . . . 
            mAvt.skel.keyframe_insert(data_path = "location", index = -1, frame = f+correction_iterations)
            
            for bone in bones:
                mAvt.skel.pose.bones[bone].keyframe_insert(data_path = "rotation_quaternion", index = -1, frame = f+correction_iterations)
            #mAvt.mesh.keyframe_insert(data_path = "location", index = -1, frame = f+correction_iterations)
            #mAvt.mesh.keyframe_insert(data_path = "rotation_euler", index = -1, frame = f+correction_iterations)
                
            f+=1


        
                        
        return {'FINISHED'}



classes  = (
            Avatar_PT_LoadPanel, 
            Avatar_OT_LoadModel, 
            Avatar_OT_ResetParams,
            Avatar_PT_MotionPanel,
            Avatar_OT_Motion3DPoints,
            Avatar_OT_UpdateStreaming,
            Avatar_OT_StreamingPose,
            Avatar_OT_MotionBVH,
            Avatar_PT_DressingPanel,
            Avatar_OT_WearCloth,
            Avatar_OT_CreateStudio
)

def register():
    
    # Create a new preview collection (only upon register)
    pcoll = bpy.utils.previews.new()

    pcoll.images_location = "%s/cloth_previews" % (avt_path)
    #print("%s/cloth_previews" % (avt_path))

    # Enable access to our preview collection outside of this function
    preview_collections["thumbnail_previews"] = pcoll

    # This is an EnumProperty to hold all of the images
    bpy.types.Scene.my_thumbnails = EnumProperty(
        items=generate_previews(),
        )	
    from bpy.utils import register_class  
    for clas in classes:
        register_class(clas)

def unregister():
    from bpy.utils import unregister_class  
    for clas in classes:
        unregister_class(clas)

    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()

    del bpy.types.Scene.my_thumbnails


if __name__ == '__main__':
    register()