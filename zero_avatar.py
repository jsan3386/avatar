#coding=utf-8

#bl_info = {
#	'name': "Avatar",
#	'author': "Jordi Sanchez-Riera",
#	'version': (0, 1, 0),
#	"blender": (2, 80, 0),
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
import bpy.utils.previews 

for p in bpy.utils.script_paths():
    sys.path.append(p)

# Set a file 'config.py' with variable avt_path that contains the
# path of the script
# need to add the root path in the blender preferences panel
from config import avt_path

# add extra paths
sys.path.append(avt_path + "/body")
sys.path.append(avt_path + "/dressing")
sys.path.append(avt_path + "/dressing/materials")
sys.path.append(avt_path + "/motion")


#import load as load
import zmq

import importlib

from iAvatar import Avatar

import iAvatar
importlib.reload(iAvatar)

import shape_utils
importlib.reload(shape_utils)

import dressing
importlib.reload(dressing)

import movement_280
importlib.reload(movement_280)

mAvt = iAvatar.Avatar(addon_path=avt_path)

preview_collections = {}

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

def update_weights (self, context):
    #obj = context.active_object
    global mAvt

    obj = mAvt.body
    
    # calculate new shape with PCA shapes
    mAvt.val_breast = self.val_breast
    mAvt.val_torso = self.val_torso
    mAvt.val_hips = - self.val_hips
    mAvt.val_armslegs = self.val_limbs
    mAvt.val_weight = - self.val_weight
    mAvt.val_muscle = - self.val_muscle
    mAvt.val_strength = self.val_strength

    mAvt.refresh_shape()



class AVATAR_OT_LoadModel(bpy.types.Operator):

    bl_idname = "avt.load_model"
    bl_label = "Load human model"
    bl_description = "Loads a parametric naked human model"

    def execute(self, context):
        global mAvt
        global avt_path
        scn = context.scene
        obj = context.active_object
        
        # load makehuman model
        model_file = "%s/body/models/standard.mhx2" % avt_path
        bpy.ops.import_scene.makehuman_mhx2(filepath=model_file)

        mAvt.load_shape_model()
        mAvt.body = bpy.data.objects["Standard:Body"]
        mAvt.skel = bpy.data.objects["Standard"]

        # Create skin material
        skin_material = importlib.import_module('skin_material')
        skin_mat = skin_material.create_material(0)
        path_tex_img = "%s/dressing/textures/skin/tex00.png" % (avt_path)
        path_tex_norm = "%s/dressing/textures/skin/norm00.jpg" % (avt_path)
        path_tex_spec = "%s/dressing/textures/skin/spec00.jpg" % (avt_path)
        skin_material.assign_textures(mAvt.body, skin_mat, path_tex_img, path_tex_norm, path_tex_spec)

        return {'FINISHED'}

class AVATAR_OT_ResetParams(bpy.types.Operator):
    
    bl_idname = "avt.reset_params"
    bl_label = "Reset Parameters"
    bl_description = "Reset original parameters of body shape"
    
    def execute(self, context):	
        global mAvt

        obj = mAvt.body
    
        # calculate new shape with PCA shapes
        mAvt.val_breast = self.val_breast = 0.0
        mAvt.val_torso = self.val_torso = 0.0
        mAvt.val_hips = self.val_hips = 0.0
        mAvt.val_armslegs = self.val_limbs = 0.0
        mAvt.val_weight = self.val_weight = 0.0
        mAvt.val_muscle = self.val_muscle = 0.0
        mAvt.val_strength = self.val_strength = 0.0

        mAvt.refresh_shape()

        for object in bpy.data.objects:
            if object.name != "Standard" and object.name != "Standard:Body" and object.name != "Camera" and object.name != "Light":
            
                mAvt.deform_cloth(cloth_name=str(object.name))

        return {'FINISHED'}

class AVATAR_PT_LoadPanel(bpy.types.Panel):

    bl_idname = "AVATAR_PT_LoadPanel"
    bl_label = "Load model"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"

    bpy.types.Object.val_breast = FloatProperty(name="Breast Size", description="Breasts Size", default=0, min=-0.2, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.val_torso = FloatProperty(name="Shoulders Fat", description="Shoulders Fat", default=0, min=-0.3, max=0.3, precision=2, update=update_weights)
    bpy.types.Object.val_limbs = FloatProperty(name="Limbs Fat", description="Limbs Fat", default=0, min=-0.8, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.val_hips = FloatProperty(name="Hip Fat", description="Hips Fat", default=0, min=-0.5, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.val_weight = FloatProperty(name="Weight", description="Overall Weight", default=0, min=-1.0, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.val_muscle = FloatProperty(name="Musculature", description="Musculature", default=0, min=-1.0, max=0.3, precision=2, update=update_weights)
    bpy.types.Object.val_strength = FloatProperty(name="Strength", description="Body Strength", default=0, min=-0.5, max=0.5, precision=2, update=update_weights)
    

    def draw(self, context):
        layout = self.layout
        obj = context.object 
        scene = context.scene

        row = layout.row()
        row.operator('avt.load_model', text="Load human")

        if (obj is None or (obj.type not in ['MESH', 'ARMATURE'])):
            return

        layout.separator()
        layout.prop(obj, "val_breast")
        layout.prop(obj, "val_torso")
        layout.prop(obj, "val_limbs")
        layout.prop(obj, "val_hips")
        layout.prop(obj, "val_weight")
        layout.prop(obj, "val_muscle")
        layout.prop(obj, "val_strength")
        layout.separator()
        row = layout.row()
        row.operator('avt.reset_params', text="Reset parameters")		

class AVATAR_OT_CreateStudio (bpy.types.Operator):

    bl_idname = "avt.create_studio"
    bl_label = "Create Studio"
    bl_description = "Set up a lighting studio for high quality renderings"

    def execute(self, context):
        global avt_path

        dressing.load_studio(avt_path)
        
        return {'FINISHED'}

class AVATAR_OT_WearCloth (bpy.types.Operator):
    
    bl_idname = "avt.wear_cloth"
    bl_label = "Wear Cloth"
    bl_description = "Dress human with selected cloth"
    
    def execute(self, context):
        global iconname
        global avt_path
        scn = context.scene
        obj = context.active_object
        #
        # Unselect everything to make sure changes are applied to iconname object
        for o in bpy.context.scene.objects:
            o.select_set(False)
        c_file = "%s/dressing/models/clothes/%s.obj" % (avt_path, iconname)
        dressing.load_cloth(c_file, iconname)
        cloth = bpy.data.objects[iconname]
        cloth.select_set(True)

        # Create skin material
        cloth_material = importlib.import_module(iconname)
        importlib.reload(cloth_material)
        cloth_mat = cloth_material.create_material(iconname, 0)
        tex_img, tex_norm, tex_spec = dressing.read_file_textures(avt_path, iconname)
        cloth_material.assign_textures(cloth, cloth_mat, tex_img, tex_norm, tex_spec)
                            
        return {'FINISHED'}


class AVATAR_PT_DressingPanel(bpy.types.Panel):
    
    bl_idname = "AVATAR_PT_DressingPanel"
    bl_label = "Dress Human"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"
    
    def draw(self, context):
        layout = self.layout
        obj = context.object
        scn = context.scene
        global iconname

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

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    #buf = memoryview(msg)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])


class AVATAR_OT_StreamingPose(bpy.types.Operator):
    bl_idname = "avt.streaming_pose"
    bl_label = "Connect socket"  # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'} 

    socket = None 

    def execute(self, context):  # execute() is called when running the operator.

        if not context.window_manager.socket_connected:
            self.zmq_ctx = zmq.Context().instance()  # zmq.Context().instance()  # Context
            self.socket = self.zmq_ctx.socket(zmq.SUB)
            self.socket.connect(f"tcp://127.0.0.1:5667")  # publisher connects to this (subscriber)
            self.socket.setsockopt(zmq.SUBSCRIBE, ''.encode('ascii'))
            print("Waiting for data...")

            # poller socket for checking server replies (synchronous)
            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)

            # let Blender know our socket is connected
            context.window_manager.socket_connected = True

            bpy.app.timers.register(self.timed_msg_poller)

        # stop ZMQ poller timer and disconnect ZMQ socket
        else:
            # cancel timer function with poller if active
            if bpy.app.timers.is_registered(self.timed_msg_poller):
                bpy.app.timers.unregister(self.timed_msg_poller())

            try:
                # close connection
                self.socket.close()
                print("Subscriber socket closed")
                # remove reference
            except AttributeError:
                print("Subscriber socket was not active")

            # let Blender know our socket is disconnected
            self.socket = None
            context.window_manager.socket_connected = False

        return {'FINISHED'}  # Lets Blender know the operator finished successfully.

    def timed_msg_poller(self):  # context
        global mAvt
        socket_sub = self.socket
        # only keep running if socket reference exist (not None)
        if socket_sub:
            # get sockets with messages (0: don't wait for msgs)
            sockets = dict(self.poller.poll(0))
            # check if our sub socket has a message
            if socket_sub in sockets:
                # get the message
                points3d = recv_array(socket_sub)
                print(points3d)
                M_mb = movement_280.get_trans_mat_blend_to_matlab()
                pts_skel = np.matmul(points3d, M_mb)
                #pts_skel = movement.correct_pose(pts_skel,trans_correction,w10)
                correction_params = np.zeros((14,3),dtype=np.float32)
                params = movement_280.get_skeleton_parameters(mAvt.skel, pts_skel, correction_params)

        # keep running
        return 0.001

class AVATAR_PT_MotionPanel(bpy.types.Panel):
    
    bl_idname = "AVATAR_PT_MotionPanel"
    bl_label = "Motion"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"

    bpy.types.WindowManager.socket_connected = BoolProperty(name="Connect status", description="Boolean", default=False)

    def draw(self, context):
        layout = self.layout
        obj = context.object
        wm = context.window_manager
        
        row = layout.row()
        if not wm.socket_connected:
            layout.operator("avt.streaming_pose")  # , text="Connect Socket"
        else:
            layout.operator("avt.streaming_pose", text="Disconnect Socket")


classes  = (
            AVATAR_PT_LoadPanel, 
            AVATAR_OT_LoadModel,
            AVATAR_OT_ResetParams,
            AVATAR_PT_DressingPanel,
            AVATAR_OT_WearCloth,
            AVATAR_OT_CreateStudio,
            AVATAR_PT_MotionPanel,
            AVATAR_OT_StreamingPose,
)

def register():

    # Create a new preview collection (only upon register)
    pcoll = bpy.utils.previews.new()
    pcoll.images_location = "%s/dressing/cloth_previews" % (avt_path)

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