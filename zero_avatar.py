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

import subprocess
import signal

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
sys.path.append(avt_path + "/motion/net_models/cpm_pose")
sys.path.append(avt_path + "/motion/retarget_motion")

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

import bvh_utils
importlib.reload(bvh_utils)

import retarget
importlib.reload(retarget)

import load
importlib.reload(load)


from bpy_extras.io_utils import axis_conversion

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

    # set previous mesh vertices values
    cp_vals = obj.data.copy()
    mAvt.np_mesh_prev = mAvt.read_verts(cp_vals)
    
    # calculate new shape with PCA shapes
    mAvt.val_breast = self.val_breast
    mAvt.val_torso = self.val_torso
    mAvt.val_hips = - self.val_hips
    mAvt.val_armslegs = self.val_limbs
    mAvt.val_weight = - self.val_weight
    mAvt.val_strength = self.val_strength

    mAvt.refresh_shape()

    mAvt.np_mesh = mAvt.read_verts(obj.data)
    mAvt.np_mesh_diff = mAvt.np_mesh - mAvt.np_mesh_prev

    for object in bpy.data.objects:
        if ((object.type == 'MESH') and (object.name != "Standard:Body")):
            mAvt.deform_cloth(cloth_name=str(object.name))



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
        mAvt.armature = bpy.data.armatures["Standard"]
        mAvt.skel_ref = movement_280.get_rest_pose(mAvt.skel, mAvt.list_bones)
        mAvt.hips_pos = (mAvt.skel.matrix_world @ Matrix.Translation(mAvt.skel.pose.bones["Hips"].head)).to_translation()

        # Info to compute deformation of clothes in fast manner
        size = len(mAvt.body.data.vertices)
        mAvt.body_kdtree = mathutils.kdtree.KDTree(size)
        for i, v in enumerate (mAvt.body.data.vertices):
            mAvt.body_kdtree.insert(v.co, i)
        mAvt.body_kdtree.balance()

        # Set collision body
        bpy.context.view_layer.objects.active = mAvt.body
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.modifier_add(type='COLLISION')

        # Create skin material
        skin_material = importlib.import_module('skin_material')
        importlib.reload(skin_material)
        skin_mat = skin_material.create_material('skin', 0)
        tex_img, tex_norm, tex_spec = dressing.read_file_textures(avt_path, 'skin')
        skin_material.assign_textures(mAvt.body, skin_mat, tex_img, tex_norm, tex_spec)

        return {'FINISHED'}

class AVATAR_OT_ResetParams(bpy.types.Operator):
    
    bl_idname = "avt.reset_params"
    bl_label = "Reset Parameters"
    bl_description = "Reset original parameters of body shape"
    
    def execute(self, context):	
        global mAvt

        obj = mAvt.body

        # set previous mesh vertices values
        cp_vals = obj.data.copy()
        mAvt.np_mesh_prev = mAvt.read_verts(cp_vals)

    
        # calculate new shape with PCA shapes
        mAvt.val_breast = self.val_breast = 0.0
        mAvt.val_torso = self.val_torso = 0.0
        mAvt.val_hips = self.val_hips = 0.0
        mAvt.val_armslegs = self.val_limbs = 0.0
        mAvt.val_weight = self.val_weight = 0.0
        mAvt.val_strength = self.val_strength = 0.0

        mAvt.refresh_shape()

        mAvt.np_mesh = mAvt.read_verts(obj.data)
        mAvt.np_mesh_diff = mAvt.np_mesh - mAvt.np_mesh_prev

        for object in bpy.data.objects:
            if ((object.type == 'MESH') and (object.name != "Standard:Body")):
                mAvt.deform_cloth(cloth_name=str(object.name))

        return {'FINISHED'}

class AVATAR_PT_LoadPanel(bpy.types.Panel):

    bl_idname = "AVATAR_PT_LoadPanel"
    bl_label = "Load model"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"

    bpy.types.Object.val_breast = FloatProperty(name="Breast Size", description="Breasts Size", default=0, 
                                                min=0.0, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.val_torso = FloatProperty(name="Shoulders Fat", description="Shoulders Fat", default=0, 
                                               min=-0.3, max=0.3, precision=2, update=update_weights)
    bpy.types.Object.val_limbs = FloatProperty(name="Limbs Fat", description="Limbs Fat", default=0, 
                                               min=0.0, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.val_hips = FloatProperty(name="Hips Fat", description="Hips Fat", default=0, 
                                              min=0.0, max=1.0, precision=2, update=update_weights)
    bpy.types.Object.val_weight = FloatProperty(name="Weight", description="Weight", default=0, 
                                                min=-0.5, max=1.5, precision=2, update=update_weights)
    bpy.types.Object.val_strength = FloatProperty(name="Strength", description="Body Strength", default=0, 
                                                  min=0.0, max=0.5, precision=2, update=update_weights)
    

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
        print(tex_img)
        print(tex_norm)
        print(tex_spec)
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

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

class AVATAR_OT_SetRestPose(bpy.types.Operator):
    bl_idname = "avt.set_rest_pose"
    bl_label = "Connect socket"  # Display name in the interface.
#    bl_options = {'REGISTER', 'UNDO'} 
    bl_options = {'REGISTER'} 

    def execute(self, context):  # execute() is called when running the operator.
        global mAvt

        movement_280.set_rest_pose(mAvt.skel, mAvt.skel_ref, mAvt.list_bones)
        mAvt.frame = 1

        return {'FINISHED'}


class AVATAR_OT_StreamingPose(bpy.types.Operator):
    bl_idname = "avt.streaming_pose"
    bl_label = "Connect socket"  # Display name in the interface.
#    bl_options = {'REGISTER', 'UNDO'} 
    bl_options = {'REGISTER'} 

    def execute(self, context):  # execute() is called when running the operator.
        global mAvt

        if not context.window_manager.socket_connected:
            self.zmq_ctx = zmq.Context().instance()  # zmq.Context().instance()  # Context
            bpy.types.WindowManager.socket = self.zmq_ctx.socket(zmq.SUB)
            bpy.types.WindowManager.socket.connect(f"tcp://127.0.0.1:5667")  # publisher connects to this (subscriber)
            bpy.types.WindowManager.socket.setsockopt(zmq.SUBSCRIBE, ''.encode('ascii'))
            print("Waiting for data...")

            # poller socket for checking server replies (synchronous)
            self.poller = zmq.Poller()
            self.poller.register(bpy.types.WindowManager.socket, zmq.POLLIN)

            # let Blender know our socket is connected
            context.window_manager.socket_connected = True
            mAvt.frame = 1
            mAvt.start_origin = context.window_manager.start_origin
            mAvt.write_timeline = context.window_manager.write_timeline

            bpy.app.timers.register(self.timed_msg_poller)

        # stop ZMQ poller timer and disconnect ZMQ socket
        else:
            # cancel timer function with poller if active
            if bpy.app.timers.is_registered(self.timed_msg_poller):
                bpy.app.timers.unregister(self.timed_msg_poller())

            try:
                # close connection
                bpy.types.WindowManager.socket.close()
                print("Subscriber socket closed")
                # remove reference
            except AttributeError:
                print("Subscriber socket was not active")

            # let Blender know our socket is disconnected
            bpy.types.WindowManager.socket = None
            context.window_manager.socket_connected = False
            context.window_manager.pid = 0

        return {'FINISHED'}  # Lets Blender know the operator finished successfully.

    def timed_msg_poller(self):  # context
        global mAvt
        socket_sub = bpy.types.WindowManager.socket
#        write_timeline = bpy.types.WindowManager.write_timeline
#        start_origin = bpy.types.WindowManager.start_origin
        # only keep running if socket reference exist (not None)
        if socket_sub:
            # get sockets with messages (0: don't wait for msgs)
            sockets = dict(self.poller.poll(0))
            # check if our sub socket has a message
            if socket_sub in sockets:
                # get the message
                points3d = recv_array(socket_sub)
                #print(points3d)
                M_mb = movement_280.get_trans_mat_blend_to_matlab()
                pts_skel = np.matmul(points3d, M_mb)
                if mAvt.start_origin:
                    # translate points
                    new_pts_skel = []
                    if mAvt.frame == 1:
                        hips = pts_skel[14,:]
                        mAvt.trans = hips - np.array(mAvt.hips_pos)
                    for pt in pts_skel:
                        new_pts_skel.append( [pt[0]-mAvt.trans[0], pt[1]-mAvt.trans[1], pt[2]-mAvt.trans[2]])
                    pts_skel = np.array(new_pts_skel)

                # set skeleton rest position: MAYBE MOVE ALL THIS TO SERVER.PY IN ORDER TO MAKE FASTER UPDATES
                movement_280.set_rest_pose(mAvt.skel, mAvt.skel_ref, mAvt.list_bones)
                movement_280.calculate_rotations(mAvt.skel, pts_skel)

                if mAvt.write_timeline:
                    bpy.context.view_layer.update()
                    mAvt.skel.keyframe_insert(data_path = "location", index = -1, frame = mAvt.frame)
            
                    for bone in mAvt.list_bones:
                        mAvt.skel.pose.bones[bone].keyframe_insert(data_path = "rotation_quaternion", index = -1, frame = mAvt.frame)

                    mAvt.frame += 1

        # keep running
        return 0.001


class AVATAR_OT_StreamingPublisher(bpy.types.Operator):
    bl_idname = "avt.streaming_publisher"
    bl_label = "Start streaming"  # Display name in the interface.
#    bl_options = {'REGISTER', 'UNDO'} 
    bl_options = {'REGISTER'} 

    bpy.types.WindowManager.pid = IntProperty(default=0)

    def execute(self, context):  # execute() is called when running the operator.
        global avt_path

        if not context.window_manager.streaming:
            str_fps = str(context.window_manager.fps)
            path_frames = "%s/motion/frames" % avt_path
            prog = "%s/motion/server.py" % avt_path
            proc = subprocess.Popen(["python", prog, "-frames", path_frames, str_fps]) 
            context.window_manager.pid = proc.pid
            context.window_manager.streaming = True
            mAvt.start_origin = context.window_manager.start_origin
            mAvt.write_timeline = context.window_manager.write_timeline

        else:
            if context.window_manager.pid != 0:
                os.kill(context.window_manager.pid, signal.SIGTERM)
            context.window_manager.streaming = False

        return {'FINISHED'}

class AVATAR_OT_LoadBVH (bpy.types.Operator):
    
    bl_idname = "avt.load_bvh"
    bl_label = "Load BVH"
    bl_description = "Transfer motion to human model"

    filepath = bpy.props.StringProperty(subtype="FILE_PATH") 

    def invoke(self, context, event):
        bpy.context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        global avt_path
        global mAvt
        scn = context.scene
        obj = context.active_object
        #
        mAvt.bvh_offset = obj.bvh_offset
        mAvt.bvh_start_origin = obj.bvh_start_origin

        print("VALUES")
        print(mAvt.bvh_offset)
        print(mAvt.bvh_start_origin)

        arm2 = mAvt.skel
        mesh_arm2 = mAvt.body
        original_position = []
        rest_pose_3D = []
        
        bones = ["Hips","LHipJoint","LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase","LowerBack","Spine","Spine1","LeftShoulder","LeftArm","LeftForeArm","LeftHand","LThumb","LeftFingerBase","LeftHandFinger1","Neck","Neck1","Head","RightShoulder","RightArm","RightForeArm","RightHand","RThumb","RightFingerBase","RightHandFinger1","RHipJoint","RightUpLeg","RightLeg","RightFoot","RightToeBase"]        
            
        for i in range(len(bones)):
            bone = bones[i]
            matrix = arm2.pose.bones[bone].matrix
            original_position.append(matrix)
            
        file_path = self.filepath 
        
        retarget.loadRetargetSimplify(context,file_path,original_position,mAvt.bvh_offset,mAvt.bvh_start_origin) 

        return {'FINISHED'}



# class AVATAR_OT_LoadBVH (bpy.types.Operator):
    
#     bl_idname = "avt.load_bvh"
#     bl_label = "Load BVH"
#     bl_description = "Transfer motion to human model"

#     filepath = bpy.props.StringProperty(subtype="FILE_PATH") 

#     def invoke(self, context, event):
#         bpy.context.window_manager.fileselect_add(self)
#         return {'RUNNING_MODAL'}

#     def execute(self, context):
#         global avt_path
#         global mAvt
#         scn = context.scene
#         obj = context.active_object
#         #
#         reference_body = "%s/body/Reference.bvh" % avt_path
#         file_bone_corresp = "%s/motion/bones/avt_corrsp_01.txt" % avt_path
#         file_path = self.filepath 

#         bvh_nodes, bvh_frame_time, bvh_frame_count = bvh_utils.read_bvh(context, file_path) 
#         avt_nodes, _, _ = bvh_utils.read_bvh(context, reference_body)

#         bvh_name = bpy.path.display_name_from_filepath(file_path)
#         global_matrix = axis_conversion(from_forward='-Z', from_up='Y').to_4x4()        
#         bones_eq = bvh_utils.bone_equivalence(file_bone_corresp)

#         bvh_utils.transfer_motion(avt_nodes, bvh_nodes, mAvt.skel, mAvt.armature, bones_eq, global_matrix)
# #        bvh_utils.bvh_node_dict2armature(context, bvh_name, bvh_nodes, bvh_frame_time, mAvt.armature, mAvt.skel, 
# #                                         global_matrix=global_matrix)
# #        print(bvh_nodes)
#         print(bvh_frame_time)
#         print(bvh_frame_count)

#         return {'FINISHED'}

class AVATAR_PT_MotionPanel(bpy.types.Panel):
    
    bl_idname = "AVATAR_PT_MotionPanel"
    bl_label = "Motion"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"

    # NOTE:
    # For the moment we don't implement write_bvh. This can be done registering motin to timeline and the export to bvh

    bpy.types.WindowManager.socket_connected = BoolProperty(name="Connect status", description="Boolean", default=False)
    bpy.types.WindowManager.streaming = BoolProperty(name="Streaming status", description="Boolean", default=False)
    bpy.types.WindowManager.fps = IntProperty(name="FPS", description="Streaming frame rate", default=30, min=1, max=60)
#    bpy.types.WindowManager.write_bvh = BoolProperty(name = "wBvh", description="Start at origin", default = False)
    bpy.types.WindowManager.write_timeline = BoolProperty(name = "wTimeline", description="Start at origin", default = False)
    bpy.types.WindowManager.start_origin = BoolProperty(name = "sOrigin", description="Start at origin", default = False)

    bpy.types.Object.bvh_offset = IntProperty(name = "Offset", description="Start motion offset", default = 0, min = 0, max = 250)
    bpy.types.Object.bvh_start_origin = BoolProperty(name = "Origin", description="Start at origin", default = False)


    def draw(self, context):
        layout = self.layout
        obj = context.object
        wm = context.window_manager
        
        row = layout.row()
        if not wm.socket_connected:
            layout.operator("avt.streaming_pose")  # , text="Connect Socket"
        else:
            layout.operator("avt.streaming_pose", text="Disconnect Socket")

#        row = layout.row()
        layout.operator("avt.set_rest_pose", text="Reset pose")
#        layout.prop(wm, "write_bvh", text="Write BVH file")
        layout.operator("avt.load_bvh", text="Load BVH")
        layout.prop(obj, "bvh_offset", text="Motion offset")
        layout.prop(obj, "bvh_start_origin", text="Start origin")

        if not wm.socket_connected:
            return

        # row = layout.row()
        # layout.operator("avt.streaming_publisher")  # , text="Connect Socket"

#        row = layout.row()
        if not wm.streaming:
            layout.operator("avt.streaming_publisher")  # , text="Start streaming"
        else:
            layout.operator("avt.streaming_publisher", text="Stop streaming")

        layout.prop(wm, "fps")
        layout.prop(wm, "write_timeline", text="Write timeline keypoints")
        layout.prop(wm, "start_origin", text="Start at origin")




classes  = (
            AVATAR_PT_LoadPanel, 
            AVATAR_OT_LoadModel,
            AVATAR_OT_ResetParams,
            AVATAR_PT_DressingPanel,
            AVATAR_OT_WearCloth,
            AVATAR_OT_CreateStudio,
            AVATAR_PT_MotionPanel,
            AVATAR_OT_StreamingPose,
            AVATAR_OT_StreamingPublisher,
            AVATAR_OT_SetRestPose,
            AVATAR_OT_LoadBVH,
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