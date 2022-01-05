#coding=utf-8

bl_info = {
    'name': "Avatar",
    'author': "Jordi Sanchez-Riera",
    'version': (0, 1, 0),
    "blender": (2, 80, 0),
    'location': "View3D",
    'description': "Create and move a simple avatar",
    'warning': '',
    'wiki_url': 'https://jsan3386.github.io/avatar/',
    'category': 'Avatar'
}



import os
import sys
import bpy

import math
from mathutils import Vector, Quaternion, Matrix

import numpy as np

import subprocess
import signal

#from numpy import *

from bpy.props import BoolProperty, IntProperty, FloatProperty, StringProperty, EnumProperty

import mathutils 
#from bpy.props import * 
import bpy.utils.previews 

# for p in bpy.utils.script_paths():
#     sys.path.append(p)

# # Set a file 'config.py' with variable avt_path that contains the
# # path of the script
# # need to add the root path in the blender preferences panel
# from config import avt_path

import addon_utils

for mod in addon_utils.modules():
    if mod.bl_info['name'] == "Avatar":
        avt_path = mod.__file__
    else:
        pass

# add extra paths
sys.path.append(os.path.join(avt_path, "body"))
sys.path.append(os.path.join(avt_path, "dressing"))
sys.path.append(os.path.join(avt_path, "dressing", "materials"))
sys.path.append(os.path.join(avt_path, "motion"))
sys.path.append(os.path.join(avt_path, "motion", "retarget_motion"))


import importlib

from iAvatar import Avatar

import iAvatar
importlib.reload(iAvatar)

import shape_utils
importlib.reload(shape_utils)

import dressing
importlib.reload(dressing)

import motion_utils
importlib.reload(motion_utils)

import bvh_utils
importlib.reload(bvh_utils)

import retarget
importlib.reload(retarget)


from bpy_extras.io_utils import axis_conversion

mAvt = iAvatar.Avatar(addon_path=avt_path)

avt_preview_collections = {}

def generate_previews():

    # We are accessing all of the information that we generated in the register function below
    gcoll = avt_preview_collections["thumbnail_previews"]
    image_location = gcoll.images_location

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
        thumb = gcoll.load(filepath, filepath, 'IMAGE')
        enum_items.append((i, i, imagename, thumb.icon_id, a))

    return enum_items



def update_weights (self, context):
    #obj = context.active_object
    global mAvt

    if mAvt.body is not None:
        obj = mAvt.body
    else:
        # for now assume avatar already there
        reload_avatar()
        # obj = context.active_object
    
    # calculate new shape with PCA shapes
    mAvt.val_breast = self.val_breast
    mAvt.val_torso = self.val_torso
    mAvt.val_hips = - self.val_hips
    mAvt.val_armslegs = self.val_limbs
    mAvt.val_weight = - self.val_weight
    mAvt.val_strength = self.val_strength
    mAvt.val_belly = self.val_belly

    mAvt.refresh_shape(obj)

    mAvt.np_mesh = mAvt.read_verts(obj.data)
    mAvt.np_mesh_diff = mAvt.np_mesh - mAvt.np_mesh_prev

    for object in bpy.data.objects:
        if ((object.type == 'MESH') and (object.name != "Avatar:Body")):
            mAvt.deform_cloth(cloth_name=str(object.name))


def load_model_from_blend_file(filename):

    with bpy.data.libraries.load(filename) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects]
        # print('These are the objs: ', data_to.objects)

    # Objects have to be linked to show up in a scene
    for obj in data_to.objects:
        bpy.context.scene.collection.objects.link(obj) 


def reload_avatar():
    global mAvt

    mAvt.load_shape_model()
    mAvt.eyes = bpy.data.objects["Avatar:High-poly"]
    mAvt.body = bpy.data.objects["Avatar:Body"]
    mAvt.skel = bpy.data.objects["Avatar"]
    mAvt.armature = bpy.data.armatures["Avatar"]
    mAvt.skel_ref = motion_utils.get_rest_pose(mAvt.skel, mAvt.list_bones)
    mAvt.hips_pos = (mAvt.skel.matrix_world @ Matrix.Translation(mAvt.skel.pose.bones["Hips"].head)).to_translation()

    # Info to be used to compute body rotations is a faster manner
    list_matrices2 = []
    for bone in mAvt.skel.pose.bones:
        list_matrices2.append(bone.matrix_basis.copy())
    mAvt.list_matrices_basis = list_matrices2

    list_matrices3 = []
    for bone in mAvt.skel.data.bones:
        list_matrices3.append(bone.matrix_local.copy())
    mAvt.list_matrices_local = list_matrices3 

    # bvh_file = "%s/body/Reference.bvh" % avt_path
    # bvh_nodes, _, _ = bvh_utils.read_bvh(bpy.context, bvh_file)
    # mAvt.list_nodes = bvh_utils.sorted_nodes(bvh_nodes)

    # Info to compute deformation of clothes in fast manner
    size = len(mAvt.body.data.vertices)
    mAvt.body_kdtree = mathutils.kdtree.KDTree(size)
    for i, v in enumerate (mAvt.body.data.vertices):
        mAvt.body_kdtree.insert(v.co, i)
    mAvt.body_kdtree.balance()





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
        # model_file = "%s/body/models/standard.mhx2" % avt_path
        # bpy.ops.import_scene.makehuman_mhx2(filepath=model_file)
        model_file = "%s/body/models/avatar.blend" % avt_path
        load_model_from_blend_file(model_file)

        mAvt.load_shape_model()
        mAvt.eyes = bpy.data.objects["Avatar:High-poly"]
        mAvt.body = bpy.data.objects["Avatar:Body"]
        mAvt.skel = bpy.data.objects["Avatar"]
        mAvt.armature = bpy.data.armatures["Avatar"]
        mAvt.skel_ref = motion_utils.get_rest_pose(mAvt.skel, mAvt.list_bones)
        mAvt.hips_pos = (mAvt.skel.matrix_world @ Matrix.Translation(mAvt.skel.pose.bones["Hips"].head)).to_translation()

        # Info to be used to compute body rotations is a faster manner
        list_matrices2 = []
        for bone in mAvt.skel.pose.bones:
            list_matrices2.append(bone.matrix_basis.copy())
        mAvt.list_matrices_basis = list_matrices2

        list_matrices3 = []
        for bone in mAvt.skel.data.bones:
            list_matrices3.append(bone.matrix_local.copy())
        mAvt.list_matrices_local = list_matrices3 

        # bvh_file = "%s/body/Reference.bvh" % avt_path
        # bvh_nodes, _, _ = bvh_utils.read_bvh(bpy.context, bvh_file)
        # mAvt.list_nodes = bvh_utils.sorted_nodes(bvh_nodes)

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

        # Create skin material: eyes material should be created too
        # importlib.import_module('material_utils')
        import material_utils
        importlib.reload(material_utils)
        skin_mat = material_utils.create_material_generic('skin', 0, 1)
        tex_img, tex_norm, tex_spec = dressing.read_file_textures(avt_path, 'skin')
        material_utils.assign_textures_generic_mat(mAvt.body, skin_mat, tex_img, tex_norm, tex_spec)
        eyes_mat = material_utils.create_material_generic('eyes', 0, 1)
        tex_img, tex_norm, tex_spec = dressing.read_file_textures(avt_path, 'eyes')
        material_utils.assign_textures_generic_mat(mAvt.eyes, eyes_mat, tex_img, tex_norm, tex_spec)


        return {'FINISHED'}

class AVATAR_OT_SetBodyShape(bpy.types.Operator):
    
    bl_idname = "avt.set_body_shape"
    bl_label = "Set Body Shape"
    bl_description = "Set Body Shape"
    
    # This operator is only to use for external program to generate bodies with different shapes

    def execute(self, context):	
        global mAvt

        obj = mAvt.body

        # set previous mesh vertices values
        cp_vals = obj.data.copy()
        mAvt.np_mesh_prev = mAvt.read_verts(cp_vals)

        mAvt.refresh_shape(obj)

        mAvt.np_mesh = mAvt.read_verts(obj.data)
        mAvt.np_mesh_diff = mAvt.np_mesh - mAvt.np_mesh_prev

        for object in bpy.data.objects:
            if ((object.type == 'MESH') and (object.name != "Avatar:Body")):
                mAvt.deform_cloth(cloth_name=str(object.name))

        return {'FINISHED'}


class AVATAR_OT_ResetParams(bpy.types.Operator):
    
    bl_idname = "avt.reset_params"
    bl_label = "Reset Parameters"
    bl_description = "Reset original parameters of body shape"
    
    def execute(self, context):	
        global mAvt

        # obj = mAvt.body
        obj = bpy.data.objects["Avatar:Body"]

        # set previous mesh vertices values
        cp_vals = obj.data.copy()
        mAvt.np_mesh_prev = mAvt.read_verts(cp_vals)

        # calculate new shape with PCA shapes
        obj.val_breast = obj.val_torso = obj.val_hips = obj.val_limbs = 0.0
        obj.val_weight = obj.val_strength = obj.val_belly = 0.0

        mAvt.refresh_shape(obj)

        mAvt.np_mesh = mAvt.read_verts(obj.data)
        mAvt.np_mesh_diff = mAvt.np_mesh - mAvt.np_mesh_prev

        for object in bpy.data.objects:
            if ((object.type == 'MESH') and (object.name != "Avatar:Body")):
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
        #global iconname
        global avt_path
        scn = context.scene
        obj = context.active_object
        #
        iconname = bpy.context.scene.avt_thumbnails
        iconname = iconname.split(".")[0]

        # Unselect everything to make sure changes are applied to iconname object
        for o in bpy.context.scene.objects:
            o.select_set(False)
        c_file = "%s/dressing/models/clothes/%s.obj" % (avt_path, iconname)
        dressing.load_cloth(c_file, iconname)
        cloth = bpy.data.objects[iconname]
        cloth.select_set(True)

        # Create cloth material
        import material_utils
        importlib.reload(material_utils)
        mat_id = dressing.get_material_id (iconname)
        cloth_mat = material_utils.create_material_generic(iconname, 0, mat_id)
        tex_img, tex_norm, tex_spec = dressing.read_file_textures(avt_path, iconname)
        material_utils.assign_textures_generic_mat(cloth, cloth_mat, tex_img, tex_norm, tex_spec)
                            
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
        # global iconname

        row = layout.row()
        #Presets
        row.template_icon_view(context.scene, "avt_thumbnails")
        row = layout.row()

        # Just a way to access which one is selected
        # iconname = bpy.context.scene.avt_thumbnails
        # iconname = iconname.split(".")[0]
        #print(iconname)
        col = row.column()
        cols = col.row()  #True
        # Activate item icons
        row = layout.row()
        row.operator('avt.wear_cloth', text="Load selected cloth")	
        layout.separator()
        row = layout.row()
        row.operator('avt.create_studio', text="Create studio")	


class AVATAR_OT_SetRestPose(bpy.types.Operator):
    bl_idname = "avt.set_rest_pose"
    bl_label = "Reset Pose"  # Display name in the interface.
    bl_options = {'REGISTER'} 

    def execute(self, context):  # execute() is called when running the operator.
        global mAvt

        motion_utils.set_rest_pose(mAvt.skel, mAvt.skel_ref, mAvt.list_bones)
        mAvt.frame = 1

        return {'FINISHED'}


class AVATAR_OT_LoadBVH (bpy.types.Operator):
    
    bl_idname = "avt.load_bvh"
    bl_label = "Load BVH"
    bl_description = "Transfer motion to human model"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH") 

    act_x: bpy.props.BoolProperty(name="X")
    act_y: bpy.props.BoolProperty(name="Y")
    act_z: bpy.props.BoolProperty(name="Z")

    def invoke(self, context, event):
        bpy.context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        global avt_path
        global mAvt
        scn = context.scene
        obj = context.active_object
        #
        # print(mAvt)
        # mAvt.bvh_offset = obj.bvh_offset
        # mAvt.bvh_start_origin = obj.bvh_start_origin

        # arm2 = mAvt.skel
        
        # bones = ["Hips","LHipJoint","LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase","LowerBack","Spine","Spine1","LeftShoulder","LeftArm","LeftForeArm","LeftHand","LThumb","LeftFingerBase","LeftHandFinger1","Neck","Neck1","Head","RightShoulder","RightArm","RightForeArm","RightHand","RThumb","RightFingerBase","RightHandFinger1","RHipJoint","RightUpLeg","RightLeg","RightFoot","RightToeBase"]        
            
        # for i in range(len(bones)):
        #     bone = bones[i]
        #     matrix = arm2.pose.bones[bone].matrix
        #     original_position.append(matrix)

        # original_position = []
        # for b in obj.pose.bones:
        #     original_position.append(b.matrix.copy())
            
        file_path_bvh = self.filepath 
        
        bone_corresp_file = "%s/motion/rigs/%s.txt" % (avt_path, scn.skel_rig)

        if obj is not None:
            # retarget calculating rotaions from 3D positions: very slow
            # retarget.retarget_skeleton(bone_corresp_file, file_path_bvh, obj)
            # retarget copying the bvh constraints: very slow
            # retarget.retarget_constraints(bone_corresp_file, file_path_bvh, obj, self.act_x, self.act_y, self.act_z)
            # retarget using matrix world for every joint: faster
            # retarget.retarget_matworld (bone_corresp_file, file_path_bvh, obj, scn.skel_rig)
            # retarget based on bvh_retarget addon
            retarget.retarget_addon(bone_corresp_file, file_path_bvh, obj, scn.skel_rig)
        else:
            print("Please, select a model to transfer the bvh action")

        return {'FINISHED'}


class AVATAR_PT_MotionPanel(bpy.types.Panel):
    
    bl_idname = "AVATAR_PT_MotionPanel"
    bl_label = "Motion"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"

    bpy.types.Object.bvh_offset = IntProperty(name = "Offset", description="Start motion offset", default = 0, min = 0, max = 250)
    bpy.types.Object.bvh_start_origin = BoolProperty(name = "Origin", description="Start at origin", default = False)

    def draw(self, context):
        layout = self.layout
        obj = context.object
        wm = context.window_manager
        
        layout.operator("avt.set_rest_pose", text="Reset pose")
        layout.prop(context.scene, 'skel_rig', text='')
        layout.operator("avt.load_bvh", text="Load BVH")


classes  = (
            AVATAR_PT_LoadPanel, 
            AVATAR_OT_LoadModel,
            AVATAR_OT_ResetParams,
            AVATAR_OT_SetBodyShape,
            AVATAR_PT_DressingPanel,
            AVATAR_OT_WearCloth,
            AVATAR_OT_CreateStudio,
            AVATAR_PT_MotionPanel,
            AVATAR_OT_SetRestPose,
            AVATAR_OT_LoadBVH,
)

def enum_menu_items():
    global avt_path

    rigs_folder = "%s/motion/rigs" % avt_path

    rigs_names = [f for f in os.listdir(rigs_folder) if f.endswith('.txt')]

    menu_items = []
    i = 0
    for rig in rigs_names:

        i = i + 1
        rigsplit = rig.split('.')
        name = rigsplit[0]

        menu_items.append((name, name, '', i))

    return menu_items

def register():

    # Create a new preview collection (only upon register)
    gcoll = bpy.utils.previews.new() # garment collections
    gcoll.images_location = "%s/dressing/cloth_previews" % (avt_path)

    # Enable access to our preview collection outside of this function
    avt_preview_collections["thumbnail_previews"] = gcoll

    # This is an EnumProperty to hold all of the images
    bpy.types.Scene.avt_thumbnails = EnumProperty(
        items=generate_previews(),
        )

    bpy.types.Scene.skel_rig = bpy.props.EnumProperty(items=enum_menu_items())

    from bpy.utils import register_class  
    for clas in classes:
        register_class(clas)

def unregister():
    from bpy.utils import unregister_class  
    for clas in classes:
        unregister_class(clas)

    for gcoll in avt_preview_collections.values():
        bpy.utils.previews.remove(gcoll)
    avt_preview_collections.clear()

    del bpy.types.Scene.avt_thumbnails
    del bpy.types.Scene.skel_rig



if __name__ == '__main__':
    register()