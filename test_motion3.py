import bpy
import sys
import importlib
from mathutils import Quaternion, Matrix, Vector
import numpy as np

sys.path.append("/home/jsanchez/Software/gitprojects/avatar/motion")


import bvh_utils
importlib.reload(bvh_utils)

# import movement_280
# importlib.reload(movement_280)



#bvh_file = "/Users/jsanchez/Software/gitprojects/avatar/body/Reference.bvh"
bvh_file = "/mnt/data/jsanchez/BlenderAssets/mocaps/mixamo/walking.bvh"

bvh_nodes, _, _ = bvh_utils.read_bvh(bpy.context, bvh_file)
bvh_nodes_list = bvh_utils.sorted_nodes(bvh_nodes)

for bvh_node in bvh_nodes_list:
    print(bvh_node.name)
    if bvh_node.name == "mixamorig:Hips":
        for f in range(len(bvh_node.anim_data)):
            print(f)
            print(bvh_node.anim_data[f])

# STEPS:
# 1) set source and target skeletons to rest pose
# 2) scale and calculate offset source skeleton respect target skeleton
# 3) match skeleton bone axis
# 4) transfer rotations to target skeleton
# 5) adjust bone positions (mixamo also has different bone positions) [optional]

skel = bpy.data.objects["walking"]
target = bpy.data.objects["Standard"]

def set_rest_pose(skeleton):

    for bone in skeleton.pose.bones:
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = (0,0,0)

def set_hips_origin(skeleton, hips_name):

    hips_bone = skeleton.pose.bones[hips_name]
    hips_bone.location = (0,0,0)

def find_scale_factor(skel, trg_skel, hips_name_skel, hips_name_target):

    hips_pos_skel = (skel.matrix_world @ Matrix.Translation(skel.pose.bones[hips_name_skel].head)).to_translation()
    hips_pos_targ = (trg_skel.matrix_world @ Matrix.Translation(trg_skel.pose.bones[hips_name_target].head)).to_translation()

    print(hips_pos_skel)
    print(hips_pos_targ)

    return hips_pos_targ[2] / hips_pos_skel[2]  

def read_text_lines(filename):

    list_bones = []

    text_file = open(filename, "r")
    lines = text_file.readlines()
    for line in lines:
        line_split = line.split()
        if len(line_split) == 2:
            list_bones.append([line_split[0], line_split[1]])
        else: # only 1 element
            list_bones.append([line_split[0], "none"])

    return list_bones

def find_bone_match(list_bones, bone_name):

    bone_match = "none"
    for b in list_bones:
        if b[0] == bone_name:
            bone_match = b[1]
            break
    return bone_match

file_bone_correspondences = "/home/jsanchez/Software/gitprojects/avatar/bone_correspondance_mixamo.txt"
bone_corresp = read_text_lines(file_bone_correspondences)

bmatch = find_bone_match(bone_corresp, "LeftFoot")
print("Bone match", bmatch)

# focus only one bone:
# get rotations of bone through all sequence

hips_name_skel = "mixamorig:Hips"
hips_name_target = "Hips"

skel.location = (0,0,0)
target.location = (0,0,0)
set_rest_pose(skel)
set_rest_pose(target)
set_hips_origin(skel, "mixamorig:Hips")
set_hips_origin(target, "Hips")
bpy.context.view_layer.update()
bbox_corners_skel = [skel.matrix_world @ Vector(corner) for corner in skel.bound_box]
bbox_corners_target = [target.matrix_world @ Vector(corner) for corner in target.bound_box]
dist_skel = bbox_corners_skel[1][2] - bbox_corners_skel[0][2]
dist_target = bbox_corners_target[1][2] - bbox_corners_target[0][2]
fscale = dist_target / dist_skel
print(bbox_corners_skel)
print(bbox_corners_target)
#fscale = find_scale_factor (skel, target, hips_name_skel, hips_name_target)
skel.scale = (fscale, fscale, fscale)
a = (skel.matrix_world @ Matrix.Translation(skel.pose.bones[hips_name_skel].head)).to_translation()
print(a)