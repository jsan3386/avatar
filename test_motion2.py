import bpy
import importlib
from mathutils import Quaternion, Matrix, Vector
import numpy as np

import bvh_utils
importlib.reload(bvh_utils)

import movement_280
importlib.reload(movement_280)

skel = bpy.data.objects["Standard"]

pb_hips = skel.pose.bones["Hips"]
pb_neck = skel.pose.bones["Neck"]
pb_join = skel.pose.bones["LHipJoint"]
pb_uple = skel.pose.bones["LeftUpLeg"]
pb_head = skel.pose.bones["Head"]

# code from : https://ipfs-sec.stackexchange.cloudflare-ipfs.com/blender/A/question/44637.html

def matrix_world(armature_ob, bone_name):
    local = armature_ob.data.bones[bone_name].matrix_local
    basis = armature_ob.pose.bones[bone_name].matrix_basis

    parent = armature_ob.pose.bones[bone_name].parent
    if parent == None:
        return  local @ basis
    else:
        parent_local = armature_ob.data.bones[parent.name].matrix_local
        return matrix_world(armature_ob, parent.name) @ (parent_local.inverted() @ local) @ basis


bvh_file = "/Users/jsanchez/Software/gitprojects/avatar/body/Reference.bvh"

bvh_nodes, _, _ = bvh_utils.read_bvh(bpy.context, bvh_file)
bvh_nodes_list = bvh_utils.sorted_nodes(bvh_nodes)

bvh_utils.set_bone_matrices(skel, bvh_nodes_list)

# for node in bvh_nodes_list:
#     print(node.parent)

hips_pos = movement_280.get_bvh_node_val(bvh_nodes_list, "Hips", "HEAD")
vT = Vector((-2.8643, -9.4030, 0.5685))
mR = Matrix(((0.9765, -0.2106, -0.0460),
            (0.2147,  0.9696,  0.1170),
            (0.0200, -0.1242,  0.9921)))


# motion
loc = Vector((-2.8643, -11.4028, 4.7693))
q1 = Quaternion((0.9922, -0.0608, 0.0891, 0.0618)) # Hips
q2 = Quaternion((0.9787, -0.2051, 0.0000, 0.0058)) # LHipJoint
q3 = Quaternion((0.9990, -0.0326, -0.0000, 0.0306)) # LeftUpLeg

bpy.context.view_layer.update()

print("Before start anything")



ref_arm = movement_280.get_skeleton_joints(skel)
pt1 = np.array(ref_arm[11])
pt2 = np.array(ref_arm[12])
print(pt1)
print(pt2)

vec_pt2 = Vector((pt2[0], pt2[1], pt2[2]))
orig_knee = Vector((1.42479992, -0.37378985,  4.75250053))

# set hips loc, rot
#pb_hips.location = loc
# pb_hips.rotation_mode = "QUATERNION"
# pb_hips.rotation_quaternion = q1

# pb_join.rotation_mode = "QUATERNION"
# pb_join.rotation_quaternion = q2

pb_uple.rotation_mode = "QUATERNION"
pb_uple.rotation_quaternion = q2

mat1 = matrix_world(skel, "LeftUpLeg")
mat2 = matrix_world(skel, "LeftLeg")
print(mat1)
print(mat2)

new_knee = mat1.inverted() @ (q2.to_matrix().to_4x4() @ orig_knee)

print(new_knee)
