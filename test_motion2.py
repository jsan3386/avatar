import bpy
import importlib
from mathutils import Quaternion, Matrix, Vector

import bvh_utils
importlib.reload(bvh_utils)

import movement_280
importlib.reload(movement_280)

skel = bpy.data.objects["Standard"]

pb_hips = skel.pose.bones["Hips"]
pb_neck = skel.pose.bones["Neck"]
pb_join = skel.pose.bones["LHipJoint"]
pb_uple = skel.pose.bones["LeftUpLeg"]

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


bvh_file = "/home/jsanchez/Software/gitprojects/avatar/body/Reference.bvh"

bvh_nodes, _, _ = bvh_utils.read_bvh(bpy.context, bvh_file)
bvh_nodes_list = bvh_utils.sorted_nodes(bvh_nodes)

bvh_utils.set_bone_matrices(skel, bvh_nodes_list)

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

#mat0 = movement_280.get_bvh_node_matrix(bvh_nodes_list, "Hips")
#mat1 = movement_280.get_bvh_node_matrix(bvh_nodes_list, "LHipJoint")
#print(mat0)
print(pb_hips.matrix)
#print(q1.to_matrix())
#print(mat1)
print(pb_join.matrix)
#print(q2.to_matrix())

ref_hips = pb_hips.bone.matrix_local.copy()
par_hips = pb_hips.matrix.copy()
ref_join = pb_join.bone.matrix_local.copy()
mat_join = pb_join.matrix.copy()

# set hips loc, rot
#pb_hips.location = loc
pb_hips.rotation_mode = "QUATERNION"
pb_hips.rotation_quaternion = q1

# pb_join.rotation_mode = "QUATERNION"
# pb_join.rotation_quaternion = q2

#movement_280.translate_bvh_nodes(bvh_nodes_list, vT - hips_pos)
#movement_280.rotate_bvh_joint(bvh_nodes_list, q1.to_matrix().to_4x4(), "Hips")
# movement_280.rotate_bvh_joint(bvh_nodes_list, q2.to_matrix().to_4x4(), "LHipJoint")


bpy.context.view_layer.update()

print("Check values 2 sides")

#mat1 = movement_280.get_bvh_node_matrix(bvh_nodes_list, "Hips")
#mat2 = movement_280.get_bvh_node_matrix(bvh_nodes_list, "LHipJoint")
#print(mat1)
print(pb_hips.matrix)
#print(mat2)
print(pb_join.matrix)
print(pb_join.matrix_basis)


print("Reproduce matrices")
hips_trans = par_hips.to_4x4() @ q1.to_matrix().to_4x4()
join_trans = hips_trans.to_4x4() @ q1.to_matrix().to_4x4()
rep_mat = join_trans.inverted().to_4x4() @ skel.matrix_world @ mat_join
print(rep_mat)

mat_world = matrix_world(skel, "LHipJoint")
print(mat_world)



