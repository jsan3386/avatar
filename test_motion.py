import os
import sys
import importlib
import bpy
import time

import numpy as np

from mathutils import Vector, Matrix, Quaternion

sys.path.append("/home/jsanchez/Software/gitprojects/avatar/motion")

import movement_280
importlib.reload(movement_280)

import bvh_utils
importlib.reload(bvh_utils)

frames_folder = "/home/jsanchez/Software/gitprojects/avatar/motion/frames"

skel = bpy.data.objects["Standard"]

list_bones = ["Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LowerBack",
                "Spine", "Spine1", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LThumb", 
                "LeftFingerBase", "LeftHandFinger1", "Neck", "Neck1", "Head", "RightShoulder", "RightArm",
                "RightForeArm", "RightHand", "RThumb", "RightFingerBase", "RightHandFinger1", "RHipJoint",
                "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]


skel_ref = movement_280.get_rest_pose(skel, list_bones)

#movement_280.get_rest_pose2(skel, list_bones)

point_files = [f for f in os.listdir(frames_folder) if f.endswith('.txt')]
point_files.sort()
        
num_packg = 0

bvh_file = "/home/jsanchez/Software/gitprojects/avatar/body/Reference.bvh"

# poseBone = skel.pose.bones["Neck"]
# print(poseBone.rotation_quaternion)
# #q2 = Quaternion([1,0.1,0.3,0])
# q1 = Quaternion([1,0,0,0])
# q2 = Quaternion([0.5931, 0.7688, 0.0000, 0.2392])
# q3 = q1 @ q2
# print(q3)

# list_cpm_bones = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
#                   "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
#                   "Neck", "Head"]

bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", 
             "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]

poseBone = skel.pose.bones["Hips"]
boneRefPoseMtx = poseBone.bone.matrix_local.copy()

list_pb_matrices = []

list_pb_matrices.append(boneRefPoseMtx)

for bone in bone_name:
    pb = skel.pose.bones[bone]
    list_pb_matrices.append(pb.matrix)

#print(list_pb_matrices)

working = 0

bvh_nodes, _, _ = bvh_utils.read_bvh(bpy.context, bvh_file)
bvh_nodes_list = bvh_utils.sorted_nodes(bvh_nodes)

bvh_utils.set_bone_matrices(skel, bvh_nodes_list)

# for node in bvh_nodes_list:
#     print("OHOHO")
#     print(node.name)
#     print(node.rest_head_world)
#     print(node.rest_tail_world)
#     print(node.children)
#     print(node.matrix)


#for f in point_files:
for f in range(1,2):

#    start = time.time()
    #fpname = "%s/%s" % (frames_folder,f)
    fname = "frame_SA%02d_%05d.txt" % (2, f)
    fpname = "%s/%s" % (frames_folder,fname)
    pts_skel = np.loadtxt(fpname)

    M_mb = movement_280.get_trans_mat_blend_to_matlab()
    pts_skel = np.matmul(pts_skel, M_mb)
    
    # poseBone.rotation_mode = "QUATERNION"
    # poseBone.rotation_quaternion = q2
    poseBone = skel.pose.bones["Hips"]
    mat1 = poseBone.bone.matrix
    mat2 = poseBone.bone.matrix_local

    m_world = skel.matrix_world
#    print(m_world)


    if working:
        print("WORKING")
        # set skeleton rest position: MAYBE MOVE ALL THIS TO SERVER.PY IN ORDER TO MAKE FASTER UPDATES
        movement_280.set_rest_pose(skel, skel_ref, list_bones)
        loc, list_q = movement_280.calculate_rotations(skel, pts_skel)

        # print("hips matrices")
        # poseBone = skel.pose.bones["Hips"]
        # mat1 = poseBone.bone.matrix
        # mat2 = poseBone.bone.matrix_local
 
        # print("rotate joints")
        # mR = Matrix([[1,0,0], [0,0,-1], [0,1,0]])
        # poseBone.rotation_mode = "QUATERNION"
        # rotMtx = boneRefPoseMtx.to_3x3().inverted() @ mR @ boneRefPoseMtx.to_3x3()
        # poseBone.rotation_quaternion = rotMtx.to_quaternion()

        bpy.context.view_layer.update()
        # print("joints")
        # ref_arm = movement_280.get_skeleton_joints(skel)
        # print(np.array(ref_arm))


    else:
        print("FAILING")
        # Try to implement faster way
        # print("BVH JOINT NODES")
        # jnts_bvh = movement_280.get_skeleton_bvh_joints(bvh_nodes_list)
        # print(np.array(jnts_bvh))
        # print("JOINT NODES")
        # jnts = movement_280.get_skeleton_joints(skel)
        # print(np.array(jnts))
        hips_loc, hips_rot, rotations = movement_280.calculate_rotations2(bvh_nodes_list, list_pb_matrices, pts_skel)
        #print(rotations)
#        movement_280.apply_rotations(skel, hips_loc, hips_rot, rotations)

        # # # print("bvh joints")
        # # jnts_bvh = movement_280.get_skeleton_bvh_joints(bvh_nodes_list)
        # # print(np.array(jnts_bvh))
        # print("rotate jnts")
        # rot_mat = Matrix([[1,0,0], [0,0,-1], [0,1,0]])
        # movement_280.rotate_bvh_joint (bvh_nodes_list, rot_mat, "Hips")
        # print("bvh joints")
        # jnts_bvh = movement_280.get_skeleton_bvh_joints(bvh_nodes_list)
        # print(np.array(jnts_bvh))

    # get rest pose nodes

#    for node in bvh_nodes_list:
#        print("OHOHO")
#        print(node.name)
#        print(node.rest_head_world)
#        print(node.rest_tail_world)
#        print(node.children)

    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

#    time.sleep(max(1./30 - (time.time() - start), 0))
