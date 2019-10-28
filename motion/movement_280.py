import os
import sys
import bpy
import math
from mathutils import Vector, Quaternion, Matrix
import numpy as np
from numpy import *
import mathutils
from bpy.props import *

#from config import avt_path

def get_trans_mat_blend_to_matlab ():

    rx = math.radians(90)
    ry = math.radians(0)
    rz = math.radians(0)

    M_rx = np.array([[1,0,0],[0,math.cos(rx),math.sin(rx)],[0,-math.sin(rx),math.cos(rx)]])
    M_ry = np.array([[math.cos(ry),0,-math.sin(ry)],[0,1,0],[math.sin(ry),0,math.cos(ry)]])
    M_rz = np.array([[math.cos(rz),math.sin(rz),0],[-math.sin(rz),math.cos(rz),0],[0,0,1]])

    ## Rotation Matrix from rotations rx ry rz
    M_mb1 = np.matmul(M_rx, M_ry)
    M_mb = np.matmul(M_mb1, M_rz)

    return M_mb


def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0]; # total points
    #print(A)
    #print(B)
    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = transpose(AA) @ BB
    U, S, Vt = linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if linalg.det(R) < 0:
        #print ("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A.T + centroid_B.T
    #print (t)

    return R, t


def trans_coord_system(p1, o1, o2, M1, M2):

    # note is necessary to use a copy version of the matrix
    # otherwise it modifies the content of M2 outside the function
    # Actually it transposes matrix M2 outside the function. Must be the way
    # Blender handles the transform operators
    M2t = M2.copy()
    M2t.transpose()
    return M2t @ (o1 - o2 + M1 @ p1)

def compute_rotation(pb_matrix, pt0, pt1, pt2):
    #pt0 -> start_point_bone
    #pt1 -> end_point_bone
    #pt2 -> goal_point_bone

    M1 = Matrix([[1,0,0], [0,1,0], [0,0,1]])
    ### He afegit un canvi aquí (el .to_3x3(), a priori sembla que ha millorat.. )
    #M2 = poseBone.matrix.copy()
    M2 = pb_matrix.to_3x3()

    v1 = trans_coord_system(pt1, Vector((0,0,0)), pt0, M1, M2) # sempre és (0,l,0)
    v2 = trans_coord_system(pt2, Vector((0,0,0)), pt0, M1, M2)

    a = v1.normalized()
    b = v2.normalized()

    c = a.cross(b)

    # c.magnitude from normalized vectors should not be greater than 1 ?
    # in some cases c.magnitude > 1 then asin fails [-1,+1]

    # check for cases > 1 and round to 1
    v_magnitude = c.magnitude
    if (v_magnitude > 1):
        v_magnitude = 1

    # to determine if angle in [0, pi/2] or in [pi/2, pi]
    l = np.linalg.norm(pt1 - pt0)
    dist = np.linalg.norm(pt1 - pt2)
    dist_max = math.sqrt(2*l*l)

    if (dist < dist_max): theta = math.asin(v_magnitude)
    else: theta = math.pi - math.asin(v_magnitude)

    if (c.magnitude>0):
        axis = c.normalized()
        st2 = math.sin(theta/2)
        q = Quaternion( [math.cos(theta/2), st2*axis.x, st2*axis.y, st2*axis.z] )
    else:
        q = Quaternion( [1,0,0,0] )

    return q

def rotate_point(point, matrix, rot_center):
    # move point to origin
    v1 = point - rot_center
    v2 = matrix @ v1
    v3 = v2 + rot_center
    return v3

def get_bone_tail_position(obj, bone_name):
    return (obj.matrix_world @ Matrix.Translation(obj.pose.bones[bone_name].tail)).to_translation()

def get_bone_head_position(obj, bone_name):
    return (obj.matrix_world @ Matrix.Translation(obj.pose.bones[bone_name].head)).to_translation()

def set_point_bvh_nodes(bvh_nodes, node_name, pt):

    for node in bvh_nodes:
        if node.name == node_name:
            node.rest_tail_world = pt

def get_bvh_node (bvh_nodes, name):
    sel_node = []
    for node in bvh_nodes:
        if node.name == name:
            sel_node = node
    return sel_node

def get_bvh_node_matrix(bvh_nodes, name):
    for bvh_node in bvh_nodes:
        if bvh_node.name == name :
            return bvh_node.matrix



def get_bvh_node_val(bvh_nodes, name, btype):

    value = []
    for bvh_node in bvh_nodes:
        if bvh_node.name == name :
            if btype == 'HEAD' :
                value = bvh_node.rest_head_world
            elif btype == 'TAIL' :
                value = bvh_node.rest_tail_world
            else:
                value = [0,0,0]
                print("Error evaluating bvh nodes")
    return value

def translate_bvh_nodes (bvh_nodes, displacement):
    for node in bvh_nodes:
        node.rest_head_world = node.rest_head_world + displacement
        node.rest_tail_world = node.rest_tail_world + displacement

def get_child_names(bvh_node):
    child = []
    for chld in bvh_node.children:
        child.append(chld.name)
    return child

def get_children(bvh_nodes, name):

    res = []
    bvh_node = get_bvh_node(bvh_nodes, name)
    child = get_child_names(bvh_node)
    res.append(child)
    if not child:
        return []
    else:
        for ch in child:
            res += get_children(bvh_nodes, ch)

    return res

def rotate_bvh_joint (bvh_nodes, matrix, name):
    # 
    rot_center = get_bvh_node_val(bvh_nodes, name, 'HEAD')
    # translate all points
    translate_bvh_nodes (bvh_nodes, -rot_center)
    # rotate children
    children = get_children(bvh_nodes, name)
    # flatten list
    f_children = [val for sublist in children for val in sublist]
    # update matrix value of node name
    t_node = get_bvh_node(bvh_nodes, name)
    t_node.matrix = matrix @ t_node.matrix.to_3x3()

    for child in f_children:
        node = get_bvh_node(bvh_nodes, child)
        pt_head = node.rest_head_world
        pt_tail = node.rest_tail_world
        #new_pt_head = rotate_point(pt_head, matrix, rot_center)
        #new_pt_tail = rotate_point(pt_tail, matrix, rot_center)
        new_pt_head = matrix @ pt_head
        new_pt_tail = matrix @ pt_tail
        node.rest_head_world = new_pt_head
        node.rest_tail_world = new_pt_tail
        # update matrix
        node.matrix = matrix @ node.matrix.to_3x3()
    # set coordinates back
    translate_bvh_nodes (bvh_nodes, rot_center)




def get_skeleton_bvh_joints(bvh_nodes):

    jnts = []

    jnts.append(get_bvh_node_val(bvh_nodes, "Head", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "Neck", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "RightArm", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "RightForeArm", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "RightHand", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "LeftArm", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "LeftForeArm", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "LeftHand", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "RightUpLeg", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "RightLeg", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "RightFoot", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "LeftUpLeg", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "LeftLeg", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "LeftFoot", "HEAD"))
    jnts.append(get_bvh_node_val(bvh_nodes, "Hips", "HEAD"))

    return jnts

def get_skeleton_joints (obj):

    jnts = []

    jnts.append(get_bone_head_position(obj, "Head"))
    jnts.append(get_bone_head_position(obj, "Neck"))
    jnts.append(get_bone_head_position(obj, "RightArm"))
    jnts.append(get_bone_head_position(obj, "RightForeArm"))
    jnts.append(get_bone_head_position(obj, "RightHand"))
    jnts.append(get_bone_head_position(obj, "LeftArm"))
    jnts.append(get_bone_head_position(obj, "LeftForeArm"))
    jnts.append(get_bone_head_position(obj, "LeftHand"))
    jnts.append(get_bone_head_position(obj, "RightUpLeg"))
    jnts.append(get_bone_head_position(obj, "RightLeg"))
    jnts.append(get_bone_head_position(obj, "RightFoot"))
    jnts.append(get_bone_head_position(obj, "LeftUpLeg"))
    jnts.append(get_bone_head_position(obj, "LeftLeg"))
    jnts.append(get_bone_head_position(obj, "LeftFoot"))
    jnts.append(get_bone_head_position(obj, "Hips"))

    return jnts


def checkError(R,A,T,B): # UNUSED

    print("Calculating the error....")
    error = (R*A.T+tile(T,(1,4))).T-B
    print(error)
    error = multiply(error,error)
    error = sum(error)
    rmse = sqrt(error/4)
    print(rmse)
    print("IF RMSE IS NEAR ZERO, THE FUNCTION IS CORRECT")


def correct_pose(pts_skel,correction,scale):
    new_pts_skel = []
    pts_skel = pts_skel*scale
    hips = pts_skel[14,:]
    translation = hips - [correction[0],correction[1],correction[2]]
    for i in pts_skel:
        new_pts_skel.append([i[0]-translation[0],i[1]-translation[1],i[2]-translation[2]])

    return np.array(new_pts_skel)


def slerp(v0, v1, t_array):
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 @ v1)

    if (dot < 0.0):
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995

    if (dot > DOT_THRESHOLD):
        result = v0[np.newaxis,:] + t_array[:,np.newaxis] @ (v1 - v0)[np.newaxis,:]
        return (result.T / np.linalg.norm(result, axis=1)).T

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0[:,np.newaxis] @ v0[np.newaxis,:]) + (s1[:,np.newaxis] @ v1[np.newaxis,:])


def transition_to_desired_motion(q_list,initial_rotation,skel_basis,correction_iteration,mesh):

    list_of_rotations = []
    initial_quaternion = Quaternion((1,0,0,0))
    bone_name = ["Hips","Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]

    initial = slerp(initial_quaternion,initial_rotation,np.arange(0,1,0.05))
    list_of_rotations.append(initial)
    for i in range(len(q_list)):
        movements = slerp(initial_quaternion,q_list[i],np.arange(0,1,0.05))
        list_of_rotations.append(movements)

    scene = bpy.context.scene
    bones = ["Hips","LHipJoint","LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase","LowerBack","Spine","Spine1","LeftShoulder","LeftArm","LeftForeArm","LeftHand","LThumb","LeftFingerBase","LeftHandFinger1","Neck","Neck1","Head","RightShoulder","RightArm","RightForeArm","RightHand","RThumb","RightFingerBase","RightHandFinger1","RHipJoint","RightUpLeg","RightLeg","RightFoot","RightToeBase"]
    for step in range(len(list_of_rotations[0])):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        for i in range(len(bone_name)):
            #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            poseBone = skel_basis.pose.bones[bone_name[i]]
            poseBone.rotation_mode = "QUATERNION"
            poseBone.rotation_quaternion = list_of_rotations[i][step]

        #bpy.context.scene.frame_set(2+correction_iteration)
        skel_basis.keyframe_insert(data_path = "location", frame = 2 + correction_iteration)
        for bone in bones:
            skel_basis.pose.bones[bone].keyframe_insert(data_path = "rotation_quaternion", frame = 2 + correction_iteration)
        correction_iteration+=1
    return correction_iteration


def get_rest_pose (skel, list_bones):

    bone_mat = []
    for bone in list_bones:
        matrix = skel.pose.bones[bone].matrix.copy()
        bone_mat.append(matrix)

    return bone_mat

def get_rest_pose2 (skel, list_bones):

    for bone in list_bones:
        pb = skel.pose.bones[bone]
        print(pb.rotation_quaternion)

def set_rest_pose (skel, mat_bones, list_bones):
    for b_idx, bone in enumerate(list_bones):
        poseBone = skel.pose.bones[bone]
        poseBone.matrix = mat_bones[b_idx]
        bpy.context.view_layer.update()

def set_rest_pose2 (skel, mat_bones, list_bones):
    q1 = Quaternion([1,0,0,0])
    for b_idx, bone in enumerate(list_bones):
        poseBone = skel.pose.bones[bone]
        poseBone.rotation_mode = 'QUATERNION'
        poseBone.rotation_quaternion = q1
    bpy.context.view_layer.update()



def getcoords(Vector):
    points = []
    for i in Vector:
        points.append([i.x,i.y,i.z])
    return points

def set_pose (skel_basis):

#, Quaternion((0.9945854544639587, -0.0917830541729927, 1.519468355581921e-07, -0.04874023050069809)), Quaternion((0.9761632680892944, 0.14361903071403503, -2.0232151598520431e-07, -0.16272307932376862)), Quaternion((0.9989520907402039, -0.04325992614030838, -1.7270473051667068e-07, -0.014944829046726227)), Quaternion((0.9902194142341614, -0.006000734865665436, 0.011777707375586033, -0.1388911008834839)), Quaternion((0.9856129288673401, 0.09215943515300751, 1.2596402712006238e-06, -0.1416822075843811)), Quaternion((0.9865731000900269, -0.16097018122673035, -2.0918021164106904e-06, -0.027606511488556862)), Quaternion((0.9889397621154785, -0.09117575734853745, -0.04149548336863518, 0.10937654227018356)), Quaternion((0.9890621304512024, 0.07755535840988159, -1.1207237093913136e-06, 0.12546411156654358)), Quaternion((0.9873631000518799, -0.15127789974212646, 2.5151682621071814e-06, 0.04721295088529587))]

    poseBone = skel_basis.pose.bones["Hips"]
    poseBone.location = Vector((-2.8643, -11.4028, 4.7693))
    poseBone.rotation_mode = "QUATERNION"
    poseBone.rotation_quaternion = Quaternion((0.9922428131103516, -0.06077069416642189, 0.08913875371217728, 0.061769038438797))

    poseBone = skel_basis.pose.bones["Neck"]
    poseBone.rotation_mode = "QUATERNION"
    poseBone.rotation_quaternion = Quaternion((0.9787302613258362, -0.20506875216960907, 1.3352967620505751e-08, 0.005822303704917431))

    poseBone = skel_basis.pose.bones["LHipJoint"]
    poseBone.rotation_mode = "QUATERNION"
    poseBone.rotation_quaternion = Quaternion((0.9989989995956421, -0.03258729726076126, -1.0428419017216584e-07, 0.030644865706562996))

    poseBone = skel_basis.pose.bones["LeftUpLeg"]
    poseBone.rotation_mode = "QUATERNION"
    poseBone.rotation_quaternion = Quaternion((0.9887815713882446, 0.12731005251407623, 2.0658987409660767e-07, 0.07812276482582092))

    poseBone = skel_basis.pose.bones["LeftLeg"]
    poseBone.rotation_mode = "QUATERNION"
    poseBone.rotation_quaternion = Quaternion((0.9999021291732788, -0.013957100920379162, 5.3658322229921396e-08, 0.0009937164140865207))


def calculate_rotations2(bvh_nodes, list_matrices, goal_pts):

    # Use bvh nodes structure so we don't need to update scene to see changes (very slow)
    rotations = []

    ref_arm = get_skeleton_bvh_joints(bvh_nodes)
    ref_skel = np.array(ref_arm)
#    print(ref_arm)
    hips_pos = get_bvh_node_val(bvh_nodes, "Hips", "HEAD")

    A = np.mat((ref_skel[14,:], ref_skel[8,:], ref_skel[11,:], ref_skel[1,:]))
    B = np.mat((goal_pts[14,:], goal_pts[8,:], goal_pts[11,:], goal_pts[1,:]))
    R, T = rigid_transform_3D(A,B)

    mR = Matrix([[R[0,0],R[0,1],R[0,2]], [R[1,0],R[1,1],R[1,2]], [R[2,0],R[2,1],R[2,2]]])
    vT = Vector(T)

    # move arm2 to orient with pts_skel
    pts_r1 = []
    for vec in ref_skel: pts_r1.append(mR @ Vector(vec))
    pts_tr1 = []
    for vec in pts_r1: pts_tr1.append(vT+Vector(vec))
    skel_coords = pts_tr1

    vT = Vector(getcoords(skel_coords)[-1])
    # bone_translate_matrix = Matrix.Translation(vT)
    # boneRefPoseMtx = list_matrices[0]
    # loc = (boneRefPoseMtx.inverted() @ bone_translate_matrix).to_translation()
    # In the blender skeleton hips pivot point is in (0,0,0), therefore when we transform the 3D poinst
    # we need to accound for the real hips distance
    translate_bvh_nodes(bvh_nodes, vT - hips_pos)

    # rZ = Matrix([[0,1,0], [-1,0,0], [0,0,1]])
    # rotMtx = boneRefPoseMtx.to_3x3().inverted() @ mR @ boneRefPoseMtx.to_3x3()
    rotate_bvh_joint(bvh_nodes, mR, "Hips")

    hips_loc = vT
    hips_rot = mR    

    bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", 
                 "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]
    begin = [1, 14, 11, 12, 14, 8, 9, 1, 5, 6, 1, 2, 3]
    end = [0, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]

    # skel_coords = get_skeleton_bvh_joints(bvh_nodes)
    # print(skel_coords)


    for x in range(0, 3):
#    for x in range(0, 1):

        skel_coords = get_skeleton_bvh_joints(bvh_nodes)

        start_point_bone = Vector(skel_coords[begin[x]])
        end_point_bone = Vector(skel_coords[end[x]])
        goal_point_end_bone = Vector(goal_pts[end[x]])

        # print(start_point_bone)
        # print(end_point_bone)
        # print(goal_point_end_bone)

        # posebone only needed for matrix: need to save this in another structure
        # Need to pass Matrix structure from blender
        pbmatrix = list_matrices[x+1] # list_matrices[0] corresponds to hips
#        print(pbmatrix)
        mat = get_bvh_node_matrix(bvh_nodes, bone_name[x])
        if x == 2:
            mat2 = get_bvh_node_matrix(bvh_nodes, "Hips")
            print(mat2)
            mat1 = get_bvh_node_matrix(bvh_nodes, "LHipJoint")
            print(mat1)
            print(mat)
        q2 = compute_rotation(mat, start_point_bone, end_point_bone, goal_point_end_bone)
        rotations.append(q2)
        print(q2)

        rotate_bvh_joint(bvh_nodes, mat, bone_name[x])
        #list_matrices[x+1] = pbmatrix.to_3x3() @ q2.to_matrix()

    return hips_loc, hips_rot, rotations

def apply_rotations(skel_basis, hips_loc, hips_rot, rotations):

    poseBone = skel_basis.pose.bones["Hips"]
    boneRefPoseMtx = poseBone.bone.matrix_local.copy()
    bonePoseMtx = poseBone.matrix.to_3x3().copy()

#    #vT = Vector(getcoords(skel_coords)[-1])
    vT = hips_loc
    bone_translate_matrix = Matrix.Translation(vT)
    loc = (boneRefPoseMtx.inverted() @ bone_translate_matrix).to_translation()
    poseBone.location = loc

    mR = hips_rot
    poseBone.rotation_mode = "QUATERNION"
    rotMtx = boneRefPoseMtx.to_3x3().inverted() @ mR @ boneRefPoseMtx.to_3x3()
    poseBone.rotation_quaternion = rotMtx.to_quaternion()

    bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", 
                 "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]

    for x in range(0, 13):

        poseBone = skel_basis.pose.bones[bone_name[x]]

        poseBone.rotation_mode = "QUATERNION"
        poseBone.rotation_quaternion = rotations[x]

    bpy.context.view_layer.update()



def calculate_rotations(skel_basis, goal_pts):

    list_quaternions = []
    location = []

    ref_arm = get_skeleton_joints(skel_basis)
    ref_skel = np.array(ref_arm)

    A = np.mat((ref_skel[14,:], ref_skel[8,:], ref_skel[11,:], ref_skel[1,:]))
    B = np.mat((goal_pts[14,:], goal_pts[8,:], goal_pts[11,:], goal_pts[1,:]))
    R, T = rigid_transform_3D(A,B)

    mR = Matrix([[R[0,0],R[0,1],R[0,2]], [R[1,0],R[1,1],R[1,2]], [R[2,0],R[2,1],R[2,2]]])
    vT = Vector(T)

    # move arm2 to orient with pts_skel
    pts_r1 = []
    for vec in ref_skel: pts_r1.append(mR @ Vector(vec))
    pts_tr1 = []
    for vec in pts_r1: pts_tr1.append(vT+Vector(vec))
    skel_coords = pts_tr1

    poseBone = skel_basis.pose.bones["Neck"]
    print(poseBone.matrix)


    #mR.resize_4x4()
    # Update Hips position
    poseBone = skel_basis.pose.bones["Hips"]
    boneRefPoseMtx = poseBone.bone.matrix_local.copy()
#    bonePoseMtx = poseBone.matrix.to_3x3().copy()

    vT = Vector(getcoords(skel_coords)[-1])  # not sure what this line is doing here!!
    # When translate posebone hips actually is translating the whole object and takes origin as pivot point
    # need to account for this difference
    hips_bone_pos = get_bone_head_position(skel_basis, "Hips")
    bone_translate_matrix = Matrix.Translation(vT)
    loc = (boneRefPoseMtx.inverted() @ bone_translate_matrix).to_translation()
    poseBone.location = loc
    location = loc 

    #mR = Matrix([[0,-1,0], [1,0,0], [0,0,1]])
    poseBone.rotation_mode = "QUATERNION"
    rotMtx = boneRefPoseMtx.to_3x3().inverted() @ mR @ boneRefPoseMtx.to_3x3()
    poseBone.rotation_quaternion = rotMtx.to_quaternion()
    list_quaternions.append(rotMtx.to_quaternion())

    bpy.context.view_layer.update()

    # new_pts = get_skeleton_joints(skel_basis)
    # print(new_pts)

    #compute other rotations[
    bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", 
                 "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]
    begin = [1, 14, 11, 12, 14, 8, 9, 1, 5, 6, 1, 2, 3]
    end = [0, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]

    # rotation = []

    for x in range(0, 3):
#    for x in range(0, 1):

        bpy.context.view_layer.update()

        skel_coords = get_skeleton_joints(skel_basis)
        poseBone = skel_basis.pose.bones[bone_name[x]]

        start_point_bone = Vector(skel_coords[begin[x]])
        end_point_bone = Vector(skel_coords[end[x]])
        goal_point_end_bone = Vector(goal_pts[end[x]])

        # print(start_point_bone)
        # print(end_point_bone)
        # print(goal_point_end_bone)

        pb_matrix = poseBone.matrix # doesn't seem we need a copy version of matrix .copy()
        if x == 2:
            t_pb2 = skel_basis.pose.bones["Hips"]
            print(t_pb2.matrix)
            t_pb = skel_basis.pose.bones["LHipJoint"]
            print(t_pb.matrix)
            print(pb_matrix)
        q2 = compute_rotation(pb_matrix, start_point_bone, end_point_bone, goal_point_end_bone)
        poseBone.rotation_mode = "QUATERNION"
        poseBone.rotation_quaternion = q2
        list_quaternions.append(q2)
        print(q2)

    return location, list_quaternions

        # if (write_bvh):
        #     # convert rotation to bvh format
        #     mat_final = parentRefPoseMtx * parentPoseMtx.inverted() * bonePoseMtx * 
        #                 q2.to_matrix().to_4x4() * boneRefPoseMtx.inverted()	

        #     p = [degrees(mat_final.to_euler().z), degrees(mat_final.to_euler().y), degrees(mat_final.to_euler().x)]
        #     rotation.append(p)
           

    # # At this point we have the rotations for all bones
    # if (write_bvh):

    #     params = [p_hips_loc, p_hips_rot, rotation[0], rotation[1], [0,0,0], rotation[2], rotation[3], [0,0,0], [0,0,0],
    #               rotation[4], rotation[5], rotation[6], rotation[7], rotation[8], rotation[9], rotation[10]]
    #     # save params results
    #     list_params = [item for sublist in params for item in sublist]
    #     flat_params = ['{:.2f}'.format(x) for x in list_params]
    #     str_params = ' '.join(str(e) for e in flat_params)

    #     with open(bvh_ref_file, "a") as myfile:
    #         myfile.write(str_params)
    #         myfile.write("\n")

    # if (write_timeline):

    




def get_skeleton_parameters(skel_basis, goal_pts, correction_params):
    skel_params = []
    ref_arm = get_skeleton_joints(skel_basis)
    ref_skel = np.array(ref_arm)
    q_list = []

    A = np.mat((ref_skel[14,:], ref_skel[8,:], ref_skel[11,:], ref_skel[1,:]))
    B = np.mat((goal_pts[14,:], goal_pts[8,:], goal_pts[11,:], goal_pts[1,:]))
    R, T = rigid_transform_3D(A,B)

    mR = Matrix([[R[0,0],R[0,1],R[0,2]], [R[1,0],R[1,1],R[1,2]], [R[2,0],R[2,1],R[2,2]]])
    vT = Vector(T)

    # move arm2 to orient with pts_skel
    pts_r1 = []
    for vec in ref_skel: pts_r1.append(mR @ Vector(vec))
    pts_tr1 = []
    for vec in pts_r1: pts_tr1.append(vT+Vector(vec))
    skel_coords = pts_tr1
    #apply translation and first rotation to all skeleton
    bpy.context.view_layer.update()
    #mR.resize_4x4()
    poseBone = skel_basis.pose.bones["Hips"]
    boneRefPoseMtx = poseBone.bone.matrix_local.copy()
    bonePoseMtx = poseBone.matrix.to_3x3().copy()
    vT = Vector(getcoords(skel_coords)[-1])
    bone_translate_matrix = Matrix.Translation(vT)
    loc = (boneRefPoseMtx.inverted() @ bone_translate_matrix).to_translation()
    poseBone.location = loc
    rotMtx = boneRefPoseMtx.to_3x3().inverted() @ mR @ boneRefPoseMtx.to_3x3()
    initial_rotation = rotMtx.to_quaternion()
    poseBone.rotation_mode = "QUATERNION"

    poseBone.rotation_quaternion = rotMtx.to_quaternion()
    q_list.append(rotMtx.to_quaternion())
    p_hips_rot = [degrees(mR.to_euler().z), degrees(mR.to_euler().y), degrees(mR.to_euler().x)]
    p_hips_loc = [vT.x, vT.y, vT.z]
    reference = get_skeleton_joints(skel_basis)
    #compute other rotations[
    bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]
    begin = [1, 14, 11, 12, 14, 8, 9, 1, 5, 6, 1, 2, 3]
    end = [0, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]
    rotation = []
    previous_q = Quaternion([1,0,0,0])
    skel_coords = get_skeleton_joints(skel_basis)
    for x in range(0, 13):
        bpy.context.view_layer.update()
        #skel_coords = get_skeleton_coords(skel_basis)
        skel_coords = get_skeleton_joints(skel_basis)
        poseBone = skel_basis.pose.bones[bone_name[x]]
        boneRefPoseMtx = poseBone.bone.matrix_local.copy()
        parentRefPoseMtx = poseBone.parent.bone.matrix_local.copy()
        parentPoseMtx = poseBone.parent.matrix.copy()
        bonePoseMtx = poseBone.matrix.copy()
        #print(bonePoseMtx)
        # print("############ " + bone_name[x] + "  #############")
        start_point_bone = Vector(skel_coords[begin[x]])
        end_point_bone = Vector(skel_coords[end[x]])
        goal_point_end_bone = Vector(goal_pts[end[x]])
        #q2 = compute_rotation(poseBone, start_point_bone, end_point_bone, goal_point_end_bone)
        q2 = compute_rotation(bonePoseMtx.to_3x3(), start_point_bone, end_point_bone, goal_point_end_bone)
        poseBone.rotation_mode = "QUATERNION"
        poseBone.rotation_quaternion = q2
        q_list.append(q2)
        #mat_final = parentRefPoseMtx * parentPoseMtx.inverted() * bonePoseMtx * q2.to_matrix().to_4x4() * boneRefPoseMtx.inverted()
        mat_final = parentRefPoseMtx.to_3x3() @ parentPoseMtx.to_3x3().inverted() @ bonePoseMtx.to_3x3() @ q2.to_matrix().to_3x3() @ boneRefPoseMtx.to_3x3().inverted()
        #p = [degrees(mat_final.to_euler().z), degrees(mat_final.to_euler().y), degrees(mat_final.to_euler().x)]
        p =  [mat_final.to_euler().z,mat_final.to_euler().y,mat_final.to_euler().x]
        newp = [x - y for x, y in zip(p, correction_params[x])]

            ##POCA BROMA QUE AIXÒ CONVERGIA (GAIREBÉ TOT)
#        poseBone.rotation_mode = "ZYX"
#        poseBone.rotation_euler = Vector(newp)
        rotation.append(newp)



    skel_params.append(p_hips_loc)
    skel_params.append(p_hips_rot)
    skel_params.append(rotation[0])
    skel_params.append(rotation[1])
    skel_params.append([0,0,0])
    skel_params.append(rotation[2])
    skel_params.append(rotation[3])
    skel_params.append([0,0,0])
    skel_params.append([0,0,0])
    skel_params.append(rotation[4])
    skel_params.append(rotation[5])
    skel_params.append(rotation[6])
    skel_params.append(rotation[7])
    skel_params.append(rotation[8])
    skel_params.append(rotation[9])
    skel_params.append(rotation[10])
    previous_q = q2


    ## SKEL PARAMS ARE QUITE IMPORTANT TOO. THERE ARE THE ROTATIONS OF THE BONES (INVERSE CAN BE COMPUTED EZI)

    return q_list


def get_skeleton_parameters_correction(skel_basis, goal_pts, correction_params):
    skel_params = []
    ref_arm = get_skeleton_joints(skel_basis)
    ref_skel = np.array(ref_arm)
    q_list = []

    A = np.mat((ref_skel[14,:], ref_skel[8,:], ref_skel[11,:], ref_skel[1,:]))
    B = np.mat((goal_pts[14,:], goal_pts[8,:], goal_pts[11,:], goal_pts[1,:]))
    R, T = rigid_transform_3D(A,B)

    mR = Matrix([[R[0,0],R[0,1],R[0,2]], [R[1,0],R[1,1],R[1,2]], [R[2,0],R[2,1],R[2,2]]])
    vT = Vector(T)

    # move arm2 to orient with pts_skel
    pts_r1 = []
    for vec in ref_skel: pts_r1.append(mR @ Vector(vec))
    pts_tr1 = []
    for vec in pts_r1: pts_tr1.append(vT+Vector(vec))
    skel_coords = pts_tr1
    #apply translation and first rotation to all skeleton
    bpy.context.view_layer.update()
    #mR.resize_4x4()
    poseBone = skel_basis.pose.bones["Hips"]
    boneRefPoseMtx = poseBone.bone.matrix_local.copy()
    bonePoseMtx = poseBone.matrix.to_3x3().copy()
    vT = Vector(getcoords(skel_coords)[-1])
    bone_translate_matrix = Matrix.Translation(vT)
    loc = (boneRefPoseMtx.inverted() @ bone_translate_matrix).to_translation()
    poseBone.location = loc
    rotMtx = boneRefPoseMtx.to_3x3().inverted() @ mR @ boneRefPoseMtx.to_3x3()
    initial_rotation = rotMtx.to_quaternion()
    poseBone.rotation_mode = "QUATERNION"
    poseBone.rotation_quaternion = rotMtx.to_quaternion()
    p_hips_rot = [degrees(mR.to_euler().z), degrees(mR.to_euler().y), degrees(mR.to_euler().x)]
    p_hips_loc = [vT.x, vT.y, vT.z]
    reference = get_skeleton_joints(skel_basis)
    #compute other rotations[
    bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]
    begin = [1, 14, 11, 12, 14, 8, 9, 1, 5, 6, 1, 2, 3]
    end = [0, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]
    rotation = []
    previous_q = Quaternion([1,0,0,0])
    skel_coords = get_skeleton_joints(skel_basis)
    for x in range(len(bone_name)):
        bpy.context.view_layer.update()
        #skel_coords = get_skeleton_coords(skel_basis)
        skel_coords = get_skeleton_joints(skel_basis)
        poseBone = skel_basis.pose.bones[bone_name[x]]
        boneRefPoseMtx = poseBone.bone.matrix_local.copy()
        parentRefPoseMtx = poseBone.parent.bone.matrix_local.copy()
        parentPoseMtx = poseBone.parent.matrix.copy()
        bonePoseMtx = poseBone.matrix.copy()
        #print(bonePoseMtx)
        # print("############ " + bone_name[x] + "  #############")
        start_point_bone = Vector(skel_coords[begin[x]])
        end_point_bone = Vector(skel_coords[end[x]])
        goal_point_end_bone = Vector(goal_pts[end[x]])
        #q2 = compute_rotation(poseBone, start_point_bone, end_point_bone, goal_point_end_bone)
        q2 = compute_rotation(bonePoseMtx.to_3x3(), start_point_bone, end_point_bone, goal_point_end_bone)
        poseBone.rotation_mode = "QUATERNION"
        poseBone.rotation_quaternion = q2
        q_list.append(q2)
        #mat_final = parentRefPoseMtx * parentPoseMtx.inverted() * bonePoseMtx * q2.to_matrix().to_4x4() * boneRefPoseMtx.inverted()
        mat_final = parentRefPoseMtx.to_3x3() @ parentPoseMtx.to_3x3().inverted() @ bonePoseMtx.to_3x3() @ q2.to_matrix().to_3x3() @ boneRefPoseMtx.to_3x3().inverted()
        #p = [degrees(mat_final.to_euler().z), degrees(mat_final.to_euler().y), degrees(mat_final.to_euler().x)]
        p =  [mat_final.to_euler().z,mat_final.to_euler().y,mat_final.to_euler().x]
        newp = [x - y for x, y in zip(p, correction_params[x])]
        rotation.append(newp)




    skel_params.append(p_hips_loc)
    skel_params.append(p_hips_rot)
    skel_params.append(rotation[0])
    skel_params.append(rotation[1])
    skel_params.append([0,0,0])
    skel_params.append(rotation[2])
    skel_params.append(rotation[3])
    skel_params.append([0,0,0])
    skel_params.append([0,0,0])
    skel_params.append(rotation[4])
    skel_params.append(rotation[5])
    skel_params.append(rotation[6])
    skel_params.append(rotation[7])
    skel_params.append(rotation[8])
    skel_params.append(rotation[9])
    skel_params.append(rotation[10])



    ## SKEL PARAMS ARE QUITE IMPORTANT TOO. THERE ARE THE ROTATIONS OF THE BONES (INVERSE CAN BE COMPUTED EZI)

    return q_list, initial_rotation


def get_skeleton_parameters_correction_BVH(skel_basis, goal_pts, correction_params,extra):
    skel_params = []
    ref_arm = get_skeleton_joints(skel_basis)
    ref_skel = np.array(ref_arm)
    q_list = []
    provisional_list=[]
    Deg2Rad = pi/180

    for vector in goal_pts:
        provisional_list.append([vector.x,vector.y,vector.z])
    goal_pts = np.array(provisional_list)


    A = np.mat((ref_skel[14,:], ref_skel[8,:], ref_skel[11,:], ref_skel[1,:]))
    B = np.mat((goal_pts[14,:], goal_pts[8,:], goal_pts[11,:], goal_pts[1,:]))
    R, T = rigid_transform_3D(A,B)

    mR = Matrix([[R[0,0],R[0,1],R[0,2]], [R[1,0],R[1,1],R[1,2]], [R[2,0],R[2,1],R[2,2]]])
    vT = Vector(T)

    # move arm2 to orient with pts_skel
    pts_r1 = []
    for vec in ref_skel: pts_r1.append(mR @ Vector(vec))
    pts_tr1 = []
    for vec in pts_r1: pts_tr1.append(vT+Vector(vec))
    skel_coords = pts_tr1
    #apply translation and first rotation to all skeleton
    bpy.context.view_layer.update()
    #mR.resize_4x4()
    poseBone = skel_basis.pose.bones["Hips"]
    boneRefPoseMtx = poseBone.bone.matrix_local.copy()
    bonePoseMtx = poseBone.matrix.to_3x3().copy()
    vT = Vector(getcoords(skel_coords)[-1])
    bone_translate_matrix = Matrix.Translation(vT)
    loc = (boneRefPoseMtx.inverted() @ bone_translate_matrix).to_translation()
    poseBone.location = loc
    rotMtx = boneRefPoseMtx.to_3x3().inverted() @ mR @ boneRefPoseMtx.to_3x3()
    initial_rotation = rotMtx.to_quaternion()
    poseBone.rotation_mode = "QUATERNION"
    poseBone.rotation_quaternion = rotMtx.to_quaternion()
    p_hips_rot = [degrees(mR.to_euler().z), degrees(mR.to_euler().y), degrees(mR.to_euler().x)]
    p_hips_loc = [vT.x, vT.y, vT.z]
    reference = get_skeleton_joints(skel_basis)
    #compute other rotations[
    bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]
    begin = [1, 14, 11, 12, 14, 8, 9, 1, 5, 6, 1, 2, 3]
    end = [0, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]
    rotation = []
    previous_q = Quaternion([1,0,0,0])
    skel_coords = get_skeleton_joints(skel_basis)
    for x in range(len(bone_name)):
        bpy.context.view_layer.update()
        #skel_coords = get_skeleton_coords(skel_basis)
        skel_coords = get_skeleton_joints(skel_basis)
        poseBone = skel_basis.pose.bones[bone_name[x]]
        boneRefPoseMtx = poseBone.bone.matrix_local.copy()
        parentRefPoseMtx = poseBone.parent.bone.matrix_local.copy()
        parentPoseMtx = poseBone.parent.matrix.copy()
        bonePoseMtx = poseBone.matrix.copy()
        #print(bonePoseMtx)
        # print("############ " + bone_name[x] + "  #############")
        start_point_bone = Vector(skel_coords[begin[x]])
        end_point_bone = Vector(skel_coords[end[x]])
        goal_point_end_bone = Vector(goal_pts[end[x]])
        #q2 = compute_rotation(poseBone, start_point_bone, end_point_bone, goal_point_end_bone)
        q2 = compute_rotation(bonePoseMtx.to_3x3(), start_point_bone, end_point_bone, goal_point_end_bone)
        poseBone.rotation_mode = "QUATERNION"
        poseBone.rotation_quaternion = q2
        q_list.append(q2)
        #mat_final = parentRefPoseMtx * parentPoseMtx.inverted() * bonePoseMtx * q2.to_matrix().to_4x4() * boneRefPoseMtx.inverted()
        mat_final = parentRefPoseMtx.to_3x3() @ parentPoseMtx.to_3x3().inverted() @ bonePoseMtx.to_3x3() @ q2.to_matrix().to_3x3() @ boneRefPoseMtx.to_3x3().inverted()
        #p = [degrees(mat_final.to_euler().z), degrees(mat_final.to_euler().y), degrees(mat_final.to_euler().x)]
        p =  [mat_final.to_euler().z,mat_final.to_euler().y,mat_final.to_euler().x]
        newp = [x - y for x, y in zip(p, correction_params[x])]
        rotation.append(newp)




    skel_params.append(p_hips_loc)
    skel_params.append(p_hips_rot)
    skel_params.append(rotation[0])
    skel_params.append(rotation[1])
    skel_params.append([0,0,0])
    skel_params.append(rotation[2])
    skel_params.append(rotation[3])
    skel_params.append([0,0,0])
    skel_params.append([0,0,0])
    skel_params.append(rotation[4])
    skel_params.append(rotation[5])
    skel_params.append(rotation[6])
    skel_params.append(rotation[7])
    skel_params.append(rotation[8])
    skel_params.append(rotation[9])
    skel_params.append(rotation[10])



    ## SKEL PARAMS ARE QUITE IMPORTANT TOO. THERE ARE THE ROTATIONS OF THE BONES (INVERSE CAN BE COMPUTED EZI)

    return q_list, initial_rotation

def transition_to_desired_motion_BVH(q_list,initial_rotation,skel_basis,correction_iteration,mesh, initial_quaternion,offset):

    list_of_rotations = []

    bone_name = ["Hips","Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]

    initial = slerp(initial_quaternion,initial_rotation,np.arange(0,1,1/(offset+1)))
    list_of_rotations.append(initial)
    for i in range(len(q_list)):
        movements = slerp(initial_quaternion,q_list[i],np.arange(0,1,1/(offset+1)))
        list_of_rotations.append(movements)

    scene = bpy.context.scene
    bones = ["Hips","LHipJoint","LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase","LowerBack","Spine","Spine1","LeftShoulder","LeftArm","LeftForeArm","LeftHand","LThumb","LeftFingerBase","LeftHandFinger1","Neck","Neck1","Head","RightShoulder","RightArm","RightForeArm","RightHand","RThumb","RightFingerBase","RightHandFinger1","RHipJoint","RightUpLeg","RightLeg","RightFoot","RightToeBase"]
    for step in range(len(list_of_rotations[0])):
        for i in range(len(bone_name)):
            #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            poseBone = skel_basis.pose.bones[bone_name[i]]
            poseBone.rotation_mode = "QUATERNION"
            poseBone.rotation_quaternion = list_of_rotations[i][step]

        #bpy.context.scene.frame_set(2+correction_iteration)
        skel_basis.keyframe_insert(data_path = "location", frame = correction_iteration)
        for bone in bones:
            skel_basis.pose.bones[bone].keyframe_insert(data_path = "rotation_quaternion", frame = correction_iteration)
        correction_iteration+=1
    return correction_iteration
