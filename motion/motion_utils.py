import os
import sys
import bpy
import math
from mathutils import Vector, Quaternion, Matrix
import numpy as np
from numpy import *
import mathutils
from bpy.props import *

import importlib

import bvh_utils
importlib.reload(bvh_utils)

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

def ralign(X,Y):
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1  
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t

    R = np.dot( np.dot(U, S ), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return c, R, t

def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(S, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t



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

def set_rest_pose3 (skel, mat_bones, mat_local):
    i = 0
    for bone in skel.pose.bones:
        bone.matrix_basis = mat_bones[i]
        i = i + 1
    i = 0
    for bone in skel.data.bones:
        bone.matrix_local = mat_local[i]
        i = i + 1

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


def matrix_world(armature_ob, bone_name):
    local = armature_ob.data.bones[bone_name].matrix_local
    basis = armature_ob.pose.bones[bone_name].matrix_basis

    parent = armature_ob.pose.bones[bone_name].parent
    if parent == None:
        return  local @ basis
    else:
        parent_local = armature_ob.data.bones[parent.name].matrix_local
        return matrix_world(armature_ob, parent.name) @ (parent_local.inverted() @ local) @ basis

def calculate_rotations_fast(skel_basis, bvh_nodes, goal_pts):

    # we keep 2 parallel structures, blender skeleton, and bvh_nodes
    # blender skeleton update matrices of bodies, but when reading values of skeleton are not updated
    # only way to update is to use scene_update function which makes calculations very slow
    # for this reason we need to update skeleton joints in a different structure
    # Other way to update matrices
    # https://stackoverflow.com/questions/13840418/force-matrix-world-to-be-recalculated-in-blender

    ref_arm = bvh_utils.get_skeleton_bvh_joints(bvh_nodes)
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

    poseBone = skel_basis.pose.bones["Hips"]
    boneRefPoseMtx = poseBone.bone.matrix_local.copy()

    # In the blender skeleton hips pivot point is in (0,0,0), therefore when we transform the 3D poinst
    # we need to accound for the real hips distance
    # hips_pos = bvh_utils.get_bvh_node_val(bvh_nodes, "Hips", "HEAD")
    # bvh_utils.translate_bvh_nodes(bvh_nodes, vT - hips_pos)
    vT = Vector(getcoords(skel_coords)[-1])
    bone_translate_matrix = Matrix.Translation(vT)
    loc = (boneRefPoseMtx.inverted() @ bone_translate_matrix).to_translation()
    poseBone.location = loc

    poseBone.rotation_mode = "QUATERNION"
    rotMtx = boneRefPoseMtx.to_3x3().inverted() @ mR @ boneRefPoseMtx.to_3x3()
    poseBone.rotation_quaternion = rotMtx.to_quaternion()
    # bvh_utils.rotate_bvh_joint(bvh_nodes, mR, "Hips")


    bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", 
                 "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]
    bone_end = ["Head", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
                "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand"]
    begin = [1, 14, 11, 12, 14, 8, 9, 1, 5, 6, 1, 2, 3]
    end = [0, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]


    for x in range(0, 13):

        # get current bone
        poseBone = skel_basis.pose.bones[bone_name[x]]

        # find head and tail bone matrices and locations
        mat1 = matrix_world(skel_basis, bone_name[x]) # bone head
        pt_head = mat1.to_translation()
        mat2 = matrix_world(skel_basis, bone_end[x]) # bone tail (acutally bone's head child)
        pt_tail = mat2.to_translation()

        # find rotation to apply to bone
        q2 = compute_rotation(mat1, pt_head, pt_tail, Vector(goal_pts[end[x]]))
        poseBone.rotation_mode = "QUATERNION"
        poseBone.rotation_quaternion = q2



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

    # Update Hips position
    poseBone = skel_basis.pose.bones["Hips"]
    boneRefPoseMtx = poseBone.bone.matrix_local.copy()

    vT = Vector(getcoords(skel_coords)[-1])  
    # When translate posebone hips actually is translating the whole object and takes origin as pivot point
    # need to account for this difference
    hips_bone_pos = get_bone_head_position(skel_basis, "Hips")
    bone_translate_matrix = Matrix.Translation(vT)
    loc = (boneRefPoseMtx.inverted() @ bone_translate_matrix).to_translation()
    poseBone.location = loc
    location = loc 

    poseBone.rotation_mode = "QUATERNION"
    rotMtx = boneRefPoseMtx.to_3x3().inverted() @ mR @ boneRefPoseMtx.to_3x3()
    poseBone.rotation_quaternion = rotMtx.to_quaternion()
    list_quaternions.append(rotMtx.to_quaternion())

    bpy.context.view_layer.update()

    #compute other rotations[
    bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", 
                 "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]
    begin = [1, 14, 11, 12, 14, 8, 9, 1, 5, 6, 1, 2, 3]
    end = [0, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]

    for x in range(0, 13):

        bpy.context.view_layer.update()

        skel_coords = get_skeleton_joints(skel_basis)
        poseBone = skel_basis.pose.bones[bone_name[x]]

        start_point_bone = Vector(skel_coords[begin[x]])
        end_point_bone = Vector(skel_coords[end[x]])
        goal_point_end_bone = Vector(goal_pts[end[x]])

        poseBone = skel_basis.pose.bones[bone_name[x]]
        pb_matrix = poseBone.matrix # doesn't seem we need a copy version of matrix .copy()

        q2 = compute_rotation(pb_matrix, start_point_bone, end_point_bone, goal_point_end_bone)
        poseBone.rotation_mode = "QUATERNION"
        poseBone.rotation_quaternion = q2
        list_quaternions.append(q2)

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

