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
