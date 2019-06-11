import bpy
import math
from mathutils import Vector, Quaternion, Matrix
from numpy import *
#from math import sqrt
import numpy as np
import os
import time
from time import sleep

### This code works only if the original_position of the avatar is the position when imported.

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
       #print ("Reflection detected")
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    #print (t)

    return R, t


def trans_coord_system(p1, o1, o2, M1, M2):

	# note is necessary to use a copy version of the matrix
	# otherwise it modifies the content of M2 outside the function
	# Actually it transposes matrix M2 outside the function. Must be the way
	# Blender handles the transform operators
	M2t = M2.copy()
	M2t.transpose()
	return M2t * (o1 - o2 + M1 * p1)

def compute_rotation(poseBone, pt0, pt1, pt2):
    #pt0 -> start_point_bone
    #pt1 -> end_point_bone
    #pt2 -> goal_point_bone

	M1 = Matrix([[1,0,0], [0,1,0], [0,0,1]])
    ### He afegit un canvi aquí (el .to_3x3(), a priori sembla que ha millorat.. )
	#M2 = poseBone.matrix.to_3x3().copy()
	M2 = poseBone.copy()

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


def get_bone_tail_position(obj, bone_name):
    return (obj.matrix_world * Matrix.Translation(obj.pose.bones[bone_name].tail)).to_translation()

def get_bone_head_position(obj, bone_name):
    return (obj.matrix_world * Matrix.Translation(obj.pose.bones[bone_name].head)).to_translation()

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

def get_skeleton_coords (skel):

	pts_skel = []

	pt1 = skel.location + skel.pose.bones["Head"].head
	pts_skel.append(pt1)
	pt2 = skel.location + skel.pose.bones["Spine1"].tail
	pts_skel.append(pt2)
	pt3 = skel.location + skel.pose.bones["RightShoulder"].tail
	pts_skel.append(pt3)
	pt4 = skel.location + skel.pose.bones["RightArm"].tail
	pts_skel.append(pt4)
	pt5 = skel.location + skel.pose.bones["RightForeArm"].tail
	pts_skel.append(pt5)
	pt6 = skel.location + skel.pose.bones["LeftShoulder"].tail
	pts_skel.append(pt6)
	pt7 = skel.location + skel.pose.bones["LeftArm"].tail
	pts_skel.append(pt7)
	pt8 = skel.location + skel.pose.bones["LeftForeArm"].tail
	pts_skel.append(pt8)
	pt9 = skel.location + skel.pose.bones["RightUpLeg"].head
	pts_skel.append(pt9)
	pt10 = skel.location + skel.pose.bones["RightLeg"].head
	pts_skel.append(pt10)
	pt11 = skel.location + skel.pose.bones["RightLeg"].tail #--> food.head
	pts_skel.append(pt11)
	pt12 = skel.location + skel.pose.bones["LeftUpLeg"].head
	pts_skel.append(pt12)
	pt13 = skel.location + skel.pose.bones["LeftLeg"].head
	pts_skel.append(pt13)
	pt14 = skel.location + skel.pose.bones["LeftLeg"].tail #--> food.head
	pts_skel.append(pt14)
	pt15 = skel.location + skel.pose.bones["Hips"].head
	pts_skel.append(pt15)
	return pts_skel

def checkError(R,A,T,B):

    print("Calculating the error.... won't take long")
    error = (R*A.T+tile(T,(1,4))).T-B
    print(error)
    error = multiply(error,error)
    error = sum(error)
    rmse = sqrt(error/4)
    print(rmse)
    print("IF RMSE IS NEAR ZERO, THE FUNCTION IS CORRECT")


def getcoords(Vector):
	points = []
	for i in Vector:
		points.append([i.x,i.y,i.z])
	return points

def get_skeleton_parameters (skel_basis, goal_pts, correction_params):

    # output variable
    skel_params = []

#    arm2 = skel_basis
#    pts_skel = goal_points

    # need to get skeleton points (ref_skel)
    #ref_arm = get_skeleton_coords(skel_basis)
    ref_arm = get_skeleton_joints(skel_basis)
    ref_skel = np.array(ref_arm)

    # compute translation and first rotation between rest position and desired points
    A = np.mat((ref_skel[14,:], ref_skel[8,:], ref_skel[11,:], ref_skel[1,:]))
    B = np.mat((goal_pts[14,:], goal_pts[8,:], goal_pts[11,:], goal_pts[1,:]))
    R, T = rigid_transform_3D(A,B)
   # checkError(R,A,T,B)



    mR = Matrix([[R[0,0],R[0,1],R[0,2]], [R[1,0],R[1,1],R[1,2]], [R[2,0],R[2,1],R[2,2]]])
    vT = Vector(T)

    # move arm2 to orient with pts_skel
    pts_r1 = []
    for vec in ref_skel: pts_r1.append(mR*Vector(vec))
    pts_tr1 = []
    for vec in pts_r1: pts_tr1.append(vT+Vector(vec))

    skel_coords = pts_tr1

    #apply translation and first rotation to all skeleton
    bpy.context.scene.update()

    #mR.resize_4x4()

    poseBone = skel_basis.pose.bones["Hips"]

    boneRefPoseMtx = poseBone.bone.matrix_local.copy()
    bonePoseMtx = poseBone.matrix.to_3x3().copy()


    vT = Vector(getcoords(skel_coords)[-1])

    bone_translate_matrix = Matrix.Translation(vT)
    loc = (boneRefPoseMtx.inverted() * bone_translate_matrix).to_translation()
    poseBone.location = loc

    rotMtx = boneRefPoseMtx.to_3x3().inverted() * mR * boneRefPoseMtx.to_3x3()

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

    for x in range(0, 13):

        bpy.context.scene.update()
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

        #mat_final = parentRefPoseMtx * parentPoseMtx.inverted() * bonePoseMtx * q2.to_matrix().to_4x4() * boneRefPoseMtx.inverted()
        mat_final = parentRefPoseMtx.to_3x3() * parentPoseMtx.to_3x3().inverted() * bonePoseMtx.to_3x3() * q2.to_matrix().to_3x3() * boneRefPoseMtx.to_3x3().inverted()
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

    return reference




def get_skeleton_parameters2(skel_basis, goal_pts, correction_params, reference):

    # output variable
    skel_params = []
    reference = reference.copy()

#    arm2 = skel_basis
#    pts_skel = goal_points

    # need to get skeleton points (ref_skel)
    #ref_arm = get_skeleton_coords(skel_basis)
    #ref_arm = get_skeleton_joints(reference)
    ref_skel = np.array(reference)

    # compute translation and first rotation between rest position and desired points
    A = np.mat((ref_skel[14,:], ref_skel[8,:], ref_skel[11,:], ref_skel[1,:]))
    B = np.mat((goal_pts[14,:], goal_pts[8,:], goal_pts[11,:], goal_pts[1,:]))
    R, T = rigid_transform_3D(A,B)
   # checkError(R,A,T,B)



    mR = Matrix([[R[0,0],R[0,1],R[0,2]], [R[1,0],R[1,1],R[1,2]], [R[2,0],R[2,1],R[2,2]]])
    vT = Vector(T)

    # move arm2 to orient with pts_skel
    pts_r1 = []
    for vec in ref_skel: pts_r1.append(mR*Vector(vec))
    pts_tr1 = []
    for vec in pts_r1: pts_tr1.append(vT+Vector(vec))

    skel_coords = pts_tr1

    #apply translation and first rotation to all skeleton
    bpy.context.scene.update()

    #mR.resize_4x4()

    poseBone = skel_basis.pose.bones["Hips"]

    boneRefPoseMtx = poseBone.bone.matrix_local.copy()
    bonePoseMtx = poseBone.matrix.to_3x3().copy()


    vT = Vector(getcoords(skel_coords)[-1])

    bone_translate_matrix = Matrix.Translation(vT)
    loc = (boneRefPoseMtx.inverted() * bone_translate_matrix).to_translation()
    poseBone.location = loc

    rotMtx = boneRefPoseMtx.to_3x3().inverted() * mR * boneRefPoseMtx.to_3x3()

    poseBone.rotation_mode = "QUATERNION"
    poseBone.rotation_quaternion = rotMtx.to_quaternion()

    p_hips_rot = [degrees(mR.to_euler().z), degrees(mR.to_euler().y), degrees(mR.to_euler().x)]
    p_hips_loc = [vT.x, vT.y, vT.z]


    #compute other rotations[
    bone_name = ["Neck","LHipJoint","LeftUpLeg", "LeftLeg", "RHipJoint", "RightUpLeg", "RightLeg", "LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"]
    begin = [1, 14, 11, 12, 14, 8, 9, 1, 5, 6, 1, 2, 3]
    end = [0, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]

    rotation = []
    previous_q = Quaternion([1,0,0,0])



#    print("|||||***** This are the goal points for the end points bones ****|||||")
#    for x in range(0,13):
#        print(goal_pts[end[x]])


    for x in range(0, 13):

        bpy.context.scene.update()
        #skel_coords = get_skeleton_coords(skel_basis)
        skel_coords = get_skeleton_joints(skel_basis)



        poseBone = skel_basis.pose.bones[bone_name[x]]

        boneRefPoseMtx = poseBone.bone.matrix_local.copy()
        parentRefPoseMtx = poseBone.parent.bone.matrix_local.copy()
        parentPoseMtx = poseBone.parent.matrix.copy()
        bonePoseMtx = poseBone.matrix.copy()
        print(bonePoseMtx)

        start_point_bone = Vector(skel_coords[begin[x]])
        end_point_bone = Vector(skel_coords[end[x]])
        goal_point_end_bone = Vector(goal_pts[end[x]])


        #q2 = compute_rotation(poseBone, start_point_bone, end_point_bone, goal_point_end_bone)
        q2 = compute_rotation(bonePoseMtx.to_3x3(), start_point_bone, end_point_bone, goal_point_end_bone)

        poseBone.rotation_mode = "QUATERNION"
        poseBone.rotation_quaternion = q2
#        print("THE QUATERNION FOR THIS BONE IS")
#        print(q2)

        #mat_final = parentRefPoseMtx * parentPoseMtx.inverted() * bonePoseMtx * q2.to_matrix().to_4x4() * boneRefPoseMtx.inverted()
        mat_final = parentRefPoseMtx.to_3x3() * parentPoseMtx.to_3x3().inverted() * bonePoseMtx.to_3x3() * q2.to_matrix().to_3x3() * boneRefPoseMtx.to_3x3().inverted()
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

    return skel_params


##############################################################################################################

# MAIN CODE

##############################################################################################################

context = bpy.context
path_input="/home/aniol/IRI/DadesMarta/frames"
path_output="/home/aniol/IRI/DadesMarta/sequences"

# Set rotation matrix to transform from Matlab coordinates to Blender coordinates
#
rx = math.radians(90)
ry = math.radians(0)
rz = math.radians(0)

M_rx = np.array([[1,0,0],[0,math.cos(rx),math.sin(rx)],[0,-math.sin(rx),math.cos(rx)]])
M_ry = np.array([[math.cos(ry),0,-math.sin(ry)],[0,1,0],[math.sin(ry),0,math.cos(ry)]])
M_rz = np.array([[math.cos(rz),math.sin(rz),0],[-math.sin(rz),math.cos(rz),0],[0,0,1]])


## Rotation Matrix from rotations rx ry rz
M_mb1 = np.matmul(M_rx, M_ry)
M_mb = np.matmul(M_mb1, M_rz)
model = "Standard"


correction_params = np.zeros((14,3),dtype=np.float32)
f = 1
#bpy.ops.import_scene.makehuman_mhx2(filter_glob = "Model02.mhx2",filepath = "/home/aniol/IRI/DadesMarta/models/Model02.mhx2")
arm2 = bpy.data.objects[model]
original_position = []
print("**** INITIAL MATRIX DISTRIBUTION ****")
bones = ["Hips","LHipJoint","LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase","LowerBack","Spine","Spine1","LeftShoulder","LeftArm","LeftForeArm","LeftHand","LThumb","LeftFingerBase","LeftHandFinger1","Neck","Neck1","Head","RightShoulder","RightArm","RightForeArm","RightHand","RThumb","RightFingerBase","RightHandFinger1","RHipJoint","RightUpLeg","RightLeg","RightFoot","RightToeBase"]
for i in range(len(bones)):
    #print(bones[i])
    bone = bones[i]
    matrix = arm2.pose.bones[bone].matrix.copy()
    original_position.append(matrix)
    #print(matrix)

while f<300:

    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    # Blender Modal Operator  (buscar més info)
    #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1, time_limit=0.0)

    if f == 1:
        ref = original_position.copy()
        for i in range(len(bones)):
            #print(bones[i])
            bone = bones[i]
            poseBone = arm2.pose.bones[bone]
            poseBone.matrix = ref[i]
            bpy.context.scene.update()
        print("RUNNING EXPERIMENT NUMBER: " + str(f))
        bpy.context.scene.update()
        fname = "frame_SA%02d_%05d.txt" % (2, f)
        print(fname)
        fpname = "%s/%s" % (path_input,fname)
        pts_skel = loadtxt(fpname)
        # adjust 3D points axis to Blender axis
        pts_skel = np.matmul(pts_skel, M_mb)
#        print("############### ORIGINAL skeleton params ################")
####        reference_skel_coords = get_skeleton_parameters(arm2,pts_skel,correction_params)
#        for x in range(0,15):
#            print([reference_skel_coords[x].x,reference_skel_coords[x].y,reference_skel_coords[x].z])
#        print("%%%%%%%%%% Coords at step 40 %%%%%%%%%%%%%%%%")
        skel_coords = get_skeleton_joints(arm2)
#        for x in range(0,15):
#            print([skel_coords[x].x,skel_coords[x].y,skel_coords[x].z])


    else:
        ref = original_position.copy()
        for i in range(len(bones)):
            #print(bones[i])
            bone = bones[i]
            poseBone = arm2.pose.bones[bone]
            poseBone.matrix = ref[i]
            bpy.context.scene.update()
        print("RUNNING EXPERIMENT NUMBER: " + str(f))
        bpy.context.scene.update()
        fname = "frame_SA%02d_%05d.txt" % (2, f)
        print(fname)
        fpname = "%s/%s" % (path_input,fname)
        pts_skel = loadtxt(fpname)
        # adjust 3D points axis to Blender axis
        pts_skel = np.matmul(pts_skel, M_mb)
        #print("############### ORIGINAL skeleton params ################")
        params = get_skeleton_parameters(arm2,pts_skel,correction_params)

    f+=1
