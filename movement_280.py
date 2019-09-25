import os
import sys
import bpy
import math
from mathutils import Vector, Quaternion, Matrix
import numpy as np
from numpy import *
import mathutils
from bpy.props import *
from config import avt_path



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
	print("translation")
	print(translation)
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





def getcoords(Vector):
	points = []
	for i in Vector:
		points.append([i.x,i.y,i.z])
	return points

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


def get_skeleton_parameters_correction_BVH(skel_basis, goal_pts, correction_params):
	skel_params = []
	ref_arm = get_skeleton_joints(skel_basis)
	ref_skel = np.array(ref_arm)
	q_list = []
	provisional_list=[]

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

def transition_to_desired_motion_BVH(q_list,initial_rotation,skel_basis,correction_iteration,mesh, initial_quaternion):

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
		skel_basis.keyframe_insert(data_path = "location", frame = correction_iteration)
		for bone in bones:
			skel_basis.pose.bones[bone].keyframe_insert(data_path = "rotation_quaternion", frame = correction_iteration)
		correction_iteration+=1
	return correction_iteration
