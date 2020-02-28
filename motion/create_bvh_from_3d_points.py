import bpy
import math
from shutil import copyfile
from mathutils import Vector, Quaternion, Matrix
from math import sqrt, degrees
import numpy as np

# This file takes as input a set of files which contain 3D points in format x1 y1 z1; x2 y2 z2; ... ; x15 y15 z15;
# and writes the necessary transformations to match a skeleton with the 3d points in bvh format.

def rigid_transform_3D(A, B):
	assert len(A) == len(B)

	N = A.shape[0] # total points

	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)
    
	# centre the points
	AA = A - np.tile(centroid_A, (N, 1))
	BB = B - np.tile(centroid_B, (N, 1))

	# dot is matrix multiplication for array
	H = np.transpose(AA) * BB

	U, S, Vt = np.linalg.svd(H)

	R = Vt.T * U.T

	# special reflection case
	if np.linalg.det(R) < 0:
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
	
	M1 = Matrix([[1,0,0], [0,1,0], [0,0,1]])
	M2 = poseBone.matrix.copy() 

	v1 = trans_coord_system(pt1, Vector((0,0,0)), pt0, M1, M2)
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
	else: theta = theta = math.pi - math.asin(v_magnitude)
	
	if (c.magnitude>0):
		axis = c.normalized()
		st2 = math.sin(theta/2)
		q = Quaternion( [math.cos(theta/2), st2*axis.x, st2*axis.y, st2*axis.z] )
	else:
		q = Quaternion( [1,0,0,0] )
	
	return q

def get_bone_head_position(obj, bone_name):
	return (obj.matrix_world * Matrix.Translation(obj.pose.bones[bone_name].head)).to_translation()

def get_bone_tail_position(obj, bone_name):
	return (obj.matrix_world * Matrix.Translation(obj.pose.bones[bone_name].tail)).to_translation()

def get_skeleton_coords (skel):

	pts_skel = []

	pt1 = get_bone_tail_position(skel, "Neck")
	pts_skel.append(pt1)
	pt2 = get_bone_tail_position(skel, "Chest")
	pts_skel.append(pt2)
	pt3 = get_bone_tail_position(skel, "RightCollar")
	pts_skel.append(pt3)
	pt4 = get_bone_tail_position(skel, "RightUpArm")
	pts_skel.append(pt4)
	pt5 = get_bone_tail_position(skel, "RightLowArm")
	pts_skel.append(pt5)
	pt6 = get_bone_tail_position(skel, "LeftCollar")
	pts_skel.append(pt6)
	pt7 = get_bone_tail_position(skel, "LeftUpArm")
	pts_skel.append(pt7)
	pt8 = get_bone_tail_position(skel, "LeftLowArm")
	pts_skel.append(pt8)
	pt9 = get_bone_head_position(skel, "RightUpLeg")
	pts_skel.append(pt9)
	pt10 = get_bone_head_position(skel, "RightLowLeg")
	pts_skel.append(pt10)
	pt11 = get_bone_tail_position(skel, "RightLowLeg")
	pts_skel.append(pt11)
	pt12 = get_bone_head_position(skel, "LeftUpLeg")
	pts_skel.append(pt12)
	pt13 = get_bone_head_position(skel, "LeftLowLeg")
	pts_skel.append(pt13)
	pt14 = get_bone_tail_position(skel, "LeftLowLeg")
	pts_skel.append(pt14)
	pt15 = get_bone_head_position(skel, "Hips")
	pts_skel.append(pt15)

	return pts_skel

def get_skeleton_parameters (skel_basis, goal_points):
    
	# output variable
	skel_params = []
	
	arm2 = skel_basis
	pts_skel = goal_points
	
	# need to get skeleton points (ref_skel)
	ref_arm = get_skeleton_coords (arm2)
	ref_skel = np.array(ref_arm)
	
	# compute translation and first rotation between rest position and desired points
	A = np.mat((ref_skel[14,:], ref_skel[8,:], ref_skel[11,:], ref_skel[1,:]))
	B = np.mat((pts_skel[14,:], pts_skel[8,:], pts_skel[11,:], pts_skel[1,:]))
	R, T = rigid_transform_3D(A,B)
	
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
	
	mR.resize_4x4()

	poseBone = arm2.pose.bones["Hips"]

	boneRefPoseMtx = poseBone.bone.matrix_local.copy()
	bonePoseMtx = poseBone.matrix.copy()
	
	bone_translate_matrix = Matrix.Translation(vT)
	loc = (boneRefPoseMtx.inverted() * bone_translate_matrix).to_translation()
	poseBone.location = loc	
	
	rotMtx = boneRefPoseMtx.inverted() * mR * boneRefPoseMtx

	poseBone.rotation_mode = "QUATERNION"
	poseBone.rotation_quaternion = rotMtx.to_quaternion()

	p_hips_rot = [degrees(mR.to_euler().z), degrees(mR.to_euler().y), degrees(mR.to_euler().x)] 
	p_hips_loc = [vT.x, vT.y, vT.z]
	
	#compute other rotations
	bone_name = ["LeftUpLeg", "LeftLowLeg", "RightUpLeg", "RightLowLeg", "LeftCollar", "LeftUpArm", "LeftLowArm", "RightCollar", "RightUpArm", "RightLowArm", "Neck"]
	begin = [11, 12, 8, 9, 1, 5, 6, 1, 2, 3, 1]
	end = [12, 13, 9, 10, 5, 6, 7, 2, 3, 4, 0]
	
	rotation = []
	
	for x in range(0, 11):
		bpy.context.scene.update()
		skel_coords = get_skeleton_coords (arm2)

		poseBone = arm2.pose.bones[bone_name[x]]
		
		boneRefPoseMtx = poseBone.bone.matrix_local.copy()
		parentRefPoseMtx = poseBone.parent.bone.matrix_local.copy()
		parentPoseMtx = poseBone.parent.matrix.copy()
		bonePoseMtx = poseBone.matrix.copy()

		pt0 = skel_coords[begin[x]]
		pt1 = skel_coords[end[x]]
		pt2 = Vector(pts_skel[end[x],:])
		
		q2 = compute_rotation(poseBone, pt0, pt1, pt2)

		poseBone.rotation_mode = "QUATERNION"
		poseBone.rotation_quaternion = q2

		mat_final = parentRefPoseMtx * parentPoseMtx.inverted() * bonePoseMtx * q2.to_matrix().to_4x4() * boneRefPoseMtx.inverted()	

		p = [degrees(mat_final.to_euler().z), degrees(mat_final.to_euler().y), degrees(mat_final.to_euler().x)]
		rotation.append(p)
	
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
	return skel_params

def total_error(pts1, pts2):
	
	vect_error = []
	for x in range(0,15):
		vect_error.append(np.linalg.norm(pts1[x]-pts2[x]))
	return sum(vect_error)


#####################
# Begin Code


print("START")

path="/home/jsanchez/Software/gitprojects/avatar/motion/frames"
context = bpy.context

# Write a reference skeleton using 3d points of the first frame
fname = "frame_SA02_00001.txt"
fpname = "%s/%s" % (path, fname)

skelref = np.loadtxt(fpname)

for x in range(0,15):
	skelref[x] = skelref[x]-skelref[14]

with open("/home/jsanchez/Software/gitprojects/avatar/motion/frames/Reference.bvh", "a") as myfile:
	myfile.write("HIERARCHY\n")
	myfile.write("ROOT Hips\n")
	myfile.write("{\n")
	myfile.write("\t OFFSET 0.00 0.00 0.00\n")
	myfile.write("\t CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n")
	myfile.write("\t JOINT LeftUpLeg\n")
	myfile.write("\t {\n")
	myfile.write("\t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[11][0],skelref[11][1],skelref[11][2]))
	myfile.write("\t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t JOINT LeftLowLeg\n")
	myfile.write("\t \t {\n")
	myfile.write("\t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[12][0]-skelref[11][0],skelref[12][1]-skelref[11][1],skelref[12][2]-skelref[11][2]))
	myfile.write("\t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t JOINT LeftFoot\n")
	myfile.write("\t \t \t {\n")
	myfile.write("\t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[13][0]-skelref[12][0],skelref[13][1]-skelref[12][1],skelref[13][2]-skelref[12][2]))
	myfile.write("\t \t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t \t End Site\n")
	myfile.write("\t \t \t \t {\n")
	myfile.write("\t \t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (0, -0.1, 0))
	myfile.write("\t \t \t \t }\n")
	myfile.write("\t \t \t }\n")
	myfile.write("\t \t }\n")
	myfile.write("\t }\n")

	myfile.write("\t JOINT RightUpLeg\n")
	myfile.write("\t {\n")
	myfile.write("\t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[8][0],skelref[8][1],skelref[8][2]))
	myfile.write("\t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t JOINT RightLowLeg\n")
	myfile.write("\t \t {\n")
	myfile.write("\t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[9][0]-skelref[8][0],skelref[9][1]-skelref[8][1],skelref[9][2]-skelref[8][2]))
	myfile.write("\t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t JOINT RightFoot\n")
	myfile.write("\t \t \t {\n")
	myfile.write("\t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[10][0]-skelref[9][0],skelref[10][1]-skelref[9][1],skelref[10][2]-skelref[9][2]))
	myfile.write("\t \t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t \t End Site\n");
	myfile.write("\t \t \t \t {\n")
	myfile.write("\t \t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (0, -0.1, 0))
	myfile.write("\t \t \t \t }\n")
	myfile.write("\t \t \t }\n")
	myfile.write("\t \t }\n")
	myfile.write("\t }\n")

	myfile.write("\t JOINT Chest\n")
	myfile.write("\t {\n")
	myfile.write("\t \t OFFSET %2.2f %2.2f %2.2f\n" % (0, 0, 0))
	myfile.write("\t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t JOINT LeftCollar\n")
	myfile.write("\t \t {\n")
	myfile.write("\t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[1][0],skelref[1][1],skelref[1][2]))
	myfile.write("\t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t JOINT LeftUpArm\n")
	myfile.write("\t \t \t {\n")
	myfile.write("\t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[5][0]-skelref[1][0],skelref[5][1]-skelref[1][1],skelref[5][2]-skelref[1][2]))
	myfile.write("\t \t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t \t JOINT LeftLowArm\n")
	myfile.write("\t \t \t \t {\n")
	myfile.write("\t \t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[6][0]-skelref[5][0],skelref[6][1]-skelref[5][1],skelref[6][2]-skelref[5][2]))
	myfile.write("\t \t \t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t \t \t End Site\n")
	myfile.write("\t \t \t \t \t {\n")
	myfile.write("\t \t \t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[7][0]-skelref[6][0],skelref[7][1]-skelref[6][1],skelref[7][2]-skelref[6][2]))
	myfile.write("\t \t \t \t \t }\n")
	myfile.write("\t \t \t \t }\n")
	myfile.write("\t \t \t }\n")
	myfile.write("\t \t }\n")

	myfile.write("\t \t JOINT RightCollar\n")
	myfile.write("\t \t {\n")
	myfile.write("\t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[1][0],skelref[1][1],skelref[1][2]))
	myfile.write("\t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t JOINT RightUpArm\n")
	myfile.write("\t \t \t {\n")
	myfile.write("\t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[2][0]-skelref[1][0],skelref[2][1]-skelref[1][1],skelref[2][2]-skelref[1][2]))
	myfile.write("\t \t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t \t JOINT RightLowArm\n")
	myfile.write("\t \t \t \t {\n")
	myfile.write("\t \t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[3][0]-skelref[2][0],skelref[3][1]-skelref[2][1],skelref[3][2]-skelref[2][2]))
	myfile.write("\t \t \t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t \t \t End Site\n")
	myfile.write("\t \t \t \t \t {\n")
	myfile.write("\t \t \t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[4][0]-skelref[3][0],skelref[4][1]-skelref[3][1],skelref[4][2]-skelref[3][2]))
	myfile.write("\t \t \t \t \t }\n")
	myfile.write("\t \t \t \t }\n")
	myfile.write("\t \t \t }\n")
	myfile.write("\t \t }\n")

	myfile.write("\t \t JOINT Neck\n")
	myfile.write("\t \t {\n")
	myfile.write("\t \t \t OFFSET %2.2f %2.2f %2.2f\n" % (skelref[1][0],skelref[1][1],skelref[1][2]))
	myfile.write("\t \t \t CHANNELS 3 Zrotation Yrotation Xrotation\n")
	myfile.write("\t \t \t End Site\n")
	myfile.write("\t \t \t {\n")
	myfile.write("\t \t \t \t OFFSET %2.2f %2.2f %2.2f\n" %(skelref[0][0]-skelref[1][0],skelref[0][1]-skelref[1][1],skelref[0][2]-skelref[1][2]))
	myfile.write("\t \t \t }\n")
	myfile.write("\t \t }\n")
	myfile.write("\t }\n")
	myfile.write("}\n")
	myfile.write("\n")
	myfile.write("MOTION\n")
	myfile.write("Frames: %d\n" % (993))
	myfile.write("Frame Time: %2.2f\n" % (0.03))
	for x in range (0,47):
		myfile.write("0.00 ")
	myfile.write("0.00\n")


afname = "Sequence"
afpname = "%s/%s.bvh" % (path, afname)

copyfile("/home/jsanchez/Software/gitprojects/avatar/motion/frames/Reference.bvh", afpname)

# try to read num frames from bvh file
bvh_file_txt = open(afpname,'r')
# skip 99 first lines
for l in range(1,99):
	bvh_file_txt.readline()
line = bvh_file_txt.readline()
line_split = line.split()
num_frames = int(line_split[1])

for f in range(1,num_frames):
#for f in range(1,3):

	# load goal 3d points
	fname = "frame_SA02_%05d.txt" % (f)
	fpname = "%s/%s" % (path, fname)

	pts_skel = np.loadtxt(fpname)

	fname = "Reference"
	fpname = "%s/%s.bvh" % (path, fname)

	bpy.ops.import_anim.bvh(filepath=fpname, axis_forward='Y', axis_up='Z', filter_glob="*.bvh", target='ARMATURE', global_scale=1, frame_start=1, use_fps_scale=False, use_cyclic=False, rotate_mode='NATIVE')

	arm2 = bpy.data.objects[fname]

	params = get_skeleton_parameters (arm2, pts_skel)
	
	error = total_error(pts_skel, get_skeleton_coords(arm2))
	print("Frame %d. Error: %f" %(f, error))

	# save params results
	list_params = [item for sublist in params for item in sublist]
	flat_params = ['{:.2f}'.format(x) for x in list_params]
	str_params = ' '.join(str(e) for e in flat_params)

	with open(afpname, "a") as myfile:
		myfile.write(str_params)
		myfile.write("\n")

	# remove arm2
	arm2.select = True
	bpy.ops.object.delete()

print("END")
