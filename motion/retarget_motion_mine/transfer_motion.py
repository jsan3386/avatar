import bpy
from mathutils import Matrix, Vector
import numpy as np

source = bpy.data.objects["02_01_walk"]
target = bpy.data.objects["Standard"]

file_bone_correspondences = "/home/jsanchez/Software/gitprojects/avatar/motion/retarget_motion_mine/bone_correspondance_cmu.txt"
#file_bone_correspondences = "/home/jsanchez/Software/gitprojects/avatar/bone_correspondance_mixamo.txt"
#file_bone_correspondences = "/Users/jsanchez/Software/gitprojects/avatar/bone_correspondance_mixamo.txt"


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

def get_bone_head_position(obj, bone_name):
    return (obj.matrix_world @ Matrix.Translation(obj.pose.bones[bone_name].head)).to_translation()


# create target animation
target.animation_data_clear()

# get frames of action
act_size = source.animation_data.action.frame_range
print(act_size)

nfirst = int(act_size[0])
nlast = int(act_size[1])

bpy.context.scene.frame_start = nfirst
bpy.context.scene.frame_end = nlast


bone_corresp = read_text_lines(file_bone_correspondences)

# store pose bone matrices target
matrices_target= {}
#for to_match in goal.data.bones:
for bone in target.pose.bones:
    matrices_target[bone.name] = bone.matrix_basis.copy()
    #print([ "matrix", bone.name, matrix_os[bone.name] ] )
#    if bone.name == "RightArm":
#        print(bone.matrix_basis.decompose()[0])
#        print(bone.matrix_basis.decompose()[1].to_euler())

matrices_source = {}
for bone in source.data.bones:
    matrices_source[bone.name] = bone.matrix_local


# target bones in rest position
trg_bone_loc_hips = get_bone_head_position(target, "Hips")
trg_bone_loc_lefthips = get_bone_head_position(target, "LeftUpLeg")
trg_bone_loc_righthips = get_bone_head_position(target, "RightUpLeg")
trg_bone_loc_neck = get_bone_head_position(target, "Neck")


# read source animation
for f in range(nfirst, nlast):

    bpy.context.scene.frame_set(f)

    # get bvh bone locations
    source_bone_name = find_bone_match(bone_corresp, "Hips")
    src_bone_loc_hips = get_bone_head_position(source, source_bone_name)
    source_bone_name = find_bone_match(bone_corresp, "LeftUpLeg")
    src_bone_loc_lefthips = get_bone_head_position(source, source_bone_name)
    source_bone_name = find_bone_match(bone_corresp, "RightUpLeg")
    src_bone_loc_righthips = get_bone_head_position(source, source_bone_name)
    source_bone_name = find_bone_match(bone_corresp, "Neck")
    src_bone_loc_neck = get_bone_head_position(source, source_bone_name)

    # read source motion
    for pb in target.pose.bones:
        
        bone_name = find_bone_match(bone_corresp, pb.name)
        if bone_name is not "none":
            
            # source bone
            spb = source.pose.bones[bone_name]
        
            # insert keyframe
            loc = spb.location

            if pb.parent is None:
                # print(f, (source.pose.bones["mixamorig:Hips"].matrix_basis).to_translation())
                # bone_translate_matrix = Matrix.Translation(source.pose.bones["mixamorig:Hips"].matrix_basis).to_translation()
                # loca = (source.data.bones["mixamorig:Hips"].matrix_local.inverted() @ Vector(bone_translate_matrix)).to_translation()
                # print(f, loca)
                #loc = source.matrix_world @ source.pose.bones["mixamorig:Hips"].head
                loc = source.matrix_world @ source.pose.bones["hip"].head
                pb.location = target.matrix_world.inverted() @ pb.bone.matrix_local.inverted() @ loc
                pb.keyframe_insert('location', frame=f, group=pb.name)

                # compute translation and first rotation between rest position and desired points
                A = np.mat((trg_bone_loc_hips, trg_bone_loc_lefthips, trg_bone_loc_righthips, trg_bone_loc_neck)) # my skeleton
                B = np.mat((src_bone_loc_hips, src_bone_loc_lefthips, src_bone_loc_righthips, src_bone_loc_neck)) # bvh skeleton
                R, T = rigid_transform_3D(A,B)

                mR = Matrix([[R[0,0],R[0,1],R[0,2]], [R[1,0],R[1,1],R[1,2]], [R[2,0],R[2,1],R[2,2]]])
                mR.resize_4x4()

                boneRefPoseMtx = pb.bone.matrix_local.copy()
                rotMtx = boneRefPoseMtx.inverted() @ mR @ boneRefPoseMtx
                pb.rotation_mode = 'XYZ'
                pb.rotation_euler = rotMtx.to_euler()
                pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)

                # spb.rotation_mode = 'XYZ'
                # pb.rotation_mode = 'XYZ'
                # #rot = spb.rotation_euler
                # rot = (spb.matrix_basis @ matrices_target[pb.name]).to_euler()
                # pb.rotation_euler = rot
                # pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)



            else:
                pb.location = loc
                pb.keyframe_insert('location', frame=f, group=pb.name)

            # if pb.parent is None:        
            #     pb.location = matrices_source[bone_name].to_translation()
            #     print(f, loc)
            #     print(f, pb.location)
            #     pb.keyframe_insert('location', frame=f, group=pb.name)
            # else:
            #     pb.location = loc
            #     pb.keyframe_insert('location', frame=f, group=pb.name)
                
            # pb.location = spb.location
            # pb.keyframe_insert('location', frame=f, group=pb.name)
        
                spb.rotation_mode = 'XYZ'
                pb.rotation_mode = 'XYZ'
                #rot = spb.rotation_euler
                rot = (spb.matrix_basis @ matrices_target[pb.name]).to_euler()
                pb.rotation_euler = rot
                pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)
