import bpy
from mathutils import Matrix, Vector, Quaternion
import numpy as np
import math

# Try to make faster the retargeting.
# The algorithm is based in 2 parts.
# 1. From unkonwn armature transfer bone rotations to a known armature. Since sometimes this rotation transfer
# has rotations that mess with the associated mesh of the armature
# 2. Once armatures are equal, use a function to align 2 vectors. This function guarantees shortest rotations.

# The method is slow due to compute vectors, we need current position of bones and this can only be done using
# view_layer.update() which at the same time, slows down loops a lot!

# Ideas to improve and make it faster. Try to remove bvh skeleton between steps 1 and 2, when bvh skeleton is
# not needed anymore.
# Use matrix_world.to_translation() instead of get_bone_head_position. Matrix world compute all matrices. The
# problem with this function or equivalents is that the positions obtained are slightly different. This causes
# method to fail. The different positions probably comes from the precision of the matrices in Blender. Not sure.

source = bpy.data.objects["walking"]
target = bpy.data.objects["Avatar"]
target_cp = bpy.data.objects["Skel_cp"]
# me_cp = target.data.copy()

# target_cp = bpy.data.objects.new("Skel_cp", me_cp)
# target_cp.location = target.location

# bpy.context.scene.collection.objects.link(target_cp)
# bpy.context.view_layer.update()

#file_bone_correspondences = "/home/jsanchez/Software/gitprojects/avatar/motion/retarget_motion_mine/bone_correspondance_cmu.txt"
#file_bone_correspondences = "/home/jsanchez/Software/gitprojects/avatar/bone_correspondance_mixamo.txt"
file_bone_correspondences = "/Users/jsanchez/Software/gitprojects/avatar/motion/rigs/mixamo.txt"


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

def get_pose_bone_head_position(obj, pose_bone):
    return (obj.matrix_world @ Matrix.Translation(pose_bone.head)).to_translation()

def get_pose_bone_tail_position(obj, pose_bone):
    return (obj.matrix_world @ Matrix.Translation(pose_bone.tail)).to_translation()

def matrix_scale(scale_vec):
    return Matrix([[scale_vec[0],0,0,0],
                   [0,scale_vec[1],0,0],
                   [0,0,scale_vec[2],0],
                   [0,0,0,1]
    ])

def matrix_for_bone_from_parent(bone, ao):
    eb1 = ao.data.bones[bone.name]
    E = eb1.matrix_local # * Matrix.Scale(eb1.length,4)
    ebp = ao.data.bones[bone.name].parent
    E_p = ebp.matrix_local # * Matrix.Scale(ebp.length,4)
    return E_p.inverted() @ E

def matrix_the_hard_way(pose_bone, ao):
    if pose_bone.rotation_mode == 'QUATERNION':
        mr = pose_bone.rotation_quaternion.to_matrix().to_4x4()
    else:
        mr = pose_bone.rotation_euler.to_matrix().to_4x4()
    m1 = Matrix.Translation(pose_bone.location) @ mr @ matrix_scale(pose_bone.scale)

    E = ao.data.bones[pose_bone.name].matrix_local
    if pose_bone.parent is None:
        return E @ m1
    else:
        m2 = matrix_the_hard_way(pose_bone.parent, ao)
        E_p = ao.data.bones[pose_bone.parent.name].matrix_local
        return m2 @ E_p.inverted() @ E @ m1

def matrix_world(armature_ob, bone_name):
    local = armature_ob.data.bones[bone_name].matrix_local
    basis = armature_ob.pose.bones[bone_name].matrix_basis

    parent = armature_ob.pose.bones[bone_name].parent
    if parent == None:
        return  local @ basis
    else:
        parent_local = armature_ob.data.bones[parent.name].matrix_local
        return matrix_world(armature_ob, parent.name) @ (parent_local.inverted() @ local) @ basis


def worldMatrix(ArmatureObject,Bone):
# simplified version of the matrix_the_hard_way
# To Test
# Probably can't use without update of the bones, since bone.matrix does not updates
# automatically
    _bone = ArmatureObject.pose.bones[Bone]
    _obj = ArmatureObject
    return _obj.matrix_world @ _bone.matrix

def trans_coord_system(p1, o1, o2, M1, M2):

    # note is necessary to use a copy version of the matrix
    # otherwise it modifies the content of M2 outside the function
    # Actually it transposes matrix M2 outside the function. Must be the way
    # Blender handles the transform operators
    M2t = M2.copy()
    M2t.transpose()
    return M2t @ (o1 - o2 + M1 @ p1)

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

# # create target animation
# target_cp.animation_data_clear()

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
    matrices_source[bone.name] = bone.matrix_local.copy()


# target bones in rest position
trg_bone_loc_hips = get_bone_head_position(target_cp, "Hips")
trg_bone_loc_lefthips = get_bone_head_position(target_cp, "LeftUpLeg")
trg_bone_loc_righthips = get_bone_head_position(target_cp, "RightUpLeg")
trg_bone_loc_neck = get_bone_head_position(target_cp, "Neck")


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


    matrix_os= {}
    #for to_match in goal.data.bones:
    for bone in target_cp.data.bones:
        bone_match = find_bone_match(bone_corresp, bone.name)
        if bone_match is not "none":
            #matrix_os[bone_match] = goal.data.bones[bone_match].matrix_local # if we want to match rest pose
            ebp = source.pose.bones[bone_match]
            matrix_os[bone_match] = matrix_the_hard_way(ebp, source)
            #print([ "matrix", bone_match, matrix_os[bone_match] ] )

    # read source motion
    for pb in target_cp.pose.bones:
        
        bone_name = find_bone_match(bone_corresp, pb.name)
        if bone_name is not "none":
            goal_bone = bone_name
            
            # # source bone
            # spb = source.pose.bones[bone_name]
        
            # # insert keyframe
            # loc = spb.location

            if pb.parent is None:
                # print(f, (source.pose.bones["mixamorig:Hips"].matrix_basis).to_translation())
                # bone_translate_matrix = Matrix.Translation(source.pose.bones["mixamorig:Hips"].matrix_basis).to_translation()
                # loca = (source.data.bones["mixamorig:Hips"].matrix_local.inverted() @ Vector(bone_translate_matrix)).to_translation()
                # print(f, loca)
                loc = source.matrix_world @ source.pose.bones["mixamorig:Hips"].head
                # loc = source.matrix_world @ source.pose.bones["hip"].head
                pb.location = target_cp.matrix_world.inverted() @ pb.bone.matrix_local.inverted() @ loc
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
            #     pb.location = loc
            #     pb.keyframe_insert('location', frame=f, group=pb.name)

            # # if pb.parent is None:        
            # #     pb.location = matrices_source[bone_name].to_translation()
            # #     print(f, loc)
            # #     print(f, pb.location)
            # #     pb.keyframe_insert('location', frame=f, group=pb.name)
            # # else:
            # #     pb.location = loc
            # #     pb.keyframe_insert('location', frame=f, group=pb.name)
                
            # # pb.location = spb.location
            # # pb.keyframe_insert('location', frame=f, group=pb.name)
        
            #     #spb.rotation_mode = 'XYZ'
            #     pb.rotation_mode = 'XYZ'
            #     #rot = spb.rotation_euler
            #     rot = matrices_target[pb.name] @ spb.matrix_basis.copy()
            #     #rot = spb.matrix_basis.copy()
            #     pb.rotation_euler = rot.to_euler()
            #     pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)

                # we can not set .matrix, because a lot of stuff behind the scenes has not yet
                # caught up with our alterations, and it ends up doing math on outdated numbers
                mp = matrix_the_hard_way(pb.parent, target_cp) @ matrix_for_bone_from_parent(pb, target_cp)
                m2 = mp.inverted() @ matrix_os[goal_bone] # @ Matrix.Scale(goal.data.bones[goal_bone].length, 4)
                #m2 = matrix_os[goal_bone] # @ Matrix.Scale(goal.data.bones[goal_bone].length, 4)
                loc,rot,scale = m2.decompose()
                # to_pose.location = loc
                if 'QUATERNION' == pb.rotation_mode:
                    pb.rotation_quaternion = rot
                    pb.keyframe_insert('rotation_quaternion', frame=f, group=pb.name)
                else:
                    pb.rotation_euler = rot.to_euler(pb.rotation_mode)
                    pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)
                # to_pose.scale = scale / arm.data.bones[to_pose.name].length

                print("last debug")
                print(rot)

ept0 = bpy.data.objects["ept0"]
ept1 = bpy.data.objects["ept1"]
ept2 = bpy.data.objects["ept2"]

# copy rotations from 3d points
# now skeletons are equal (same name bones, same length bones)
#for f in range(nfirst, nlast):
for f in range(1, 2):

    # set target in rest position
    for bone in target.pose.bones:
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = (0, 0, 0)

    bpy.context.scene.frame_set(f)

    pb_list = ["Hips", "LowerBack", "Spine", "Spine1", "RightShoulder", "RightArm", "RightForeArm", "RightHand"]

    # for pb in target.pose.bones:
    for pbname in pb_list:
        pb = target.pose.bones[pbname]
        print(pbname)

        # bpy.context.view_layer.update()

        pb_cp = target_cp.pose.bones[pb.name]

        if pb.parent is None:

            pb.location = pb_cp.location
            pb.keyframe_insert('location', frame=f, group=pb.name)

            pb_cp.rotation_mode = 'XYZ'
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = pb_cp.rotation_euler
            pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)

        else:

            if pb.children:

                # recalculate rotations to avoid strange mesh deformations
                pt0 = get_pose_bone_head_position(target, pb)
                pt1 = get_pose_bone_tail_position(target, pb)
                pt2 = get_pose_bone_tail_position(target_cp, pb_cp)
                print("points blender update")
                print(pt0)
                print(pt1)
                print(pt2)

                # print(pb.children)
                pb_child = pb.children[0]   # we assume each bone only have one children !!
                pb_cp_child = pb_cp.children[0]
                pt0 = matrix_world(target, pb.name).to_translation()
                pt1 = matrix_world(target, pb_child.name).to_translation()
                pt2 = matrix_world(target_cp, pb_cp_child.name).to_translation()
                print("points matrix_world update")
                print(pt0)
                print(pt1)
                print(pt2)

                # print(pb.children)
                pb_child = pb.children[0]   # we assume each bone only have one children !!
                pb_cp_child = pb_cp.children[0]
                # pt0 = worldMatrix(target, pb.name).to_translation()
                # pt1 = worldMatrix(target, pb_child.name).to_translation()
                pt0 = matrix_the_hard_way(pb, target).to_translation()
                pt1 = matrix_the_hard_way(pb_child, target).to_translation()
                pt2 = matrix_the_hard_way(pb_cp_child, target_cp).to_translation()
                print("points matrix_the_hard_way update")
                print(pt0)
                print(pt1)
                print(pt2)

                # print(pb.children)
                pb_child = pb.children[0]   # we assume each bone only have one children !!
                pb_cp_child = pb_cp.children[0]
                pt0 = worldMatrix(target, pb.name).to_translation()
                pt1 = worldMatrix(target, pb_child.name).to_translation()
                # pt0 = matrix_world(target, pb.name).to_translation()
                # pt1 = matrix_world(target, pb_child.name).to_translation()
                pt2 = worldMatrix(target_cp, pb_cp_child.name).to_translation()
                print("points worldMatrix update")
                print(pt0)
                print(pt1)
                print(pt2)


                ept0.location = pt0
                ept1.location = pt1
                ept2.location = pt2

                q2 = compute_rotation(pb, pt0, pt1, pt2)

                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = q2
                pb.keyframe_insert('rotation_quaternion', frame=f, group=pb.name)


