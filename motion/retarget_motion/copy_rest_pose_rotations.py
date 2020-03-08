import bpy
from mathutils import *

#file_bone_correspondences = "/home/jsanchez/Software/gitprojects/avatar/bone_correspondance_mixamo.txt"
file_bone_correspondences = "/Users/jsanchez/Software/gitprojects/avatar/motion/skeletons/mixamo.txt"


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

def worldMatrix(ArmatureObject,Bone):
# simplified version of the matrix_the_hard_way
# To Test
# Probably can't use without update of the bones, since bone.matrix does not updates
# automatically
    _bone = ArmatureObject.pose.bones[Bone]
    _obj = ArmatureObject
    return _obj.matrix_world * _bone.matrix

def pose_to_match(arm, goal, bc):
    """
    pose arm so that its bones line up with the REST pose of goal
    """

    matrix_os= {}
    #for to_match in goal.data.bones:
    for bone in arm.data.bones:
        bone_match = find_bone_match(bc, bone.name)
        if bone_match is not "none":
            #matrix_os[bone_match] = goal.data.bones[bone_match].matrix_local # if we want to match rest pose
            ebp = goal.pose.bones[bone_match]
            matrix_os[bone_match] = matrix_the_hard_way(ebp, goal)
            #print([ "matrix", bone_match, matrix_os[bone_match] ] )

    #xyz' = s * m * m(parent) * xyz

    for to_pose in arm.pose.bones:
            
        bone_match = find_bone_match(bc, to_pose.name)
        if bone_match is not "none":
            goal_bone = bone_match

            if to_pose.parent is None:
                len2 = arm.data.bones[to_pose.name].length
                len1 = goal.data.bones[goal_bone].length
                print(goal_bone)
                #to_pose.matrix = matrix_os[goal_bone] @ Matrix.Scale(0.076, 4)
                m1 = arm.matrix_world @ matrix_os[goal_bone] @ to_pose.bone.matrix_local
                loc,rot,scale = m1.decompose()
                # # to_pose.location = loc
                if 'QUATERNION' == to_pose.rotation_mode:
                    to_pose.rotation_quaternion = rot
                else:
                    to_pose.rotation_euler = rot.to_euler(to_pose.rotation_mode)

            else:
                # we can not set .matrix, because a lot of stuff behind the scenes has not yet
                # caught up with our alterations, and it ends up doing math on outdated numbers
                mp = matrix_the_hard_way(to_pose.parent, arm) @ matrix_for_bone_from_parent(to_pose, arm)
                m2 = mp.inverted() @ matrix_os[goal_bone] # @ Matrix.Scale(goal.data.bones[goal_bone].length, 4)
                #m2 = matrix_os[goal_bone] # @ Matrix.Scale(goal.data.bones[goal_bone].length, 4)
                loc,rot,scale = m2.decompose()
                # to_pose.location = loc
                if 'QUATERNION' == to_pose.rotation_mode:
                    to_pose.rotation_quaternion = rot
                else:
                    to_pose.rotation_euler = rot.to_euler(to_pose.rotation_mode)
                # to_pose.scale = scale / arm.data.bones[to_pose.name].length



#
#
#

bone_corresp = read_text_lines(file_bone_correspondences)

pose_to_match(bpy.data.objects['Standard'], bpy.data.objects['aerial_evade'], bone_corresp)