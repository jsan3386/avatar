import bpy
from mathutils import Matrix, Vector

source = bpy.data.objects["aerial_evade"]
target = bpy.data.objects["Standard"]

file_bone_correspondences = "/home/jsanchez/Software/gitprojects/avatar/bone_correspondance_mixamo.txt"
#file_bone_correspondences = "/Users/jsanchez/Software/gitprojects/avatar/bone_correspondance_mixamo.txt"


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

# read source animation
for f in range(nfirst, nlast):

    bpy.context.scene.frame_set(f)

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
                print(f, loc) 
                loc = source.matrix_world @ source.pose.bones["mixamorig:Hips"].head
                print(loc)
                pb.location = target.matrix_world.inverted() @ pb.bone.matrix_local.inverted() @ loc
                pb.keyframe_insert('location', frame=f, group=pb.name)

                spb.rotation_mode = 'XYZ'
                pb.rotation_mode = 'XYZ'
                #rot = spb.rotation_euler
                rot = (spb.matrix_basis).to_euler()
                pb.rotation_euler = rot
                pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)



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
