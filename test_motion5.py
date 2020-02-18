import bpy

source = bpy.data.objects["walking"]
target = bpy.data.objects["Armature"]

# create target animation
target.animation_data_clear()

nfirst = bpy.context.scene.frame_start 
nlast = bpy.context.scene.frame_end

# read source animation
for f in range(nfirst, nlast):

    bpy.context.scene.frame_set(f)

    # read source motion
    for pb in source.pose.bones:
        
        # insert keyframe
        loc = pb.location
        
        trgpb = target.pose.bones[pb.name]
        
        target.pose.bones[pb.name].location = loc
        trgpb.keyframe_insert('location', frame=f, group=pb.name)
        
        pb.rotation_mode = 'XYZ'
        trgpb.rotation_mode = 'XYZ'
        rot = pb.rotation_euler
        target.pose.bones[pb.name].rotation_euler = rot
        trgpb.keyframe_insert('rotation_euler', frame=f, group=pb.name)
    
    
    