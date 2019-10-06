#
#
import bpy
import math

def load_cloth (cloth_file, cloth_name):

    bpy.ops.import_scene.obj(filepath=cloth_file)
        
    # change name to object
    bpy.context.selected_objects[0].name = cloth_name
    bpy.context.selected_objects[0].data.name = cloth_name
        
    b = bpy.data.objects[cloth_name]
    b.select_set(True)
    bpy.context.view_layer.objects.active = b
    bpy.ops.object.mode_set(mode='OBJECT')
    
    if bpy.data.objects.get("Standard") is not None:
        a = bpy.data.objects["Standard"]
        b = bpy.data.objects[cloth_name]
        a.select_set(True)
        b.select_set(True)
        bpy.context.view_layer.objects.active = a
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    for obj in bpy.data.objects:
        obj.select_set(False)

def read_file_textures (root_path, fold_name):

    tex_col = tex_norm = tex_spec = None 
    ftex = open('%s/dressing/textures/%s/default.txt' % (root_path, fold_name), 'r')
    lines = []
    # Note: is important to use strip() to remove undesired characters from string when reading file
    for line in ftex:
        lines.append(line.strip())
    ftex.close()
    num_lines = len(lines)
    if num_lines == 1:
        tex_col = "%s/dressing/textures/%s/%s" % (root_path, fold_name, lines[0])
    elif num_lines == 2:
        tex_col = "%s/dressing/textures/%s/%s" % (root_path, fold_name, lines[0])
        tex_norm = "%s/dressing/textures/%s/%s" % (root_path, fold_name, lines[1])
    elif num_lines == 3:
        tex_col = "%s/dressing/textures/%s/%s" % (root_path, fold_name, lines[0])
        tex_norm = "%s/dressing/textures/%s/%s" % (root_path, fold_name, lines[1])
        tex_spec = "%s/dressing/textures/%s/%s" % (root_path, fold_name, lines[2])
    else:
        print("Error reading default texture file")

    return tex_col, tex_norm, tex_spec

def load_studio(root_path):

    # Load studio plane
    s_file = "%s/dressing/models/studio_plane.obj" % root_path
    bpy.ops.import_scene.obj(filepath=s_file)
        
    # change name to object
    bpy.context.selected_objects[0].name = "studio_plane"
    bpy.context.selected_objects[0].data.name = "studio_plane"

    # remove lights and cameras
    for o in bpy.context.scene.objects:
        if o.type == 'CAMERA':
            o.select_set(True)
        elif o.type == 'LIGHT':
            o.select_set(True)
        else:
            o.select_set(False)

    # Call the operator only once
    bpy.ops.object.delete()

    # create camera and lights
    cam_data = bpy.data.cameras.new("CameraData")
    cam_object = bpy.data.objects.new(name="Camera", object_data=cam_data)
    bpy.context.collection.objects.link(cam_object)
    #bpy.data.cameras["Camera"].clip_end = 1000
    cam_object.location = (0, -66.2, 9.28)
    cam_object.rotation_euler = (math.radians(90), 0, 0)

    fill_data = bpy.data.lights.new(name="FillData", type='SUN')
    fill_data.energy = 1
    fill_object = bpy.data.objects.new(name="fill", object_data=fill_data)
    bpy.context.collection.objects.link(fill_object)
    bpy.context.view_layer.objects.active = fill_object
    fill_object.location = (32.29, -25.6, 48.17)
    fill_object.rotation_euler = (math.radians(-15), math.radians(30), math.radians(-14))

    back_data = bpy.data.lights.new(name="BackData", type='SUN')
    back_data.energy = 1
    back_object = bpy.data.objects.new(name="back", object_data=back_data)
    bpy.context.collection.objects.link(back_object)
    bpy.context.view_layer.objects.active = back_object
    back_object.location = (33.46, 46.93, 41.5)
    back_object.rotation_euler = (math.radians(45), math.radians(-23), math.radians(31))

    key_data = bpy.data.lights.new(name="KeyData", type='SUN')
    key_data.energy = 1
    key_object = bpy.data.objects.new(name="key", object_data=key_data)
    bpy.context.collection.objects.link(key_object)
    bpy.context.view_layer.objects.active = key_object
    key_object.location = (-36.88, -30.55, 49.1)
    key_object.rotation_euler = (math.radians(14), math.radians(-54), math.radians(11))

    # update scene, if needed
    dg = bpy.context.evaluated_depsgraph_get() 
    dg.update()


