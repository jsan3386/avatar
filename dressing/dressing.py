#
#

def load_cloth (cloth_file, cloth_name):

    bpy.ops.import_scene.obj(filepath=cloth_file)
        
    # change name to object
    bpy.context.selected_objects[0].name = cloth_name
    bpy.context.selected_objects[0].data.name = cloth_name
        
    b = bpy.data.objects[cloth_name]
    b.select_set(True)
    bpy.context.view_layer.objects.active = b
    bpy.ops.object.mode_set(mode='OBJECT')
    
    if bpy.data.objects.get("Standard") is not False:
        a = bpy.data.objects["Standard"]
        b = bpy.data.objects[cloth_name]
        a.select_set(True)
        b.select_set(True)
        bpy.context.view_layer.objects.active = a
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    for obj in bpy.data.objects:
        obj.select_set(False)
    #########
    #######
    #####
    ## Aqui s'ha de posar que sigui cloth tot excepte sabates i gorres	
    if "shoe" in cloth_name or "hat" in cloth_name or "glass" in cloth_name:
        print(" ************* Objecte Rígid **************** ")
    else:
        b = bpy.data.objects[cloth_name]
        b.select_set(True)
        bpy.context.view_layer.objects.active = b
        
        bpy.ops.object.modifier_add(type='CLOTH')
    
    cloth = bpy.data.objects[cloth_name]

    for obj in bpy.data.objects:
        obj.select_set(False)
    
    return cloth	

