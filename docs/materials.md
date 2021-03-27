### Materials

In Blender there are two main Renders: Cycles and EEVEE. Both renders need to have a material associated for each mesh in order to show realistic textures. This materials can be created and modified by python scripts as shown below. These changes can be seen under the Tab "Shading" in the Blender Viewport.

We set an example assuming we have a Plane mesh and we want to apply some textures to it. The steps would be the following:
1. Create material
2. Add textures, normal maps
3. Release memory usage



```
def create_floor_material (index, mat_id):

    # Everytime we load and object a Default material is created
    # we force to remove this materials
    for m in bpy.data.materials:
        if "Default" in m.name:
            bpy.data.materials.remove(m)

    mat_name = 'floor_material%02d' % index
    floorMat = (bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name))
    floorMat.pass_index = mat_id

    floorMat.use_nodes= True

    #Clears default nodes
    floorMat.node_tree.nodes.clear()

    # Add new nodes
    tex_image = floorMat.node_tree.nodes.new(type="ShaderNodeTexImage")
    tex_image.location = -35, 377
    norm_image = floorMat.node_tree.nodes.new(type="ShaderNodeTexImage") # .001
    norm_image.location = -303, -190
    rough_image = floorMat.node_tree.nodes.new(type="ShaderNodeTexImage") # .002
    rough_image.location = -126, 92 

    nmap = floorMat.node_tree.nodes.new(type="ShaderNodeMapping")
    nmap.location = -778, 207 
    tex_coord = floorMat.node_tree.nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = -1027, 218

    norm_map = floorMat.node_tree.nodes.new(type="ShaderNodeNormalMap")
    norm_map.location = 155, -210
    col_ramp = floorMat.node_tree.nodes.new(type="ShaderNodeValToRGB")
    col_ramp.location = 225, 135

    principled = floorMat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.location = 536, 348

    out_mat = floorMat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    out_mat.location = 826, 348

    # link nodes
    floorMat.node_tree.links.new(tex_image.outputs['Color'], principled.inputs['Base Color'])
    floorMat.node_tree.links.new(tex_coord.outputs['UV'], nmap.inputs['Vector'])
    floorMat.node_tree.links.new(nmap.outputs['Vector'], tex_image.inputs['Vector'])
    floorMat.node_tree.links.new(nmap.outputs['Vector'], norm_image.inputs['Vector'])
    floorMat.node_tree.links.new(nmap.outputs['Vector'], rough_image.inputs['Vector'])
    floorMat.node_tree.links.new(norm_image.outputs['Color'], norm_map.inputs['Color'])
    floorMat.node_tree.links.new(norm_map.outputs['Normal'], principled.inputs['Normal'])
    floorMat.node_tree.links.new(rough_image.outputs['Color'], col_ramp.inputs['Fac'])
    floorMat.node_tree.links.new(col_ramp.outputs['Color'], principled.inputs['Roughness'])

    floorMat.node_tree.links.new(principled.outputs['BSDF'], out_mat.inputs['Surface'])

    return floorMat
```

asñdlkjfañsldkjf

```
def assign_tex_floor_mat (floor_obj, floor_mat, col_img, norm_img, rough_img):

    floor_obj.select_set(True)
    #bpy.context.scene.objects.active = floor_obj

    if len(floor_obj.material_slots) == 0:
        bpy.context.view_layer.objects.active = floor_obj
        bpy.ops.object.material_slot_add()
    floor_obj.material_slots[0].material = floor_mat

    img_tex_img = img_tex_norm = img_tex_rough = None

    if col_img is not None:
        img_name = os.path.basename(col_img)
        img_tex_img = (bpy.data.images.get(img_name) or bpy.data.images.load(col_img))
    if norm_img is not None:
        img_name = os.path.basename(norm_img)
        img_tex_norm = (bpy.data.images.get(img_name) or bpy.data.images.load(norm_img))
    if rough_img is not None:
        img_name = os.path.basename(rough_img)
        img_tex_rough = (bpy.data.images.get(img_name) or bpy.data.images.load(rough_img))

    matnodes = floor_mat.node_tree.nodes

    for n in matnodes:

        if n.type == 'NORMAL_MAP':
            # set normal map strength to 3
            matnodes.active = n
            n.select = True
            n.inputs[0].default_value = 1.0  # strength

        if n.type == 'TEX_IMAGE':
            if n.name == "Image Texture": # gloss
                if img_tex_img is not None:
                    matnodes.active = n
                    n.select = True
                    n.image = img_tex_img
            if n.name == "Image Texture.001": # normal
                if img_tex_norm is not None:
                    matnodes.active = n
                    n.select = True
                    n.image = img_tex_norm
                    n.image.colorspace_settings.name = 'Non-Color'
            if n.name == "Image Texture.002": # spec
                if img_tex_rough is not None:
                    matnodes.active = n
                    n.select = True
                    n.image = img_tex_rough
                    n.image.colorspace_settings.name = 'Non-Color'

    # unselect object
    floor_obj.select_set(False)    
```

    
3. In the case we want to load many textures in our script, these textures will accumulate on the Blender memory making execution slower or even lead to a segmentation fault. 

```
def release_texture(texture_name):

    for img in bpy.data.images:
        if img.name == texture_name:
            bpy.data.images.remove(img)
```

