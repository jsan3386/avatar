# Here will be the different materials for skin and clothes
#

import bpy
import os

def create_material(matname, index):
    
    # Everytime we load and object a Default material is created
    # we force to remove this materials
    for m in bpy.data.materials:
        if "Default" in m.name:
            bpy.data.materials.remove(m)

    mat_name = '%s_mat%02d' % (matname, index)
    skinMat = (bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name))

    skinMat.use_nodes= True

    #Clears default nodes
    skinMat.node_tree.nodes.clear()

    #Adds new nodes
    tex_image = skinMat.node_tree.nodes.new(type = 'ShaderNodeTexImage')
    tex_image.location = 0 , 0
    tex_norm = skinMat.node_tree.nodes.new(type = 'ShaderNodeTexImage')
    tex_norm.location = 0, -600
    tex_spec = skinMat.node_tree.nodes.new(type = 'ShaderNodeTexImage')
    tex_spec.location = 0, -300
    norm_map = skinMat.node_tree.nodes.new(type = 'ShaderNodeNormalMap')
    norm_map.location = 300, -600
    principled = skinMat.node_tree.nodes.new(type = 'ShaderNodeBsdfPrincipled')
    principled.location = 600, 0
    output = skinMat.node_tree.nodes.new(type = 'ShaderNodeOutputMaterial')
    output.location = 1000, 0

    # link nodes
    skinMat.node_tree.links.new(tex_image.outputs['Color'], principled.inputs['Base Color'])
    skinMat.node_tree.links.new(tex_norm.outputs['Color'], norm_map.inputs['Color'])
    skinMat.node_tree.links.new(norm_map.outputs['Normal'], principled.inputs['Normal'])
    skinMat.node_tree.links.new(tex_spec.outputs['Color'], principled.inputs['Specular'])
    skinMat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return skinMat

def assign_textures(body, cmat, tex_img, tex_norm, tex_spec):

    body.select_set(True)
    #bpy.context.scene.objects.active = sub_obj

    if len(body.material_slots) == 0:
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.material_slot_add()
    body.material_slots[0].material = cmat

    img_tex_img = img_tex_norm = img_tex_spec = None 

    # Need to add strip to remove strange characters from string
    # not sure how this appeared, probably due to single/double quotes ??
    if os.path.exists(tex_img):
        # this might not work for windows platforms. 
        # see: https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
        img_name = os.path.basename(tex_img)
        img_tex_img = (bpy.data.images.get(img_name) or bpy.data.images.load(tex_img))
    if os.path.exists(tex_norm):
        img_name = os.path.basename(tex_norm)
        img_tex_norm = (bpy.data.images.get(img_name) or bpy.data.images.load(tex_norm))
    if os.path.exists(tex_spec):
        img_name = os.path.basename(tex_spec)
        img_tex_spec = (bpy.data.images.get(img_name) or bpy.data.images.load(tex_spec))

    matnodes = cmat.node_tree.nodes
    #imgnodes = [n for n in matnodes if n.type == 'TEX_IMAGE']
    #mapnode = [n for n in matnodes if n.type == 'MAPPING']
    #normnode = [n for n in matnodes if n.type == 'NORMAL_MAP']
        
    for n in matnodes:
        
        if n.type == 'NORMAL_MAP':
            # set normal map strength to 3
            matnodes.active = n
            n.select = True
            n.inputs[0].default_value = 1.0  # strength
        
        if n.type == 'TEX_IMAGE':	
            if n.name == "Image Texture": # gloss
                matnodes.active = n
                n.select = True
                if img_tex_img is not None:
                    n.image = img_tex_img
            if n.name == "Image Texture.001": # normal
                matnodes.active = n
                n.select = True
                if img_tex_norm is not None:
                    n.image = img_tex_norm
                n.image.colorspace_settings.name = 'Non-Color'
            if n.name == "Image Texture.002": # spec
                matnodes.active = n
                n.select = True
                if img_tex_spec is not None:
                    n.image = img_tex_spec
                n.image.colorspace_settings.name = 'Non-Color'
            
    # unselect object
    body.select_set(False)
        

