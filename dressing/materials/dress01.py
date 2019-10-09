
import material_utils

def create_material(matname, index):
    
    mat = material_utils.create_material_generic(matname, index)
    return mat

def assign_textures(body, cmat, tex_img, tex_norm, tex_spec):

    material_utils.assign_textures_generic_mat(body, cmat, tex_img, tex_norm, tex_spec)
