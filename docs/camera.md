### Camera


The camera script is taken from [here](https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera)

The camera projection matrix is calculated from P = K [R|T], where K is the intrinsics matrix and R, T the rotation and translation matrices.


def get_calibration_matrix_K_from_blender(camd):

    scene = bpy.context.scene

    f_in_mm = camd.lens
    sensor_width_in_mm = camd.sensor_width

    w = scene.render.resolution_x
    h = scene.render.resolution_y
    
    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect

    # yes, shift_x is inverted. WTF blender?
    c_x = w * (0.5 - camd.shift_x)
    c_y = h * (0.5 + camd.shift_y)

    K = Matrix(((f_x, 0, c_x),
                (0, f_y, c_y),
                (0,   0,   1)))


    return K
