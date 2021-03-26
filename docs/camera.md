### Camera


The camera script is taken from [here](https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera)

The camera projection matrix is calculated from P = K [R|T], where K is the intrinsics matrix and R, T the rotation and translation matrices.
We use a slightly different code to calculate the intrinsic matrix. We found the linked code has errors for some camera resolutions.

```
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
```

The extrinsics calibration matrix remains the same.

```
# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT, location, rotation
```

