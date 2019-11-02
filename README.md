IRIsAvatar: Project Avatar for Blender
======================================


# Pre-requisites

- Need to install addon in Blender. Follow instructions in: http://www.makehumancommunity.org/content/downloads.html
- Need to install zmq outside and inside Blender (pyzmq). Outside Blender normal install with pip. Inside blender follow: 
https://blender.stackexchange.com/questions/56011/how-to-install-pip-for-blenders-bundled-python




# Shape Panel


#Motion Panel


#Dressing Panel

- Download textures from https://drive.google.com/open?id=133n9ZpfK3DGlQIPOhnC94tbTFBDR_b3U



# Extras

Pot funcionar aixo en comptes de la instruccio per a forcar el drawing.

for area in bpy.context.screen.areas:
    if area.type in ['IMAGE_EDITOR', 'VIEW_3D']:
        area.tag_redraw()

# Notes and comments

#On 3D points transfer motion

- Blender skeleton updates matrices but when trying to read positions of joints using pose.bones[bone_name].head/tail, this values are not updated. To get and updated value is necessary to use scene_update() function.
This makes the whole algorithm quite slow 
- A solution to make it faster is to compute/update values on different structures. The file bvh_utils is an attempt of that. Somehow though, I was unable to calculate rotations of a bone with a parent correctly.
- The final solution consists in to use world matrices for every bone. These matrices also contains the bone head positions. However, in order to compute rotations is necessary to know also the bone tail. Since bone tail is the same as child bone head, we use that as value for tail. Only problem could be in termination bones. To fix that will be necessary to create new bones, this way blender can keep updated the values of everything. For CPM skeletons this is not necessary.
    - For this last solution I encountered several things to take into account:
    - 1) For some reason in my first attemps to solve rotations I used neck head position as left/right shoulder positions
