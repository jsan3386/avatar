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
