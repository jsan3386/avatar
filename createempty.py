import bpy
import math
from mathutils import Vector, Quaternion, Matrix
from numpy import *
#from math import sqrt
import numpy as np
import os


context = bpy.context
path_input="/home/aniol/avatar/frames"

rx = math.radians(90)
ry = math.radians(0)
rz = math.radians(0)

M_rx = np.array([[1,0,0],[0,math.cos(rx),math.sin(rx)],[0,-math.sin(rx),math.cos(rx)]])
M_ry = np.array([[math.cos(ry),0,-math.sin(ry)],[0,1,0],[math.sin(ry),0,math.cos(ry)]])
M_rz = np.array([[math.cos(rz),math.sin(rz),0],[-math.sin(rz),math.cos(rz),0],[0,0,1]])


## Rotation Matrix from rotations rx ry rz
M_mb1 = np.matmul(M_rx, M_ry)
M_mb = np.matmul(M_mb1, M_rz)

fname = "frame_SA%02d_%05d.txt" % (2, 2)
#           fpname = "%s/A%d/%s" % (path_input, action, fname)
fpname = "%s/%s" % (path_input,fname)
pts_skel = loadtxt(fpname)
# adjust 3D points axis to Blender axis
pts_skel = np.matmul(pts_skel, M_mb)
new_pts_skel = []
hips = pts_skel[14,:]
arm2 = bpy.data.objects["Standard"]
correction = arm2.pose.bones['Hips'].head
translation = hips - [correction[0],correction[1],correction[2]]

for i in pts_skel:
		bpy.ops.object.empty_add(location=([i[0]-translation[0],i[1]-translation[1],i[2]-translation[2]]))

