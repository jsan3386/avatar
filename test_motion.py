import os
import sys
import importlib
import bpy

import numpy as np

from mathutils import Vector, Matrix, Quaternion

sys.path.append("/home/jsanchez/Software/gitprojects/avatar/motion")

import movement_280
importlib.reload(movement_280)

frames_folder = "/home/jsanchez/Software/gitprojects/avatar/motion/frames"

skel = bpy.data.objects["Standard"]

list_bones = ["Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LowerBack",
                "Spine", "Spine1", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LThumb", 
                "LeftFingerBase", "LeftHandFinger1", "Neck", "Neck1", "Head", "RightShoulder", "RightArm",
                "RightForeArm", "RightHand", "RThumb", "RightFingerBase", "RightHandFinger1", "RHipJoint",
                "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]


skel_ref = movement_280.get_rest_pose(skel, list_bones)


point_files = [f for f in os.listdir(frames_folder) if f.endswith('.txt')]
point_files.sort()
        
num_packg = 0

poseBone = skel.pose.bones["LeftArm"]
q2 = Quaternion([1,0,0,0])

#for f in point_files:
for f in range(1,2):

    #start = time.time()
    #fpname = "%s/%s" % (frames_folder,f)
    fname = "frame_SA%02d_%05d.txt" % (2, f)
    fpname = "%s/%s" % (frames_folder,fname)
    pts_skel = np.loadtxt(fpname)
    #time.sleep(max(1./fps - (time.time() - start), 0))

    M_mb = movement_280.get_trans_mat_blend_to_matlab()
    pts_skel = np.matmul(pts_skel, M_mb)

    
    poseBone.rotation_mode = "QUATERNION"
    poseBone.rotation_quaternion = q2


    # set skeleton rest position: MAYBE MOVE ALL THIS TO SERVER.PY IN ORDER TO MAKE FASTER UPDATES
#    movement_280.set_rest_pose(skel, skel_ref, list_bones)
#    movement_280.calculate_rotations(skel, pts_skel)

    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
