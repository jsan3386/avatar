IRIsAvatar: Project Avatar for Blender
======================================

# List TODOs, things to fix

- skin normal norm00.png is not correct, this provokes some transparent faces on rendered image
- problems when appending model and skeleton from blend file 


# Pre-requisites

- Need to install addon in Blender. Follow instructions in: http://www.makehumancommunity.org/content/downloads.html
- Need to install zmq outside and inside Blender (pyzmq). Outside Blender normal install with pip. Inside blender follow: 
https://blender.stackexchange.com/questions/56011/how-to-install-pip-for-blenders-bundled-python




# Shape Panel


# Motion Panel

- Motion from 3D points

Note: Makehuman loaded in 2.79 is 1x while in 2.8 is 10x. This means 3D point coordinates must be resized accordingly. 
Also note 3D coordinates coming from matlab have different axis orientations, a transformation must be applied to correct that. Matlab Y up X forward. Blender Z up  Y forward.

Current status:

1. Some blender sequences are generated to simulate 2D and 3D points.
    1.1. If we pass 3D points to the algorithm, the motion is correct
2. In jordi_tf user there are the training files to detect 2D pose and to find 2D - 3D conversions
    2.1. 2D-3D conversions use Julieta's code (https://github.com/una-dinosauria/3d-pose-baseline)
         go to /home/jordi_tf/Software/3d-pose-baseline/src, activate venv_cpm
         training_avatar.py is for training, inference_avatar.py is for inference
    2.2. 2D detections
         Openpose can't make it work. Some problems with google protobuf when installing caffe
         AlphaPose-master: can calculate poses from images (working) 
            in /home/jordi_tf/Software/AlphaPose-master/examples/demo are my generated synthetic images
            in /home/jordi_tf/Software/AlphaPose-master/results are the results
         AlphaPose: this code should work for webcam, but it can't detect camera when using code with ssh
            some other error with cuda version (not sure I can fix). Needs python3.6 to run 


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

![Alt text](./figures/skeletons.jpg?raw=true "Skeletons")

For a better motion transfer in our skeleton we should format the CPM points to match better our standard skeleton joints. This is specially problematic in shoulders and head. If we observe in our skeleton shoulders are in the middle of the bone, for now they are approximated using neck position. Also now, head is attached to the neck bone, this should not be like this. If we observe our skeleton, head should be detached from bone neck.