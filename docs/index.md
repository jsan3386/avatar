## Avatar Add-on

Integration of Makehuman inside Blender with a intuitive interface for a fast prototyping of dressed human models. The add-on includes a small subset of assets available in Makehuman. The 3D model shape can be slightly modified by a set of different parameters. The purpose of this add-on is to have a ready to use tool to generate large datasets that can be used to train neural networks. Everything can be scripted in python, most useful functions and explanations are given below.


- There is also available a MoCap software using this add-on.
- A new version integrating SMPL model for researchers will be made public soon.


### Useful python code

1. [Camera](camera.md)
2. [Body](body.md)
3. [Materials](materials.md)


### Avatar: 3D human modeler suite

Human 3D model avatar based on Makehuman. I created this project with the intention to accelerate human dataset
creation. The project has a set of elements to give some variance to 3D human model in shape as well as in clothes.

The code has many parts that can be improved and the project also has the potential to grow including more clothes, new shape controls, increasing performance or work in collaboration with other addons. (mocap, correct bvh)

If you are programmer, graphic designer or just think you can collaborate to improve the project you are wellcome to message me. (email)

Enjoy Avatar!!

[![Avatar](http://img.youtube.com/vi/RLZ4DafZ9JM/0.jpg)](http://www.youtube.com/watch?v=RLZ4DafZ9JM "Avatar")

### List TODOs, things to fix

- [] skin normal norm00.png is not correct, this provokes some transparent faces on rendered image
- [] need complete clothes masks
- [] shape parameter belly is wrong 
- [] change names for cloth icon collection (in code), to not overlap with ClothWeaver

### Installation

Requirements: Blender >= 2.8

1) Create a file in the avatar github folder named config.py with the following content:  
    avt_path = "/path/of/root/avatar/github/project"

2) Go to Edit>Preferences>FilePaths and add to Scripts the path to the github avatar folder

3) Go to Edit>Preferences>Add-ons>Install and choose the file avatar_addon_b280.py 

The new addon should appear in the right tab (press N on the 3Dviewer screen)

### Shape Panel

There are several parameters to control the shape of the body. Each one of the parameters is a PCA from several bodies created when modifying the correspondent parameter in Makehuman.

Reset parameters, change the body weights to set the original body shape


### Motion Panel

Before loading a motion file, you need to select which kind of rig is defined in the BVH file. Currently, only CMU and Mixamo skeleton rigs are available. If you have a file with different rig, you need to create a .txt file
with the bone correspondences manually and add it to $avatar_path/motion/rigs

Note that the file can take some time to load. Calculations are very slow because update() function needs to be called quite often.

Note that usual function to load a motion file in Makehuman provided in Makewalk addon, is working only in Blender 2.79. Even I modified some parts of the code to make it compatible with Blender 2.80, there are a lot of mistakes when importing motions. Probably is due to the fact that now frame 0 is not Tpose anymore.
Finally, I decided to implement my own function. Is not optimal and I'm sure it can be greatly improved, but it works for all the cases I've tried.  

Another way to load an action to Avatar model is to use the Blender Addon [BHV Retargeter](http://diffeomorphic.blogspot.com/p/bvh-retargeter.html).  


### Dressing Panel

There is a set of clothes downloaded from Makehuman website. These clothes are slightly modified to fit the makehuman body without having to remove vertices from the model.

Original textures can be downloaded in the Makehuman website or [here](https://drive.google.com/open?id=133n9ZpfK3DGlQIPOhnC94tbTFBDR_b3U)

If you want to use your own texture in one of the clothes:
    1. Set the image or images in the cloth folder > $avatar_path/dressing/textures/cloth_folder
    2. Change the image name in file > default.txt
    3. Default file assumes: 1st line texture image; 2nd line normal map; 3rd line specular map. If your texture has no normal map neither specular map, you can leave the line in blank.


<!-- # Extras

Pot funcionar aixo en comptes de la instruccio per a forcar el drawing.

for area in bpy.context.screen.areas:
    if area.type in ['IMAGE_EDITOR', 'VIEW_3D']:
        area.tag_redraw() -->

<!-- # Notes and comments

#On 3D points transfer motion

- Blender skeleton updates matrices but when trying to read positions of joints using pose.bones[bone_name].head/tail, this values are not updated. To get and updated value is necessary to use scene_update() function.
This makes the whole algorithm quite slow 
- A solution to make it faster is to compute/update values on different structures. The file bvh_utils is an attempt of that. Somehow though, I was unable to calculate rotations of a bone with a parent correctly.
- The final solution consists in to use world matrices for every bone. These matrices also contains the bone head positions. However, in order to compute rotations is necessary to know also the bone tail. Since bone tail is the same as child bone head, we use that as value for tail. Only problem could be in termination bones. To fix that will be necessary to create new bones, this way blender can keep updated the values of everything. For CPM skeletons this is not necessary.
    - For this last solution I encountered several things to take into account:
    - 1) For some reason in my first attemps to solve rotations I used neck head position as left/right shoulder positions

![Alt text](./figures/skeletons.jpg?raw=true "Skeletons")

For a better motion transfer in our skeleton we should format the CPM points to match better our standard skeleton joints. This is specially problematic in shoulders and head. If we observe in our skeleton shoulders are in the middle of the bone, for now they are approximated using neck position. Also now, head is attached to the neck bone, this should not be like this. If we observe our skeleton, head should be detached from bone neck. -->




