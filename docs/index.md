## Avatar Add-on

Integration of Makehuman inside Blender with a intuitive interface for a fast prototyping of dressed human models. The add-on includes a small subset of assets available in Makehuman. The 3D model shape can be slightly modified by a set of different parameters. The purpose of this add-on is to have a ready to use tool to generate large datasets that can be used to train neural networks. Everything can be scripted in python, most useful functions and explanations are given below.


- There is also available a MoCap software using this add-on. (to release soon)


### Useful python code

1. [Camera](camera.md)
2. [Body](body.md)
3. [Materials](materials.md)


### Avatar: 3D human modeler suite (youtube video)

[![Avatar](http://img.youtube.com/vi/RLZ4DafZ9JM/0.jpg)](http://www.youtube.com/watch?v=RLZ4DafZ9JM "Avatar")


### Installation

Requirements: Blender >= 2.8

1. Go to Edit>Preferences>Add-ons>Install and choose zip file 

The new addon should appear in the right tab (press N on the 3Dviewer screen)

### Shape Panel

There are several parameters to control the shape of the body. Each one of the parameters is a PCA from several bodies created when modifying the correspondent parameter in Makehuman.

Reset parameters, change the body weights to set the original body shape


### Motion Panel

There are currently 3 different ways to import an action to Avatar model

  1. Using blender Location/Rotation constraints (default)
  2. Using 3D points (implemented but not activated)
  3. Using external addon [BHV Retargeter](http://diffeomorphic.blogspot.com/p/bvh-retargeter.html) (recommended)

Methods 1, and 2 are very slow, and they need a list of bone correspondences between the target and source. There are examples in the ``$avatar_path/motion/rigs`` folder. Might be necessary to mark "x", "y" or "z" in the select file windown when loading the BVH file depending on how is defined the skeleton in that file.
Method 3 is completely automatic and the recommended method.

#### Motion file resources

 1. [CMU](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/cmu-bvh-conversion)
 2. [Mixamo](https://www.mixamo.com/)
 3. [SFU Mocap](http://mocap.cs.sfu.ca/)
 4. [OHIO](https://accad.osu.edu/research/motion-lab/mocap-system-and-data)
 5. [DanceDB](http://dancedb.eu/main/performances)

 


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




