### Body

#### Load

```
def load_model_to_scene(model_file, s=1.0):
    # We assume that the model name is Armature, need to change the name if it's different
    
    bpy.ops.import_scene.fbx(filepath=model_file, axis_forward='Y', axis_up='Z')

    hmodel = bpy.data.objects["Armature"]
    hmodel.scale = Vector((s,s,s))		

    return hmodel
```



#### Shape

```
bpy.ops.avt.reset_params()
```

```
model.val_breast = val   # breast values goes from 0 to 1
model.val_torso = np.random.exponential(2, 1)  # torso values goes from -0.3 to 0.3
model.val_limbs = val  # limbs values goes from 0 to 1
model.val_hips = val # hips values goes from 0 to 1
model.val_strength = val # strength values goes from 0 to 0.5
model.val_weight = val # weight values goes from -0.5 to 1.5

bpy.ops.avt.set_body_shape()
```

```
import bpy
import numpy as np
import random


out_path = "/some/user/path"

model = bpy.data.objects["Avatar"]

scene = bpy.context.scene

bpy.ops.avt.reset_params()

for f in range(100):
    
    print(f)

    model.val_breast = random.uniform(0, 1)
    model.val_weight = random.uniform(0, 1)
    model.val_hips = random.uniform(0, 1)

    bpy.ops.avt.set_body_shape()
    
    # set output path so render won't get overwritten
    scene.render.filepath = "%s/%04d.jpg" % (out_path, f)
    bpy.ops.render.render(write_still=True) # render still
```


#### Pose

List of Avatar joints. For some reason, if you save different models and load them again, blender modifies the joint order, so it is important to read them from a list or sort joints to obtain them always in the same order.

```
bones_avatar_rig = ["Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LowerBack",
                    "Spine", "Spine1", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LThumb",
                    "LeftFingerBase", "LeftHandFinger1", "Neck", "Neck1", "Head", "RightShoulder", 
                    "RightArm", "RightForeArm", "RightHand", "RThumb", "RightFingerBase", "RightHandFinger1",
                    "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "HeadTop"]
```
```
def get_bone_head_position(obj, bone_name):
    return (obj.matrix_world @ Matrix.Translation(obj.pose.bones[bone_name].head)).to_translation()
```
```
def get_bone_tail_position(obj, bone_name):
    return (obj.matrix_world @ Matrix.Translation(obj.pose.bones[bone_name].tail)).to_translation()
```

Load BVH file with avatar addon. First if you are using avatar functions. Second if you use BVH add-on.

```
bpy.ops.avt.load_bvh(filepath=motion_file)
```
```
bpy.ops.mcp.load_and_retarget(filepath=action_file)
```
