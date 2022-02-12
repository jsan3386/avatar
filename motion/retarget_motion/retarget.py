import bpy
from mathutils import Matrix, Vector, Quaternion, Euler
import numpy as np
import math
from collections import OrderedDict

def get_bone_head_position(obj, bone_name):
    return (obj.matrix_world @ Matrix.Translation(obj.pose.bones[bone_name].head)).to_translation()

def get_pose_bone_head_position(obj, pose_bone):
    return (obj.matrix_world @ Matrix.Translation(pose_bone.head)).to_translation()

def get_pose_bone_tail_position(obj, pose_bone):
    return (obj.matrix_world @ Matrix.Translation(pose_bone.tail)).to_translation()

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0] # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       #print ("Reflection detected")
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    #print (t)

    return R, t

def read_text_lines(filename):

    list_bones = []

    text_file = open(filename, "r")
    lines = text_file.readlines()
    for line in lines:
        line_split = line.split()
        if len(line_split) == 2:
            list_bones.append([line_split[0], line_split[1]])
        else: # only 1 element
            list_bones.append([line_split[0], "none"])

    return list_bones

def find_bone_match(list_bones, bone_name):

    bone_match = "none"
    for b in list_bones:
        if b[0] == bone_name:
            bone_match = b[1]
            break
    return bone_match

def matrix_for_bone_from_parent(bone, ao):
    eb1 = ao.data.bones[bone.name]
    E = eb1.matrix_local # * Matrix.Scale(eb1.length,4)
    ebp = ao.data.bones[bone.name].parent
    E_p = ebp.matrix_local # * Matrix.Scale(ebp.length,4)
    return E_p.inverted() @ E

def matrix_the_hard_way(pose_bone, ao):
    if pose_bone.rotation_mode == 'QUATERNION':
        mr = pose_bone.rotation_quaternion.to_matrix().to_4x4()
    else:
        mr = pose_bone.rotation_euler.to_matrix().to_4x4()
    m1 = Matrix.Translation(pose_bone.location) @ mr # @ matrix_scale(pose_bone.scale)

    E = ao.data.bones[pose_bone.name].matrix_local
    if pose_bone.parent is None:
        return E @ m1
    else:
        m2 = matrix_the_hard_way(pose_bone.parent, ao)
        E_p = ao.data.bones[pose_bone.parent.name].matrix_local
        return m2 @ E_p.inverted() @ E @ m1


def matrix_scale(scale_vec):
    return Matrix([[scale_vec[0],0,0,0],
                   [0,scale_vec[1],0,0],
                   [0,0,scale_vec[2],0],
                   [0,0,0,1]
    ])

def worldMatrix(ArmatureObject,Bone):
# simplified version of the matrix_the_hard_way
# To Test
# Probably can't use without update of the bones, since bone.matrix does not updates
# automatically
    _bone = ArmatureObject.pose.bones[Bone]
    _obj = ArmatureObject
    return _obj.matrix_world * _bone.matrix

# working version
def matrix_world(armature_ob, bone_name):
    local = armature_ob.data.bones[bone_name].matrix_local
    basis = armature_ob.pose.bones[bone_name].matrix_basis

    parent = armature_ob.pose.bones[bone_name].parent
    if parent == None:
        return  local @ basis
    else:
        parent_local = armature_ob.data.bones[parent.name].matrix_local
        return matrix_world(armature_ob, parent.name) @ (parent_local.inverted() @ local) @ basis         


def trans_coord_system(p1, o1, o2, M1, M2):

    # note is necessary to use a copy version of the matrix
    # otherwise it modifies the content of M2 outside the function
    # Actually it transposes matrix M2 outside the function. Must be the way
    # Blender handles the transform operators
    M2t = M2.copy()
    M2t.transpose()
    return M2t @ (o1 - o2 + M1 @ p1)

def compute_rotation(poseBone, pt0, pt1, pt2):

    M1 = Matrix([[1,0,0], [0,1,0], [0,0,1]])
    M2 = poseBone.matrix.copy() 

    v1 = trans_coord_system(pt1, Vector((0,0,0)), pt0, M1, M2)
    v2 = trans_coord_system(pt2, Vector((0,0,0)), pt0, M1, M2)

    a = v1.normalized()
    b = v2.normalized()

    c = a.cross(b)

    # c.magnitude from normalized vectors should not be greater than 1 ?
    # in some cases c.magnitude > 1 then asin fails [-1,+1]

    # check for cases > 1 and round to 1
    v_magnitude = c.magnitude
    if (v_magnitude > 1):
        v_magnitude = 1

    # to determine if angle in [0, pi/2] or in [pi/2, pi]
    l = np.linalg.norm(pt1 - pt0)
    dist = np.linalg.norm(pt1 - pt2)
    dist_max = math.sqrt(2*l*l)

    if (dist < dist_max): theta = math.asin(v_magnitude) 
    else: theta = theta = math.pi - math.asin(v_magnitude)

    if (c.magnitude>0):
        axis = c.normalized()
        st2 = math.sin(theta/2)
        q = Quaternion( [math.cos(theta/2), st2*axis.x, st2*axis.y, st2*axis.z] )
    else:
        q = Quaternion( [1,0,0,0] )

    return q

# set skel in T pose
D = math.pi / 180
TPose2 = {
    "LeftShoulder" : (0, 0, -90*D, 'XYZ'),
    "LeftArm" : (0, 0, -90*D, 'XYZ'),
    "LeftForeArm" :   (0, 0, -90*D, 'XYZ'),
    "LeftHand" :      (0, 0, -90*D, 'XYZ'),

    "RightShoulder" : (0, 0, 90*D, 'XYZ'),
    "RightArm" : (0, 0, 90*D, 'XYZ'),
    "RightForeArm" :   (0, 0, 90*D, 'XYZ'),
    "RightHand" :      (0, 0, 90*D, 'XYZ'),

    "LeftUpLeg" :     (-90*D, 0, 0, 'XYZ'),
    "LeftLeg" :      (-90*D, 0, 0, 'XYZ'),
    #"foot.L" :      (None, 0, 0, 'XYZ'),
    #"toe.L" :       (pi, 0, 0, 'XYZ'),

    "RightUpLeg" :     (-90*D, 0, 0, 'XYZ'),
    "RightLeg" :      (-90*D, 0, 0, 'XYZ'),
    #"foot.R" :      (None, 0, 0, 'XYZ'),
    #"toe.R" :       (pi, 0, 0, 'XYZ'),
}

def insertRotation(pb, mat, frame=None):
    if frame is None:
        frame = bpy.context.scene.frame_current
    if pb.rotation_mode == 'QUATERNION':
        pb.rotation_quaternion = mat.to_quaternion()
        pb.keyframe_insert("rotation_quaternion", frame=frame, group=pb.name)
    elif pb.rotation_mode == "AXIS_ANGLE":
        pb.rotation_axis_angle = mat.to_axis_angle()
        pb.keyframe_insert("rotation_axis_angle", frame=frame, group=pb.name)
    else:
        pb.rotation_euler = mat.to_euler(pb.rotation_mode)
        pb.keyframe_insert("rotation_euler", frame=frame, group=pb.name)


def updateScene():
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()

def put_in_rest_pose(rig):
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix()
    updateScene()


def put_in_tpose(rig):
    for pb in rig.pose.bones:
        if pb.name in TPose2.keys():
            ex,ey,ez,order = TPose2[pb.name]
        else:
            continue

        euler = pb.matrix.to_euler(order)
        if ex is None:
            ex = euler.x
        if ey is None:
            ey = euler.y
        if ez is None:
            ez = euler.z
        euler = Euler((ex,ey,ez), order)
        mat = euler.to_matrix().to_4x4()
        mat.col[3] = pb.matrix.col[3]

        loc = pb.bone.matrix_local
        if pb.parent:
            mat = pb.parent.matrix.inverted() @ mat
            loc = pb.parent.bone.matrix_local.inverted() @ loc
        mat =  loc.inverted() @ mat
        euler = mat.to_euler('YZX')
        euler.y = 0
        pb.matrix_basis = euler.to_matrix().to_4x4()
        updateScene()


def retarget_constraints(bone_corresp_file, action, trg_skel, act_x, act_y, act_z):

    # load bvh file
    bvh_file = action
    bpy.ops.import_anim.bvh(filepath=bvh_file, axis_up='Y', axis_forward='-Z', filter_glob="*.bvh",
                                    target='ARMATURE', global_scale=1.0, frame_start=1, use_fps_scale=True,
                                    use_cyclic=False, update_scene_duration=True, rotate_mode='NATIVE')

    # create target animation
    trg_skel.animation_data_clear()

    # get frames of action
    fbvh = bpy.path.basename(bvh_file)
    spbvh = fbvh.split('.')
    bvh_obj_name = spbvh[0]
    bvh_obj = bpy.data.objects[bvh_obj_name]
    act_size = bvh_obj.animation_data.action.frame_range

    nfirst = int(act_size[0])
    nlast = int(act_size[1])

    # read correspondence bones
    list_bones_src = []
    list_bones_trg = []
    with open(bone_corresp_file, 'r') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 

    for text in content:
        line = text.split()
        if len(line) == 2:
            list_bones_trg.append(line[0])
            list_bones_src.append(line[1])

    # Scale bvh skeleton to match size of our model
    # This step is important to transfer correctly translations to target, otherwise human steps
    # are too big or too small
    bbox_corners_skel = [bvh_obj.matrix_world @ Vector(corner) for corner in bvh_obj.bound_box]
    bbox_corners_target = [trg_skel.matrix_world @ Vector(corner) for corner in trg_skel.bound_box]
    dist_skel = bbox_corners_skel[1][2] - bbox_corners_skel[0][2]
    dist_target = bbox_corners_target[1][2] - bbox_corners_target[0][2]
    fscale = dist_target / dist_skel
    #fscale = 0.062
    bvh_obj.scale = (fscale, fscale, fscale) 

    # create constraints for all bones
    for bone_idx, bone in enumerate(list_bones_trg):
        pb_trg = trg_skel.pose.bones[bone]

        if bone == "Hips":
            if not pb_trg.constraints.get("Copy Location"):
                pb_trg.constraints.new("COPY_LOCATION")
                pb_trg.constraints['Copy Location'].target = bvh_obj
                pb_trg.constraints['Copy Location'].subtarget = list_bones_src[bone_idx]
            if not pb_trg.constraints.get("Copy Rotation"):
                pb_trg.constraints.new("COPY_ROTATION")
                pb_trg.constraints['Copy Rotation'].target = bvh_obj
                pb_trg.constraints['Copy Rotation'].subtarget = list_bones_src[bone_idx]
                if act_x:
                    pb_trg.constraints['Copy Rotation'].use_x = False
                if act_y:
                    pb_trg.constraints['Copy Rotation'].use_x = False
                if act_z:
                    pb_trg.constraints['Copy Rotation'].use_x = False
        else:
            if not pb_trg.constraints.get("Copy Rotation"):
                pb_trg.constraints.new("COPY_ROTATION")
                pb_trg.constraints['Copy Rotation'].target = bvh_obj
                pb_trg.constraints['Copy Rotation'].subtarget = list_bones_src[bone_idx]


    for bone in list_bones_trg:

        pb_trg = trg_skel.pose.bones[bone]

        mats = []
        for f in range(nfirst, nlast):
            bpy.context.scene.frame_set(f)
            mats.append([pb_trg.matrix.copy()])

        # transfer matrices to bone
        for f in range(nfirst, nlast):
            bpy.context.scene.frame_set(f)
            if bone == "Hips":
                # pb_trg.constraints['Copy Location'].influence = 0
                # pb_trg.constraints['Copy Rotation'].influence = 0
                pb_trg.matrix = mats[f-1][0]
                pb_trg.keyframe_insert(data_path='location', frame=f)
                pb_trg.keyframe_insert(data_path='rotation_quaternion', frame=f)
            else:
                # pb_trg.constraints['Copy Rotation'].influence = 0
                pb_trg.matrix = mats[f-1][0]
                pb_trg.keyframe_insert(data_path='rotation_quaternion', frame=f)

    # remove all bone constraints
    for bone in list_bones_trg:
        pb_trg = trg_skel.pose.bones[bone]
        if bone == "Hips":
            pb_ct = pb_trg.constraints.get('Copy Location')
            pb_trg.constraints.remove(pb_ct)
            pb_ct = pb_trg.constraints.get('Copy Rotation')
            pb_trg.constraints.remove(pb_ct)
        else:
            pb_ct = pb_trg.constraints.get('Copy Rotation')
            pb_trg.constraints.remove(pb_ct)

    bpy.data.objects.remove(bvh_obj)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)
    for block in bpy.data.actions:
        if block.users == 0:
            bpy.data.actions.remove(block)


class CAnimation:

    def __init__(self, srcRig, trgRig, list_bones_trg, list_bones_src, obj, skel):
        self.srcRig = srcRig
        self.trgRig = trgRig
        self.scene = bpy.context.scene
        self.boneAnims = OrderedDict()
        self.list_bones_trg = list_bones_trg
        self.list_bones_src = list_bones_src
        self.obj = obj
        self.skel = skel

        # for b_idx, b_name in enumerate(bones_skel):
        for b_idx, b_name in enumerate(self.list_bones_trg):
            trgBone = trgRig.pose.bones[b_name]
            srcBone = srcRig.pose.bones[self.list_bones_src[b_idx]]
            parent = self.get_parent_name(trgBone, self.list_bones_trg)
            self.boneAnims[b_name] = CBoneAnim(srcBone, trgBone, parent, self.obj, self.skel)
            # print("InitCanimation", parent, trgBone, srcBone)

    def get_parent_name (self, p_bone, list_bones):
    # find parent
        pb_parent = p_bone.parent
        if pb_parent:
            list_parent = pb_parent.name
            while (list_parent not in list_bones):
                new_bone = self.skel.pose.bones[list_parent].parent
                list_parent = new_bone.name
            return self.boneAnims[list_parent]
        else:
            return None

    def putInTPoses(self):
        scn = bpy.context.scene
        scn.frame_set(0)
        put_in_rest_pose(self.srcRig)
        put_in_tpose(self.srcRig)
        put_in_rest_pose(self.trgRig)
        put_in_tpose(self.trgRig)
        for banim in self.boneAnims.values():
            banim.getTPoseMatrix()


    def retarget(self, frames):
        scn = bpy.context.scene
        for frame in frames:
            scn.frame_set(frame)
            for banim in self.boneAnims.values():
                banim.retarget(frame)


class CBoneAnim:

    def __init__(self, srcBone, trgBone, parent, obj, skel):
        self.srcMatrix = None
        self.trgMatrix = None
        self.srcBone = srcBone
        self.trgBone = trgBone
        self.parent = parent
        self.aMatrix = None
        self.obj = obj
        self.skel = skel
        # matrix_local: rest pose matrix in bone local space
        if self.parent:
            self.bMatrix = trgBone.bone.matrix_local.inverted() @ self.parent.trgBone.bone.matrix_local
        else:
            self.bMatrix = trgBone.bone.matrix_local.inverted()


    def getTPoseMatrix(self):
        trgrot = self.trgBone.matrix.decompose()[1]
        trgmat = trgrot.to_matrix().to_4x4()
        srcrot = self.srcBone.matrix.decompose()[1]
        srcmat = srcrot.to_matrix().to_4x4()
        self.aMatrix = srcmat.inverted() @ trgmat


    def retarget(self, frame):
        self.srcMatrix = self.srcBone.matrix.copy()
        self.trgMatrix = self.srcMatrix @ self.aMatrix
        # self.trgMatrix.col[3] = self.srcMatrix.col[3]
        if self.parent:
            mat1 = self.parent.trgMatrix.inverted() @ self.trgMatrix
        else:
            mat1 = self.trgMatrix
        mat2 = self.bMatrix @ mat1

        insertRotation(self.trgBone, mat2, frame)
        if not self.parent:
            pb = self.trgBone
            pb_src = self.srcBone
            loc = self.obj.matrix_world @ pb_src.matrix.to_translation()
            pb.location =  self.skel.matrix_world.inverted() @ pb.bone.matrix_local.inverted() @ loc
            pb.keyframe_insert(data_path='location', frame=frame)

        return



def retarget_addon(bone_corresp_file, action, trg_skel, rig):
    # retarget based on bvh_retarget addon by Thomas Larsson
    # https://bitbucket.org/Diffeomorphic/retarget_bvh/src/master/

    print("Using Retarget Addon")

    # load bvh file
    bvh_file = action
    bpy.ops.import_anim.bvh(filepath=bvh_file)

    # create target animation
    trg_skel.animation_data_clear()

    # get frames of action
    fbvh = bpy.path.basename(bvh_file)
    spbvh = fbvh.split('.')
    bvh_obj_name = spbvh[0]
    bvh_obj = bpy.data.objects[bvh_obj_name]
    act_size = bvh_obj.animation_data.action.frame_range

    nfirst = int(act_size[0])
    nlast = int(act_size[1])

    # read correspondence bones
    list_bones_src = []
    list_bones_trg = []
    with open(bone_corresp_file, 'r') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 

    for text in content:
        line = text.split()
        if len(line) == 2:
            list_bones_trg.append(line[0])
            list_bones_src.append(line[1])

    # Scale bvh skeleton to match size of our model
    # This step is important to transfer correctly translations to target, otherwise human steps
    # are too big or too small
    # # Not working if action starts lay down
    # bbox_corners_skel = [bvh_obj.matrix_world @ Vector(corner) for corner in bvh_obj.bound_box]
    # bbox_corners_target = [trg_skel.matrix_world @ Vector(corner) for corner in trg_skel.bound_box]
    # dist_skel = bbox_corners_skel[1][2] - bbox_corners_skel[0][2]
    # dist_target = bbox_corners_target[1][2] - bbox_corners_target[0][2]
    # fscale = dist_target / dist_skel
    # # Calculate scale using bone length
    bidx = list_bones_trg.index("LeftUpLeg")
    trg_bone_length = trg_skel.pose.bones[list_bones_trg[bidx]].length
    src_bone_length = bvh_obj.pose.bones[list_bones_src[bidx]].length
    fscale = trg_bone_length / src_bone_length

    #fscale = 0.062
    bvh_obj.scale = (fscale, fscale, fscale) 

    # bpy.context.view_layer.update()

    anim = CAnimation(bvh_obj, trg_skel, list_bones_trg, list_bones_src, bvh_obj, trg_skel)
    anim.putInTPoses()
    vals = np.arange(nfirst, nlast + 1)
    anim.retarget(vals)

    bpy.context.scene.frame_start = nfirst
    bpy.context.scene.frame_end = nlast

    act = trg_skel.animation_data.action
    act.name = bvh_obj.name

    # set frame to 1
    bpy.context.scene.frame_set(1)


    bpy.data.objects.remove(bvh_obj)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)
    for block in bpy.data.actions:
        if block.users == 0:
            bpy.data.actions.remove(block)



def retarget_matworld(bone_corresp_file, action, trg_skel, rig):

    # load bvh file
    bvh_file = action
    bpy.ops.import_anim.bvh(filepath=bvh_file, axis_up='Y', axis_forward='-Z', filter_glob="*.bvh",
                                    target='ARMATURE', global_scale=1.0, frame_start=1, use_fps_scale=True,
                                    use_cyclic=False, update_scene_duration=True, rotate_mode='NATIVE')

    # create target animation
    trg_skel.animation_data_clear()

    # get frames of action
    fbvh = bpy.path.basename(bvh_file)
    spbvh = fbvh.split('.')
    bvh_obj_name = spbvh[0]
    bvh_obj = bpy.data.objects[bvh_obj_name]
    act_size = bvh_obj.animation_data.action.frame_range

    nfirst = int(act_size[0])
    nlast = int(act_size[1])

    # read correspondence bones
    list_bones_src = []
    list_bones_trg = []
    with open(bone_corresp_file, 'r') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 

    for text in content:
        line = text.split()
        if len(line) == 2:
            list_bones_trg.append(line[0])
            list_bones_src.append(line[1])

    # Scale bvh skeleton to match size of our model
    # This step is important to transfer correctly translations to target, otherwise human steps
    # are too big or too small
    bbox_corners_skel = [bvh_obj.matrix_world @ Vector(corner) for corner in bvh_obj.bound_box]
    bbox_corners_target = [trg_skel.matrix_world @ Vector(corner) for corner in trg_skel.bound_box]
    dist_skel = bbox_corners_skel[1][2] - bbox_corners_skel[0][2]
    dist_target = bbox_corners_target[1][2] - bbox_corners_target[0][2]
    fscale = dist_target / dist_skel
    #fscale = 0.062
    bvh_obj.scale = (fscale, fscale, fscale) 

    # if rig == 'mixamo':
    #     list1 = ["mixamorig:LeftUpLeg", "mixamorig:LeftLeg"]
    #     list2 = ["LeftUpLeg", "LeftLeg"]

    #     # Set Avatar bone rolls to be the same as the input bvh motion
    #     bpy.context.view_layer.objects.active = bvh_obj
    #     bpy.ops.object.mode_set(mode = 'EDIT')
    #     src_bones_roll = []
    #     for bone in list1:
    #         src_ebone = bvh_obj.data.edit_bones[bone]
    #         src_bones_roll.append(src_ebone.roll)
    #     src_bones_roll = np.array(src_bones_roll)
    #     bpy.ops.object.mode_set(mode = 'OBJECT')


    #     bpy.context.view_layer.objects.active = trg_skel
    #     bpy.ops.object.mode_set(mode = 'EDIT')
    #     for bidx, bone in enumerate(list2):
    #         # select bone to copy roll
    #         trg_skel.data.edit_bones[bone].roll = src_bones_roll[bidx]
    #     bpy.ops.object.mode_set(mode = 'OBJECT')

    # bpy.context.view_layer.update()

    for f in range(nfirst, nlast):

        bpy.context.scene.frame_set(f)

        for bidx, bone in enumerate(list_bones_trg):        

            pb_trg = trg_skel.pose.bones[bone]

            mw_src = matrix_world(bvh_obj, list_bones_src[bidx])
            mw_trg = matrix_world(trg_skel, bone)

            t_src, mw_src_R, s_src = mw_src.decompose()
            t_trg, mw_trg_R, s_trg = mw_trg.decompose()

            basis_trg = trg_skel.pose.bones[bone].matrix_basis
            basis_t, basis_trg_R, basis_s = basis_trg.decompose()

            T1 = basis_trg_R @ mw_trg_R.inverted() @ mw_src_R

            if rig == "mixamo" and bone == 'Hips':
                # We need to invert X axis
                S1 = mw_src
                t, R, s = S1.decompose() # important decompose R to get
                                         # correct angles later
                rot_x = R.to_euler('XYZ')[0]
                rot_y = R.to_euler('XYZ')[1]
                rot_z = R.to_euler('XYZ')[2]
                mat_x = Matrix.Rotation(rot_x, 4, 'X')
                mat_y = Matrix.Rotation(rot_y, 4, 'Y')
                mat_z = Matrix.Rotation(rot_z, 4, 'Z')
                S1 = mat_z @ mat_y @ mat_x.inverted()
                S1 = basis_trg @ mw_trg.inverted() @ S1

                # apply rotation to bone
                pb_trg.rotation_mode = 'XYZ'
                pb_trg.rotation_euler = S1.to_euler('XYZ')
                pb_trg.keyframe_insert(data_path='rotation_euler', frame=f)
            else:
                pb_trg.rotation_mode = 'XYZ'
                pb_trg.rotation_euler = T1.to_euler('XYZ')
                pb_trg.keyframe_insert(data_path='rotation_euler', frame=f)
            
            # need to introduce some rotation offsets to bone
            # depending on the motion file (bvh) skeleton
            # TODO

            # Code assumes objects are created in world origin
            # otherwise function need to change to account for the displacement
            if bone == "Hips":
                pb_src = bvh_obj.pose.bones[list_bones_src[bidx]]
                loc = bvh_obj.matrix_world @ pb_src.matrix.to_translation()
                pb_trg.location =  trg_skel.matrix_world.inverted() @ pb_trg.bone.matrix_local.inverted() @ loc
                pb_trg.keyframe_insert(data_path='location', frame=f)


    bpy.data.objects.remove(bvh_obj)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)
    for block in bpy.data.actions:
        if block.users == 0:
            bpy.data.actions.remove(block)



def retarget_skeleton(source_skel_type, action, target):

    # This is a very slow version. An attempt to debug the code and make it faster, check file
    # transfer_motion.py

    # load bvh file
    bvh_file = action
    bpy.ops.import_anim.bvh(filepath=bvh_file, axis_up='Y', axis_forward='-Z', filter_glob="*.bvh",
                                    target='ARMATURE', global_scale=1.0, frame_start=1, use_fps_scale=False,
                                    use_cyclic=False, rotate_mode='NATIVE')

    # create target animation
    target.animation_data_clear()

    me_cp = target.data.copy()

    target_cp = bpy.data.objects.new("Skel_cp", me_cp)
    target_cp.location = target.location

    bpy.context.scene.collection.objects.link(target_cp)
    bpy.context.view_layer.update()

    target_cp.animation_data_clear()

    # get frames of action
    fbvh = bpy.path.basename(bvh_file)
    spbvh = fbvh.split('.')
    bvh_obj_name = spbvh[0]
    bvh_obj = bpy.data.objects[bvh_obj_name]
    act_size = bvh_obj.animation_data.action.frame_range

    nfirst = int(act_size[0])
    nlast = int(act_size[1])

    bpy.context.scene.frame_start = nfirst
    bpy.context.scene.frame_end = nlast

    bone_corresp = read_text_lines(source_skel_type)

    # Scale bvh skeleton to match size of our model
    # This step is important to transfer correctly translations to target, otherwise human steps
    # are too big or too small
    bbox_corners_skel = [bvh_obj.matrix_world @ Vector(corner) for corner in bvh_obj.bound_box]
    bbox_corners_target = [target.matrix_world @ Vector(corner) for corner in target.bound_box]
    dist_skel = bbox_corners_skel[1][2] - bbox_corners_skel[0][2]
    dist_target = bbox_corners_target[1][2] - bbox_corners_target[0][2]
    fscale = dist_target / dist_skel
    #fscale = 0.062
    bvh_obj.scale = (fscale, fscale, fscale)    


    # target bones in rest position: to compute global rotation
    trg_bone_loc_hips = get_bone_head_position(target, "Hips")
    trg_bone_loc_lefthips = get_bone_head_position(target, "LeftUpLeg")
    trg_bone_loc_righthips = get_bone_head_position(target, "RightUpLeg")
    trg_bone_loc_neck = get_bone_head_position(target, "Neck")

    # Go through animation and compute rotations of each bone
    for f in range(nfirst, nlast):

        bpy.context.scene.frame_set(f)

        # get bvh bone locations
        source_bone_name = find_bone_match(bone_corresp, "Hips")
        src_bone_loc_hips = get_bone_head_position(bvh_obj, source_bone_name)
        source_bone_name = find_bone_match(bone_corresp, "LeftUpLeg")
        src_bone_loc_lefthips = get_bone_head_position(bvh_obj, source_bone_name)
        source_bone_name = find_bone_match(bone_corresp, "RightUpLeg")
        src_bone_loc_righthips = get_bone_head_position(bvh_obj, source_bone_name)
        source_bone_name = find_bone_match(bone_corresp, "Neck")
        src_bone_loc_neck = get_bone_head_position(bvh_obj, source_bone_name)

        matrix_os= {}
        #for to_match in goal.data.bones:
        for bone in target_cp.data.bones:
            bone_match = find_bone_match(bone_corresp, bone.name)
            if bone_match != "none":
                #matrix_os[bone_match] = goal.data.bones[bone_match].matrix_local # if we want to match rest pose
                ebp = bvh_obj.pose.bones[bone_match]
                matrix_os[bone_match] = matrix_the_hard_way(ebp, bvh_obj)
                #print([ "matrix", bone_match, matrix_os[bone_match] ] )

        hips_match_name = find_bone_match(bone_corresp, "Hips")

        # read source motion
        for pb in target_cp.pose.bones:
        
            bone_name = find_bone_match(bone_corresp, pb.name)
            if bone_name != "none":
                goal_bone = bone_name
            
                # source bone
                spb = bvh_obj.pose.bones[bone_name]
        
                # # insert keyframe
                # loc = spb.location

                if pb.parent is None:
                    #loc = source.matrix_world @ source.pose.bones["mixamorig:Hips"].head
                    loc = bvh_obj.matrix_world @ bvh_obj.pose.bones[hips_match_name].head
                    pb.location = target_cp.matrix_world.inverted() @ pb.bone.matrix_local.inverted() @ loc
                    pb.keyframe_insert('location', frame=f, group=pb.name)

                    # compute translation and first rotation between rest position and desired points
                    A = np.mat((trg_bone_loc_hips, trg_bone_loc_lefthips, trg_bone_loc_righthips, trg_bone_loc_neck)) # my skeleton
                    B = np.mat((src_bone_loc_hips, src_bone_loc_lefthips, src_bone_loc_righthips, src_bone_loc_neck)) # bvh skeleton
                    R, T = rigid_transform_3D(A,B)

                    mR = Matrix([[R[0,0],R[0,1],R[0,2]], [R[1,0],R[1,1],R[1,2]], [R[2,0],R[2,1],R[2,2]]])
                    mR.resize_4x4()

                    boneRefPoseMtx = pb.bone.matrix_local.copy()
                    rotMtx = boneRefPoseMtx.inverted() @ mR @ boneRefPoseMtx
                    pb.rotation_mode = 'XYZ'
                    pb.rotation_euler = rotMtx.to_euler()
                    pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)

                    # # To try in the future. Not working I think because connection between hips and legs
                    # # in source armatures don't exists and this creates strange rotations
                    # len2 = target.data.bones[pb.name].length
                    # len1 = bvh_obj.data.bones[goal_bone].length
                    # #to_pose.matrix = matrix_os[goal_bone] @ Matrix.Scale(0.076, 4)
                    # m1 = target.matrix_world @ matrix_os[goal_bone] @ pb.bone.matrix_local
                    # loc,rot,scale = m1.decompose()
                    # # # to_pose.location = loc
                    # if 'QUATERNION' == pb.rotation_mode:
                    #     pb.rotation_quaternion = rot
                    # else:
                    #     pb.rotation_euler = pb.to_euler(pb.rotation_mode)

                else:
                    # pb.location = loc
                    # pb.keyframe_insert('location', frame=f, group=pb.name)

                    # we can not set .matrix, because a lot of stuff behind the scenes has not yet
                    # caught up with our alterations, and it ends up doing math on outdated numbers
                    mp = matrix_the_hard_way(pb.parent, target_cp) @ matrix_for_bone_from_parent(pb, target_cp)
                    m2 = mp.inverted() @ matrix_os[goal_bone] # @ Matrix.Scale(goal.data.bones[goal_bone].length, 4)
                    #m2 = matrix_os[goal_bone] # @ Matrix.Scale(goal.data.bones[goal_bone].length, 4)
                    loc,rot,scale = m2.decompose()
                    # # pb.location = loc
                    # if 'QUATERNION' == pb.rotation_mode:
                    #     pb.rotation_quaternion = rot
                    # else:
                    #     pb.rotation_euler = rot.to_euler(pb.rotation_mode)
                    # # pb.scale = scale / target.data.bones[pb.name].length

                    # print("last debug 2")
                    # print(rot)

                    pb.rotation_mode = 'XYZ'
                    #rot = matrices_target[pb.name] @ spb.matrix_basis.copy()
                    pb.rotation_euler = rot.to_euler(pb.rotation_mode)
                    pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)


    # copy rotations from 3d points
    # now skeletons are equal (same name bones, same length bones)
    for f in range(nfirst, nlast):
    #for f in range(1, 2):

        # set target in rest position
        for bone in target.pose.bones:
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = (0, 0, 0)

        bpy.context.scene.frame_set(f)
        # bpy.context.view_layer.update()

        for pb in target.pose.bones:

            pb_cp = target_cp.pose.bones[pb.name]

            if pb.parent is None:

                pb.location = pb_cp.location
                pb.keyframe_insert('location', frame=f, group=pb.name)

                pb_cp.rotation_mode = 'XYZ'
                pb.rotation_mode = 'XYZ'
                pb.rotation_euler = pb_cp.rotation_euler
                pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)

            else:
                # we only recalculate bones that have children
                # if pb.children:

                # Using view_layer.update() . Very slow !!!
                # recalculate rotations to avoid strange mesh deformations
                pt0 = get_pose_bone_head_position(target, pb)
                pt1 = get_pose_bone_tail_position(target, pb)
                pt2 = get_pose_bone_tail_position(target_cp, pb_cp)

                # # Update matrices ourselves . Faster
                # # recalculate rotations to avoid strange mesh deformations
                # pb_child = pb.children[0]   # we assume each bone only have one children !!
                # pb_cp_child = pb_cp.children[0]
                # # pt0 = matrix_the_hard_way(pb, target).to_translation()
                # # pt1 = matrix_the_hard_way(pb_child, target).to_translation()
                # pt0 = get_pose_bone_head_position(target, pb)
                # pt1 = get_pose_bone_tail_position(target, pb)
                # pt2 = matrix_the_hard_way(pb_cp_child, target_cp).to_translation()

                q2 = compute_rotation(pb, pt0, pt1, pt2)

                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = q2
                pb.keyframe_insert('rotation_quaternion', frame=f, group=pb.name)

            bpy.context.view_layer.update()


    # Remove bvh skeleton (bvh_obj)
    bpy.data.objects.remove(bvh_obj)
    bpy.data.objects.remove(target_cp)


