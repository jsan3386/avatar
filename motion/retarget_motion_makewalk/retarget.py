#!/usr/bin/python
# -*- coding: utf-8 -*-

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  Authors:             Thomas Larsson
#  Script copyright (C) Thomas Larsson 2014
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

#
#   M_b = global bone matrix, relative world (PoseBone.matrix)
#   L_b = local bone matrix, relative parent and rest (PoseBone.matrix_local)
#   R_b = bone rest matrix, relative armature (Bone.matrix_local)
#   T_b = global T-pose marix, relative world
#
#   M_b = M_p R_p^-1 R_b L_b
#   M_b = A_b M'_b
#   T_b = A_b T'_b
#   A_b = T_b T'^-1_b
#   B_b = R^-1_b R_p
#
#   L_b = R^-1_b R_p M^-1_p A_b M'_b
#   L_b = B_b M^-1_p A_b M'_b
#


import bpy
import mathutils
import time
from collections import OrderedDict
from mathutils import *
from bpy.props import *

from simplify import simplifyFCurves, rescaleFCurves
from utils import *
import t_pose



class CAnimation:

    def __init__(self, srcRig, trgRig, boneAssoc, context):
        self.srcRig = srcRig
        self.trgRig = trgRig
        self.scene = context.scene
        self.boneAnims = OrderedDict()

        for (trgName, srcName) in boneAssoc:
            try:
                trgBone = trgRig.pose.bones[trgName]
                srcBone = srcRig.pose.bones[srcName]
            except KeyError:
                print("  -", trgName, srcName)
                continue
            banim = self.boneAnims[trgName] = CBoneAnim(srcBone, trgBone, self, context)


    def printResult(self, scn, frame):
        scn.frame_set(frame)
        for name in ["LeftHip"]:
            banim = self.boneAnims[name]
            banim.printResult(frame)


    def setTPose(self, context):
        putInRestPose(self.srcRig, True)
        t_pose.setTPose(self.srcRig, context)
        putInRestPose(self.trgRig, True)
        t_pose.setTPose(self.trgRig, context)
        #for banim in self.boneAnims.values():
        #    banim.insertTPoseFrame()
        context.scene.frame_set(0)
        for banim in self.boneAnims.values():
            banim.getTPoseMatrix()
        putInRestPose(self.srcRig, True)


    def retarget(self, frames, context):
        objects = hideObjects(context, self.srcRig)
        scn = context.scene
        try:
            for frame in frames:
                scn.frame_set(frame)#scn.frame_set(frame+50) Quan entres a aquest frame, hips ja te un location en aquest frame.
                for banim in self.boneAnims.values():
                    banim.retarget(frame)#banim.retarget(frame+50)
        finally:
            unhideObjects(objects)


class CBoneAnim:

    def __init__(self, srcBone, trgBone, anim, context):
        self.name = srcBone.name
        self.srcMatrices = {}
        self.trgMatrices = {}
        self.srcMatrix = None
        self.trgMatrix = None
        self.srcBone = srcBone
        self.trgBone = trgBone
        self.order,self.locks = getLocks(trgBone, context)
        self.aMatrix = None
        self.parent = self.getParent(trgBone, anim)
        if self.parent:
            self.trgBone.McpParent = self.parent.trgBone.name
            trgParent = self.parent.trgBone
            self.bMatrix = Mult2(trgBone.bone.matrix_local.inverted(), trgParent.bone.matrix_local)
        else:
            self.bMatrix = trgBone.bone.matrix_local.inverted()
        self.useLimits = anim.scene.McpUseLimits


    def __repr__(self):
        if self.parent:
            parname = self.parent.name
        else:
            parname = None
        return (
            "<CBoneAnim %s\n" % self.name +
            "  src %s\n" % self.srcBone.name +
            "  trg %s\n" % self.trgBone.name +
            "  par %s\n" % parname +
            "  A %s\n" % self.aMatrix +
            "  B %s\n" % self.bMatrix)


    def printResult(self, frame):
        print(
            "Retarget %s => %s\n" % (self.srcBone.name, self.trgBone.name) +
            "S %s\n" % self.srcBone.matrix +
            "T %s\n" % self.trgBone.matrix +
            "R %s\n" % Mult2(self.trgBone.matrix, self.srcBone.matrix.inverted())
            )


    def getParent(self, pb, anim):
        pb = pb.parent
        while pb:
            if pb.McpBone:
                try:
                    return anim.boneAnims[pb.name]
                except KeyError:
                    pass

            subtar = None
            for cns in pb.constraints:
                if (cns.type[0:4] == "COPY" and
                    cns.type != "COPY_SCALE" and
                    cns.influence > 0.8):
                    subtar = cns.subtarget

            if subtar:
                pb = anim.trgRig.pose.bones[subtar]
            else:
                pb = pb.parent
        return None


    def insertKeyFrame(self, mat, frame):
        pb = self.trgBone
        setRotation(pb, mat, frame, pb.name)
        if not self.parent:
            pb.location = mat.to_translation()
            pb.keyframe_insert("location", frame=frame, group=pb.name)


    def insertTPoseFrame(self):
        mat = t_pose.getStoredBonePose(self.trgBone)
        self.insertKeyFrame(mat, 10)


    def getTPoseMatrix(self):
        self.aMatrix = Mult2(self.srcBone.matrix.inverted(), self.trgBone.matrix)
        if not isRotationMatrix(self.trgBone.matrix):
            raise RuntimeError("Target %s not rotation matrix %s" % (self.trgBone.name, self.trgBone.matrix))
        if not isRotationMatrix(self.srcBone.matrix):
            raise RuntimeError("Source %s not rotation matrix %s" % (self.srcBone.name, self.srcBone.matrix))
        if not isRotationMatrix(self.aMatrix):
            raise RuntimeError("A %s not rotation matrix %s" % (self.trgBone.name, self.aMatrix.matrix))


    def retarget(self, frame):
        if not self.parent:
            print("Retarget frame", frame)
        self.srcMatrix = self.srcBone.matrix.copy()
        if not self.parent:
            print(self.name, self.srcMatrix)
        self.trgMatrix = Mult2(self.srcMatrix, self.aMatrix)
        if not self.parent:
            print(self.name, self.trgMatrix)
        self.trgMatrix.col[3] = self.srcMatrix.col[3]
        if not self.parent:
            print(self.name, self.trgMatrix)
        if self.parent:
            mat1 = Mult2(self.parent.trgMatrix.inverted(), self.trgMatrix)
        else:
            mat1 = self.trgMatrix
        if not self.parent:
            print(self.name, mat1)
        mat2 = Mult2(self.bMatrix, mat1)
        if not self.parent:
            print(self.name, mat2)
        mat3 = correctMatrixForLocks(mat2, self.order, self.locks, self.trgBone, self.useLimits)
        if not self.parent:
            print(self.name, mat3)
        self.insertKeyFrame(mat3, frame)


        self.srcMatrices[frame] = self.srcMatrix
        mat1 = Mult2(self.bMatrix.inverted(), mat3)
        if self.parent:
            self.trgMatrix = Mult2(self.parent.trgMatrix, mat1)
        else:
            self.trgMatrix = mat1
        self.trgMatrices[frame] = self.trgMatrix

        return

        if self.name == "upper_arm.L":
            print()
            print(self)
            print("S ", self.srcMatrix)
            print("T ", self.trgMatrix)
            print(self.parent.name)
            print("TP", self.parent.trgMatrix)
            print("M1", mat1)
            print("M2", mat2)
            print("MB2", self.trgBone.matrix)


def getLocks(pb, context):
    scn = context.scene
    locks = []
    order = 'XYZ'
    if scn.McpClearLocks:
        pb.lock_rotation[0] = pb.lock_rotation[2] = False
        for cns in pb.constraints:
            if cns.type == 'LIMIT_ROTATION':
                cns.use_limit_x = cns.use_limit_z = 0

    if pb.lock_rotation[1]:
        locks.append(1)
        order = 'YZX'
        if pb.lock_rotation[0]:
            order = 'YXZ'
            locks.append(0)
        if pb.lock_rotation[2]:
            locks.append(2)
    elif pb.lock_rotation[2]:
        locks.append(2)
        order = 'ZYX'
        if pb.lock_rotation[0]:
            order = 'ZXY'
            locks.append(0)
    elif pb.lock_rotation[0]:
        locks.append(0)
        order = 'XYZ'

    if pb.rotation_mode != 'QUATERNION':
        order = pb.rotation_mode

    return order,locks


def correctMatrixForLocks(mat, order, locks, pb, useLimits):
    head = Vector(mat.col[3])

    if locks:
        euler = mat.to_3x3().to_euler(order)
        for n in locks:
            euler[n] = 0
        mat = euler.to_matrix().to_4x4()

    if not useLimits:
        mat.col[3] = head
        return mat

    for cns in pb.constraints:
        if (cns.type == 'LIMIT_ROTATION' and
            cns.owner_space == 'LOCAL' and
            not cns.mute and
            cns.influence > 0.5):
            euler = mat.to_3x3().to_euler(order)
            if cns.use_limit_x:
                euler.x = min(cns.max_x, max(cns.min_x, euler.x))
            if cns.use_limit_y:
                euler.y = min(cns.max_y, max(cns.min_y, euler.y))
            if cns.use_limit_z:
                euler.z = min(cns.max_z, max(cns.min_z, euler.z))
            mat = euler.to_matrix().to_4x4()

    mat.col[3] = head
    return mat


def hideObjects(context, rig):
    if bpy.app.version >= (2,80,0):
        return None
    objects = []
    for ob in getSceneObjects(context):
        if ob != rig:
            objects.append((ob, list(ob.layers)))
            ob.layers = 20*[False]
    return objects


def unhideObjects(objects):
    if bpy.app.version >= (2,80,0):
        return
    for (ob,layers) in objects:
        ob.layers = layers


def clearMcpProps(rig):
    keys = list(rig.keys())
    for key in keys:
        if key[0:3] == "Mcp":
            del rig[key]

    for pb in rig.pose.bones:
        keys = list(pb.keys())
        for key in keys:
            if key[0:3] == "Mcp":
                del pb[key]


def retargetAnimation(context, srcRig, trgRig):
    import source, target
    from fkik import setMhxIk, setRigifyFKIK, setRigify2FKIK

    startProgress("Retargeting")
    scn = context.scene
    setMhxIk(trgRig, True, True, 0.0)
    frames = getActiveFrames(srcRig)
    nFrames = len(frames)
    setActiveObject(context, trgRig)
    if trgRig.animation_data:
        trgRig.animation_data.action = None

    if isRigify(trgRig):
        setRigifyFKIK(trgRig, 0.0)
    elif isRigify2(trgRig):
        setRigify2FKIK(trgRig, 1.0)

    try:
        #print("*******************")
        #print("M'he posat al frames[0]")
        #print(frames)
        #print(frames[0])
        #print("*******************")
        #scn.frame_current = frames[2]
        scn.frame_current = frames[0]
    except:
        raise MocapError("No frames found.")
    oldData = changeTargetData(trgRig, scn)

    source.ensureSourceInited(scn)
    source.setArmature(srcRig, scn)
    print("Retarget %s --> %s" % (srcRig.name, trgRig.name))

    print("ENSURE TARGET INITED")
    target.ensureTargetInited(scn)
    boneAssoc = target.getTargetArmature(trgRig, context)
    anim = CAnimation(srcRig, trgRig, boneAssoc, context)
    anim.setTPose(context)

    setCategory("Retarget")
    frameBlock = frames[0:100]#frames[0:100]
    index = 0
    try:
        while frameBlock:
            showProgress(index, frames[index], nFrames)
            anim.retarget(frameBlock, context)
            index += 100
            frameBlock = frames[index:index+100]
        scn.frame_current = frames[0]
    finally:
        restoreTargetData(trgRig, oldData)

    #anim.printResult(scn, 1)

    setInterpolation(trgRig)
    act = trgRig.animation_data.action
    act.name = trgRig.name[:4] + srcRig.name[2:]
    act.use_fake_user = True
    clearCategory()
    endProgress("Retargeted %s --> %s" % (srcRig.name, trgRig.name))


#
#   changeTargetData(rig, scn):
#   restoreTargetData(rig, data):
#

def changeTargetData(rig, scn):
    tempProps = [
        ("MhaRotationLimits", 0.0),
        ("MhaArmIk_L", 0.0),
        ("MhaArmIk_R", 0.0),
        ("MhaLegIk_L", 0.0),
        ("MhaLegIk_R", 0.0),
        ("MhaSpineIk", 0),
        ("MhaSpineInvert", 0),
        ("MhaElbowPlant_L", 0),
        ("MhaElbowPlant_R", 0),
        ]

    props = []
    for (key, value) in tempProps:
        try:
            props.append((key, rig[key]))
            rig[key] = value
        except KeyError:
            pass

    permProps = [
        ("MhaElbowFollowsShoulder", 0),
        ("MhaElbowFollowsWrist", 0),
        ("MhaKneeFollowsHip", 0),
        ("MhaKneeFollowsFoot", 0),
        ("MhaArmHinge", 0),
        ("MhaLegHinge", 0),
        ]

    for (key, value) in permProps:
        try:
            rig[key+"_L"]
            rig[key+"_L"] = value
            rig[key+"_R"] = value
        except KeyError:
            pass

    layers = list(rig.data.layers)
    #if rig.MhAlpha8:
    #    rig.data.layers = MhxLayers
    #elif isRigify(rig):
    #    rig.data.layers = RigifyLayers
    rig.data.layers = RigifyLayers

    locks = []
    for pb in rig.pose.bones:
        constraints = []
#        if not scn.McpUseLimits:
#            for cns in pb.constraints:
#                if cns.type == 'LIMIT_DISTANCE':
#                    cns.mute = True
#                elif cns.type[0:5] == 'LIMIT':
#                    constraints.append( (cns, cns.mute) )
#                    cns.mute = True
        locks.append( (pb, constraints) )

    norotBones = []
    return (props, layers, locks, norotBones)


def restoreTargetData(rig, data):
    (props, rig.data.layers, locks, norotBones) = data

    for (key,value) in props:
        rig[key] = value

    for b in norotBones:
        b.use_inherit_rotation = True

    for lock in locks:
        (pb, constraints) = lock
        for (cns, mute) in constraints:
            cns.mute = mute


#
#    loadRetargetSimplify(context, filepath):
#

def loadRetargetSimplify(context, filepath, original_position,frame_start,origin):
    import load
    from fkik import limbsBendPositive

    print("\nLoad and retarget %s" % filepath)
    time1 = time.clock()
    scn = context.scene
    #trgRig = bpy.data.objects["Standard"]
    trgRig = context.active_object
    data = changeTargetData(trgRig, scn)
    # try:
    #clearMcpProps(trgRig)
    srcRig = load.readBvhFile(context, filepath, scn, False, original_position,frame_start,origin)
    frames = getActiveFrames(srcRig)
        #
    #     #print(frames)
    #     try:
    #         print("RETARGET ANIMATION")
    load.renameAndRescaleBvh(context, srcRig, trgRig)
    #         retargetAnimation(context, srcRig, trgRig)
    #         scn = context.scene
    #         if scn.McpDoBendPositive:
    #             limbsBendPositive(trgRig, True, True, (0,1e6))
    #         if scn.McpDoSimplify:
    #             simplifyFCurves(context, trgRig, False, False)
    #         if scn.McpRescale:
    #             rescaleFCurves(context, trgRig, scn.McpRescaleFactor)
    #     finally:
    #         load.deleteSourceRig(context, srcRig, 'Y_')
    # finally:
    #     restoreTargetData(trgRig, data)
    # time2 = time.clock()
    # print("%s finished in %.3f s" % (filepath, time2-time1))
    return

