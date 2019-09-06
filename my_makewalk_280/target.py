#!/usr/bin/python
# -*- coding: utf-8 -*-

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  Authors:             Thomas Larsson
#  Script copyright (C) Thomas Larsson 2014
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; eimcp.r version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See mcp.
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

import bpy
from bpy.props import *
import math
import os

import utils
import t_pose
from utils import *
from armature import CArmature
if bpy.app.version < (2,80,0):
    from buttons27 import SaveTargetExport
else:
    from buttons28 import SaveTargetExport

#
#   Global variables
#

_target = None
_targetInfo = {}
#_targetArmatures = { "Automatic" : ([],[],[]) }
_targetArmatures = {}
_trgArmature = None
_trgArmatureEnums =[("Automatic", "Automatic", "Automatic")]
_ikBones = []
_bendTwist = {}

def getTargetInfo(rigname):
    global _targetInfo
    return _targetInfo[rigname]

def loadTargets():
    global _targetInfo
    _targetInfo = {}

def isTargetInited(scn):
    return ( _targetArmatures != {} )

def ensureTargetInited(scn):
    if not isTargetInited(scn):
        initTargets(scn)

#
#   getTargetArmature(rig, context):
#

def getTargetArmature(rig, context):
    global _target, _targetArmatures, _targetInfo, _trgArmature, _ikBones, _bendTwist

    scn = context.scene
    setCategory("Identify Target Rig")
    ensureTargetInited(scn)
    putInRestPose(rig, True)
    bones = rig.data.bones.keys()

    if scn.McpAutoTargetRig:
        name = guessTargetArmatureFromList(rig, bones, scn)
    else:
        try:
            name = scn.McpTargetRig
        except:
            raise MocapError("Initialize Target Panel first")

    if name == "Automatic":
        setCategory("Automatic Target Rig")
        amt = _trgArmature = CArmature()
        amt.findArmature(rig, ignoreHiddenLayers=scn.McpIgnoreHiddenLayers)
        _targetArmatures["Automatic"] = amt
        scn.McpTargetRig = "Automatic"
        amt.display("Target")

        boneAssoc = []
        for pb in rig.pose.bones:
            if pb.McpBone:
                boneAssoc.append( (pb.name, pb.McpBone) )

        _ikBones = []
        rig.McpTPoseFile = ""
        _targetInfo[name] = (boneAssoc, _ikBones, rig.McpTPoseFile, _bendTwist)
        clearCategory()
        return boneAssoc

    else:
        setCategory("Manual Target Rig")
        scn.McpTargetRig = name
        _target = name
        (boneAssoc, _ikBones, rig.McpTPoseFile, _bendTwist) = _targetInfo[name]
        if not testTargetRig(name, rig, boneAssoc):
            print("WARNING:\nTarget armature %s does not match armature %s.\nBones:" % (rig.name, name))
            for pair in boneAssoc:
                print("  %s : %s" % pair)
        print("Target armature %s" % name)

        for pb in rig.pose.bones:
            pb.McpBone = pb.McpParent = ""
        for bname,mhx in boneAssoc:
            try:
                rig.pose.bones[bname].McpBone = mhx
            except KeyError:
                print("  ", bname)
                pass

        clearCategory()
        return boneAssoc



def guessTargetArmatureFromList(rig, bones, scn):
    global _target, _targetArmatures, _targetInfo
    ensureTargetInited(scn)
    print("Guessing target")

    if isMhxRig(rig):
        return "MHX"
    elif isMhOfficialRig(rig):
        return "MH-Official"
    elif isRigify(rig):
        return "Rigify"
    elif isRigify2(rig):
        return "Rigify2"
    elif isMhx7Rig(rig):
        return "MH-alpha7"
    elif isGenesis(rig):
        return "Genesis"
    elif isGenesis3(rig):
        return "Genesis3"
    elif False:
        for name in _targetInfo.keys():
            if name not in ["MHX", "MH-Official", "Rigify", "Rigify2", "MH-alpha7", "Genesis", "Genesis3"]:
                (boneAssoc, _ikBones, _tpose, _bendTwist) = _targetInfo[name]
                if testTargetRig(name, rig, boneAssoc):
                    return name
    else:
        return "Automatic"


def testTargetRig(name, rig, rigBones):
    from armature import validBone
    print("Testing %s" % name)
    for (bname, mhxname) in rigBones:
        try:
            pb = rig.pose.bones[bname]
        except KeyError:
            pb = None
        if pb is None or not validBone(pb):
            print("  Did not find bone %s (%s)" % (bname, mhxname))
            return False
    return True

#
#   findTargetKeys(mhx, list):
#

def findTargetKeys(mhx, list):
    bones = []
    for (bone, mhx1) in list:
        if mhx1 == mhx:
            bones.append(bone)
    return bones

###############################################################################
#
#    Target armatures
#
###############################################################################

#    (mhx bone, text)

TargetBoneNames = [
    ('hips',         'Root bone'),
    ('spine',        'Lower spine'),
    ('spine-1',      'Middle spine'),
    ('chest',        'Upper spine'),
    ('neck',         'Neck'),
    ('head',         'Head'),
    None,
    ('shoulder.L',   'L shoulder'),
    ('upper_arm.L',  'L upper arm'),
    ('forearm.L',    'L forearm'),
    ('hand.L',       'L hand'),
    None,
    ('shoulder.R',   'R shoulder'),
    ('upper_arm.R',  'R upper arm'),
    ('forearm.R',    'R forearm'),
    ('hand.R',       'R hand'),
    None,
    ('hip.L',        'L hip'),
    ('thigh.L',      'L thigh'),
    ('shin.L',       'L shin'),
    ('foot.L',       'L foot'),
    ('toe.L',        'L toes'),
    None,
    ('hip.R',        'R hip'),
    ('thigh.R',      'R thigh'),
    ('shin.R',       'R shin'),
    ('foot.R',       'R foot'),
    ('toe.R',        'R toes'),
]

###############################################################################
#
#    Target initialization
#
###############################################################################


def initTargets(scn):
    global _targetArmatures, _targetInfo, _trgArmatureEnums
    _targetInfo = { "Automatic" : ([], [], "", {}) }
    _targetArmatures = { "Automatic" : CArmature() }
    path = os.path.join(os.path.dirname(__file__), "target_rigs")
    for fname in os.listdir(path):
        file = os.path.join(path, fname)
        (name, ext) = os.path.splitext(fname)
        if ext == ".trg" and os.path.isfile(file):
            (name, stuff) = readTrgArmature(file, name)
            _targetInfo[name] = stuff

    _trgArmatureEnums =[("Automatic", "Automatic", "Automatic")]
    keys = list(_targetInfo.keys())
    keys.sort()
    for key in keys:
        _trgArmatureEnums.append((key,key,key))

    bpy.types.Scene.McpTargetRig = EnumProperty(
        items = _trgArmatureEnums,
        name = "Target rig",
        default = 'Automatic')
    print("Defined McpTargetRig")
    return


def readTrgArmature(file, name):
    print("Read target file", file)
    fp = open(file, "r")
    status = 0
    bones = []
    tpose = ""
    ikbones = []
    bendtwist = []
    for line in fp:
        words = line.split()
        if len(words) > 0:
            key = words[0].lower()
            if key[0] == "#":
                continue
            elif key == "name:":
                name = words[1]
            elif key == "bones:":
                status = 1
            elif key == "ikbones:":
                status = 2
            elif key == "bendtwist:":
                status = 3
            elif key == "t-pose:":
                status = 0
                tpose = os.path.join("target_rigs", words[1])
            elif len(words) != 2:
                print("Ignored illegal line", line)
            elif status == 1:
                bones.append( (words[0], nameOrNone(words[1])) )
            elif status == 2:
                ikbones.append( (words[0], nameOrNone(words[1])) )
            elif status == 3:
                bendtwist.append( (words[0], nameOrNone(words[1])) )

    fp.close()
    return (name, (bones,ikbones,tpose,bendtwist))


class MCP_OT_InitTargets(bpy.types.Operator):
    bl_idname = "mcp.init_targets"
    bl_label = "Init Target Panel"
    bl_description = "(Re)load all .trg files in the target_rigs directory."
    bl_options = {'UNDO'}

    def execute(self, context):
        try:
            initTargets(context.scene)
        except MocapError:
            bpy.ops.mcp.error('INVOKE_DEFAULT')
        return{'FINISHED'}


class MCP_OT_GetTargetRig(bpy.types.Operator):
    bl_idname = "mcp.get_target_rig"
    bl_label = "Identify Target Rig"
    bl_description = "Identify the target rig type of the active armature."
    bl_options = {'UNDO'}

    def execute(self, context):
        from retarget import changeTargetData, restoreTargetData
        rig = context.object
        scn = context.scene
        data = changeTargetData(rig, scn)
        try:
            getTargetArmature(rig, context)
        except MocapError:
            bpy.ops.mcp.error('INVOKE_DEFAULT')
        finally:
            restoreTargetData(rig, data)
        return{'FINISHED'}


def saveTargetFile(filepath, context):
    from t_pose import saveTPose

    rig = context.object
    scn = context.scene
    fname,ext = os.path.splitext(filepath)
    filepath = fname + ".trg"
    fp = open(filepath, "w")
    name = os.path.basename(fname).capitalize().replace(" ","_")
    fp.write("Name:\t%s\n\nBones:\n" % name)
    for pb in rig.pose.bones:
        if pb.McpBone:
            fp.write("\t%s\t%s\n" % (pb.name, pb.McpBone))
    fp.write("\nIkBones:\n\n")
    if scn.McpSaveTargetTPose:
        fp.write("T-pose:\t%s-tpose.json\n" % name)
    fp.close()
    print("Saved %s" % filepath)

    if scn.McpSaveTargetTPose:
        tposePath = fname + "-tpose.json"
        saveTPose(context, tposePath)


class MCP_OT_SaveTargetFile(bpy.types.Operator, SaveTargetExport):
    bl_idname = "mcp.save_target_file"
    bl_label = "Save Target File"
    bl_description = "Save a .trg file for this character"
    bl_options = {'UNDO'}

    def execute(self, context):
        try:
            saveTargetFile(self.properties.filepath, context)
        except MocapError:
            bpy.ops.mcp.error('INVOKE_DEFAULT')
        return{'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

#----------------------------------------------------------
#   Initialize
#----------------------------------------------------------

classes = [
    MCP_OT_InitTargets,
    MCP_OT_GetTargetRig,
    MCP_OT_SaveTargetFile,
]

def initialize():
    for cls in classes:
        bpy.utils.register_class(cls)


def uninitialize():
    for cls in classes:
        bpy.utils.unregister_class(cls)
