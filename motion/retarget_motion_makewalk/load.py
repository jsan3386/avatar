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

import bpy, os, mathutils, math, time
from math import sin, cos
from mathutils import *
from bpy.props import *

import props
import simplify
from utils import *


###################################################################################
#    BVH importer.
#    The importer that comes with Blender had memory leaks which led to instability.
#    It also creates a weird skeleton from CMU data, with hands theat start at the wrist
#    and ends at the elbow.
#

#
#    class CNode:
#

class CNode:
    def __init__(self, words, parent):
        name = words[1]
        for word in words[2:]:
            name += ' '+word

        self.name = name
        self.parent = parent
        self.children = []
        self.head = Vector((0,0,0))
        self.offset = Vector((0,0,0))
        if parent:
            parent.children.append(self)
        self.channels = []
        self.matrix = None
        self.inverse = None
        return

    def __repr__(self):
        return "CNode %s" % (self.name)

    def display(self, pad):
        vec = self.offset
        if vec.length < Epsilon:
            c = '*'
        else:
            c = ' '
        print("%s%s%10s (%8.3f %8.3f %8.3f)" % (c, pad, self.name, vec[0], vec[1], vec[2]))
        for child in self.children:
            child.display(pad+"  ")
        return

    def build(self, amt, orig, parent):
        self.head = orig + self.offset
        if not self.children:
            return self.head

        zero = (self.offset.length < Epsilon)
        eb = amt.edit_bones.new(self.name)
        if parent:
            eb.parent = parent
        eb.head = self.head
        tails = Vector((0,0,0))
        for child in self.children:
            tails += child.build(amt, self.head, eb)
        n = len(self.children)
        eb.tail = tails/n
        #self.matrix = eb.matrix.rotation_part()
        (loc, rot, scale) = eb.matrix.decompose()
        self.matrix = rot.to_matrix()
        self.inverse = self.matrix.copy()
        self.inverse.invert()
        if zero:
            return eb.tail
        else:
            return eb.head

#
#    readBvhFile(context, filepath, scn, scan):
#    Custom importer
#

Location = 1
Rotation = 2
Hierarchy = 1
Motion = 2
Frames = 3

Epsilon = 1e-5

def readBvhFile(context, filepath, scn, scan, original_position,frame_start,origin):
    props.ensureInited(context)
    setCategory("Load Bvh File")
    scale = scn.McpBvhScale
    startFrame = scn.McpStartFrame
    endFrame = scn.McpEndFrame
    frameno = 1
    if scn.McpFlipYAxis:
        flipMatrix = Mult2(Matrix.Rotation(math.pi, 3, 'X'), Matrix.Rotation(math.pi, 3, 'Y'))
    else:
        flipMatrix = Matrix.Rotation(0, 3, 'X')
    if True or scn.McpRot90Anim:
        flipMatrix = Mult2(Matrix.Rotation(math.pi/2, 3, 'X'), flipMatrix)
    if (scn.McpSubsample):
        ssFactor = scn.McpSSFactor
    else:
        ssFactor = 1
    defaultSS = scn.McpDefaultSS

    fileName = os.path.realpath(os.path.expanduser(filepath))
    (shortName, ext) = os.path.splitext(fileName)
    if ext.lower() != ".bvh":
        raise MocapError("Not a bvh file: " + fileName)
    startProgress( "Loading BVH file "+ fileName )

    time1 = time.clock()
    level = 0
    nErrors = 0
    coll = getCollection(context)
    rig = None

    fp = open(fileName, "rU")
    print( "Reading skeleton" )
    lineNo = 0
    for line in fp:
        words= line.split()
        lineNo += 1
        if len(words) == 0:
            continue
        key = words[0].upper()
        if key == 'HIERARCHY':
            status = Hierarchy
            ended = False
        elif key == 'MOTION':
            if level != 0:
                raise MocapError("Tokenizer out of kilter %d" % level)
            if scan:
                return root
            amt = bpy.data.armatures.new("BvhAmt")
            rig = bpy.data.objects.new("BvhRig", amt)
            coll.objects.link(rig)
            setActiveObject(context, rig)
            context.view_layer.update()
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.object.mode_set(mode='EDIT')
            root.build(amt, Vector((0,0,0)), None)
            #root.display('')
            bpy.ops.object.mode_set(mode='OBJECT')
            status = Motion
            print("Reading motion")
        elif status == Hierarchy:
            if key == 'ROOT':
                node = CNode(words, None)
                root = node
                nodes = [root]
            elif key == 'JOINT':
                node = CNode(words, node)
                nodes.append(node)
                ended = False
            elif key == 'OFFSET':
                (x,y,z) = (float(words[1]), float(words[2]), float(words[3]))
                node.offset = scale * Mult2(flipMatrix, Vector((x,y,z)))
            elif key == 'END':
                node = CNode(words, node)
                ended = True
            elif key == 'CHANNELS':
                oldmode = None
                for word in words[2:]:
                    (index, mode, sign) = channelYup(word)
                    if mode != oldmode:
                        indices = []
                        node.channels.append((mode, indices))
                        oldmode = mode
                    indices.append((index, sign))
            elif key == '{':
                level += 1
            elif key == '}':
                if not ended:
                    node = CNode(["End", "Site"], node)
                    node.offset = scale * Mult2(flipMatrix, Vector((0,1,0)))
                    node = node.parent
                    ended = True
                level -= 1
                node = node.parent
            else:
                raise MocapError("Did not expect %s" % words[0])
        elif status == Motion:
            if key == 'FRAMES:':
                nFrames = int(words[1]) + frame_start
            elif key == 'FRAME' and words[1].upper() == 'TIME:':
                frameTime = float(words[2])
                frameFactor = int(1.0/(scn.render.fps*frameTime) + 0.49)
                if defaultSS:
                    ssFactor = frameFactor if frameFactor > 0 else 1
                startFrame *= ssFactor
                endFrame *= ssFactor
                status = Frames
                frame = frame_start ################# ABANS ERA 0 AIXOOOO
                frameno = 1

                #source.findSrcArmature(context, rig)
                bpy.ops.object.mode_set(mode='POSE')
                pbones = rig.pose.bones
                for pb in pbones:
                    pb.rotation_mode = 'QUATERNION'
        elif status == Frames:
            if (frame >= startFrame and frame <= endFrame and frame % ssFactor == 0 and frame < nFrames):
                if frameno ==1:
                    for node in nodes:
                        if node.name == "Hips" or node.name == 'mixamorig:Hips':
                            for (mode,indices) in node.channels:
                                if mode == Location:
                                    vec = Vector((0,0,0))
                                    for (index,sign) in indices:

                                        vec[index] = sign*float(words[index]) ## ABANS ERA WORDS[0]
                                elif mode == Rotation:
                                    mats = []
                                    for (axis,sign) in indices:
                                        angle = sign*float(words[0])*Deg2Rad
                                        mats.append(Matrix.Rotation(angle,3,axis))
                                    flipInv = flipMatrix.inverted()
                                    mat = node.inverse @ flipMatrix @ mats[0] @ mats[1] @ mats[2] @ flipInv @ node.matrix
                start_rotation = mat
                translation_vector = vec
                # start_rotation = original_position
                # translation_vector = Vector((0,0,0))
                print("ADD FRAME LÑASJDÑLKFJASÑDLKFJAÑLDSKJF")
                addFrame(words, frame, nodes, pbones, scale, flipMatrix, translation_vector, 
                                                                            start_rotation, origin)
                showProgress(frameno, frame, nFrames, step=200)
                frameno += 1
            frame += 1

    fp.close()

    if not rig:
        raise MocapError("Bvh file \n%s\n is corrupt: No rig defined" % filepath)
    setInterpolation(rig)
    time2 = time.clock()
    endProgress("Bvh file %s loaded in %.3f s" % (filepath, time2-time1))
    if frameno == 1:
        print("Warning: No frames in range %d -- %d." % (startFrame, endFrame))
    renameBvhRig(rig, filepath)
    rig.McpIsSourceRig = True
    clearCategory()
    return rig

#
#    addFrame(words, frame, nodes, pbones, scale, flipMatrix):
#

def addFrame(words, frame, nodes, pbones, scale, flipMatrix, translation_vector, start_rotation,origin):
    m = 0
    first = True
    flipInv = flipMatrix.inverted()
    extra = 0
    for node in nodes:
        name = node.name
        try:
            pb = pbones[name]
        except:
            pb = None
        if pb:
            for (mode, indices) in node.channels:

                if mode == Location:
                    vec = Vector((0,0,0))
                    for (index, sign) in indices:
                        vec[index] = sign*float(words[index])# words[m]#vec[index] = sign*float(translation_vector[m])
                        m += 1
                    # si no funciona: Descomentar if first i fisrst = false i tabular la resta
                    if first: #Hips

                        pb.location = Mult2(node.inverse, scale * Mult2(flipMatrix, vec))#pb.head)#node.head)

                        if origin:
                            pb.location[0] -= translation_vector[0] * scale
                            #pb.location[1] -= translation_vector[1] * scale    AIXÒ ÉS LA CORRECCIÓ TRANSLACIÓ EN DIRECCIÓ Z, NO ENS INTERESSA QUE HIPS ESTIGUI A L'ORIGEN.
                            pb.location[2] -= translation_vector[2] * scale
                            if extra == 0:
                                pass
                            else: # Calculus of the relative rotation around center (0,0)
                                x = pb.location[0] * math.cos(extra * Deg2Rad) - pb.location[2] * math.sin(extra * Deg2Rad)
                                y = pb.location[0] * math.sin(extra * Deg2Rad) + pb.location[2] * math.cos(extra * Deg2Rad)
                                pb.location[0] = x
                                pb.location[2] = y
                                #print(str(x) + ","+str(y))

                        else:
                            if extra == 0:
                                pass
                            else: # The relative position must be corrected, this function is made for rotations around the origin.

                                x = pb.location[0] * math.cos(extra * Deg2Rad) - pb.location[2] * math.sin(extra * Deg2Rad)
                                y = pb.location[0] * math.sin(extra * Deg2Rad) + pb.location[2] * math.cos(extra * Deg2Rad)

                                pb.location[0] = x
                                pb.location[2] = y

                        #print(pb.location)

                        pb.keyframe_insert('location', frame=frame, group=name)
                        first = False
                        #quat = Quaternion((0.707,0,0,0.707))
                        ##quat = Quaternion((0.707,0,0.707,0))
                        #pb.rotation_mode = "QUATERNION"
                        #pb.rotation_quaternion = quat

                        bpy.context.view_layer.update()

                        #pb.keyframe_insert('rotation_quaternion', frame = frame, group= name)

                    else: #mai entra al else perquè first és quan és HIPS que és l'únic amb location.
                        pb.location = Mult2(node.inverse, scale * Mult2(flipMatrix, vec) - node.head)
                        pb.location[0] += translation_vector[0] * scale
                        pb.location[1] -= translation_vector[1] * scale
                        pb.location[2] += translation_vector[2] * scale
                        pb.keyframe_insert('location', frame=frame, group=name)



                elif mode == Rotation:
                    mats = []
                    for (axis, sign) in indices:
                        angle = sign*float(words[m])*Deg2Rad
                        newangle = angle
                        # Temporal fix: only works with mixamo bvh files
                        if name == "mixamorig:LeftArm" and str(axis) == "X":
                            newangle += 0.2*Deg2Rad
                        if name == "mixamorig:LeftArm" and str(axis) == "Y":
                            newangle += 0.9*Deg2Rad
                        if name == "mixamorig:LeftArm" and str(axis) == "Z":
                            newangle += 50.6*Deg2Rad

                        if name == "mixamorig:LeftForeArm" and str(axis) == "X":
                            newangle -= 40.8*Deg2Rad
                        if name == "mixamorig:LeftForeArm" and str(axis) == "Y":
                            newangle += 0.92*Deg2Rad
                        if name == "mixamorig:LeftForeArm" and str(axis) == "Z":
                            newangle -= 4.58*Deg2Rad

                        if name == "mixamorig:RightArm" and str(axis) == "X":
                            newangle += 1.97*Deg2Rad
                        if name == "mixamorig:RightArm" and str(axis) == "Y":
                            newangle += 3.26*Deg2Rad
                        if name == "mixamorig:RightArm" and str(axis) == "Z":
                            newangle -= 50.4*Deg2Rad

                        if name == "mixamorig:RightForeArm" and str(axis) == "X":
                            newangle -= 40.5*Deg2Rad
                        if name == "mixamorig:RightForeArm" and str(axis) == "Y":
                            newangle -= 2.22*Deg2Rad
                        if name == "mixamorig:RightForeArm" and str(axis) == "Z":
                            newangle += 4.84*Deg2Rad


                        if name == "Hips" and str(axis) == "Y":

                            #body = bpy.data.objects["Standard"]
                            body = bpy.context.active_object
                            body.rotation_mode = "XYZ"
                            body.rotation_euler[2] = extra * Deg2Rad
                        mats.append(Matrix.Rotation(newangle, 3, axis))
                        m += 1
                    mat = Mult3(Mult2(node.inverse, flipMatrix),  Mult3(mats[0], mats[1], mats[2]), Mult2(flipInv, node.matrix))
                    setRotation(pb, mat, frame, name)
                else:
                    pass

    return

#
#    channelYup(word):
#    channelZup(word):
#

def channelYup(word):
    if word == 'Xrotation':
        return ('X', Rotation, +1)
    elif word == 'Yrotation':
        return ('Y', Rotation, +1)
    elif word == 'Zrotation':
        return ('Z', Rotation, +1)
    elif word == 'Xposition':
        return (0, Location, +1)
    elif word == 'Yposition':
        return (1, Location, +1)
    elif word == 'Zposition':
        return (2, Location, +1)

def channelZup(word):
    if word == 'Xrotation':
        return ('X', Rotation, +1)
    elif word == 'Yrotation':
        return ('Z', Rotation, +1)
    elif word == 'Zrotation':
        return ('Y', Rotation, -1)
    elif word == 'Xposition':
        return (0, Location, +1)
    elif word == 'Yposition':
        return (2, Location, +1)
    elif word == 'Zposition':
        return (1, Location, -1)

#
#   end BVH importer
#
###################################################################################


###################################################################################

#
#    class CEditBone():
#

class CEditBone():
    def __init__(self, bone):
        self.name = bone.name
        self.head = bone.head.copy()
        self.tail = bone.tail.copy()
        self.roll = bone.roll
        if bone.parent:
            self.parent = bone.parent.name
        else:
            self.parent = None
        if self.parent:
            self.use_connect = bone.use_connect
        else:
            self.use_connect = False
        #self.matrix = bone.matrix.copy().rotation_part()
        (loc, rot, scale) = bone.matrix.decompose()
        self.matrix = rot.to_matrix()
        self.inverse = self.matrix.copy()
        self.inverse.invert()

    def __repr__(self):
        return ("%s p %s\n  h %s\n  t %s\n" % (self.name, self.parent, self.head, self.tail))

#
#    renameBones(srcRig, context):
#

def renameBones(srcRig, context):
    from source import getSourceBoneName

    srcBones = []
    trgBones = {}

    setActiveObject(context, srcRig)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='EDIT')
    #print("Ren", bpy.context.object, srcRig.mode)
    ebones = srcRig.data.edit_bones
    for bone in ebones:
        srcBones.append( CEditBone(bone) )

    setbones = []
    adata = srcRig.animation_data
    if adata is None:
        action = None
    else:
        action = adata.action
    for srcBone in srcBones:
        srcName = srcBone.name
        trgName = getSourceBoneName(srcName)
        if isinstance(trgName, tuple):
            print("BUG. Target name is tuple:", trgName)
            trgName = trgName[0]
        eb = ebones[srcName]
        if trgName:
            if action:
                grp = action.groups[srcName]
                grp.name = trgName
            eb.name = trgName
            trgBones[trgName] = CEditBone(eb)
            setbones.append((eb, trgName))
        else:
            eb.name = '_' + srcName

    for (eb, name) in setbones:
        eb.name = name
    #createExtraBones(ebones, trgBones)
    bpy.ops.object.mode_set(mode='POSE')

#
#    renameBvhRig(srcRig, filepath):
#

def renameBvhRig(srcRig, filepath):
    base = os.path.basename(filepath)
    (filename, ext) = os.path.splitext(base)
    print("File", filename, len(filename))
    if len(filename) > 12:
        words = filename.split('_')
        if len(words) == 1:
            words = filename.split('-')
        name = 'Y_'
        if len(words) > 1:
            words = words[1:]
        for word in words:
            name += word
    else:
        name = 'Y_' + filename
    print("Name", name)

    srcRig.name = name
    adata = srcRig.animation_data
    if adata:
        adata.action.name = name
    return

#
#    deleteSourceRig(context, rig, prefix):
#

def deleteSourceRig(context, rig, prefix):
    ob = context.object
    setActiveObject(context, rig)
    bpy.ops.object.mode_set(mode='OBJECT')
    setActiveObject(context, ob)
    deleteObject(context, rig)
    if bpy.data.actions:
        for act in bpy.data.actions:
            if act.name[0:2] == prefix:
                act.use_fake_user = False
                if act.users == 0:
                    bpy.data.actions.remove(act)
                    del act
    return


#
#    rescaleRig(scn, trgRig, srcRig):
#

def rescaleRig(scn, trgRig, srcRig):
    if not scn.McpAutoScale:
        return
    if isMhOfficialRig(trgRig):
        upleg1 = trgRig.data.bones["upperleg01.L"]
        upleg2 = trgRig.data.bones["upperleg02.L"]
        trgScale = upleg1.length + upleg2.length
    else:
        upleg = getTrgBone('thigh.L', trgRig)
        trgScale = upleg.length
    srcScale = srcRig.data.bones['thigh.L'].length
    scale = trgScale/srcScale
    print("Rescale %s with factor %f" % (srcRig.name, scale))
    scn.McpBvhScale = scale

    bpy.ops.object.mode_set(mode='EDIT')
    ebones = srcRig.data.edit_bones
    for eb in ebones:
        oldlen = eb.length
        eb.head *= scale
        eb.tail *= scale
    bpy.ops.object.mode_set(mode='POSE')
    adata = srcRig.animation_data
    if adata is None:
        return
    for fcu in adata.action.fcurves:
        words = fcu.data_path.split('.')
        if words[-1] == 'location':
            for kp in fcu.keyframe_points:
                kp.co[1] *= scale
    return


#
#    renameAndRescaleBvh(context, srcRig, trgRig):
#

def renameAndRescaleBvh(context, srcRig, trgRig):
    import source, target
    setCategory("Rename And Rescale")
    try:
        if srcRig["McpRenamed"]:
            raise MocapError("%s already renamed and rescaled." % srcRig.name)
    except:
        pass

    import t_pose
    scn = context.scene
    setActiveObject(context, srcRig)
    #(srcRig, srcBones, action) =  renameBvhRig(rig, filepath)
    target.getTargetArmature(trgRig, context)
    source.findSrcArmature(context, srcRig)
    t_pose.addTPoseAtFrame0(srcRig, scn)
    renameBones(srcRig, context)
    setInterpolation(srcRig)
    rescaleRig(context.scene, trgRig, srcRig)
    srcRig["McpRenamed"] = True
    clearCategory()
