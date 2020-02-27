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
from bpy_extras.io_utils import ImportHelper, ExportHelper


class SaveTargetExport(ExportHelper):
    filename_ext = ".trg"
    filter_glob : StringProperty(default="*.trg", options={'HIDDEN'})
    filepath : StringProperty(name="File Path", description="Filepath to target file", maxlen=1024, default="")


class LoadBVH(ImportHelper):
    filename_ext = ".bvh"
    filter_glob : StringProperty(default="*.bvh", options={'HIDDEN'})
    filepath : StringProperty(name="File Path", description="Filepath used for importing the BVH file", maxlen=1024, default="")


class LoadJson(ImportHelper):
    filename_ext = ".json"
    filter_glob : StringProperty(default="*.json", options={'HIDDEN'})
    filepath : StringProperty(name="File Path", description="Filepath to tpose file", maxlen=1024, default="")


class ProblemsString:
    problems = ""


class TypeString:
    type : StringProperty()


class AnswerString:
    answer : StringProperty()


class PropString:
    prop : StringProperty()


class LocRotDel:
    loc : BoolProperty("Loc", default=False)
    rot : BoolProperty("Rot", default=False)
    delete : BoolProperty("Del", default=False)


class LeftLast:
    left : BoolProperty("Loc", default=False)
    last : BoolProperty("Rot", default=False)

