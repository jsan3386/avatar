import bpy


bvh_file = "/mnt/data/jsanchez/BlenderAssets/mocaps/mixamo/aerial_evade.bvh"

scale = 1.0
startFrame = 1
endFrame = 70
frameno = 1

fileName = bvh_file

def read_bvh (fileName): 

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
                frameno += 1
            frame += 1

    fp.close()



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


read_bvh(bvh_file)