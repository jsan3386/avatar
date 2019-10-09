# this code will compute 3d positions and send to Blender to move character

# Usage:
#  Opt1. Use camera frames to get 3D skeleton
#  Opt2. Use pre-stored 3D frame files
# 		 server.py -frames <folder_path> <fps>

import zmq
import sys
import os
import numpy as np 
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5667")

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


if __name__ == "__main__":

    # option camera stream
    if sys.argv[1] == '-camera':
        pass
    elif sys.argv[1] == '-frames':
        frames_folder = sys.argv[2]
        fps = int(sys.argv[3])
        #print(frames_folder)
        #fname = "frame_SA%02d_%05d.txt" % (2, f)
        # read frames in folder
        point_files = [f for f in os.listdir(frames_folder) if f.endswith('.txt')]
        point_files.sort()
        
        num_packg = 0

        for f in point_files:
            start = time.time()
            fpname = "%s/%s" % (frames_folder,f)
            pts_skel = np.loadtxt(fpname)
            #print(pts_skel)
            send_array(socket, pts_skel)
            #print("Packages sent: ", num_packg)
            num_packg += 1
            time.sleep(max(1./fps - (time.time() - start), 0))




