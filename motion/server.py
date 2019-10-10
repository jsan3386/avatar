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
import cv2
import tensorflow as tf
from net_models.cpm_pose import cpm_body_slim

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5667")

# Input size
input_size = 368
hmap_size = 46
joints = 14
cmap_radius = 21
#model_path = "%s/motion/net_models/cpm_pose" % avt_path
model_path = "/home/jsanchez/Software/gitprojects/avatar/motion/net_models/cpm_pose/cpm_body.pkl"

# Set color for each finger
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]


limbs = [[0, 1],
         [2, 3],
         [3, 4],
         [5, 6],
         [6, 7],
         [8, 9],
         [9, 10],
         [11, 12],
         [12, 13]]

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

def visualize_result(test_img, stage_heatmap_np, hmap_size, joints):

    demo_stage_heatmaps = []

    last_heatmap = stage_heatmap_np[-1][0, :, :, 0:joints].reshape((hmap_size, hmap_size, joints))
    last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))

    joint_coord_set = np.zeros((joints, 2))

    for joint_num in range(joints):
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]), (test_img.shape[0], test_img.shape[1]))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
        else:
            joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)

    # Plot limb colors
    for limb_num in range(len(limbs)):

        x1 = joint_coord_set[limbs[limb_num][0], 0]
        y1 = joint_coord_set[limbs[limb_num][0], 1]
        x2 = joint_coord_set[limbs[limb_num][1], 0]
        y2 = joint_coord_set[limbs[limb_num][1], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 200 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 6),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])
            cv2.fillConvexPoly(test_img, polygon, color=limb_color)

    return test_img

def read_image(oriImg, boxsize):

    if oriImg is None:
        print('oriImg is None')
        return None

    scale = boxsize / (oriImg.shape[0] * 1.0)
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((boxsize, boxsize, 3)) * 128

    img_h = imageToTest.shape[0]
    img_w = imageToTest.shape[1]
    if img_w < boxsize:
        offset = img_w % 2
        # make the origin image be the center
        output_img[:, int(boxsize / 2 - math.floor(img_w / 2)):int(
            boxsize / 2 + math.floor(img_w / 2) + offset), :] = imageToTest
    else:
        # crop the center of the origin image
        output_img = imageToTest[:,
                     int(img_w / 2 - boxsize / 2):int(img_w / 2 + boxsize / 2), :]
    return output_img


if __name__ == "__main__":

    # option camera stream
    if sys.argv[1] == '-camera':

        # Init CPM network
        # if FLAGS.color_channel == 'RGB':
        #     input_data = tf.placeholder(dtype=tf.float32, shape=[None, input_size, input_size, 3], name='input_image')
        # else:
        #     input_data = tf.placeholder(dtype=tf.float32, shape=[None, input_size, input_size, 1], name='input_image')
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, input_size, input_size, 3], name='input_image')

        center_map = tf.placeholder(dtype=tf.float32, shape=[None, input_size, input_size, 1], name='center_map')

        model = cpm_body_slim.CPM_Model(stages, joints + 1)
        model.build_model(input_data, center_map, 1)

        saver = tf.train.Saver()

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        model.load_weights_from_file(model_path, sess, False)

        test_center_map = cpm_utils.gaussian_img(input_size,
                                                 input_size,
                                                 input_size / 2,
                                                 input_size / 2,
                                                 cmap_radius)
        test_center_map = np.reshape(test_center_map, [1, input_size,
                                                       input_size, 1])

        # Check weights
        for variable in tf.trainable_variables():
            with tf.variable_scope('', reuse=True):
                var = tf.get_variable(variable.name.split(':0')[0])


        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            test_img = read_image(frame, input_size)
            test_img_resize = cv2.resize(test_img, (input_size, input_size))

#            if FLAGS.color_channel == 'GRAY':
#                test_img_resize = mgray(test_img_resize, test_img)

            test_img_input = test_img_resize / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)

            # Inference
            predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap, model.stage_heatmap,],
                                                                 feed_dict={'input_image:0': test_img_input,
                                                                            'center_map:0': test_center_map})

            # Show visualized image
            demo_img = visualize_result(test_img, stage_heatmap_np, hmap_size, joints)
            cv2.imshow('demo_img', demo_img.astype(np.uint8))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break       

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    # option read frame files
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




