import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
from utils_forrest.traj_track import filter_coord, draw_traj_points

def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes 
    ###############################
    if input_path[-4:] in ['.avi', '.mp4']: # do detection on a video  
        video_out = output_path + '/traj_'  + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        video_fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        # the main loop
        batch_size   = 1
        images       = []
        start_point  = 0 #%
        show_window  = False
        balls_coords = []
        balls_frms   = []
        quiet        = True
        labels       = config['model']['labels']
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            # stack all frames
            balls_frms.append(image)

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for k in range(len(images)):
                        # trajectory track
                        # discarded: traj_track(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)
                        for box in batch_boxes[k]:
                            label_str = ''
                            label = -1

                            for j in range(len(labels)):
                                if box.classes[j] > obj_thresh:
                                    if label_str != '': label_str += ', '
                                    label_str += (labels[j] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                                    label = j
                                if not quiet: print(label_str)

                            if label >= 0 and labels[label] == "ball":
                                # stack all ball coords
                                balls_coords.append(( int((box.xmin + box.xmax) / 2), int((box.ymin + box.ymax) / 2), i )) # i: frameNo
                        # Further: draw bounding boxes on the image using labels
                        # show the video with detection bounding boxes          
                        if show_window: cv2.imshow('video with bboxes', images[k])

                    images = []
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        # filter wrong ball coord
        balls_coords = filter_coord(balls_coords)
        print("FROM{}TO".format(balls_coords)) # TEST

        # generate trajectory video
        video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                       video_fps,
                                       (frame_w, frame_h))
        for i in range(len(balls_frms)):
            coord_stack = []
            for coord in balls_coords:
                if coord[2] <= i: coord_stack.append(coord)
            if len(coord_stack) > 0: draw_traj_points(coord_stack, balls_frms[i])
            video_writer.write(balls_frms[i])

        # generate trajectory image
        print("firstBallNo:{}, \nlastBallNo:{}, \nlengthOfFrm:{}".format(balls_coords[0][2], balls_coords[-1][2], len(balls_frms))) # test
        first_ball_no = balls_coords[0][2]
        first_ball_frm = balls_frms[first_ball_no]
        output_image = draw_traj_points(balls_coords, first_ball_frm)
        cv2.imwrite(video_out[:-3] + "jpg", output_image)

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()       
    else:
        print("Trajectory tracking can only be done on videos. ")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
