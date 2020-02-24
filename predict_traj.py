import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from keras.models import load_model
from tqdm import tqdm
from utils_forrest.traj_track import get_coord, traj_track, filter_coord, draw_traj_points
import time

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
                        # stack all ball coords
                        coords = get_coord(images[k], batch_boxes[k], labels, obj_thresh, i) # i: frame no
                        balls_coords = balls_coords + coords

                        # show the video after above operations
                        if show_window: cv2.imshow('video with bboxes', images[k])

                    images = []
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        # track trajectory
        traj_track(balls_coords, balls_frms, video_out, video_fps, frame_w, frame_h)

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
    else:
        print("Trajectory tracking can only be done on videos. ")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()

    start = time.time()
    _main_(args)
    print('It took', time.time()-start, 'seconds.')
