'''
Optimization idea:
# stack frames & move functions to utils_forrest.ocr_detect.py
# open multiple thread to process ocr operations
'''
import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import pandas as pd
import pytesseract
import textwrap
import time
from utils_forrest.ocr_detect import draw_ocr_boxes, draw_ocr_text, detect_text

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
        video_out = output_path + '/ocr_'  + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)
        csv_out = output_path + '/ocrcsv_' + input_path.split('/')[-1][:-3] + 'csv'

        video_fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               video_fps, 
                               (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        quiet       = True
        detect_text = {}
        labels      = config['model']['labels']
        # 主隊, 客隊, 局數, 好壞球, 投球數, 總球數, 球速
        ocr_labels = ['score1', 'score2', 'session', 'strikeball', 'total', 'speed']
        ocr_result = pd.DataFrame(columns = ocr_labels)
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        # detect text
                        # return a list of ocr results
                        # depracated call: images[i], ocr_rec = detect_text(images[i], batch_boxes[i], ocr_labels, config['model']['labels'], obj_thresh)
                        for box in batch_boxes[i]:
                            label_str = ''
                            label_id = -1

                            # filter label(s) that is recognized for this box
                            for j in range(len(labels)):
                                if box.classes[j] > obj_thresh:
                                    if label_str != '': label_str += ', '
                                    label_str += (labels[j] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                                    label_id = j
                                if not quiet: print(label_str)

                            # detect text from current box
                            if label_id >= 0 and labels[label_id] in ocr_labels:
                                # do detection
                                cropped = images[i][box.ymin:box.ymax, box.xmin:box.xmax]
                                detect_item = pytesseract.image_to_string(cropped) # cropped image
                                if detect_item == "":
                                    detect_text.update({labels[label_id]: "N/A"})
                                else:
                                    detect_text.update({labels[label_id]: detect_item})

                        # draw box with detected text
                        draw_ocr_text(detect_text, ocr_labels, images[i])
                        draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)

                        # write records into dataframe
                        ocr_result = ocr_result.append(detect_text, ignore_index = True)
                        detect_text = {}

                        # show the video with detection bounding boxes          
                        if show_window: cv2.imshow('video with bboxes', images[i])  

                        # write result to the output video
                        video_writer.write(images[i]) 

                    images = []

                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        print(ocr_result) # TEST output
        ocr_result.to_csv(csv_out)

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()       
    else: # do detection on an image or a set of images
        image_paths = []

        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)
            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # detect text
            detect_text(image, boxes, config['model']['labels'], obj_thresh)

            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
     
            # write the image with bounding boxes to file
            cv2.imwrite(output_path + '/pred_' + image_path.split('/')[-1], np.uint8(image))         

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    start = time.time()
    _main_(args)
    print('\n>> Total consuming time: ', time.time()-start, 'seconds.')

