import numpy as np
import os
import cv2
from .colors import get_color
import json
import textwrap
import pytesseract

def draw_ocr_boxes(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)

        i = 0
        if label >= 0 and labels[label] == "speed":
            label_str = textwrap.wrap(label_str, 10)
            for line in label_str:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 2e-4 * image.shape[0], 5)
                width, height = text_size[0][0], text_size[0][1]
                region = np.array([[box.xmin - 3,        box.ymin],
                                   [box.xmin - 3,        box.ymin - height - 26],
                                   [box.xmin + width + 13, box.ymin - height - 26],
                                   [box.xmin + width + 13, box.ymin]], dtype = 'int32')

                cv2.rectangle(img = image, pt1 = (box.xmin,box.ymin), pt2 = (box.xmax,box.ymax), color = get_color(label), thickness = 2)
                # cv2.fillPoly(img = image, pts = [region], color = get_color(label))
                cv2.putText(img = image,
                            text = line,
                            org = (box.xmin + 13, box.ymin - i * 13),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 8e-4 * image.shape[0],
                            color = get_color(label), #(216, 255,  1),
                            thickness = 2, 
                            lineType = cv2.LINE_AA)
                i += 1

    return image

def draw_ocr_text(detect_text, ocr_labels, image):
    '''
    Introduction:
    # Draw ocr text on image. 
    # Position to draw: top left on screen.

    Input accept:
    # detect_text: a dict of detected strings
    # ocr_labels: corresponding labels of text # not needed
    # image: a single image

    Output:
    # an annotated image
    '''
    # list to string
    text = json.dumps(detect_text)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2e-4 * image.shape[0], 5)
    width, height = text_size[0][0], text_size[0][1]
    cv2.putText(img = image,
                text = text,
                org = (5, 20),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 8e-4 * image.shape[0],
                color = (216, 255,  1),
                thickness = 2,
                lineType = cv2.LINE_AA)

    return image

def detect_text(image, boxes, labels, ocr_labels, obj_thresh, quiet=True):
    detect_text = []

    for box in boxes:
        label_str = ''
        label_id = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label_id = i
            if not quiet: print(label_str)

        if label_id >= 0 and labels[label_id] in ocr_labels:
            # do detection
            cropped = image[box.ymin:box.ymax, box.xmin:box.xmax]
            detect_item = pytesseract.image_to_string(cropped) # cropped image
            if detect_item == "":
                detect_text.append("N/A")
            else:
                detect_text.append(detect_item)
            print("DETECTfrom{}to".format(detect_item)) # TEST
            print("LABEL:", labels[label_id])

    # draw box with detected text
    image = draw_ocr_text(detect_text, ocr_labels, image)

    return image, detect_text
