import numpy as np
import os
import cv2
from .colors import get_color
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

def tmp_draw_ocr_text(text, box, image):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2e-4 * image.shape[0], 5)
    width, height = text_size[0][0], text_size[0][1]
    # region = np.array([[box.xmin - 3,         box.ymax],
    #                   [box.xmin - 3,          box.ymax + height + 26],
    #                   [box.xmin + width + 13, box.ymax + height + 26],
    #                  [box.xmin + width + 13, box.ymax]], dtype = 'int32')

    # cv2.rectangle(img = image, pt1 = (box.xmin,box.ymin), pt2 = (box.xmax,box.ymax), color = get_color(label), thickness = 2)
    # cv2.fillPoly(img = image, pts = [region], color = get_color(label))
    cv2.putText(img = image,
                text = text,
                org = (box.xmin + 13, box.ymax + 13),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 8e-4 * image.shape[0],
                color = (216, 255,  1),
                thickness = 2,
                lineType = cv2.LINE_AA)

    return image

def detect_text(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)

        if label >= 0 and labels[label] == "speed":
            # do detection
            cropped = image[box.ymin:box.ymax, box.xmin:box.xmax]
            detect_text = pytesseract.image_to_string(cropped) # cropped image

            # draw box with detected text
            image = tmp_draw_ocr_text(detect_text[:10], box, image)

    return image
