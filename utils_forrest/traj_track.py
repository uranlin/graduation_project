import os
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from .colors import get_color

def get_coord(image, batch_boxes, labels, obj_thresh, frame_no, quiet=True):
    '''
    Introduction:
    # Find ball coordinates (one coordinate as expected) from an input image & return them.

    Input accept:
    # image: a single image
    # batch_boxes: multiple boxes of one image
    # labels: a list of all 10 labels
    # obj_thresh: lower limit accepted as label
    # frame_no: frame number
    # quiet: to print accepted label name

    Output:
    # coords: ball coords(tuple type (x_axis, y_axis, frame_no)) in a list [(x_axis, y_axis, frame_no), ...]
    '''
    coords = [] # ideally one coord per image

    for box in batch_boxes:
        label_str = ''
        label_no = -1

        # filter valid labels, multiple labels for a image is possible
        for j in range(len(labels)):
            if box.classes[j] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[j] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label_no = j
            if not quiet: print(label_str)

        # stack all ball coords
        if label_no >= 0 and labels[label_no] == "ball":
            coords.append(( int((box.xmin + box.xmax) / 2), int((box.ymin + box.ymax) / 2), frame_no )) # i: frameNo

    return coords


def traj_track(balls_coords, balls_frms, video_out, video_fps, frame_w, frame_h, quiet=True):
    '''
    Introduction: 
    # Use all frames & all ball coordinates to generate trajectory an video & image.

    Input accept:
    # balls_coords: all ball coordinates(tuple type) in list type
    # balls_frms: all frames of video in the chache of memory
    # video_out: output path (string type)
    # video_fps: frames per second of video (int type)
    # frame_w: width of frames (int type)
    # frame_h: height of frames (int type)
    # quiet: to print ball coords & some info

    Output:
    # save the trajectory video & image to disk
    '''

    # filter ball coord on trajectory
    balls_coords, cluster = filter_coord(balls_coords, frame_w)
    if not quiet: print("All ball coords: \n{}".format(balls_coords))

    # generate trajectory video
    video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'MPEG'),
                                   video_fps,
                                   (frame_w, frame_h))
    for i in range(len(balls_frms)):
        coord_stack = []
        for coord in balls_coords:
            if coord[2] <= i: coord_stack.append(coord)
        if len(coord_stack) > 0: draw_traj_points(coord_stack, balls_frms[i], cluster)
        video_writer.write(balls_frms[i])

    # generate trajectory image
    if not quiet: print("firstBallNo:{}, \nlastBallNo:{}, \nlengthOfFrm:{}".format(balls_coords[0][2], balls_coords[-1][2], len(balls_frms)))
    first_ball_no = balls_coords[0][2]
    first_ball_frm = balls_frms[first_ball_no]
    output_image = draw_traj_points(balls_coords, first_ball_frm, cluster)
    cv2.imwrite(video_out[:-3] + "jpg", output_image)

    video_writer.release()


def filter_coord(coords, frame_w, quiet=True):
    '''
    Introduction:
    # Use DBSCAN to cluster & detect balls on a pitch trajectory. Data preprocessing is done.
    # Epsilon value best performance around: 0.3 - 0.4

    Input accept:
    # coords: multiple coordinates in a list, each one in tuple type including x_coord, y_coord, frame_no

    Output:
    # multiple coordinates in a list, each one in tuple type including x_coord, y_coord, frame_no, cluster(-1 := discarded)

    Testing data:
    # coords = [
    # (422, 455, 25), (413, 452, 26), (410, 445, 27), (410, 444, 28), (407, 448, 29),
    # (402, 455, 30), (399, 460, 31), (409, 455, 32), (426, 442, 33), (676, 227, 44),
    # (746, 333, 80), (747, 329, 81), (750, 318, 82), (754, 312, 83), (754, 307, 84),
    # (758, 301, 85), (760, 293, 86), (766, 288, 87), (772, 277, 88), (779, 271, 89),
    # (774, 259, 90), (779, 263, 90), (780, 255, 91), (904, 203, 99), (903, 204, 100),
    # (915, 206, 106), (897, 209, 107), (892, 215, 107), (896, 208, 108), (891, 214, 108),
    # (898, 207, 109), (895, 213, 109), (897, 208, 110), (895, 214, 110), (901, 206, 112)
    # ]
    '''
    min_no = 4 # min of frames to be identified as trajectory
    bound_l, bound_r = 45, 65 # %
    bound_l, bound_r = bound_l/100 * frame_w, bound_r/100 * frame_w
    valid_coords = []

    # Clustering coords data by DBSCAN
    df = pd.DataFrame(coords, columns=["x_coords", "y_coords", "frame_no"])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_normalized = normalize(df_scaled) # numpy Array of an approximately Gaussian distribution
    df_normalized = pd.DataFrame(df_normalized)
    df_normalized.iloc[:, 2] = df_normalized.iloc[:, 2] * 12
    db_default = DBSCAN(eps = 0.35, min_samples = 2).fit(df_normalized)
    labels = db_default.labels_

    # Count number of points per cluster
    labels_cnt = pd.DataFrame(labels, columns=['cluster']).groupby('cluster').size()
    
    # Paste cluster label back to coordinate
    for i in range(len(coords)):
        coords[i] = coords[i] + (labels[i], )

    # Find trajectory cluster
    for i in range(1, len(coords)):
        if coords[i][0] >= bound_l and \
           coords[i][0] <= bound_r and \
           labels_cnt.loc[coords[i][3]] >= min_no and \
           coords[i][0] > coords[i-1][1] and \
           (coords[i][3] != -1):
            valid_coords.append(coords[i])
    valid_coords.sort(key=lambda e: e[2]) # take third
    cluster = valid_coords[0][3]
    if not quiet:
        print(valid_coords)
        print("bound_l: {}\nbound_r:{} \ncluster:{}".format(bound_l, bound_r, cluster))

    return coords, cluster


def draw_traj_points(coords, image, cluster):
    '''
    Introduction:
    # Drawing all input coordinates on an input image
    # Number of coordinates unlimited

    Input accept:
    # coords: single & multiple coordinates, each coordinate in tuple type
    # image: single image

    Ouput:
    # an annotated image
    '''
    # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2e-4 * image.shape[0], 5)
    # width, height = text_size[0][0], text_size[0][1]
    # region = np.array([[box.xmin - 3,         box.ymax],
    #                   [box.xmin - 3,          box.ymax + height + 26],
    #                   [box.xmin + width + 13, box.ymax + height + 26],
    #                   [box.xmin + width + 13, box.ymax]], dtype = 'int32')
    if(isinstance(coords, (tuple)) or len(coords) == 1):
        if(len(coords) == 1):
            coord = coords[0]
        else:
            coord = coords

        if coord[3] == cluster:
            radius = 4
        else:
            radius = 2

        cv2.circle(img = image, 
                   center = coord[:2], 
                   radius = radius, 
                   color = get_color(coord[3]), # (0, 0, 255), 
                   thickness = -1)
        cv2.putText(img = image,
                    text = str(coord[3]),
                    org = (coord[0], coord[1]+15),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 8e-4 * image.shape[0],
                    color = get_color(coord[3]), # (216, 255,  1),
                    thickness = 1,
                    lineType = cv2.LINE_AA)
        cv2.putText(img = image,
                    text = "FrameNo. {}".format(coord[2]),
                    org = (5, 20),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 8e-4 * image.shape[0],
                    color = get_color(coord[3]), # (216, 255,  1),
                    thickness = 1,
                    lineType = cv2.LINE_AA)
    else:
        cv2.putText(img = image,
                    text = "FrameNo. {} 0.3500".format(coords[-1][2]),
                    org = (5, 20),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 8e-4 * image.shape[0],
                    color = get_color(coords[-1][3]), # (216, 255,  1),
                    thickness = 1,
                    lineType = cv2.LINE_AA)
        for coord in coords:
            if coord[3] == cluster:
                radius = 4
            else:
                radius = 2
            cv2.circle(img = image, 
                       center = coord[:2], 
                       radius = radius, 
                       color = get_color(coord[3]), # (0, 0, 255), 
                       thickness = -1)
            cv2.putText(img = image,
                        text = str(coord[3]),
                        org = (coord[0], coord[1]+15),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 8e-4 * image.shape[0],
                        color = get_color(coord[3]), # (216, 255,  1),
                        thickness = 1,
                        lineType = cv2.LINE_AA)

    return image
