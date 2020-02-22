import os
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .colors import get_color

def filter_coord(coords):
'''
Introduction:
# Using DBSCAN to filter balls not on trajectory. Data preprocessing done.
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

    x_coord = []
    y_coord = []
    frame_no = []
    
    # Converting data to DataFrame
    for coord in coords:
        x_coord.append(coord[0])
        y_coord.append(coord[1])
        frame_no.append(coord[2])
    data = {'x_coord': x_coord,
            'y_coord': y_coord,
            'frame_no': frame_no}
    df = pd.DataFrame(data)
    
    # Scaling the data to bring all the attributes to a comparable level
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Normalizing the data so that
    # the data approximately follows a Gaussian distribution
    df_normalized = normalize(df_scaled)
    
    # Converting the numpy array into a pandas DataFrame
    df_normalized = pd.DataFrame(df_normalized)
    
    # Reducing the dimensionality of the data to make it visualizable
    # pca = PCA(n_components = 2)
    # df_principal = pca.fit_transform(df_normalized)
    # df_principal = pd.DataFrame(df_principal)
    # df_principal.columns = ['P1', 'P2']
    # print(df_principal.head())
    
    # Numpy array of all the cluster labels assigned to each data point
    db_default = DBSCAN(eps = 0.35, min_samples = 2).fit(df_normalized)
    labels = db_default.labels_
    
    # Paste label back to coordinate
    for i in range(len(coords)):
        coords[i] = coords[i] + (labels[i], )
    
    return coords


def draw_traj_points(coords, image):
'''
Introduction:
# Drawing all input coordinates on an input image

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
        cv2.circle(img = image, 
                   center = coord[:2], 
                   radius = 3, 
                   color = get_color(coord[3]), # (0, 0, 255), 
                   thickness = -1)
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
            cv2.circle(img = image, 
                       center = coord[:2], 
                       radius = 3, 
                       color = get_color(coord[3]), # (0, 0, 255), 
                       thickness = -1)

    return image
