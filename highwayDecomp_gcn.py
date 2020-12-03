import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import regionprops
from operator import itemgetter
from numpy.linalg import norm
import os
def decomposition_highway1(centroid, highway, coherence,coordinates,input):
    #
    # print(np.asarray(centroids).shape)
    # print(highway)
    # print(np.asarray(coherence).shape)
    c = 0
    decom_highway_coh = [0]*len(highway)

    # Normalizing centroids and input_sl
    input_min = input.min(axis=(0, 1), keepdims=True)
    input_max = input.max(axis=(0, 1), keepdims=True)
    input_norm = (input - input_min)/(input_max - input_min)

    c_min = centroid.min(axis=(0, 1), keepdims=True)
    c_max = centroid.max(axis=(0, 1), keepdims=True)
    c_norm = (centroid - c_min)/(c_max - c_min)


    for cluster in coordinates:
        # print(cluster)
        clusterCenter = c_norm[0][c]
        # print(clusterCenter)
        for point in cluster:
            superPixel = input_norm[0][point]
            distance = norm(clusterCenter-superPixel)
            coh = coherence[c]
#             if distance == 0:
#                 coh = coherence[c]
#             else:
#                 coh = coherence[c]*(1-distance)
            decom_highway_coh[point] = coh
        c+=1


    return decom_highway_coh
#     return decom_highway_coh, decom_highway_label

def decomposition_super1(centroid, highway, coherence,coordinates,input):

    # print(np.asarray(centroids).shape)
    # print(np.asarray(coordinates).shape)
    # print((coherence))
    c = 0
    # decom_super_coh = [[0 for x in range(300)] for y in range(300
    decom_super_coh = []
    for i in range (0, 64):
        new = []
        for j in range (0, 64):
            new.append(0)
        decom_super_coh.append(new)
    print(np.array(decom_super_coh).shape)
    # Normalizing centroids and input_sl
    input_min = input.min(axis=(0, 1), keepdims=True)
    input_max = input.max(axis=(0, 1), keepdims=True)
    input_norm = (input - input_min)/(input_max - input_min)

    c_min = centroid.min(axis=(0, 1), keepdims=True)
    c_max = centroid.max(axis=(0, 1), keepdims=True)
    c_norm = (centroid - c_min)/(c_max - c_min)
    print(np.array(input).shape)
    print(np.array(centroid).shape)

    for cluster in coordinates:
        clusterCenter = c_norm[0][c]
        for point in cluster:
            x,y = point[0],point[1]
            superPixel = input_norm[x,y]
            distance = norm(clusterCenter-superPixel)
            coh = coherence[c]
#             if distance == 0:
#                 coh = coherence[c]
#             else:
#                 coh = coherence[c]*(1-distance)
            decom_super_coh[x][y] = coh
        c+=1

    return decom_super_coh

#     return decom_super_coh, decom_super_label
def decomposition_label(centroid, highway, coherence,coordinates,input):

    # print(np.asarray(centroids).shape)
    # print(np.asarray(coordinates).shape)
    # print((coherence))
    c = 0
    # decom_super_coh = [[0 for x in range(300)] for y in range(300
    decom_super_label = []
    for i in range (0, 64):
        new = []
        for j in range (0, 64):
            new.append(0)
        decom_super_label.append(new)
#     print(np.array(decom_super_coh).shape)
    # Normalizing centroids and input_sl
    input_min = input.min(axis=(0, 1), keepdims=True)
    input_max = input.max(axis=(0, 1), keepdims=True)
    input_norm = (input - input_min)/(input_max - input_min)

    c_min = centroid.min(axis=(0, 1), keepdims=True)
    c_max = centroid.max(axis=(0, 1), keepdims=True)
    c_norm = (centroid - c_min)/(c_max - c_min)
    print(np.array(input).shape)
    print(np.array(centroid).shape)

    for cluster in coordinates:
        clusterCenter = c_norm[0][c]
        for point in cluster:
            x,y = point[0],point[1]
            decom_super_label[x][y] = c
        c+=1

    return decom_super_label
