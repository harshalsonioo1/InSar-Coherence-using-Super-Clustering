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

def superpixel(input):
    img = input

#     segments_fz = felzenszwalb(img, scale=20, sigma=0.2, min_size=20, multichannel=True) #for 1179
#     segments_fz = felzenszwalb(img, scale=15, sigma=0.2, min_size=10, multichannel=True) #for 2676
    segments_fz = felzenszwalb(img, scale=10, sigma=0.2, min_size=5, multichannel=True) #for 6367
#     segments_fz = felzenszwalb(img, scale=5, sigma=0.1, min_size=3, multichannel=True) #for 12853
    
    #segments_slic = slic(img, n_segments=100, compactness=1, sigma=1,multichannel=True)
    segments_slic = slic(img, n_segments=6367, compactness=0.9, sigma=0.8,multichannel=True)
    
    # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    # gradient = sobel(rgb2gray(img))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    print(f"Felzenszwalb number of Super Pixels: {len(np.unique(segments_fz))}")
    print(f"SLIC number of Super Pixels: {len(np.unique(segments_slic))}")



    return segments_fz,segments_slic




def centroid(segments,input):
    centroids = []
    amp_slc1 = input[:,:,0]
    amp_slc2 = input[:,:,1]
    real_ifg_phase = input[:,:,2]
    imag_ifg_phase = input[:,:,3]
    coordinates = []
    for i in range(len(np.unique(segments))):
        indices = []
        for row_index,row in enumerate(segments):
            for col_index,item in enumerate(row):
                if(item==i):
                    indices.append([row_index,col_index])
        # print(len(indices))
        # sum+=len(indices)
        coordinates.append(indices)
    delta_list = []
    a1_list,a2_list,real_list,imag_list = [],[],[],[]
    sum=0
    # print(sum,np.asarray(coordinates).shape)
    for i in range(len(coordinates)):

        a1,a2,real,imag = [],[],[],[]
        for loc in coordinates[i]:
            x,y=loc[0],loc[1]
            a1.append(amp_slc1[x,y])
            a2.append(amp_slc2[x,y])
            real.append(real_ifg_phase[x,y])
            imag.append(imag_ifg_phase[x,y])
        sum += len(a1)
        a1_list.append(np.mean(a1))
        a2_list.append(np.mean(a2))
        real_list.append(np.mean(real))
        imag_list.append(np.mean(imag))
    centroids = np.dstack((a1_list,a2_list,real_list,imag_list))
    # for props in regionprops(segments+1):
    #     cx, cy = props.centroid  # centroid coordinates
    #     centroids.append([cx,cy])
        # break
    return centroids,a1_list,a2_list,real_list,imag_list, coordinates



def highway_pixel(input, superPixel, highwayPixel):
    new_input = np.zeros((input[:,:,0].shape),dtype=int)
    # print(new_input.shape)
    for row_index,row in enumerate(superPixel):
        for col_index,item in enumerate(row):
            superPixel_label = item
            highwayPixel_label = highwayPixel[superPixel_label]
            new_input[row_index][col_index] = highwayPixel_label

    return new_input

def centroid_single(segments,input):
    centroids = []
    amp_slc1 = input[:,:,0]
    amp_slc2 = input[:,:,1]
    real_ifg_phase = input[:,:,2]
    imag_ifg_phase = input[:,:,3]
    coordinates = []
    # print(np.asarray(amp_slc1).shape)
    # print("test",amp_slc1)
    for i in range(len(np.unique(segments))):
        indices = []
        for loc,item in enumerate(segments):
            # print(loc,item)
            if(item==i):
                indices.append(loc)
        coordinates.append(indices)

    delta_list = []
    a1_list,a2_list,real_list,imag_list = [],[],[],[]
    sum=0
    # print(coordinates)
    for i in range(len(coordinates)):

        a1,a2,real,imag = [],[],[],[]
        for loc in coordinates[i]:
            # print(loc)
            a1.append(amp_slc1[0,loc])
            a2.append(amp_slc2[0,loc])
            real.append(real_ifg_phase[0,loc])
            imag.append(imag_ifg_phase[0,loc])
        cx = np.mean(coordinates[i])
        centroids.append(cx)
        sum += len(a1)
        a1_list.append(np.mean(a1))
        a2_list.append(np.mean(a2))
        real_list.append(np.mean(real))
        imag_list.append(np.mean(imag))
    centroids = np.dstack((a1_list,a2_list,real_list,imag_list))
    # for props in regionprops(segments+1):
    #     cx, cy = props.centroid  # centroid coordinates
    #     centroids.append([cx,cy])
        # break
    return centroids,a1_list,a2_list,real_list,imag_list, coordinates

def coherence_estimator(highwayToPixel, slc1,slc2):
    coordinates = []
    sum=0
    for i in range(len(np.unique(highwayToPixel))):
        indices = []
        for row_index,row in enumerate(highwayToPixel):
            for col_index,item in enumerate(row):
                if(item==i):
                    indices.append([row_index,col_index])
        # print(len(indices))
        sum+=len(indices)
        coordinates.append(indices)
    delta_list = []
    print(sum,np.asarray(coordinates).shape)
    for i in range(len(coordinates)):

        z1,z2 = [],[]
        for coord in coordinates[i]:
            x,y=coord[0],coord[1]
            # print(">>",slc1[x,y])
            z1.append(slc1[x,y])
            z2.append(slc2[x,y])
            # break

        numerator,deno1,deno2 = 0,0,0
        # print(np.asarray(z1).shape)
#         for i in range(len(z1)):
            # numerator += z1[i]*np.conj(z2[i])
            # deno1 += np.absolute(z1[i])**2
            # deno2 += np.absolute(z2[i])**2
        delta = np.abs(np.sum(z1*np.conj(z2))/np.sqrt(np.sum(np.abs(z1)**2.)*np.sum(np.abs(z2)**2.)))

            # break
        # delta = np.abs(numerator/np.sqrt(deno1*deno2))
        # print("Delta:",delta)
        delta_list.append(delta)
        # break
    # print("Highway Coherence",delta_list)
    print(np.asarray(delta_list).shape)
    return delta_list

# def centroid(segments,input):
#     centroids = []
#     amp_slc1 = input[:,:,0]
#     amp_slc2 = input[:,:,1]
#     real_ifg_phase = input[:,:,2]
#     imag_ifg_phase = input[:,:,3]
#     coordinates = []
#     for i in range(len(np.unique(segments))):
#         indices = []
#         for row_index,row in enumerate(segments):
#             for col_index,item in enumerate(row):
#                 if(item==i):
#                     indices.append([row_index,col_index])
#         # print(len(indices))
#         # sum+=len(indices)
#         coordinates.append(indices)
#     delta_list = []
#     a1_list,a2_list,real_list,imag_list = [],[],[],[]
#     sum=0
#     # print(sum,np.asarray(coordinates).shape)
#     for i in range(len(coordinates)):
#
#         a1,a2,real,imag = [],[],[],[]
#         for loc in coordinates[i]:
#             x,y=loc[0],loc[1]
#             a1.append(amp_slc1[x,y])
#             a2.append(amp_slc2[x,y])
#             real.append(real_ifg_phase[x,y])
#             imag.append(imag_ifg_phase[x,y])
#         sum += len(a1)
#         a1_list.append(np.mean(a1))
#         a2_list.append(np.mean(a2))
#         real_list.append(np.mean(real))
#         imag_list.append(np.mean(imag))
#     for props in regionprops(segments+1):
#         cx, cy = props.centroid  # centroid coordinates
#         centroids.append([cx,cy])
#         # break
#     return centroids,a1_list,a2_list,real_list,imag_list, coordinates
#
#
#
# def highway_pixel(input, superPixel, highwayPixel):
#     new_input = np.zeros((input[:,:,0].shape),dtype=int)
#     # print(new_input.shape)
#     for row_index,row in enumerate(superPixel):
#         for col_index,item in enumerate(row):
#             superPixel_label = item
#             highwayPixel_label = highwayPixel[superPixel_label]
#             new_input[row_index][col_index] = highwayPixel_label
#
#     return new_input
#
# def centroid_single(segments,input):
#     centroids = []
#     amp_slc1 = input[:,:,0]
#     amp_slc2 = input[:,:,1]
#     real_ifg_phase = input[:,:,2]
#     imag_ifg_phase = input[:,:,3]
#     coordinates = []
#     # print(np.asarray(amp_slc1).shape)
#     # print("test",amp_slc1)
#     for i in range(len(np.unique(segments))):
#         indices = []
#         for loc,item in enumerate(segments):
#             # print(loc,item)
#             if(item==i):
#                 indices.append(loc)
#         coordinates.append(indices)
#
#     delta_list = []
#     a1_list,a2_list,real_list,imag_list = [],[],[],[]
#     sum=0
#     # print(coordinates)
#     for i in range(len(coordinates)):
#
#         a1,a2,real,imag = [],[],[],[]
#         for loc in coordinates[i]:
#             # print(loc)
#             a1.append(amp_slc1[0,loc])
#             a2.append(amp_slc2[0,loc])
#             real.append(real_ifg_phase[0,loc])
#             imag.append(imag_ifg_phase[0,loc])
#         cx = np.mean(coordinates[i])
#         centroids.append(cx)
#         sum += len(a1)
#         a1_list.append(np.mean(a1))
#         a2_list.append(np.mean(a2))
#         real_list.append(np.mean(real))
#         imag_list.append(np.mean(imag))
#     # for props in regionprops(segments+1):
#     #     cx, cy = props.centroid  # centroid coordinates
#     #     centroids.append([cx,cy])
#         # break
#     return centroids,a1_list,a2_list,real_list,imag_list, coordinates

# def coherence_estimator(highwayToPixel, slc1,slc2):
#     # print(highwayToPixel)
#     regions = regionprops(highwayToPixel)
#     # print("1",(regions))
#     # print("2",highwayToPixel.astype(int))
#     # print("Regions:",len(regions))
#     a1_list,a2_list,real_list,imag_list = [],[],[],[]
#
#     delta_list = []
#     # numerator,deno1,deno2=0
#     for props in regions:
#         # print(np.asarray(props.coords).shape)
#         print(props.coords)
#         z1 = list(itemgetter(*props.coords)(slc1))
#         z2 = list(itemgetter(*props.coords)(slc2))
#         break
#         # print("z1",len(z1))
#         # numerator,deno1,deno2 = 0,0,0
#         # for i in range(len(z1)):
#         #     numerator += z1[i]*np.conj(z2[i])
#         #     deno1 += np.absolute(z1[i])**2
#         #     deno2 += np.absolute(z2[i])**2
#         # #     print("Num",numerator,deno1,deno2)
#         # #     break
#         # delta = numerator/np.sqrt(deno1*deno2)
#         # # # print("Delta:",delta)
#         # delta_list.append(delta)
#         # print("Delta list",delta_list)
#         # break
#
#     return delta_list
        # for loc in props.coords:
        #     z1 = slc1[loc]
        #     z2 = slc2[loc]
        #     sum +=




# def centroid(segments,input):
#     centroids = []
#     amp_slc1 = input[:,:,0]
#     amp_slc2 = input[:,:,1]
#     real_ifg_phase = input[:,:,2]
#     imag_ifg_phase = input[:,:,3]
#     regions = regionprops(segments+1)
#     print("Regions:",len(regions))
#     a1_list,a2_list,real_list,imag_list = [],[],[],[]
#     sum=0
#     for props in regions:
#         cx, cy = props.centroid  # centroid coordinates
#         centroids.append([cx,cy])
#         # print(props.coords,"\ncentroid",props.centroid)
#         a1,a2,real,imag = [],[],[],[]
#         for loc in props.coords:
#             x,y=loc[0],loc[1]
#             a1.append(amp_slc1[x,y])
#             a2.append(amp_slc2[x,y])
#             real.append(real_ifg_phase[x,y])
#             imag.append(imag_ifg_phase[x,y])
#         sum += len(a1)
#         a1_list.append(np.mean(a1))
#         a2_list.append(np.mean(a2))
#         real_list.append(np.mean(real))
#         imag_list.append(np.mean(imag))
#         # break
#     return centroids,a1_list,a2_list,real_list,imag_list


def oneD_regionProp(segments):
    region = dict()
    unique = np.unique(segments)
    for i in unique:
        region[i] = []
    c=0
    temp = []
    for i in segments:
        print(i)
        region[i].append(c)
        c+=1
    return region
