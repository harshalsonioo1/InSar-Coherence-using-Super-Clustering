#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# example.py
# Copyright (c) 2020 Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils import readFloatComplex, readShortComplex, readFloat
import matplotlib.pyplot as plt
import numpy as np
import argparse
from skimage.segmentation import mark_boundaries
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim

from superpixel import superpixel, centroid, highway_pixel,coherence_estimator, oneD_regionProp, centroid_single
from decomposition import decomposition_highway, decomposition_super

parser = argparse.ArgumentParser()
parser.add_argument("--ddir", type=str, default=".")
args = parser.parse_args()

IFG_PATH = args.ddir + "/1slc1_1slc2.noisy"
COH_PATH = args.ddir + "/1slc1_1slc2.filt.coh"
SLC1_PATH = args.ddir + "/1slc1.rslc"
SLC2_PATH = args.ddir + "/1slc2.rslc"
WIDTH = 300

if __name__ == "__main__":
    # Load binary interfergram data;
    ifg = readFloatComplex(IFG_PATH, WIDTH)
    coh_3vg = readFloat(COH_PATH, WIDTH)
    slc1 = readFloatComplex(SLC1_PATH, WIDTH)
    slc2 = readFloatComplex(SLC2_PATH, WIDTH)

    # 4D representation

    # Dim-0
    amp_slc1 = np.abs(slc1)
    # Dim-1
    amp_slc2 = np.abs(slc2)

    # Phase of Ifg
    phase_ifg = np.angle(ifg)
    # Force amp to one
    phase_bar_ifg = 1*np.exp(1j*phase_ifg)

    # Dim-2
    real_ifg_phase = np.real(phase_bar_ifg)
    # Dim-3
    imag_ifg_phase = np.imag(phase_bar_ifg)

    # print(amp_slc1)
    # print(amp_slc2)
    # print(real_ifg_phase)
    # print(imag_ifg_phase)
    input = np.dstack((amp_slc1,amp_slc2,real_ifg_phase,imag_ifg_phase))
    superpixel_fz,superpixel_slic=superpixel(input)
    print(input.shape)
    # print(segments_fz)
    centroids_super_fz,a1_list,a2_list,real_list,imag_list,coordinates_super_fz = centroid(superpixel_fz,input)

    input_fz = np.dstack((a1_list,a2_list,real_list,imag_list))
    kmeans_fz = KMeans(n_clusters=800, random_state=0).fit(input_fz.reshape(-1,4))
    highway_fz = kmeans_fz.labels_
    print(input_fz.shape)


    centroids_super_sl,a1_list,a2_list,real_list,imag_list,coordinates_high_sl = centroid(superpixel_slic ,input)

    input_sl = np.dstack((a1_list,a2_list,real_list,imag_list))
    # print(input_sl.shape, input_sl.reshape(-1,4).shape)
    kmeans_sl = KMeans(n_clusters=30, random_state=0).fit(input_sl.reshape(-1,4))
    highway_sl = kmeans_sl.labels_

    print(f"Felzenszwalb number of Highway Pixels: {len(np.unique(highway_fz))}")
    print(f"SLIC number of Highway Pixels: {len(np.unique(highway_sl))}")

    # print(superpixel_fz.max(),len(highway_fz))

    new_input = highway_pixel(input,superpixel_fz,highway_fz)
    # print("Input",input_fz)
    centroids_highway_fz,a1_list,a2_list,real_list,imag_list, coordinates_high_fz = centroid_single(highway_fz ,input_fz)
    highway_coh = coherence_estimator(new_input,slc1,slc2)

    # print(("Highway Coherence",highway_coh))
    # print(centroids)

    decom_highway_coh = decomposition_highway(centroids_highway_fz, highway_fz,highway_coh,coordinates_high_fz,input_fz)
    # print("Super pixel level Coherence",decom_highway_coh)
    decom_super_coh = decomposition_super(centroids_super_fz,superpixel_fz,decom_highway_coh,coordinates_super_fz,input)
    # print("Pixel level Coherence",decom_super_coh)
    # print(np.array(decom_highway_coh).shape)
    # print(np.array(decom_super_coh).shape)
    #
    print(np.array(coh_3vg).shape)
    # a = np.array(decom_super_coh) # your x
    # b = np.array(coh_3vg) # your y
    # mses = ((a-b)**2).mean(axis=1)
    # decom_super_coh = np.array(decom_super_coh)*2
    mses=mean_squared_error(coh_3vg,decom_super_coh)
    print("MSE", mses)
    print("RMSE",np.sqrt(mses))

    (score, diff) = compare_ssim(np.array(coh_3vg), np.array(decom_super_coh), full=True)
    diff = (diff * 255).astype("uint8")

    # 6. You can print only the score if you want
    print("SSIM: {}".format(score))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(coh_3vg, cmap='gray')
    ax[0].set_title("Ground Truth")
    ax[1].imshow(decom_super_coh, cmap = 'gray', vmin=np.array(decom_super_coh).min(), vmax = np.array(decom_super_coh).max())
    ax[1].set_title("Decomposed Pixel Level Coh")
    plt.show()

    # fig, ax = plt.subplots(3, 4, figsize=(14, 7))
    #
    # # SLC1
    # ax[0, 0].imshow(amp_slc1**0.3, cmap='gray')
    # ax[0, 0].set_title('amp slc1')


    # fig, ax = plt.subplots(3, 4, figsize=(14, 7))
    #
    # # SLC1
    # ax[0, 0].imshow(amp_slc1**0.3, cmap='gray')
    # ax[0, 0].set_title('amp slc1')
    # ax[0, 1].imshow(amp_slc2**0.3, cmap='gray')
    # ax[0, 1].set_title('amp slc2')
    # ax[0, 2].imshow(real_ifg_phase, cmap='jet')
    # ax[0, 2].set_title('Real Phase ifg')
    # ax[0, 3].imshow(imag_ifg_phase, cmap='jet')
    # ax[0, 3].set_title('Imaginary Phase ifg')
    #
    #
    # # FZ clustering
    # ax[1, 0].imshow(mark_boundaries(amp_slc1**0.3, segments_fz), cmap='gray')
    # # ax[1, 0].set_title('amp slc1')
    # ax[1, 1].imshow(mark_boundaries(amp_slc2**0.3, segments_fz), cmap='gray')
    # # ax[1, 1].set_title('amp slc2')
    # ax[1, 2].imshow(mark_boundaries(real_ifg_phase,segments_fz), cmap='jet')
    # # ax[1, 2].set_title('Real Phase ifg')
    # ax[1, 3].imshow(mark_boundaries(imag_ifg_phase,segments_fz), cmap='jet')
    # # ax[1, 3].set_title('Imaginary Phase ifg')
    #
    # # SLIC clustering
    # ax[2, 0].imshow(mark_boundaries(amp_slc1**0.3, segments_slic), cmap='gray')
    # # ax[2, 0].set_title('amp slc1')
    # ax[2, 1].imshow(mark_boundaries(amp_slc2**0.3, segments_slic), cmap='gray')
    # # ax[2, 1].set_title('amp slc2')
    # ax[2, 2].imshow(mark_boundaries(real_ifg_phase,segments_slic), cmap='jet')
    # # ax[2, 2].set_title('Real Phase ifg')
    # ax[2, 3].imshow(mark_boundaries(imag_ifg_phase,segments_slic), cmap='jet')
    # ax[2, 3].set_title('Imaginary Phase ifg')



    # plt.show()
