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

parser = argparse.ArgumentParser();
parser.add_argument("--ddir", type=str, default=".");
args = parser.parse_args();

IFG_PATH = args.ddir + "/20150219_20150404.diff.orb_cor"
COH_PATH = args.ddir + "/20150219_20150404.diff.orb_cor.filt.coh"
SLC1_PATH = args.ddir + "/20150219.rslc"
SLC2_PATH = args.ddir + "/20150404.rslc"
WIDTH = 1000

if __name__ == "__main__":
    # Load binary interfergram data;
    ifg = readFloatComplex(IFG_PATH, WIDTH)
    coh_3vg = readFloat(COH_PATH, WIDTH)
    slc1 = readShortComplex(SLC1_PATH, WIDTH)
    slc2 = readShortComplex(SLC2_PATH, WIDTH)

    fig, ax = plt.subplots(2, 4, figsize=(14, 7))

    # SLC1
    ax[0, 0].imshow(np.angle(slc1), cmap='jet')
    ax[0, 0].set_title('phase slc1')
    ax[1, 0].imshow(np.abs(slc1)**0.3, cmap='gray')
    ax[1, 0].set_title('amp slc1')

    # SLC2
    ax[0, 1].imshow(np.angle(slc2), cmap='jet')
    ax[0, 1].set_title('phase slc2')
    ax[1, 1].imshow(np.abs(slc2)**0.3, cmap='gray')
    ax[1, 1].set_title('amp slc2')

    # IFG
    ax[0, 2].imshow(np.angle(ifg), cmap='jet')
    ax[0, 2].set_title('phase ifg')
    ax[1, 2].imshow(np.abs(slc2)**0.3, cmap='gray')
    ax[1, 2].set_title('amp ifg')

    # It looks different with 3vG ifg, becuase some signals have been removed in 3vG's IFG
    ax[0, 3].imshow(np.angle(slc2 * np.conj(slc1)), cmap='jet')
    ax[0, 3].set_title('phase slc2*conj(slc1)')

    ax[1, 3].imshow(coh_3vg, cmap='gray')
    ax[1, 3].set_title('3vg Stack Coherence')

    fig.show()
