Interferometric Synthetic Aperture Radar (InSAR)
is a satellite-based imaging technique which has been used to
learn about earth’s surface and sub-surface movements. It can
measure earths displacements by comparing phase information
from the SAR images taken at different points in time. But, due to
high level of noises, the wrapped phases are distorted, because of
which, it cannot be used for its intended purposes. Understanding
the coherence of the images becomes rather important in this
situation to denoise the image and extract useful information
from it.
In our project, we focus on applying Super Pixeling using
both traditional Computer Vision methods as well as Graph
Neural Networks to understand how better we can estimate image
coherence between two SAR images. By applying Super pixeling
in two stages, using specific clustering techniques, we would be
able to expose non-local similarities and measure the coherence
from a non-local perspective.


# InSAR-Starter-Kit
1. Download sample data from the google drive
2. Save the data to you local directory - $PAHT_DIR$
3. Run sample code via 
``` bash
python example.py --ddir $PATH_DIR$
```
