import numpy as np
import pandas as pd
import skimage
import os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom
import scipy.misc

from pathlib import Path

dirpath = str(Path(os.getcwd()).parent.parent) + "\subset0\\"
print("current dir " + dirpath)
