import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt


# read original CT image and return it as numpy image
def load_itk_image(filename):
    itk_image = sitk.ReadImage(filename)
    numpy_image = sitk.GetArrayFromImage(itk_image)

    numpy_origin = np.array(list(reversed(itk_image.GetOrigin())))
    numpy_spacing = np.array(list(reversed(itk_image.GetSpacing())))

    return numpy_image, numpy_origin, numpy_spacing


# read annotations CSV file
def read_csv(filename):
    lines = []
    with open(filename, newline='') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            lines.append(line)
    return lines


# change world coordinates to voxel coordinates
def world_to_voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord


# extract features from candidates
def normalize_planes(npz_array):
    maxHU = 400.
    minHU = -1000.

    npz_array = (npz_array - minHU) / (maxHU - minHU)
    npz_array[npz_array > 1] = 1.
    npz_array[npz_array < 0] = 0.
    return npz_array


# get abs path of data folder
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_path = os.path.abspath(os.path.join(data_path, os.pardir))
data_path = os.path.abspath(os.path.join(data_path, 'data'))

# get abs paths of data subsets
subset_dirs = []
for root, dirs, files in os.walk(data_path):
    for d in dirs:
        if d.startswith("subset"):
            subset_path = os.path.abspath(os.path.join(data_path, d))
            subset_dirs.append(subset_path)

all_mhd_file_paths = []
# iterate through subsets, searching for mhd files recursively
for s in subset_dirs:
    for root, dirs, files in os.walk(s):
        for file in files:
            if file.endswith(".mhd"):
                all_mhd_file_paths.append(os.path.abspath(os.path.join(s, file)))

# get abs path of candidates file
candidates_path = os.path.abspath(os.path.join(data_path, 'candidates.csv'))

numpy_image, numpy_origin, numpy_spacing = load_itk_image(all_mhd_file_paths[0])

# read candidates from csv file
cands = read_csv(candidates_path)

pics_path = os.path.abspath(os.path.join(data_path, 'pics'))
if not os.path.exists(pics_path):
    os.makedirs(pics_path)

i = 0
for cand in cands[1:]:
    world_coord = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
    voxel_coord = world_to_voxel_coord(world_coord, numpy_origin, numpy_spacing)
    voxel_width = 65
    patch = numpy_image[int(voxel_coord[0]), int(voxel_coord[1] - voxel_width / 2):int(voxel_coord[1] + voxel_width / 2),
            int(voxel_coord[2] - voxel_width / 2):int(voxel_coord[2] + voxel_width / 2)]
    patch = normalize_planes(patch)
    print(len(patch))
    # plt.imshow(patch, cmap='gray')
    # plt.show()

    Image.fromarray(patch * 255).convert('L').save(os.path.join(pics_path, 'candidate_num_' + str(i) +
                                                                '_' + cand[4] + '.tiff'))
    i += 1
