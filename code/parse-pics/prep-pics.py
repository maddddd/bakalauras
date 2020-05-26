import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image


# perskaitom originalu komp. tomografijos paveiksla ir grazinam ji kaip numpy image
def load_itk_image(filename):
    itk_image = sitk.ReadImage(filename)
    numpy_image = sitk.GetArrayFromImage(itk_image)

    numpy_origin = np.array(list(reversed(itk_image.GetOrigin())))
    numpy_spacing = np.array(list(reversed(itk_image.GetSpacing())))

    return numpy_image, numpy_origin, numpy_spacing


# perskaitom kandidatu CSV faila
def read_csv(filename):
    lines = []
    with open(filename, newline='') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            lines.append(line)
    return lines


# pakeiciam tikro pasaulio koordinates i voxel koordinates
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

# randam absoliutu kelia iki data katalogo
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_path = os.path.abspath(os.path.join(data_path, os.pardir))
data_path = os.path.abspath(os.path.join(data_path, 'data'))

# randam absoliucius kelius iki subsets katalogu
subset_dirs = []
for root, dirs, files in os.walk(data_path):
    for d in dirs:
        if d.startswith("subset"):
            subset_path = os.path.abspath(os.path.join(data_path, d))
            subset_dirs.append(subset_path)

all_mhd_file_paths = []
# iteruojam subsets, ieskodami mhd failu rekursyviai
for s in subset_dirs:
    for root, dirs, files in os.walk(s):
        for file in files:
            if file.endswith(".mhd"):
                all_mhd_file_paths.append(os.path.abspath(os.path.join(s, file)))

subsets_for_mhds = []
# randam subset numeri mhd failui
for path in all_mhd_file_paths:
    subs = path.split("subset")
    subsets_for_mhds.append(subs[1][0])

# randam absoliutu kandidatu failo kelia
candidates_path = os.path.abspath(os.path.join(data_path, 'candidates_V2.csv'))

# skaitom kandidatus is csv failo
cands = read_csv(candidates_path)

pics_path = os.path.abspath(os.path.join(data_path, 'pics'))
if not os.path.exists(pics_path):
    os.makedirs(pics_path)

# issaugom nuotraukas i kieta diska
i = 0
voxel_width = 128
for cand in cands[1:]:
    j = 0
    for path in all_mhd_file_paths:
        if path.endswith(cand[0] + ".mhd"):
            numpy_image, numpy_origin, numpy_spacing = load_itk_image(path)
            world_coord = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
            voxel_coord = world_to_voxel_coord(world_coord, numpy_origin, numpy_spacing)
            patch = numpy_image[int(voxel_coord[0]),
                    int(voxel_coord[1] - voxel_width / 2):int(voxel_coord[1] + voxel_width / 2),
                    int(voxel_coord[2] - voxel_width / 2):int(voxel_coord[2] + voxel_width / 2)]
            patch = normalize_planes(patch)
            # make sure that array does not go out of bounds
            if (int(voxel_coord[1] - voxel_width / 2) + int(voxel_coord[1] + voxel_width / 2) > voxel_width) and \
                    (int(voxel_coord[2] - voxel_width / 2) + int(voxel_coord[2] + voxel_width / 2) > voxel_width):
                Image.fromarray(patch * 255).convert('L').save(os.path.join(pics_path, 'candidate_' + str(i) +
                                                                            '_subset_' + subsets_for_mhds[j]
                                                                            + '_class_' + cand[4] + '.tiff'))
            break
        j += 1
    i += 1

