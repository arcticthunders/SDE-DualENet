import re
import numpy as np
import sys


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return readPFM(path)
import os
import random
def dataloader(filepath):
    filepath += '/'
    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    disp = [dsp for dsp in classes if dsp.find('disparity-unzipped') > -1]
    print(image,disp,classes)
    all_left_img = []
    all_right_img = []
    all_left_disp = []


    driving_dir = filepath + [x for x in image][0] + '/'
    driving_disp = filepath + [x for x in disp][0]

    subdir1 = ['35mm_focallength', '15mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
                for im in imm_l:
                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
                        all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
                    all_left_disp.append(
                        driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
                        all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)

    return all_left_img, all_right_img, all_left_disp