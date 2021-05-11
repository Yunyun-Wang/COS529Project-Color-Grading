import os
import rawpy
import imageio
from pycubelut import CubeLUT, process_image
from multiprocessing import Pool

from dataset import Img

data_path = 'fivek_dataset/processed'#'fivek_dataset/raw_photos'
output_path = 'fivek_dataset/processed'

img_splits = ['HQa1400to2100', 'HQa1to700', 'HQa2101to2800', 'HQa2801to3500', 'HQa3501to4200', 'HQa4201to5000', 'HQa701to1400']

targets = ['original_small']

def process_dng(params):
    img_split, img_name = params
    print(img_name)
    img = rawpy.imread(os.path.join(data_path, img_split, 'photos', img_name))
    rgb = img.postprocess()
    imageio.imsave(os.path.join(output_path, 'original', '{}.jpg'.format(img_name[1:5])), rgb)

def process_original(params):
    Img_c, img_name = params
    print(img_name)
    img = Img_c.read_img(os.path.join(data_path, 'original', img_name))
    Img_c.write_img(img, os.path.join(output_path, 'original_small', img_name))

def process_original_small(params):
    luts, Img_c, img_name = params
    print(img_name)
    for i in range(len(luts)):
        process_image(os.path.join(data_path, 'original_small', img_name), os.path.join(output_path, 'lut_{}'.format(i))+'/', 0, luts[i], False)


if __name__ == '__main__':
    for target in targets:
        os.makedirs(os.path.join(output_path, target), exist_ok=True)
    # param_list = []
    # for img_split in img_splits:
    #     img_names = os.listdir(os.path.join(data_path, img_split, 'photos'))
    #     for img_name in img_names:
    #         if img_name[-4:] == '.dng':
    #             param_list.append((img_split, img_name))
    #
    # p = Pool(20)
    # p.map(process_dng, param_list)
    # p.close()

    # param_list = []
    # Img_c = Img()
    # img_names = os.listdir(os.path.join(data_path, 'original'))
    # for img_name in img_names:
    #     if img_name[-4:] == '.jpg':
    #         param_list.append((Img_c, img_name))
    #
    # p = Pool(20)
    # p.map(process_original, param_list)
    # p.close()

    luts = []
    lut_files = os.listdir('luts')
    lut_files.sort()
    for i in range(len(lut_files)):
        lut = CubeLUT(os.path.join('luts', lut_files[i]))
        luts.append(lut)
        os.makedirs(os.path.join(output_path, 'lut_{}'.format(i)), exist_ok=True)

    param_list = []
    Img_c = Img()
    img_names = os.listdir(os.path.join(data_path, 'original_small'))
    for img_name in img_names:
        if img_name[-4:] == '.jpg':
            param_list.append((luts, Img_c, img_name))

    p = Pool(20)
    p.map(process_original_small, param_list)
    p.close()