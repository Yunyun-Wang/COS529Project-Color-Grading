import os
import wget
from multiprocessing import Pool
from subprocess import call

from dataset import Img

server_path = 'https://data.csail.mit.edu/graphics/fivek/img/'

data_path = 'fivek_dataset/raw_photos'
output_path = 'fivek_dataset'
processed_path = 'fivek_dataset/processed'

img_splits = ['HQa1400to2100', 'HQa1to700', 'HQa2101to2800', 'HQa2801to3500', 'HQa3501to4200', 'HQa4201to5000', 'HQa701to1400']

targets = ['tiff16_a', 'tiff16_b', 'tiff16_c', 'tiff16_d', 'tiff16_e']

def download(params):
    Img_c, img_name = params
    for target in targets:
        img_path = img_name+'.tif'
        img_local_path = os.path.join(output_path, target, img_path)
        img_processed_path = os.path.join(processed_path, target, img_path[:-4]+'.jpg')
        if not os.path.exists(img_processed_path):
            print(img_path)
            wget.download(os.path.join(server_path, target, img_path), out=img_local_path)
            img = Img_c.read_img(img_local_path)
            Img_c.write_img(img, img_processed_path)
            call('rm {}'.format(img_local_path), shell=True)


if __name__ == '__main__':
    for target in targets:
        os.makedirs(os.path.join(output_path, target), exist_ok=True)
        os.makedirs(os.path.join(processed_path, target), exist_ok=True)
    param_list = []
    Img_c = Img()
    for img_split in img_splits:
        img_names = os.listdir(os.path.join(data_path, img_split, 'photos'))
        for img_name in img_names:
            if img_name[-4:] == '.dng':
                param_list.append((Img_c, img_name[:-4]))
    p = Pool(20)
    p.map(download, param_list)
    p.close()