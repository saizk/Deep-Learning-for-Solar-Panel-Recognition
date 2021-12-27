import os
import glob
import shutil
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def split_masks(img_folder, masks_folder, img_root):
    for folder in glob.glob(img_root + r'\PV*'):
        for filename in os.listdir(folder):
            src_path = rf'{folder}\{filename}'
            if 'label' in filename:
                shutil.move(src_path, rf'{masks_folder}\{filename}')
            else:
                shutil.move(src_path, rf'{img_folder}\{filename}')

        shutil.rmtree(folder)


def convert_to_png_parallel(folder='Solar Panels'):
    for pvfolder in os.listdir(folder):
        img_path, mask_path = f'{folder}/{pvfolder}/images', f'{folder}/{pvfolder}/masks'
        with ThreadPoolExecutor(max_workers=64) as pool:
            images = pool.map(
                lambda img: Image.open(img).resize((256, 256)).save(f'{img_path}/{Path(img).stem}.png'),
                glob.glob(f'{img_path}/*.bmp')
            )
            masks = pool.map(
                lambda mask: Image.open(mask).resize((256, 256)).save(f'{mask_path}/{Path(mask).stem}.png'),
                glob.glob(f'{mask_path}/*.png')
            )
    return list(images), list(masks)


def remove_bmps(folder='Solar Panels'):
    for pvfolder in os.listdir(folder):
        img_path, mask_path = f'{folder}/{pvfolder}/images', f'{folder}/{pvfolder}/masks'
        for img, mask in zip(glob.glob(f'{img_path}/*.bmp'), glob.glob(f'{mask_path}/*.bmp')):
            os.remove(img)
            os.remove(mask)


def preprocess_folders(root_dir='Solar Panels'):

    for pvfolder in glob.glob(root_dir + '/*'):  # ['PV01', 'PV03', 'PV08']
        img_folder, masks_folder = fr'{pvfolder}\images', rf'{pvfolder}\masks'
        os.mkdir(img_folder) if not os.path.exists(img_folder) else None
        os.mkdir(masks_folder) if not os.path.exists(masks_folder) else None

        split_masks(img_folder, masks_folder, pvfolder)


if __name__ == '__main__':
    # preprocess_folders()
    # convert_to_png_parallel()
    remove_bmps()
