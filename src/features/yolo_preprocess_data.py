import os
import glob
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

from concurrent.futures import ThreadPoolExecutor
from create_yolo_annotations import create_yolo_annotation


def get_indices():
    # img_ids = [i for i in range(len(os.listdir('images')))]
    img_ids = [i for i in range(3716)]
    train_val_ids, test_ids = train_test_split(img_ids, test_size=0.2, random_state=420)
    train_ids, valid_ids = train_test_split(train_val_ids, test_size=0.2, random_state=420)
    return {'train': train_ids, 'val': valid_ids, 'test': test_ids}


def split_dataset(indices):
    images = os.listdir('images')
    masks = os.listdir('masks')

    for phase in ['train', 'test', 'val']:
        os.mkdir(f'images/{phase}')
        os.mkdir(f'masks/{phase}')

    for phase in indices.keys():
        for idx, (img, mask) in enumerate(zip(images, masks)):
            if idx in indices[phase]:
                os.rename(img, f'images/{phase}/{img}')
                os.rename(mask, f'masks/{phase}/{mask}')


def convert_to_png_parallel(folder='solar_panels'):
    for phase in ['train', 'val', 'test']:
        img_path, mask_path = f'{folder}/images/{phase}', f'{folder}/masks/{phase}'
        with ThreadPoolExecutor(max_workers=64) as pool:
            images = pool.map(
                lambda img: Image.open(img).resize((256, 256)).save(f'{img_path}/{Path(img).stem}.png'),
                glob.glob(f'{img_path}/*.png')
            )
            masks = pool.map(
                lambda mask: Image.open(mask).resize((256, 256)).save(f'{mask_path}/{Path(mask).stem}.png'),
                glob.glob(f'{mask_path}/*.png')
            )
    return list(images), list(masks)


def remove_bmps():
    for phase in ['train', 'val', 'test']:
        img_path, mask_path = f'images/{phase}', f'masks/{phase}'
        for img, mask in zip(glob.glob(f'{img_path}/*.bmp'), glob.glob(f'{mask_path}/*.bmp')):
            os.remove(img)
            os.remove(mask)


def get_annotations(data_folder='solar_panels'):
    for phase in ['train', 'val', 'test']:
        img_folder, mask_folder = f'{data_folder}/{phase}/images', f'{data_folder}/{phase}/masks'
        create_yolo_annotation(
            img_folder, mask_folder,
            filename=f'{data_folder}/{phase}_yolo.txt'
        )


def main():
    indices = get_indices()
    # split_dataset(indices)

    # imgs, masks = convert_to_png_parallel()
    # remove_bmps()

    get_annotations()


if __name__ == '__main__':
    main()
