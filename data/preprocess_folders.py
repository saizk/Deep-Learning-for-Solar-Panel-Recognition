import os
import shutil


def split_masks(img_folder, masks_folder, directory):
    img_root = f'./{directory}'

    for folder in os.listdir(img_root):
        for filename in os.listdir(f'{img_root}/{folder}'):
            src_path = f'{img_root}/{folder}/{filename}'

            if 'label' in filename:
                shutil.move(src_path, f'{masks_folder}/{filename}')
            else:
                shutil.move(src_path, f'{img_folder}/{filename}')

    shutil.rmtree(img_root)


def preprocess_folders(root_dir='Solar Panels', directories=['PV01', 'PV03', 'PV08']):

    root_dir = f'./{root_dir}'
    os.mkdir(root_dir) if not os.path.exists(root_dir) else None

    img_folder, masks_folder = f'{root_dir}/images', f'{root_dir}/masks'
    os.mkdir(img_folder) if not os.path.exists(img_folder) else None
    os.mkdir(masks_folder) if not os.path.exists(masks_folder) else None

    # directories = ['PV03', 'PV08']
    for directory in directories:
        split_masks(img_folder, masks_folder, directory)


if __name__ == '__main__':
    preprocess_folders()
