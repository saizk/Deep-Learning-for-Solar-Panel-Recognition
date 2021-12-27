# -*- coding: utf-8 -*
import os

import numpy as np
import time
from download import Sentinel2Downloader, GoogleMapsAPIDownloader, GoogleMapsHDDownloader
from _config import *


def download_sent2():
    api = Sentinel2Downloader(CPC_USERNAME, CPC_PASSWORD)
    api.request_products()
    products = api.filter_products(sort_by=['cloudcoverpercentage', 'beginposition'], remove_offline=True)
    print(products[['title', 'cloudcoverpercentage']].to_string())

    api.download('342c57d0-bde8-4391-90f6-a4192ba47a14', './data')


def download_gmaps_api():
    api = GoogleMapsAPIDownloader(GMAPS_KEY)
    api.download_map('test', center=(40.43, -3.71),
                     size=(4000, 4000), zoom=12, scale=5)
    # print(response)


def download_gmaps_hd(folder=r'.\tiles'):

    madrid = np.array([(40.65, -4.082), (40.047937, -3.292)])
    madrid_2 = np.array([(40.5, -3.94), (40.2, -3.5462)])
    madrid_4 = np.array([(40.49, -3.84), (40.2775, -3.56)])
    madrid_6 = np.array([(40.43, -3.74), (40.38, -3.68)])

    sp_madrid = [(40.414, -3.856), (40.402, -3.836)]

    bajo_b = [(40.340092, -3.777754), (40.336984, -3.769861)]

    gmaps = GoogleMapsHDDownloader(
        top_left=bajo_b[0],
        right_bottom=bajo_b[1],
        zoom=20,
        folder=folder,
    )

    print('Downloading tiles...')
    gmaps.download()
    # gmaps.sharpen()
    # print('Merging tiles...')
    # gmaps.merge(r'.\tiles\merged.png')


def main():
    # download_sent2()
    # download_gmaps_api()
    start_time = time.time()
    folder = r'.\tiles'
    download_gmaps_hd(folder)
    final_time = time.time() - start_time
    total_files = len(os.listdir(folder))
    print(f'\nDownloaded files: {total_files}')
    print(f'{total_files/final_time} files/second')
    print(f'Elapsed time: {final_time}s')


if __name__ == '__main__':
    main()
