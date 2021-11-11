# -*- coding: utf-8 -*
import numpy as np
import time
from download import Sentinel2Downloader, GoogleMapsAPIDownloader, GoogleMapsHDDownloader
from _config import *
from utils import *


def download_sent2():
    api = Sentinel2Downloader(CPC_USERNAME, CPC_PASSWORD)
    api.request_products()
    products = api.filter_products(sort_by=['cloudcoverpercentage', 'beginposition'], remove_offline=True)
    print(products[['title', 'cloudcoverpercentage']].to_string())

    api.download('342c57d0-bde8-4391-90f6-a4192ba47a14', './data')


def download_gmaps_api():
    api = GoogleMapsAPIDownloader(GMAPS_KEY)
    api.download_map('test', size=(4000, 4000), zoom=12, scale=5)
    # print(response)


def download_gmaps_hd(folder=r'.\tiles'):

    madrid = np.array([(40.65, -4.082), (40.047937, -3.292)])
    madrid_2 = np.array([(40.5, -3.94), (40.2, -3.5462)])
    madrid_4 = np.array([(40.49, -3.84), (40.2775, -3.56)])
    madrid_6 = np.array([(40.43, -3.74), (40.38, -3.68)])

    gmaps = GoogleMapsHDDownloader(
        top_left=madrid_6[0],
        right_bottom=madrid_6[1],
        zoom=18,
        folder=folder,
    )

    print('Downloading tiles...')
    gmaps.download()
    # print('Merging tiles...')
    # gmaps.merge(r'.\tiles\merged.png')


def main():
    # download_sent2()
    # download_gmaps_api()
    start_time = time.time()
    download_gmaps_hd()
    print(f'Elapsed time: {time.time() - start_time}s')


if __name__ == '__main__':
    main()
