import os
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
    api.download_map('test', size=(4000, 4000), zoom=12, scale=5)
    # print(response)


def download_gmaps_hd(folder=r'.\tiles'):

    if not os.path.exists(folder):
        os.mkdir(folder)

    gmaps = GoogleMapsHDDownloader(
        top_left=(40.65, -4.082),
        right_button=(40.047937, -3.292),
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
