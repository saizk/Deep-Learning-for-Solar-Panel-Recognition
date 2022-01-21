# -*- coding: utf-8 -*
import os
import time
import numpy as np

import gmaps
from wrappers import *


def download_sent2():
    CPC_USERNAME, CPC_PASSWORD = os.environ.get('CPC_USERNAME'), os.environ.get('CPC_PASSWORD')
    api = Sentinel2Downloader(CPC_USERNAME, CPC_PASSWORD)
    api.request_products()
    products = api.filter_products(sort_by=['cloudcoverpercentage', 'beginposition'], remove_offline=True)
    print(products[['title', 'cloudcoverpercentage']].to_string())

    api.download('342c57d0-bde8-4391-90f6-a4192ba47a14', './data')


def compute_centers(top_left, bottom_right):
    centers = []
    lat1, lon1 = top_left
    lat2, lon2 = bottom_right

    latmin, latmax = min(lat1, lat2), max(lat1, lat2)
    lonmin, lonmax = min(lon1, lon2), max(lon1, lon2)

    step_lat = 0.0012
    step_lon = 0.0017

    for i, lon in enumerate(np.arange(lonmin, lonmax, step_lon)):
        centers.append([])
        for lat in np.arange(latmin, latmax, step_lat):
            centers[i].append((lat, lon))
        centers[i] = list(reversed(centers[i]))

    return centers


def download_gmaps_api():
    GMAPS_KEY = os.environ.get('GMAPS_KEY') or 'AIzaSyCmaQVinBWJKQb5EfJaavHa8kLBwmcbdXY'
    PLACES = {
        # 'mprincipe': [(40.4123, -3.854), (40.4054, -3.841)],
        'leganes': [(40.34, -3.778), (40.337, -3.77)]
    }

    api = GoogleMapsAPIDownloader(GMAPS_KEY)

    for name, coords in PLACES.items():
        centers = compute_centers(*coords)
        print(f'Number of tiles: {len(centers)}x{len(centers[0])}')

        folder, fformat = f'data/{name}', 'png'

        # api.download_grid(centers, folder, split=True)
        api.parallel_download_grid(
            centers, folder, fformat,
            size=(1280, 1280), zoom=19, scale=2,
            split=True
        )

    # folder = 'data/tiles_leganes'
    # for img in os.listdir(folder):
    #     api.split_tile(folder + '/' + img)


def download_gmaps_web(folder='tiles', zoom=20):
    madrid = np.array([(40.65, -4.082), (40.04794, -3.292)])
    madrid_2 = np.array([(40.5, -3.94), (40.2, -3.5462)])
    madrid_4 = np.array([(40.49, -3.84), (40.2775, -3.56)])
    madrid_6 = np.array([(40.43, -3.74), (40.38, -3.68)])
    # sp_madrid = [(40.413, -3.854), (40.403, -3.841)]
    bajo_b = [(40.340092, -3.777754), (40.336984, -3.769861)]
    coords = bajo_b
    gmaps = GoogleMapsWebDownloader(
        *coords,
        zoom=zoom,
        folder=folder,
    )

    print('Downloading tiles...')
    gmaps.download()
    # gmaps.sharpen()
    # print('Merging tiles...')
    # gmaps.merge(r'.\tiles\merged.png')


def main():
    # download_sent2()

    download_gmaps_api()

    # folder = 'data/tiles'
    # start_time = time.time()
    # download_gmaps_web(folder, zoom=19)
    # final_time = time.time() - start_time
    # total_files = len(os.listdir(folder))
    # print(f'\nDownloaded files: {total_files}')
    # print(f'{total_files / final_time} files/second')
    # print(f'Elapsed time: {final_time}s')


if __name__ == '__main__':
    main()
