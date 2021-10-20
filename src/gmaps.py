# -*- coding: utf-8 -*

import math
import numpy as np
import multiprocessing as mp
import urllib.request as ur

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from utils import latlon2px, get_region_size


# %% ------------------ GET URLS ------------------------

def get_url(x, y, z, style):
    return f"http://mts0.googleapis.com/vt?lyrs={style}&x={x}&y={y}&z={z}"


def get_urls(x1, y1, len_x, len_y, z, style):
    urls = [
        get_url(i, j, z, style)
        for j in range(y1, y1 + len_y)
        for i in range(x1, x1 + len_x)
    ]
    return urls


def get_urls_and_names(urls, folder, len_x, len_y, extension):
    return [(url, fr'{folder}\tile_{i // len_x}_{i % len_y}.{extension}')
            for i, url in enumerate(urls)]


# %% ------------------- DOWNLOAD --------------------------

def download(url, name):
    _HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 Edg/88.0.705.68'}
    header = ur.Request(url, headers=_HEADERS)
    # err = 0
    while True:
        try:
            data = ur.urlopen(header).read()
            save_bytes(data, name)
            return data
        except Exception as e:
            # raise Exception(f"Bad network link: {e}")
            pass


def save_bytes(data, filename):
    with open(filename, "wb") as f:
        f.write(data)


# %% -------------------- PARALLEL DOWNLOAD ---------------------------

def download_tiles(urls_and_names):
    with ThreadPoolExecutor(max_workers=32) as pool:
        byte_images = list(pool.map(
            lambda v: download(v[0], v[1]), urls_and_names)
        )
    return byte_images


def parallel_download(urls, folder, len_x, len_y, extension='png'):
    per_process = len(urls) // mp.cpu_count()

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
        urls_and_names = get_urls_and_names(urls, folder, len_x, len_y, extension)
        split_urls = [
            urls_and_names[i:i + per_process]
            for i in range(0, len(urls_and_names), per_process)
        ]
        results = list(pool.map(download_tiles, split_urls))

    return results


# %% --------------------- MAIN -------------------------------

def download_map_hd(lat_1, lng_1, lat_2, lng_2, zoom, style='s', folder='', extension='png', tiff=False):
    """
    Download Google Maps images based on spatial extent
    ----------
    :param (float) lat_1 : left-top latitude
    :param (float) lng_1: left-top longitude
    :param (float) lat_2: right-bottom latitude
    :param (float) lng_2: right-bottom longitude
    :param (int) zoom: zoom
    :param style:
        m for map;
        s for satellite;
        y for satellite with label;
        t for terrain;
        p for terrain with label;
        h for label;
    :param folder:  Folder path for storing results
    :param extension: Format of the file
    """

    pos1x, pos1y, pos2x, pos2y = latlon2px(lat_1, lng_1, lat_2, lng_2, zoom)
    len_x, len_y = get_region_size(pos1x, pos1y, pos2x, pos2y)
    print(f"Total number of tiles：{len_x} X {len_y} = {len_x * len_y}")

    urls = get_urls(pos1x, pos1y, len_x, len_y, zoom, style)

    results = parallel_download(urls, folder, len_x, len_y, extension)
    tiles = [img for row in results for img in row]

    return tiles


if __name__ == '__main__':
    t30tvk = ((40.65, -4.18), (39.65, -2.88))
    download_map_hd(
        lat_1=40.65, lng_1=-4.082,
        lat_2=40.047937, lng_2=-3.292,
        zoom=4,
        folder=r'.\tiles'
    )  # ~= 1/4 of the original satellite region
