# -*- coding: utf-8 -*
import os
import io
import numpy as np
import googlemaps
import multiprocessing as mp
import urllib.request as ur

from PIL import Image
from pathlib import Path
from sentinelsat import SentinelAPI
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import utils


class GoogleMapsWebDownloader(object):

    def __init__(self):
        self.tiles = None

    def download(self, top_left, right_bottom, folder, **kwargs):
        if not os.path.exists(folder):
            os.mkdir(folder)

        urls, xy = self._get_urls(top_left, right_bottom, kwargs.get('zoom'), kwargs.get('style'))

        results = self._download(urls, folder, *xy, kwargs.get('format'))
        tiles = [img for row in results for img in row]

    def _download(self, urls, folder, len_x, len_y, fformat):
        path = Path(folder)
        path.mkdir(exist_ok=True, parents=True)

        per_process = len(urls) // mp.cpu_count()

        urls_and_names = [
            (url, fr'{folder}\tile_{i // len_x}_{i % len_y}.{fformat}')
            for i, url in enumerate(urls)
        ]
        split_urls = [
            urls_and_names[i:i + per_process]
            for i in range(0, len(urls_and_names), per_process)
        ]
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
            results = list(pool.map(self._download_tiles, split_urls))
        return results

    def _download_tiles(self, urls_and_names):
        with ThreadPoolExecutor(max_workers=32) as pool:
            byte_images = list(pool.map(
                lambda v: self._request(v[0], v[1]), urls_and_names)
            )
        return byte_images

    def _get_urls(self, top_left, right_bottom, zoom, style):
        pos1x, pos1y, pos2x, pos2y = utils.latlon2px(*top_left, *right_bottom, zoom)
        len_x, len_y = utils.get_region_size(pos1x, pos1y, pos2x, pos2y)

        return [self.get_url(i, j, zoom, style)
                for j in range(pos1y, pos1y + len_y)
                for i in range(pos1x, pos1x + len_x)], (len_x, len_y)

    def _request(self, url, name):
        _HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 Edg/88.0.705.68'}
        header = ur.Request(url, headers=_HEADERS)
        # err = 0
        while True:
            try:
                data = ur.urlopen(header).read()
                self._save_bytes(data, name)
                return data
            except Exception as e:
                # raise Exception(f"Bad network link: {e}")
                pass

    @staticmethod
    def get_url(x, y, z, style):
        return f"http://mts0.googleapis.com/vt?lyrs={style}&x={x}&y={y}&z={z}"

    @staticmethod
    def _save_bytes(response, output):
        with open(output, 'wb') as f:
            for x in response:
                f.write(x)

    def merge(self, filename):
        self._merge_and_save(filename)

    def _merge_and_save(self, filename):
        len_xy = int(np.rint(np.sqrt(len(self.tiles))))
        merged_pic = self._merge_tiles(self.tiles, len_xy, len_xy)
        merged_pic = merged_pic.convert('RGB')
        merged_pic.save(filename)

    @staticmethod
    def _merge_tiles(tiles, len_x, len_y):
        merged_pic = Image.new('RGBA', (len_x * 256, len_y * 256))
        for i, tile in enumerate(tiles):
            tile_img = Image.open(io.BytesIO(tile))
            y, x = i // len_x, i % len_x
            merged_pic.paste(tile_img, (x * 256, y * 256))

        return merged_pic


class GoogleMapsAPIDownloader(object):
    def __init__(self, key):
        self.api = googlemaps.Client(key=key)

    def _request(self, **kwargs):
        return self.api.static_map(**kwargs)

    def download_tile(self, filename, split, **kwargs):
        response = self._request(**kwargs)
        self._save_bytes(response, filename)
        if split:
            self.split_tile(filename)

    def download_grid(self, centers, folder, split=False, **kwargs):
        path = Path(folder)
        path.mkdir(exist_ok=True, parents=True)

        for i, row in enumerate(centers):
            for j, center in enumerate(row):
                filename = f'{folder}/{path.stem}_{j}_{i}.{kwargs.get("format")}'
                self.download_tile(
                    filename=filename,
                    center=center,
                    split=split,
                    **kwargs
                )

    def parallel_download_grid(self, centers, folder, split=False, **kwargs):
        path = Path(folder)
        path.mkdir(exist_ok=True, parents=True)

        map_params = self._gen_parallel_config(centers, folder, split, **kwargs)

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
            results = list(pool.map(self._download_tiles, map_params))

    def _gen_parallel_config(self, centers, folder, split, **kwargs):
        path = Path(folder)
        map_params = self._gen_map_params(split=split, **kwargs)
        init_args = [
            (center, f'{folder}/{path.stem}_{j}_{i}.{kwargs.get("format")}', map_params)
            for i, row in enumerate(centers)
            for j, center in enumerate(row)
        ]

        per_process = len(init_args) // mp.cpu_count() or 1
        return [
            init_args[i:i + per_process]
            for i in range(0, len(init_args), per_process)
        ]

    def _download_tiles(self, centers_and_files):
        with ThreadPoolExecutor(max_workers=32) as pool:
            tiles = list(pool.map(
                lambda v: self.download_tile(center=v[0], filename=v[1], **v[2]),
                centers_and_files)
            )
        return tiles

    @staticmethod
    def _gen_map_params(**kwargs):
        return {k: v for k, v in kwargs.items()}

    @staticmethod
    def _save_bytes(response, output):
        with open(output, 'wb') as f:
            for x in response:
                f.write(x)

    @staticmethod
    def split_tile(image, size=256):

        image = Path(image)
        path = image.parent / 'split'
        path.mkdir(parents=True, exist_ok=True)

        img = np.asarray(Image.open(image).convert('RGB'))
        M, N, *_ = img.shape
        idx = M // size

        for i, row in enumerate(range(0, M, size)):

            for j, col in enumerate(range(0, N, size)):
                tile = np.asarray(Image.new('RGB', (size, size)))
                tile[:, :, :] = img[row:row + size, col:col + size, :]
                Image.fromarray(tile).save(f'{path}/{image.stem}_{(i * idx) + j}{image.suffix}')


class Sentinel2Downloader(object):

    def __init__(self, usr, pwd):
        self.api = SentinelAPI(usr, pwd)
        self.query = None

    def request_products(
            self,
            mission='S2A', level='MSIL2A',
            baseline_no='*', relative_orbit_no='R094',
            tile='T30TVK', year=2021
    ):
        query_kwargs = {'raw': f'{mission}_{level}_{year}*_{baseline_no}_{relative_orbit_no}_{tile}_{year}*'}
        self.query = self.api.query(**query_kwargs)

    def filter_products(self, sort_by=None, remove_offline=False):
        if sort_by is None:
            sort_by = ['title']

        products_df = self.api.to_dataframe(self.query)

        if remove_offline:
            filter_func = products_df.apply(lambda col: self.api.is_online(col['uuid']), axis=1)
            products_df = products_df[filter_func]

        return products_df.sort_values([*sort_by], ascending=False)

    def download(self, product_id, path):
        self.api.download(product_id, directory_path=path)

    def download_all(self, product_df, path):
        self.api.download_all(*product_df['uuid'], path)

    def get_product_info(self, product_id):
        return self.api.get_product_odata(product_id)
