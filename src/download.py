# -*- coding: utf-8 -*

import os
import io
import numpy as np
import googlemaps
import PIL.Image as pil
from sentinelsat import SentinelAPI

from gmaps import download_map_hd


class GoogleMapsHDDownloader(object):

    def __init__(self, top_left, right_bottom, zoom, folder):
        self.lat_1, self.lng_1 = top_left
        self.lat_2, self.lng_2 = right_bottom

        self.zoom = zoom
        self.folder = folder

        self.tiles = None

    def download(self):

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        self.tiles = download_map_hd(
            lat_1=self.lat_1, lng_1=self.lng_1,
            lat_2=self.lat_2, lng_2=self.lng_2,
            zoom=self.zoom,
            folder=self.folder
        )

    def merge(self, filename):
        self._merge_and_save(filename)

    def _merge_and_save(self, filename):
        len_xy = int(np.rint(np.sqrt(len(self.tiles))))
        merged_pic = self._merge_tiles(self.tiles, len_xy, len_xy)
        merged_pic = merged_pic.convert('RGB')
        merged_pic.save(filename)

    @staticmethod
    def _merge_tiles(tiles, len_x, len_y):
        merged_pic = pil.new('RGBA', (len_x * 256, len_y * 256))
        for i, tile in enumerate(tiles):
            tile_img = pil.open(io.BytesIO(tile))
            y, x = i // len_x, i % len_x
            merged_pic.paste(tile_img, (x * 256, y * 256))

        return merged_pic


class GoogleMapsAPIDownloader(object):
    def __init__(self, key):
        self.api = googlemaps.Client(key=key)

    def download_map(self, output_file, size=(2000, 2000), zoom=10, scale=2, map_type='satellite', file_format='png'):
        response = self.api.static_map(
            size=size,
            zoom=zoom,
            center=(40.42793938593949, -3.7118178511306477),
            maptype=map_type,
            format=file_format,
            scale=scale
        )
        self.save_response(response, output_file, file_format)

    @staticmethod
    def save_response(response, output_file, file_format):
        with open(f'{output_file}.{file_format}', 'wb') as f:
            for x in response:
                f.write(x)


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

