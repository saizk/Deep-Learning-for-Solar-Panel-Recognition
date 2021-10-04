from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt


class Sentinel2Downloader(object):

    def __init__(self, usr, pwd):
        self.api = SentinelAPI(usr, pwd)
        self.query = None

    def request_products(
            self,
            mission='S2A', level='MSIL2A', baseline_no='*',
            relative_orbit_no='094', tile='30TVK',
            year=2021, sort_by=None,
            remove_offline=False
    ):
        if sort_by is None:
            sort_by = ['cloudcoverpercentage']

        query_kwargs = {'raw': f'{mission}_{level}_{year}*_N{baseline_no}_R{relative_orbit_no}_T{tile}_{year}*'}
        self.query = self.api.query(**query_kwargs)
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

