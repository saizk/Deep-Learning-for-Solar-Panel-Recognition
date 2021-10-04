from download import Sentinel2Downloader
from _config import *


def main():
    api = Sentinel2Downloader(USERNAME, PASSWORD)
    # products = api.request_products(sort_by=['cloudcoverpercentage', 'beginposition'], remove_offline=True)
    # print(products[['title', 'cloudcoverpercentage']].to_string())

    api.download('342c57d0-bde8-4391-90f6-a4192ba47a14', '../data')


if __name__ == '__main__':
    main()
