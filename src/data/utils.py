# -*- coding: utf-8 -*
import math
import numpy as np


EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * math.pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0


def compute_centers(top_left, bottom_right):
    centers = []
    lat1, lon1 = top_left
    lat2, lon2 = bottom_right

    latmin, latmax = min(lat1, lat2), max(lat1, lat2)
    lonmin, lonmax = min(lon1, lon2), max(lon1, lon2)

    step_lat = 0.0012
    step_lon = 0.0017

    for lon in np.arange(lonmin, lonmax, step_lon):
        row = [(lat, lon) for lat in np.arange(latmin, latmax, step_lat)]
        centers.append(list(reversed(row)))

    return centers


def get_xy(lat, lng, zoom):
    """
        Generates an X,Y tile coordinate based on the latitude, longitude
        and zoom level
        Returns:    An X,Y tile coordinate
    """

    tile_size = 256
    # Use a left shift to get the power of 2
    # i.e. a zoom level of 2 will have 2^2 = 4 tiles
    num_tiles = 1 << zoom

    # Find the x_point given the longitude
    point_x = (tile_size / 2 + lng * tile_size / 360.0) * num_tiles // tile_size

    # Convert the latitude to radians and take the sine
    sin_y = math.sin(lat * (math.pi / 180.0))

    # Calculate the y coordinate
    point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) *
               - (tile_size / (2 * math.pi))) * num_tiles // tile_size

    return int(point_x), int(point_y)


def get_xy2(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = (my * ORIGIN_SHIFT) / 180.0
    res = INITIAL_RESOLUTION / (2 ** zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py


def latlon2px(x1, y1, x2, y2, z):
    pos_1x, pos_1y = get_xy(x1, y1, z)
    pos_2x, pos_2y = get_xy(x2, y2, z)
    return pos_1x, pos_1y, pos_2x, pos_2y


def latlon2px2(x1, y1, x2, y2, z):
    pos_1x, pos_1y = get_xy2(x1, y1, z)
    pos_2x, pos_2y = get_xy2(x2, y2, z)
    return pos_1x, pos_1y, pos_2x, pos_2y


def get_region_size(x1, y1, x2, y2):
    len_x = abs(x2 - x1 + 1)
    len_y = abs(y2 - y1 + 1)
    return len_x, len_y
