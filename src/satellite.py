"""
Provides method to generate satellite images from coordinates.
"""
import shutil   # shutil will be used to copy the image to the local
import csv
import math
import requests  # The requests package allows use to call URLS
from os.path import exists

Z = 12  # Set the resolution (max at 15)
BASE_PATH = '../data/satellite_images/'
MP_ACCESS_TOK = 'pk.eyJ1Ijoic2lwcGVqdyIsImEiOiJja3pweWxsajUwZnBxMnBvYW1wNW53d3J2In0.Rp7u8dQ4xOsZGsT45yvvlA'
COORD_CSV = '../data/csv/subsampled_coordinates.csv'


def main():
    coords = read_from_csv()
    for lat_lng in coords:
        file_path = BASE_PATH + str(lat_lng[0]) + ',' + str(lat_lng[1]) + '.png'
        if not exists(file_path):
            tile = get_tile(lat_lng[0], lat_lng[1])
            img = get_img(tile)
            write_img(img, file_path)


def read_from_csv():
    with open(COORD_CSV) as csv_file:
        res = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            res.append([float(row[0]), float(row[1])])
        return res


"""
write_img
Write image to file path
"""
def write_img(img, file_path):
    with open(file_path, 'wb') as file_ptr:
        shutil.copyfileobj(img, file_ptr)


""""
Query image from mapbox.
"""
def get_img(tile):
    response = requests.get('https://api.mapbox.com/v4/mapbox.satellite/' +
                            str(Z) + '/' + str(tile[0]) + '/' + str(tile[1]) +
                            '@2x.pngraw?access_token=' + MP_ACCESS_TOK,
                            stream=True)
    response.raw.decode_content = True
    return response.raw


"""
Convert degrees to tiles
"""
def get_tile(lat_deg, lon_deg):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** Z
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


if __name__ == "__main__":
    main()
