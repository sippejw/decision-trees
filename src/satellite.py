"""
Provides method to generate satellite images from coordinates.
"""
import shutil   # shutil will be used to copy the image to the local
import requests  # The requests package allows use to call URLS
import math

def main():
    Z = 22  # Set the resolution (max at 15)
    lat_lng = [43.640918, -79.371478]
    tile = deg2num(lat_lng[0], lat_lng[1], Z)

    r = requests.get('https://api.mapbox.com/v4/mapbox.satellite/' +
                     str(Z) + '/' + str(tile[0]) + '/' + str(tile[1]) +
                     '@2x.pngraw?access_token=pk.eyJ1Ijoic2lwcGVqdyIsImEiOiJja3pweWxsajUwZnBxMnBvYW1wNW53d3J2In0.Rp7u8dQ4xOsZGsT45yvvlA',
                     stream=True)
    print(r)
    file_path = '../data/satellite_images/' + str(tile[1]) + '.' + str(tile[0]) + '.png'
    with open(file_path, 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)


"""
Convert degrees to tiles
"""
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


if __name__ == "__main__":
    main()
