# load gejson points
import ee
from geemap import geojson_to_ee, ee_export_image_collection_to_drive
import json

from export import export_lake


def add_ndsi(img):
    ndsi = img.normalizedDifference(['B3', 'B11']).rename('NDSI')
    return img.addBands(ndsi)


# authentificate
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

file = './random_points.geojson'
points = {}
with open(file) as f:
    points = json.load(f)

# for each point
for pointDict in points['features'][2:]:
    # point to rectangle
    point = geojson_to_ee(pointDict)
    roi = point.buffer(1_000).bounds()
    name_from_coords = f'{pointDict["geometry"]["coordinates"][0]}_{pointDict["geometry"]["coordinates"][1]}'

    # get data
    # filter by roi, date, cloudiness
    collection = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(roi) \
        .filter(ee.Filter.And(ee.Filter.calendarRange(2020, 2023, 'year'), ee.Filter.calendarRange(5, 10, 'month'))) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))

    count = int(collection.size().getInfo())
    collection = collection.toList(count)
    print(f'Found {count} images')
    for i in range(0, count):
        img = ee.Image(collection.get(i))
        date = img.date().format('YYYY-MM-dd').getInfo()
        img = img.clip(roi)
        export_lake(img, roi, f'{date}', f'samples224/{name_from_coords}',
                    crs='EPSG:3857', dimensions='224x224', scale=None)
