
from typing import List
import numpy as np
import pandas as pd
import ee
import geopandas as gpd
import geemap
from processors.native_mosaic_processor import get_native_mosaic_processor
from processors.mosaic_processor import get_mosaic_processor
from export import export_lake
from utils.dotted import dotted
import eemont
from shapely.geometry import shape


def initialize():
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()
    print('Google Earth Engine initialized!')


if __name__ == '__main__':
    initialize()

    # Define the GeoJSON FeatureCollection
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Polygon", "coordinates": [[[25.262461, 54.711222], [25.252054, 54.711222], [
                25.252054, 54.706239], [25.262461, 54.706239], [25.262461, 54.711222]]]},
            {
                "type": "Feature",
                "properties": {
                    "system:index": "1"
                },
                "geometry": {
                    "coordinates": [
                        25.25768,
                        54.70859
                    ],
                    "type": "Point"
                },
            }
        ]
    }
    # Convert GeoJSON to an Earth Engine FeatureCollection
    ee_feature_collection = ee.FeatureCollection(
        feature_collection['features'])
    # Get the first feature from the FeatureCollection
    rect = ee_feature_collection.filter(
        ee.Filter.eq("system:index", "0")).first()
    # Get the geometry from the feature
    rect = rect.geometry()
    # Convert the geometry to a Rectangle
    shape = shape(feature_collection['features'][0])
    rect = ee.Geometry.Rectangle(shape.bounds)
    bounds = shape.bounds
    maxx = bounds[2] + 0.01
    minx = bounds[0] - 0.01
    maxy = bounds[3] + 0.01
    miny = bounds[1] - 0.01
    bounds = ee.Geometry.Rectangle(
        minx, miny, maxx, maxy
    )

    points = feature_collection['features'][1]['geometry']['coordinates']

    eeProcessor = get_mosaic_processor()
    start_date, end_date = '2023-11-01', '2024-01-02'
    cloud_filter, cld_prb_thresh = 80, 40
    where = 'stadionas'
    img: ee.Image = eeProcessor(
        bounds, start_date, end_date, cloud_filter, cld_prb_thresh)

    print("Collection successfully filtered")

    # get date
    name = f'{where}_{start_date}_{end_date}_{cloud_filter}_{cld_prb_thresh}'
    # name = f'{where}_{start_date}_{end_date}_native'
    img = img.clip(rect)
    export_lake(img, rect, name, f'mosaicked/{where}', bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
                scale=10, crs='EPSG:3857')
