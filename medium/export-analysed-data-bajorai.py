
from typing import List
import numpy as np
import pandas as pd
import ee
import geopandas as gpd
import geemap
from export import export_lake
from utils.dotted import dotted
import eemont
from shapely.geometry import shape


def moving_median(series, col_name, n):
    """
    Calculate moving median
    :param df: dataframe
    :param n: window size
    :return: dataframe
    """
    df = pd.Series(series.rolling(n, min_periods=n).median(),
                   name=col_name)
    # first n values are NaN, so we need to fill them with the first available value
    df = df.bfill()
    return df


def get_median_name(band: str):
    return f'MM_{band}'


def proceed(bounds: ee.Geometry.Rectangle, date_range: [str, str], bands: List[str], pixel: [float, float], moving_steps: int):
    coll = ee.ImageCollection('COPERNICUS/S2')\
        .filterDate(date_range[0], date_range[1])\
        .filterBounds(bounds)

    # Create a region of interest (e.g., a point)
    point = ee.Geometry.Point([pixel[0], pixel[1]])

    # Create a chart of NDVI over time
    ts = coll.getTimeSeriesByRegion(
        geometry=point,
        bands=bands,
        reducer=[ee.Reducer.min()],
        scale=10
    )

    df = geemap.ee_to_pandas(ts)
    # add latitudes and longitudes new columns
    df['latitude'] = pixel[0]
    df['longitude'] = pixel[1]
    # add system:time_start
    df['system:time_start'] = coll.aggregate_array(
        'system:time_start').getInfo()

    df[df == -9999] = np.nan

    df_series = [moving_median(
        df[band], get_median_name(band), moving_steps) for band in bands]
    # add moving average to df
    df = df.join(df_series)
    return df, coll


def filter_data_by_median(bounds: ee.Geometry.Rectangle, date_range: [str, str], bands: List[str], points: [float, float], moving_steps: int):
    def preprocess(df: pd.DataFrame):
        # loop rows
        df['count'] = None
        for i, row in df.iterrows():
            count = 0
            # loop bands
            for band in bands:
                # if value is inside of the error range
                if row[band] >= row[f'MM_{band}']*error_range['lower'] and row[band] <= row[f'MM_{band}']*error_range['upper']:
                    count += 1

            df.at[i, 'count'] = count

        return df

    def add_count_band(image):
        # get count by system:time_start
        count = fc.filter(ee.Filter.eq('system:time_start', image.get(
            'system:time_start'))).first().get('count')
        # add count band to image
        return image.set('count', count)

    df, coll = proceed(bounds, date_range, bands, points, moving_steps)
    # count how many values are inside of the error range
    df = preprocess(df)
    df = df.filter(regex='count|latitude|longitude|system:time_start')
    y_median = df['count'].median()
    fc = geemap.pandas_to_ee(df)

    # add count band to coll
    coll = coll.map(add_count_band)

    # filter if count band is bigger than 8
    new_coll = coll.filter(ee.Filter.gte('count', y_median))
    return new_coll


def initialize():
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()
    print('Google Earth Engine initialized!')


def make_columns():
    _cols = {
        'FID_1': 'FID_1',
        'Shape_Leng': 'Shape_Leng',
        'Shape_Area': 'Shape_Area',
        'Pavadinima': 'Pavadinima',
        'Kodas': 'Kodas',
        'Savivaldyb': 'Savivaldyb',
        'Platuma__m': 'Platuma__m',
        'Ilguma__m_': 'Ilguma__m_',
        'VT_kodas': 'VT_kodas',
        'UBR': 'UBR',
        'Lon': 'Lon',
        'Lat': 'Lat',
        'NEAR_FID': 'NEAR_FID',
        'NEAR_DIST': 'NEAR_DIST',
        'NEAR_X': 'NEAR_X',
        'NEAR_Y': 'NEAR_Y',
        'T35UMB_201': 'T35UMB_201',
        'Distance': 'Distance',
        'ID_new': 'ID_new',
        'xMin': 'xMin',
        'xMax': 'xMax',
        'yMin': 'yMin',
        'yMax': 'yMax',
        'geometry': 'geometry',
    }
    return dotted(_cols)


if __name__ == '__main__':
    initialize()
    # cols = make_columns()

    error_range = {
        "lower": 0.8,
        "upper": 1.2
    }
    # path = '/Users/zygimantas/Documents/sources/geo/lt_data/357_Ezerai_tvenkiniai_polygon_proj.shp'
    # cleaned_df = gpd.read_file(path)
    # lakes = [
    #     cleaned_df[cleaned_df[cols.Pavadinima] == 'Padvarių tvenkinys'],
    #     cleaned_df[cleaned_df[cols.Pavadinima] == 'Arimaičių ežeras'],
    #     cleaned_df[cleaned_df[cols.Pavadinima] == 'KAUNO MARIOS'],
    #     cleaned_df[cleaned_df[cols.Pavadinima] == 'Dusia'],
    #     cleaned_df[cleaned_df[cols.Pavadinima] == 'Asveja(Dubingių ežeras)']
    # ]
    # lakes = pd.concat(lakes, ignore_index=True)
    # lakes = gpd.GeoDataFrame(lakes)
    date_range = ['2019-01-01', '2023-12-30']
    bands = ['B1', 'B5', 'B6', 'B7', 'B8', 'B8A',
             'B9', 'B11', 'B12']
    # points = [
    #     21.2562, 55.9294
    #     # 21.2426, 55.9290
    # ]
    moving_steps = 20

    # Define the GeoJSON FeatureCollection
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Polygon", "coordinates": [[[25.255466, 54.758832], [25.266452, 54.758832], [
                25.266452, 54.753235], [25.255466, 54.753235], [25.255466, 54.758832]]]},
            {
                "type": "Feature",
                "properties": {
                    "system:index": "1"
                },
                "geometry": {
                    "coordinates": [
                        25.25826,
                        54.75757
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

    filtered_coll: ee.ImageCollection = filter_data_by_median(
        bounds, date_range, bands, points, moving_steps)

    print("Collection successfully filtered")

    dois = [ee.Date('2019-09-01'), ee.Date('2023-09-01')]
    samples = [
        # 2019
        filtered_coll.filter(ee.Filter.And(ee.Filter.calendarRange(
            2019, 2020, 'year'), ee.Filter.calendarRange(3, 10, 'month'))).sort('CLOUDY_PIXEL_PERCENTAGE'),
        filtered_coll.filter(ee.Filter.And(ee.Filter.calendarRange(
            2022, 2023, 'year'), ee.Filter.calendarRange(3, 10, 'month'))).sort('CLOUDY_PIXEL_PERCENTAGE'),

        # exact date 2023-12-03
        # filtered_coll.filterDate(dois[1], dois[1].advance(1, 'day')),
        # # 2019
        # filtered_coll.filter(ee.Filter.calendarRange(
        #     2019, 2020, 'year')).sort('CLOUDY_PIXEL_PERCENTAGE'),
        # # 2020
        # filtered_coll.filter(ee.Filter.calendarRange(
        #     2020, 2021, 'year')).sort('CLOUDY_PIXEL_PERCENTAGE'),
        # # 2021
        # filtered_coll.filter(ee.Filter.calendarRange(
        #     2021, 2022, 'year')).sort('CLOUDY_PIXEL_PERCENTAGE'),
        # # 2022
        # filtered_coll.filter(ee.Filter.calendarRange(
        #     2022, 2023, 'year')).sort('CLOUDY_PIXEL_PERCENTAGE'),
        # # 2023 least cloudless
        # filtered_coll.filter(ee.Filter.calendarRange(
        #     2023, 2024, 'year')).sort('CLOUDY_PIXEL_PERCENTAGE')
    ]

    sampled_images = [
        (sample.first(), f'{i}') for i, sample in enumerate(samples)
    ]
    print("sample done")

    for img, name in sampled_images:
        print(f'Exporting {name} image')
        # get date
        date = img.date().format('YYYY-MM-dd').getInfo()
        img = img.clip(rect)
        export_lake(img, rect, f'{date}', 'analysed_final/bajorai',
                    scale=10, crs='EPSG:3857')
        export_lake(img, rect, f'{date}_rgb', 'analysed_final/bajorai', [
                    'B2', 'B3', 'B4'], scale=10, crs='EPSG:3857')
        print(f'Exported {name} image')
