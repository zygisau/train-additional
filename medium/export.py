import time
import ee


def export_lake(tile, bounding_box, image_name, image_folder, bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'], scale=10, fileFormat='GeoTIFF', crs='EPSG:4326', crs_transform=None, dimensions=None):
    multi_band_tile = tile.select(bands)

    task_multi_band_clouds = ee.batch.Export.image.toDrive(**{
        'image': multi_band_tile,
        'description': image_name,
        'folder': image_folder,
        'region': bounding_box.getInfo()['coordinates'],
        'scale': scale,
        'fileFormat': fileFormat,
        'crs': crs,
        'crsTransform': crs_transform,
        'shardSize': 2560,
        'maxPixels': 1e13,
        'dimensions': dimensions,
    })

    # starting tasks
    task_multi_band_clouds.start()
    i = 0
    while task_multi_band_clouds.active():
        i += 1
        dots = i % 4
        # print(f'Downloading file [{"." * dots}{" " * (3 - dots)}]\r', end="") with status task_multi_band_clouds.status
        print(
            f'Downloading file: {task_multi_band_clouds.status()["state"]} [{"." * dots}{" " * (3 - dots)}]\r', end="")
        time.sleep(1)

    print(
        f"Task finished with status: {task_multi_band_clouds.status()['state']}")
    if task_multi_band_clouds.status()['state'] == ee.batch.Task.State.FAILED:
        print('Task failed')
        print(task_multi_band_clouds.status()['error_message'])
