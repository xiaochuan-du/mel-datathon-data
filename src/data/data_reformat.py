import pandas as pd
import numpy as np
import geopandas as gpd
import math
import fiona
import rasterio
import glob
import os
import pickle
import affine6p
import matplotlib
import functools
import time
import datetime
import json
from matplotlib import pyplot as plt
from PIL import Image

import rasterio.mask as rmask
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping
from config import config_cls


config = config_cls[os.getenv('ENV', 'default')]

################################# Preprocessing Functions #################################
bucket = config.DATA_ROOT / 'interim' / 'sugar_files_FLI'
result_format = "jpeg" # or png

def preprocess_pngs():
    """Preprocess geo data and save intermediate results on harddisk. Run once (if files are generated, no need to run)
    """
    _ = get_corners()
    pngS_to_geotifS(png_folder=f"TCI_tiles", affine_option='4corners')
    return

################################ This is the main function ################################
# @functools.lru_cache()
def ROI_tifs(ROI, masked=True):
    """Based on input ROI and preprocess files, find relevant satellite images and clip them with ROI. 
    Save the clipped images on harddrive and return path informaiton and timestamp information

    Parameters
    ----------
    ROI : path to ROI json file, need to contain the below information (as required by front end)
        {"geometry": {"type": "Polygon",
    "coordinates": [[148.60709030303114, -20.540043246963264],
      [148.69607543743531, -20.539590412428996],
      [148.6865658493269, -20.595756032466892],
      [148.6275658455197,-20.606209452942387]]}}
    """
    # Prepare shared datasets
    # with open(ROI, "r") as fp:
    #     ROI=json.load(fp)
    ROI=json.loads(ROI)
    corners_geo, _, tif_list, xy_w_geo_wo_tif = load_preprocessed()
    ROI = Polygon(ROI["geometry"]["coordinates"])
    ROI_tiles_XY = pair_ROI_to_tiles(ROI, corners_geo)
    ROI_tiles_XY = list(set(ROI_tiles_XY) - xy_w_geo_wo_tif)
    tasks = tif_list.loc[ROI_tiles_XY].reset_index()
    dates_unix = pd.to_datetime(tasks['date']).sort_values().unique().astype(np.int64) // 10**6
    
    if len(ROI_tiles_XY)==0:
        print("No tiles matched to the ROI. Please select another region.")
        # FIXME: what is the API if no corresponding tile is found?
        return

    tasks = tif_list[tif_list.index.isin(ROI_tiles_XY)] 

    if len(ROI_tiles_XY)==1:
        tif_infos = ROI_one_tile([mapping(ROI)], ROI_tiles_XY[0][0], ROI_tiles_XY[0][1], tasks['date'].values, corners_geo, save_format=result_format, masked=masked)

    else:
        print("Loading satellite images for the selected zone...")
        tif_infos = {}
        # For each tile, clip the tile with ROI and save as TIF
        for xy in ROI_tiles_XY:
            task = tasks.loc[xy]
            _ = ROI_one_tile([mapping(ROI)], xy[0], xy[1], task['date'].values, corners_geo, masked=masked)
        for unix_x in dates_unix:
            merged_array = merge_tiles(ROI_tiles_XY, unix_x, tif_folder = f"{bucket}/results/single")
            tif_infos[unix_x] = f"{bucket}/results/final/{unix_x}.{result_format}"
            save_img(merged_array, tif_infos[unix_x])
        print("Finished!")
    
    return {"png_path":tif_infos, "start_date":dates_unix[0], "end_date":dates_unix[-1], "all_dates":dates_unix}


def load_preprocessed():
    corners_geo = gpd.read_file(f"{bucket}/intermediate/tile_geo").set_index(['X','Y'])
    with open(f"{bucket}/intermediate/tile_corners.pkl", "rb") as f:
        corners_coords = pickle.load(f).set_index(['X','Y'])
    tif_list = ls_images(f"{bucket}/intermediate/geotifs/*.tif").sort_values(['x','y','date']).set_index(['x','y'])
    xy_w_geo_wo_tif = set(corners_geo.index.unique().values) - set(tif_list.index.unique().values)
    return corners_geo, corners_coords, tif_list, xy_w_geo_wo_tif

################################# Preprocessing Functions #################################
def get_corners(folder="geometries"):
    """List all geojson files in the folder of GCS, return the four corners for all files in a pd.DataFrame indexed by tile X&Y."""    
    
    geofiles = glob.glob(f"{bucket}/{folder}/*.geojson")

    Xs = pd.Series([path.split("geo-x")[1].split("-")[0] for path in geofiles]).astype(int)
    Ys = pd.Series([path.split("geo-x")[1].split("-y")[1].split(".")[0] for path in geofiles]).astype(int)
    all_corners = pd.Series([get_corner(geofile) for geofile in geofiles])

    all_corners = pd.concat([Xs, Ys, all_corners], axis=1)
    all_corners.columns = ["X", "Y", "corners"]
    
    all_corners['geometry'] = all_corners.apply(lambda row:Polygon(row["corners"]), axis=1)
    all_corners = gpd.GeoDataFrame(all_corners, geometry=all_corners["geometry"], crs={'init': 'epsg:4326'})
    
    all_corners[["X","Y","geometry"]].to_file(f"{bucket}/intermediate/tile_geo")
    with open(f"{bucket}/intermediate/tile_corners.pkl", "wb") as f:
        pickle.dump(all_corners[["X","Y","corners"]], f)
    
    return all_corners

def get_corner(geofile:str):
    """ Open geojson file from GCS corresponding to the xy of one tile, return the long/lat of the corners of tile

    Parameters
    ----------
    geofile : str
        geofile path

    Returns
    -------
    list of tuples
        Each tuple is the long/lat of TL, TR, BR and BL corner
    """
    with fiona.open(geofile, "r") as shapefile:
        tile_corner = [x["geometry"] for x in shapefile][0]["coordinates"][0]
    return tile_corner

def pngS_to_geotifS(png_folder="TCI_tiles", affine_option='4corners', mask_folder="masks"):
    """Convert all PNGs to geoTIFs with sugarcane field mask applied to the original png"""
    # Load preprocessed inputs and list pngs to convert
    with open(f"{bucket}/intermediate/tile_corners.pkl", "rb") as f:
        all_corners = pickle.load(f).set_index(["X", "Y"])
    tasks = ls_images(f"{bucket}/TCI_tiles/*.png")
    
    for index, row in tasks.iterrows():
        try:
            # Prepare inputs
            tile_x, tile_y, tile_date = row['x'], row['y'], row['date']
            png_name = f"{tile_x}-{tile_y}-TCI-{tile_date}.png"
            save_name = f"{tile_x}-{tile_y}-TCI-{tile_date}.tif"
            #field_mask_path = f"{bucket}/{mask_folder}/mask-x{tile_x}-y{tile_y}.png"

            # Open png file corresponding to the xy of tiles, return the array of raster values (3*512*512) with sugarcane field mask applied
            with rasterio.open(f"{bucket}/{png_folder}/{png_name}", "r") as src:
                array_data = src.read()# * load_filed_mask(field_mask_path)

            # Save array as geotif
            array_to_geotif(array_data, save_name, all_corners.loc[(tile_x, tile_y), "corners"][:4], affine_option=affine_option)
            
        except:
            print([index, row["x"], row["y"], row["date"]])
    return

################################# Geo Functions #################################
def ROI_one_tile(ROI, tile_x, tile_y, dates, corners_geo, source_folder="intermediate/geotifs", mask_folder="masks", save_format='tif', masked=True):
    """Clip all geoTIFs (timesteps) for an tile by the ROI polygon. 
       Save the clipped raster arrays as geotif.
    """

    # Clip ROI with tile square to get the intersection polygon bounds and corresponding rectangular corners
    #FIXME: Only single polygon is supported, complex polygon not supported yet
    inter_poly_bounds = Polygon(ROI[0]["coordinates"][0]).intersection(corners_geo.loc[(tile_x, tile_y)]).bounds
    intersection_corner = [[inter_poly_bounds[0], inter_poly_bounds[3]],
         [inter_poly_bounds[2], inter_poly_bounds[3]],
         [inter_poly_bounds[2], inter_poly_bounds[1]],
         [inter_poly_bounds[0], inter_poly_bounds[1]]
        ]

    # Use the mask and indices to select rasters of all TIFs
    results = {}

    # Find the indices of non-zero rasters on the rectangular intersection this tile and ROI
    source_path = f"{bucket}/{source_folder}/{tile_x}-{tile_y}-TCI-{dates[0]}.tif"
    ROI_full_array = ROI_on_geotif(ROI, source_path, False)[0]
    nonzeros = bbox2(ROI_full_array.sum(axis=0)) #FIXME: this will also fix the error with rasterio

    # Load sugarcane field mask
    if masked:
        field_mask_path = f"{bucket}/{mask_folder}/mask-x{tile_x}-y{tile_y}.png"
        field_mask_array = load_filed_mask(field_mask_path)[nonzeros[0]:nonzeros[1], nonzeros[2]:nonzeros[3]]
    else:
        field_mask_array = 1
        
    for x in dates:
        unix_x = int(time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple()) * 1000)
        source_path = f"{bucket}/{source_folder}/{tile_x}-{tile_y}-TCI-{x}.tif"

        with rasterio.open(source_path, "r") as src:
            polygon_array = src.read()
        polygon_array = polygon_array[:, nonzeros[0]:nonzeros[1], nonzeros[2]:nonzeros[3]] * field_mask_array

        if save_format=='tif':
            array_to_geotif(polygon_array, f"{tile_x}-{tile_y}-{unix_x}.tif", intersection_corner, 
            save_folder=f"results/single", affine_option='4corners')            
        elif save_format==result_format:
            save_img(polygon_array, f"{bucket}/results/final/{unix_x}.{result_format}")
        else:
            print("Save format not supported, aborting.")
            return

        results[unix_x] = f"results/final/{unix_x}.{result_format}"

    return results


def ROI_on_geotif(ROI, geotif_path, crop):
    """Clip one tile(geoTIF) by the ROI polygon. Return the cropped raster array and tif meta.
    
    Parameters
    ----------
    ROI : list of geojson polygons, e.g.:
        [{'type': 'Polygon',
        'coordinates': [[(148.1126437690792, -20.0084977141844666),
        (148.13147206605388, -20.004663808059437),
        (148.131814713494, -20.010831583258326),
        (148.11297164191616, -20.01114679490517)]]}]
    geotif_path : str
        full GCS path of the tif
    crop : boolean
        If true, return the clipped array; or else, return the whole array but with regions outside ROI set to nodata.
    save_clipped : None or str, optional
        If a str is given, save the result to the path defined by the str, by default None
    
    Returns
    -------
    np.array
        array of the clilpped image
    """
    # Open TIF and add mask using ROI polygon
    with rasterio.open(geotif_path, "r") as src:
        out_image, out_transform = rasterio.mask.mask(src, ROI, crop=crop)
        out_meta = src.meta

    # Update TIF meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    return out_image, out_meta

def pair_ROI_to_tiles(ROI, gdf):
    """For a given ROI, find its intersected tiles by sjoin with tile bounding box polygons.
    Return an np array of tile x&y indices.

    Parameters
    ----------
    ROI : shapely.geometry.Polygon
    gdf : geopandas.dataframe
    """    
    precise_matched_gdf = gdf[gdf.intersects(ROI)]
    
    return precise_matched_gdf.index.values


def array_to_geotif(array_data, tif_name, tile_corner, save_folder=f"intermediate/geotifs", affine_option='4corners'):
    
    """Convert an array into a geoTIF, which is a format required to intersect with ROI polygon.
    """
    if affine_option=='4corners':    
        origin = [[1,1], [1, array_data.shape[2]], [array_data.shape[1], array_data.shape[2]], [array_data.shape[1], 1]]
        convert = tile_corner
        trans_matrix = affine6p.estimate(origin, convert).get_matrix()
        transform = Affine(trans_matrix[0][1], trans_matrix[0][0], trans_matrix[0][2], 
                    trans_matrix[1][1], trans_matrix[1][0], trans_matrix[1][2])  

    elif affine_option=='4bounds': 
        west = min(tile_corner[0][0], tile_corner[3][0])
        south = min(tile_corner[2][1], tile_corner[3][1])
        east = max(tile_corner[1][0], tile_corner[2][0])
        north = max(tile_corner[0][1], tile_corner[1][1])
        bearing = GetBearing(tile_corner[0], tile_corner[1])
        transform = rasterio.transform.from_bounds(west, south, east, north, array_data.shape[1], array_data.shape[2]) #* Affine.rotation(bearing)
        #FIXME: to be further tested

    else:
        print(f"Affine option {affine_option} not supported...Aborting")
        return

    # Save array as geoTIF
    with rasterio.open(f"{bucket}/{save_folder}/{tif_name}", 'w', driver='GTiff', height=array_data.shape[1], 
                        width=array_data.shape[2], count=3, dtype=array_data.dtype, crs='EPSG:4326', 
                        transform=transform) as dst:
        dst.write(array_data)

    return transform


# Missing data not handled
def merge_tiles(ROI_tiles_XY, unix_date, tif_folder = f"{bucket}/results/single"):
    src_files_to_mosaic = []

    for xy in ROI_tiles_XY:
        tifpath = f"{tif_folder}/{xy[0]}-{xy[1]}-{unix_date}.tif"
        src = rasterio.open(tifpath)
        src_files_to_mosaic.append(src)

    mosaic, _ = merge(src_files_to_mosaic)
    
    return mosaic

################################# Utility Functions #################################
def ls_images(path, flag="TCI-"):
    """List images in a gcs/local path"""
    imgS = glob.glob(path)
    imgS = [x.split("/")[-1].split(".")[0] for x in imgS]
    Xs = pd.Series([x.split("-")[0] for x in imgS]).astype(int)
    Ys = pd.Series([x.split("-")[1] for x in imgS]).astype(int)
    Dates = pd.Series([x.split(flag)[1] for x in imgS])
    
    tasks = pd.concat([Xs, Ys, Dates], axis=1)
    tasks.columns = ["x","y","date"]
    return tasks
        
def bbox2(np_array):
    """Return the indices of the bounding box of non-zero elements in a np array"""
    rows = np.any(np_array, axis=1)
    cols = np.any(np_array, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [ymin, ymax+1, xmin, xmax+1]

def load_filed_mask(path):
    with rasterio.open(path, "r") as src:
        masked_array = src.read()[3]
    masked_array[masked_array>0] = 1
    return masked_array

def plot_an_array(x):
    fig, ax = plt.subplots()
    plt.imshow(np.transpose(x,(1,2,0)))
    return

def GetBearing(pointA, pointB):
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180Â° to + 180Â° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def swapLatLon(coord):
    return (coord[1],coord[0])

def save_img(input_array, save_path):
    """Transpose a (3 or 4, x, y) array into (x, y, 3 or 4) array and save it as an image.
    
    Parameters
    ----------
    input_array : [type]
        [description]
    save_path : [type]
        [description]
    """
    #matplotlib.image.imsave(save_path, np.transpose(input_array, (1, 2, 0)))
    #imsave(save_path, np.transpose(input_array, (1, 2, 0)))
    Image.fromarray(np.transpose(input_array, (1, 2, 0))).save(save_path)
