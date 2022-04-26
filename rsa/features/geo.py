import math
from os.path import exists
from haversine import haversine, Unit
from pandas import Series, read_csv
from geopandas import GeoDataFrame, GeoSeries, points_from_xy
from shapely.geometry import Point, MultiLineString
from ..constants import enums

def geo_is_gosafe(point :Point, gosafe_gdp :GeoDataFrame):
    """Compute for input  point if distance to gosafe location is smaller than 0.003

    Args:
        point (Point): point on EPS:4326
        gosafe_gdp (GeoDataFrame): GoSafe GeoDataFrame

    Returns:
        _type_: _description_
    """    
    if isinstance(point, Point):
        for line in gosafe_gdp.geometry:
            if (line.distance(point) < 1e-3 ) == True:
                return True
    return False

def geo_is_dcc(obj :any, dcc_gdp :GeoDataFrame):
    """Check point if belongs to any of the DCC Administrative areas

    Args:
        obj (Geometry): Point, MultiLineString
        dublin_shp (GeoDataFrame): dublin shape for the 5 administrative areas

    Returns:
        bool: Returns True if the point belogs to DCC
    """    
    if isinstance(obj, Point):
        point = GeoSeries(obj)
        for row in dcc_gdp.geometry:
            gs = GeoSeries(row)
            if point.intersects(gs)[0] == True:
                return True
    if isinstance(obj, MultiLineString):
        rp = Point(obj.representative_point().x, obj.representative_point().y)
        for row in dcc_gdp.geometry:
            gs = GeoSeries(row)            
            if gs.intersects(rp)[0] == True:
                return True

    return False

def geo_convert_gps(row):
    '''
    Convert gps coordinates from column gps
    Args:
        row (_type_): _description_

    Returns:
        tuple: rowId, latitude, longitude
    '''  
    
    gps_str = str(row['gps'])
    x , y = gps_str.split(',', 1)
    x = float(x) # Easting
    y = float(y) # Northing
    lon, lat = geo_epsg_900913_to_4326(x, y)
    return (row['id'], lat, lon)

def geo_epsg_900913_to_4326(x: float, y: float):
    '''
    Convert EPSG:900913 to EPS:4326
    Standarise RSA map gps location use EPSG:900913 (OR EPSG:3857) grid
    - https://epsg.org/home.html
    - https://epsg.org/crs_4326/WGS-84.html
    - https://epsg.org/crs_3857/WGS-84-Pseudo-Mercator.html?sessionkey=msgy6wxv94
    - https://www.iogp.org/wp-content/uploads/2019/09/373-07-02.pdf

    Args:
        x (float): Easting
        y (float): Northing

    Returns:
        tuple: longitued, latitude
    '''   
    lon = x * 180 / 20037508.34
    lat = (360 / math.pi) * math.atan(math.exp(y * math.pi / 20037508.34)) - 90
    return (lon, lat)


def haversine_distance(loc1,loc2, unit = Unit.KILOMETERS):
    """_summary_

    Args:
        loc1 (_type_): (53.346039995657755, -6.258880035966194) # (lat, lon) 
        loc2 (_type_): (53.346039995657755, -6.259380267947484) # (lat, lon)
        unit (_type_, optional): _description_. Defaults to Unit.KILOMETERS.

    Returns:
        float: Distance between the two locations
    """    
    return haversine(loc1, loc2, unit)


def geo_distance_to_closest_fire_station(loc1, fire_stations_gdf: GeoDataFrame):
    """Return distance in meters to the closest fire station
    Args:
        point (tuple): RSA casualy point (lat,long)
        fire_stations_gdf (GeoDataFrame): Fire Station GeoDataFrame

    Returns:
        Series: Station and Distance in meters
    """
    distance = 9999
    closest_station = None
    for _, station in fire_stations_gdf.iterrows():
        loc2 = (station['latitude'],station['longitude'])
        distance_to_point = haversine(loc1, loc2, Unit.METERS)
        if (distance > distance_to_point ) == True:
            distance = distance_to_point            
            closest_station = station['name']

    res = Series([distance, closest_station])
    return res


def total_number_accident_on_radius(geo_df: GeoDataFrame, row: Series, radius: int):
    loc1 = (row[enums.feature_names.LATITUDE], row[enums.feature_names.LONGITUDE] )
    count = 0
    iter = 0
    for row1 in geo_df.itertuples():
        loc2 = (row1.latitude, row1.longitude)
        distance = haversine(loc1, loc2, unit=Unit.METERS)
        count = count+1 if distance < radius else count;
        iter = iter+1
    return count

def geo_calculate_radius(geo_df: GeoDataFrame, save_to_file: str, distance = 500):
    """Calculate number of casualties occurance for each row in a radius of the input distance. If the file already exists calculation will be omitted.
    In order recalculate the points please delete the csv file.

    Args:
        geo_df (GeoDataFrame): RSA Dataframe with geo Point.
        save_to_file (str): File path to save
        distance (int, optional): Distance in meters. Defaults to 500
    """    

    # Save result to file to save calculation time. If required plase reomve the file to run distance calculation again it might take few minutes to complete
    if(exists(save_to_file)):
        print("load from file")
        #Load previously saved dataset. use existing index on file
        tmp_df = read_csv(save_to_file, index_col=0)
        geo_df = GeoDataFrame(tmp_df, geometry = points_from_xy(tmp_df.longitude, tmp_df.latitude)
    )
    else:
        print("Calculating radious 500m")
        geo_df['radius_500m'] = geo_df.apply(lambda x: total_number_accident_on_radius(geo_df, x, distance), axis=1)
        geo_df.to_csv(save_to_file)
        
    return geo_df



def run_gps_test():
    print("Testing epsg_900913_to_4326....")
    # Tests GPS conversion
    x, y = geo_epsg_900913_to_4326(-20037508.34, 20037508.34)
    assert x == -180.0
    assert round(y,8) == 85.05112878

    x, y = geo_epsg_900913_to_4326(0, 0)
    assert x == 0.0
    assert y == 0.0
    # CCT collegue: https://epsg.io/map#srs=900913&x=-696737.796923&y=7047282.851274&z=20&layer=osm
    x, y = geo_epsg_900913_to_4326(-696737.79692334, 7047282.85127419)
    assert round(x,8) == -6.25890212
    assert round(y,8) == 53.34613987

    print("Testing haversine_distance....")
    # Test distance from CCT Collegue to front Cafe
    # https://www.google.es/maps/place/CCT+College+Dublin/@53.3460352,-6.2594205,20z/data=!4m5!3m4!1s0x48670e84cfcc9cbf:0x689c7d1c132a0ddf!8m2!3d53.3460406!4d-6.2588753
    cct_loc = (53.346051204244965, -6.2588746715481625)
    cafe_1920 = (53.34607602324896, -6.259365515797901)
    assert round(haversine_distance(cct_loc, cafe_1920, Unit.METERS)) == 33





## Tests
run_gps_test()


__all__ = ['geo_convert_gps','geo_epsg_900913_to_4326', 'geo_is_dcc', 'geo_calculate_radius', 'geo_distance_to_closest_fire_station']


