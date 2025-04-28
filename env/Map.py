import geopandas as gpd
import numpy as np
import pandas as pd 
import os 
from shapely.geometry import Point, box
from shapely import wkt


AREA_DISTRIBUTION_PATH = "./Area Distribution"

class Map(object): 
    def __init__(self,  corners : tuple = (9.02, 45.35, 9.32, 45.615), map_index : int = 0): 
        """
        

        Args:
            corners (tuple, optional): _description_. Defaults to (9.02, 45.35, 9.32, 45.615).
        """
        
        
        
        self.min_lat =  corners[0]
        self.min_lon = corners[1]
        self.max_lat = corners[2]
        self.max_lon = corners[3]

        self.areas_configurations = pd.read_csv(os.path.join(AREA_DISTRIBUTION_PATH,f"clustered_areas_r_{map_index}.csv")).drop(columns=['area_id'])
        self.areas_configurations["geometry"] = self.areas_configurations['geometry'].apply(wkt.loads)
        self.areas_configurations["centroid"] = self.areas_configurations["centroid"].apply(wkt.loads)


    def __len__(self):
        return len(self.areas_configurations)
    
    
    def __getitem__(self, index):
        return self.areas_configurations.iloc[index]
    
    
    
    def __iter__(self):
        for index, row in self.areas_configurations.iterrows():
            yield index, row
    
  
        

    def Get_Map_Bounds(self): 
        

        # Extract area types from the dictionary (order matters here)
        
        minx, miny = 9.02, 45.35  # Southwest corner
        maxx, maxy = 9.32, 45.615  # Northeast corner

        # Create a GeoDataFrame with a single rectangle (bounding box) in EPSG:4326
        bbox = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)
        return  bbox.total_bounds

    def get_max_distance_by_index(self,  area_index):
        """
        Given a GeoDataFrame `gdf` and an area index (or name),
        retrieve the corresponding row, calculate, and return the maximum distance.
        """
        # Retrieve the row corresponding to the given area_index
        row = self.areas_configurations.loc[area_index]
        max_distance = self.calculate_max_distance_from_centroid(row)
        return max_distance

    @staticmethod
    def calculate_max_distance_from_centroid(row):
        """
        Given a GeoDataFrame row with a polygon in the 'geometry' column,
        compute the maximum distance from the centroid to the polygon's exterior boundary.
        Uses the 'centroid' column if available; otherwise, computes it.
        """
        polygon = row['geometry']
        
        # Use provided centroid or compute it from the polygon
        center = row['centroid'] if 'centroid' in row else polygon.centroid
        
        # Get all coordinates from the polygon's exterior ring
        exterior_coords = list(polygon.exterior.coords)
        
        # Calculate the maximum distance from the centroid to any vertex on the exterior
        max_distance = max(center.distance(Point(x, y)) for x, y in exterior_coords)
        return max_distance
    @staticmethod
    def lat_lon_to_cartesian(lat, lon):
        """
        Convert latitude and longitude to 3D Cartesian coordinates.

        :param lat: Latitude in degrees.
        :param lon: Longitude in degrees.
        :return: Tuple (x, y, z) Cartesian coordinates.
        """
        # Convert degrees to radians
        lat = np.radians(lat)
        lon = np.radians(lon)

        # Earth's radius (mean radius)
        R = 6371.0  # in kilometers

        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        z = R * np.sin(lat)

        return x, y, z

    @staticmethod
    def cartesian_to_lat_lon(x, y, z):
        """
        Convert 3D Cartesian coordinates to latitude and longitude.

        :param x: X coordinate.
        :param y: Y coordinate.
        :param z: Z coordinate.
        :return: Tuple (lat, lon) in degrees.
        """
        R = np.sqrt(x**2 + y**2 + z**2)
        lat = np.arcsin(z / R)
        lon = np.arctan2(y, x)

        # Convert radians to degrees
        lat = np.degrees(lat)
        lon = np.degrees(lon)

        return lat, lon