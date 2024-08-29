import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import seaborn as sns
import math
from locations_coords import LOCATIONS
from conv_calculation import conv_factor_kc


def calc_manhattan_dist(lat1, long1, lat2, long2):
    return abs(lat2 - lat1) + abs(long2 - long1)

def add_distance_columns(df, locations):
    for col_name, coords in locations.items():
        df[col_name] = df.apply(
            lambda r: calc_manhattan_dist(r['lat'], r['long'], coords[0], coords[1]), axis=1)
    return df

def analyze_dist_col(df, columns, conv_factor_kc):
    
    for col in columns:
        df[f'{col}_km'] = df[col] * conv_factor_kc
        
        print(f'\nStats for {col}_km')
        print(df[f'{col}_km'].describe())
        
        print(f'Correlation {col}_km with price (in general) column equals: '
              + str(df[f'{col}_km'].corr(df['price'])))
        
        print(f'Correlation {col}_km with price by square foot column equals: ' 
              + str(df[f'{col}_km'].corr(df['price_by_sqft'])))

    return df

def plot_dist_analysis(df):
    
    bins_km = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    labels = ['0-5 km', '5-10 km', '10-20 km', '20-30 km', '30-40 km', '40-50 km',
              '50-60 km', '60-70 km', '70-80 km', '80-90 km']
    
    for col in ['dist_to_downtown_km', 'dist_to_bellevue_km', 'dist_to_northwest_seattle_km']:
        df[f'{col}_bins'] = pd.cut(df[col], bins=bins_km, labels=labels)
        
        location_name = col.replace('_km', '').replace('dist_to_', '').replace('_', ' ').title()
        
        plt.figure()
        sns.barplot(x=f'{col}_bins', y='price_by_sqft', data=df)
        plt.title(f'Price by square foot by distance to {location_name}')
        plt.xlabel('Distances')
        plt.ylabel('Price by Square Foot')
        plt.xticks(rotation=45)
        plt.show()
        
        plt.figure()
        sns.barplot(x=f'{col}_bins', y='price', data=df)
        plt.title(f'Price by distance to {location_name}')
        plt.xlabel('Distances')
        plt.ylabel('Price by Square Foot')
        plt.xticks(rotation=45)
        plt.show()
        
    return df








