import pandas as pd
import matplotlib.pyplot as plt
import math
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import seaborn as sns

df_houseprice = pd.read_csv(r"C:\Users\Wiktoria\Desktop\Python Basics\Projekt\Regresja liniowa\kc_house_data.csv")

df_houseprice['sqft_living'].isna().any()
df_houseprice['sqft_living'].isnull().any()

df_houseprice['sqft_living'] = df_houseprice['sqft_living'].fillna(df_houseprice['sqft_living'].median())

df_houseprice['sqft_living'].isnull().any()
df_houseprice['sqft_living'].isna().any()

df_houseprice['price'].isna().any()
df_houseprice['price'].isnull().any()

df_houseprice['price'] = df_houseprice['price'].fillna(df_houseprice['price'].median())

df_houseprice['price'].isna().any()
df_houseprice['price'].isnull().any()

df_houseprice['price_by_sqft'] = df_houseprice['price'] / df_houseprice['sqft_living']

df_houseprice['price_by_sqft'].isnull().any()

df_houseprice['price_by_sqft'].describe()

#Downtown Seattle
seattle_downtown = (47.6062, -122.3321)

#Bellevue City
bellevue_city = (47.6104, -122.2007)

#Northwest Seattle
northwest_seattle = (47.6495, -122.4045)

locations = {
    'dist_to_downtown': seattle_downtown,
    'dist_to_bellevue': bellevue_city,
    'dist_to_northwest_seattle': northwest_seattle
    }


def calc_manhattan_dist(lat1, long1, lat2, long2):
    return abs(lat2 - lat1) + abs(long2 - long1)

def add_distance_columns(df, locations):
    for col_name, coords in locations.items():
        df[col_name] = df.apply(
            lambda r: calc_manhattan_dist(r['lat'], r['long'], coords[0], coords[1]), axis=1)
    return df

df_houseprice = add_distance_columns(df_houseprice, locations)


#conversion rate for Seattle:
#conv= 111*cos(lat)
angle_deg = 47.6
angle_rad = math.radians(angle_deg)
conv_factor_kc = 111 * math.cos(angle_rad) #1 degree of latitude difference-74.85km for King County

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

columns_to_convert = ['dist_to_downtown', 'dist_to_bellevue', 'dist_to_northwest_seattle']

df_houseprice = analyze_dist_col(df_houseprice, columns_to_convert, conv_factor_kc)

bins_km = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
labels = ['0-5 km', '5-10 km', '10-20 km', '20-30 km', '30-40 km', '40-50 km',
          '50-60 km', '60-70 km', '70-80 km', '80-90 km']

for col in ['dist_to_downtown_km', 'dist_to_bellevue_km', 'dist_to_northwest_seattle_km']:
    df_houseprice[f'{col}_bins'] = pd.cut(df_houseprice[col], bins=bins_km, labels=labels)
    
    location_name = col.replace('_km', '').replace('dist_to_', '').replace('_', ' ').title()
    
    plt.figure()
    sns.barplot(x=f'{col}_bins', y='price_by_sqft', data=df_houseprice)
    plt.title(f'Price by square foot by distance to {location_name}')
    plt.xlabel('Distances')
    plt.ylabel('Price by Square Foot')
    plt.xticks(rotation=45)
    plt.show()
    
    plt.figure()
    sns.barplot(x=f'{col}_bins', y='price', data=df_houseprice)
    plt.title(f'Price by distance to {location_name}')
    plt.xlabel('Distances')
    plt.ylabel('Price by Square Foot')
    plt.xticks(rotation=45)
    plt.show()





