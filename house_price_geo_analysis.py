import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import math
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import seaborn as sns

df_houseprice = pd.read_csv(r"C:\Users\Wiktoria\Desktop\Python Basics\Projekt\Regresja liniowa\kc_house_data.csv")

#Downtown Seattle
seattle_downtown = (47.6062, -122.3321)

#Coal Creek Natural Area (Park)
coal_park = (47.5593, -122.1488)

def calc_manhattan_dist(lat1, long1, lat2, long2):
    return abs(lat2 - lat1) + abs(long2 - long1)

#distance calculation
df_houseprice['dist_to_downtown'] = df_houseprice.apply(
    lambda r: calc_manhattan_dist(r['lat'], r['long'], seattle_downtown[0],
                                  seattle_downtown[1]), axis=1)
df_houseprice['dist_to_park'] = df_houseprice.apply(
    lambda r: calc_manhattan_dist(r['lat'], r['long'], coal_park[0],
                                  coal_park[1]), axis=1)

#conversion rate for Seattle:
#conv= 111*cos(lat)
angle_deg = 47.6
angle_rad = math.radians(angle_deg)
conv_factor_kc = 111 * math.cos(angle_rad) #1 degree of latitude difference-74.85km for King County Seattle

df_houseprice['dist_to_downtown_km'] = df_houseprice['dist_to_downtown'] * conv_factor_kc
df_houseprice['dist_to_park_km'] = df_houseprice['dist_to_park'] * conv_factor_kc

df_houseprice['dist_to_downtown_km'].describe()
df_houseprice['dist_to_park_km'].describe()

corr = df_houseprice['dist_to_downtown_km'].corr(df_houseprice['price'])
corr

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

df_houseprice['Price by Square Foot'] = df_houseprice['price'] / df_houseprice['sqft_living']

df_houseprice['Price by Square Foot'].isnull().any()

df_houseprice['Price by Square Foot'].describe()

corr_price_sq_foot = df_houseprice['dist_to_downtown_km'].corr(df_houseprice['Price by Square Foot'])
corr_price_sq_foot

df_houseprice['dist_to_downtown_km'].describe()

df_houseprice['distances'] = pd.cut(df_houseprice['dist_to_downtown_km'], bins=10)

plt.figure()
sns.barplot(x='distances', y='Price by Square Foot', data=df_houseprice)
plt.title('Zależność ceny od odległości')
plt.xlabel('Odległość')
plt.ylabel('Cena')
plt.show()

plt.figure()
sns.barplot(x='distances', y='price', data=df_houseprice)
plt.title('Zależność ceny od odległości')
plt.xlabel('Odległość')
plt.ylabel('Cena')
plt.show()