import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import math
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

df_houseprice = pd.read_csv(r"C:\Users\Wiktoria\Desktop\Python Basics\Projekt\Regresja liniowa\kc_house_data.csv")


df_houseprice['geometry'] = df_houseprice.apply(lambda x: Point(x['long'], x['lat']), axis=1)

df_houseprice['sqft_living'] = df_houseprice['sqft_living'].fillna(df_houseprice['sqft_living'].median())

df_houseprice['sqft_living'].isna().any()
df_houseprice['sqft_living'].isnull().any()
df_houseprice['sqft_living'].isnull().sum()
df_houseprice['price'].isna().any()
df_houseprice['price'].isnull().any()

df_houseprice['price'] = df_houseprice['price'].fillna(df_houseprice['price'].median())

df_houseprice['price'].isna().any()
df_houseprice['price'].isnull().any()

df_houseprice['sqft_living'].isnull().any()
df_houseprice['sqft_living'].isnull().sum()

bins = [0.0, 200.0, 300.0, 400.0, 500.0, float('inf')]
labels = ['Low budget: <$200', 'Mid-Range: $200-$300', 
          'High-End: $300-$400', 'Premium: $400-$500', 
          'Luxury: >$500']

df_houseprice['Price by Square Foot'] = df_houseprice['price'] / df_houseprice['sqft_living']

df_houseprice['Price by Square Foot'].isnull().any()

df_houseprice['Price by Square Foot'].describe()

df_houseprice['Price Range (by square foot)'] = pd.cut(df_houseprice['Price by Square Foot'], bins=bins, labels=labels)

gdf = gpd.GeoDataFrame(df_houseprice, geometry='geometry', crs='EPSG:4326')

colors = {'Low budget: <$200': 'blue', 'Mid-Range: $200-$300': 'green',
          'High-End: $300-$400': 'yellow', 'Premium: $400-$500': 'orange',
          'Luxury: >$500': 'red'}
sizes = {'Low budget: <$200': 30, 'Mid-Range: $200-$300': 40,
          'High-End: $300-$400': 50, 'Premium: $400-$500': 60,
          'Luxury: >$500': 70}

fig, ax = plt.subplots(figsize=(10, 10))

for category in labels:
    subset = gdf[gdf['Price Range (by square foot)'] == category]
    subset.plot(ax=ax, color=colors[category], markersize=subset['Price Range (by square foot)'].map(sizes),
                alpha=0.7, edgecolor='black', label=category)


ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
#ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.DarkMatter)

ax.set_xlim(gdf.total_bounds[0] - 0.1, gdf.total_bounds[2] + 0.1)
ax.set_ylim(gdf.total_bounds[1] - 0.1, gdf.total_bounds[3] + 0.1)

#ax.set_aspect('auto')

ax.set_title('Map of King County: Property Prices by Square Foot')
ax.legend(title='Price Range (by square foot)', loc='best')
