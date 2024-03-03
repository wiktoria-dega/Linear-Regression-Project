# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:56:15 2024

@author: Wiktoria
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pymongo import MongoClient

df_houseprice = pd.read_csv(r"C:\Users\Wiktoria\Desktop\Python Basics\Projekt\Regresja liniowa\kc_house_data.csv")


#printing summary of all columns
pd.set_option('display.max_columns', None)

df_houseprice.info()

print(df_houseprice.describe())

print(df_houseprice.isna().any())
print(df_houseprice.isna().sum())

#Filling in missing data
#PRICE
plt.figure()
df_houseprice['price'].hist(bins=50)

df_houseprice['price'] = df_houseprice['price'].fillna(df_houseprice['price'].median())

#BATHROOMS
plt.figure()
df_houseprice['bathrooms'].hist(bins=50)

plt.figure()
df_houseprice['bathrooms'].plot(kind='box')
df_houseprice['bathrooms'] = df_houseprice['bathrooms'].fillna(df_houseprice['bathrooms'].median())


#FLOORS
plt.figure()
df_houseprice['floors'].hist(bins=30)
print(df_houseprice['floors'].mode()[0]) #most frequently 1.0
df_houseprice['floors'] = df_houseprice['floors'].fillna(df_houseprice['floors'].mode()[0])


#LIVING SPACE ABOVE GROUND LEVEL
plt.figure()
df_houseprice['sqft_above'].hist(bins=60)
df_houseprice['sqft_above'] = df_houseprice['sqft_above'].fillna(df_houseprice['sqft_above'].median())


#BASEMENT
plt.figure()
df_houseprice['sqft_basement'].hist(bins=25)
df_houseprice['sqft_basement'] = df_houseprice['sqft_basement'].fillna(df_houseprice['sqft_basement'].median())


#YR OF BUILT
plt.figure()
df_houseprice['yr_built'].hist(bins=40)
df_houseprice['yr_built'] = df_houseprice['yr_built'].fillna(method = 'ffill')


#LIVING SPACE OF NEAREST 15
plt.figure()
df_houseprice['sqft_living15'].hist(bins=35)

#median vs mean
#mean=1986,489
median_sqft_living15 = df_houseprice['sqft_living15'].median()
print(median_sqft_living15) #median=1840

df_houseprice['sqft_living15'] = df_houseprice['sqft_living15'].fillna(df_houseprice['sqft_living15'].median())

#PLOT AREA OF NEAREST 15
plt.figure()
df_houseprice['sqft_lot15'].hist(bins=15)
df_houseprice['sqft_lot15'] = df_houseprice['sqft_lot15'].fillna(df_houseprice['sqft_lot15'].median())


#LIVING SPACE
plt.figure()
df_houseprice['sqft_living'].hist(bins=100)
df_houseprice['sqft_living'] = df_houseprice['sqft_living'].fillna(df_houseprice['sqft_living'].median())


#PLOT AREA
plt.figure()
df_houseprice['sqft_lot'].hist(bins=50)
df_houseprice['sqft_lot'] = df_houseprice['sqft_lot'].fillna(df_houseprice['sqft_lot'].median())
print(df_houseprice.isna().sum())


#EDA

#TARGET
plt.figure()
print(df_houseprice['price'].hist(bins=50))
plt.figure()
df_houseprice['price'].plot(kind='box')

upper_bound = 4000000.0

df_houseprice = df_houseprice[df_houseprice['price'] <= upper_bound ]

plt.figure()
print(df_houseprice['price'].hist(bins=30))

plt.figure()
df_houseprice['price'].plot(kind='box')

df_houseprice['price_log'] = np.log(df_houseprice['price'])

plt.figure()
df_houseprice['price_log'].hist(bins=40)

plt.figure()
df_houseprice['price_log'].plot.box()

df_houseprice = df_houseprice[(df_houseprice['price_log'] <= 14.5) & (df_houseprice['price_log'] >= 11.5)]

df_houseprice['price_log'].hist(bins=40)

#df_houseprice['price_log'].plot.box()

#What is the range of price?

min_price = df_houseprice['price'].min()
max_price = df_houseprice['price'].max()
f'The lowest price for the house is {min_price} and the highest is {max_price}'

#Does living space affect the price?
plt.figure()
df_houseprice['sqft_living'].hist(bins=30)
plt.title('Living space by number of observations')

plt.figure()
df_houseprice['sqft_living'].plot.box()
plt.title('Living space')
plt.xlabel('living space')
plt.show()

living_space_price = df_houseprice.groupby(df_houseprice['sqft_living'])['price'].mean()

plt.figure()
living_space_price.plot(kind='line')
plt.title('Average price by living space')
plt.xlabel('living space')
plt.ylabel('price')

plt.figure()
plt.scatter(df_houseprice['sqft_living'], df_houseprice['price'], alpha=0.2)
plt.title('Price by living space')
plt.xlabel('living space')
plt.ylabel('price')

corr_living_space = df_houseprice['sqft_living'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_living_space}'

#Does plot area affect the price?
plt.figure()
df_houseprice['sqft_lot'].hist(bins=50)
plt.title('Plot area by number of observations')

plt.figure()
df_houseprice['sqft_lot'].plot.box()
plt.title('Plot area')
plt.xlabel('plot area')
plt.show()


plot_area_price = df_houseprice.groupby(df_houseprice['sqft_lot'])['price'].mean()

plt.figure()
plot_area_price.plot(kind='line')
plt.title('Average price by plot area')
plt.xlabel('plot area')
plt.ylabel('price')

plt.figure()
plt.scatter(df_houseprice['sqft_lot'], df_houseprice['price'], alpha=0.2)
plt.title('Price by plot area')
plt.xlabel('plot area')
plt.ylabel('price')

corr_plot_area = df_houseprice['sqft_lot'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_plot_area}'


#How the number of bedrooms and bathrooms affect the price?
plt.figure()
df_houseprice['bedrooms'].hist(bins=80)

number_of_records_per_bedroom = df_houseprice.groupby('bedrooms').size()
number_of_records_per_bedroom

plt.figure()
number_of_records_per_bedroom.plot(kind='bar')
plt.ylim(0, 11000)
plt.title('Number of bedrooms by group')
plt.xlabel('number of bedrooms')
plt.ylabel('observations (houses)')

number_of_bedrooms = df_houseprice.groupby('bedrooms')['price'].mean()

plt.figure()
number_of_bedrooms.plot(kind='line')
plt.title('Average price by number of bedrooms')
plt.xlabel('number of bedrooms')
plt.ylabel('price')

plt.figure()
plt.scatter(df_houseprice['bedrooms'], df_houseprice['price'], alpha=0.2)
plt.title('Price by number of bedrooms')
plt.xlabel('number of bedrooms')
plt.ylabel('price')

plt.figure()
df_houseprice['bedrooms'].plot.box()
plt.title('Number of bedrooms')

corr_bedrooms_price = df_houseprice['bedrooms'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_bedrooms_price}'


plt.figure()
df_houseprice['bathrooms'].hist(bins=80)

number_of_records_per_bathroom = df_houseprice.groupby('bathrooms').size()

plt.figure()
number_of_records_per_bathroom.plot(kind='bar')
plt.ylim(0, 11000)
plt.title('Number of bathroom by group')
plt.xlabel('number of bathroom')
plt.ylabel('observations (houses)')


plt.figure()
number_of_bathrooms = df_houseprice.groupby('bathrooms')['price'].mean()
number_of_bathrooms.plot(kind='line')
plt.title('Average price by number of bathrooms')
plt.xlabel('number of bathrooms')
plt.ylabel('price')

plt.figure()
plt.scatter(df_houseprice['bathrooms'], df_houseprice['price'], alpha=0.2)
plt.title('Price by number of bathrooms')
plt.xlabel('number of bathrooms')
plt.ylabel('price')

plt.figure()
df_houseprice['bathrooms'].plot.box()
plt.title('Number of bathrooms')

corr_bedrooms_price = df_houseprice['bathrooms'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_bedrooms_price}'


#Does the year of built affect the price?
plt.figure()
df_houseprice['yr_built'].hist(bins=30)
plt.title('Year of built by number of observations')

plt.figure()
df_houseprice['yr_built'].plot.box()
plt.title('Year of built by number of observations')

plt.figure()
plt.scatter(df_houseprice['yr_built'], df_houseprice['price'], alpha=0.1) 
plt.title('Does the year of built affect the price?')
plt.xlabel('year of built')
plt.ylabel('price of house')

#nowa kolumna
df_houseprice['decade'] = (df_houseprice['yr_built'] // 10) * 10
decade_price_mean = df_houseprice.groupby('decade')['price'].mean()


plt.figure()
decade_price_mean.plot(kind='bar') 
plt.title('Average house price by decade of built')
plt.xlabel('decade of built')
plt.ylabel('average price')

plt.figure()
decade_price_mean.plot(kind='line')
plt.title('Average price by decade of built')
plt.xlabel('decade of built')
plt.ylabel('price')

corr_decade_built = df_houseprice['decade'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_decade_built}'

#Have house prices changed over time?
df_houseprice['date'] = pd.to_datetime(df_houseprice['date'])

df_houseprice['date'].info()

#mean price for day
daily_price_mean = df_houseprice.groupby(df_houseprice['date'].dt.date)['price'].mean()

plt.figure()
daily_price_mean.plot(kind='line')
plt.title('Average house price over time')
plt.xlabel('date')
plt.ylabel('price')

#Are renovated houses more expensive?

renovated_houses = df_houseprice[df_houseprice['yr_renovated'] != 0]
non_renovated_houses = df_houseprice[df_houseprice['yr_renovated'] == 0]
avg_price_of_renovated = renovated_houses['price'].mean()
avg_price_of_non_renovated = non_renovated_houses['price'].mean()
median_of_renovated = renovated_houses['price'].median()
median_of_non_renovated = non_renovated_houses['price'].median()

f'Average price of renoavted houses is {avg_price_of_renovated}, and average price of non renovated is {avg_price_of_non_renovated}'
f'Median price of renovated houses is {median_of_renovated}, and non renovated houses is {median_of_non_renovated}'


plt.figure()
plt.hist(renovated_houses['price'], bins=30, alpha=0.5, label='Renovated')
plt.title('Prices for renovated houses')
plt.xlabel('price')


plt.figure()
plt.hist(non_renovated_houses['price'], bins=30, alpha=0.5, label='Non Renovated')
plt.title('Prices for non renovated houses')
plt.xlabel('price')


plt.figure()
renovated_houses['price'].plot(kind='density', label='Renovated')
non_renovated_houses['price'].plot(kind='density', label='Non-Renovated')
plt.title('Prices for Renovated and Non-Renovated Houses')
plt.xlabel('Price')
plt.legend()
plt.show()

plt.figure()
plt.boxplot([renovated_houses['price'], non_renovated_houses['price']], labels=['Renovated', 'Non-Renovated'])
plt.title('Prices of renovated and non renovated houses')
plt.xlabel('renovation Status')
plt.ylabel('price')


#Do additional factors like basement space increase the price?
houses_with_basement = df_houseprice[df_houseprice['sqft_basement'] != 0]
houses_without_basement = df_houseprice[df_houseprice['sqft_basement'] == 0]

avg_price_with_basement = houses_with_basement['price'].mean()
avg_price_without_basement = houses_without_basement['price'].mean()

f'Average price of house with basement is {avg_price_with_basement}, and without basement is {avg_price_without_basement}'

plt.figure()
houses_with_basement['price'].hist(bins=50)

plt.figure()
houses_without_basement['price'].hist(bins=50)

plt.figure()
plt.boxplot([houses_with_basement['price'], houses_without_basement['price']], labels=['with basement', 'without basement'])
plt.title('Prices of houses with basement and houses without basement')
plt.xlabel('presence of a basement')
plt.ylabel('price')

median_price_with_basement = houses_with_basement['price'].median()
median_price_with_basement = houses_with_basement['price'].median()
f'Median price of house with basement is {median_price_with_basement}, and without basement is {median_price_with_basement}'

Q1 = houses_with_basement['price'].quantile(0.25)
Q3 = houses_with_basement['price'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR #bounds for outliers
upper_bound = Q3 + 1.5 * IQR

outliers = houses_with_basement[(houses_with_basement['price'] < lower_bound) | (houses_with_basement['price'] > upper_bound)]

number_of_outliers = outliers.shape[0]
f'Numbers of outliers for group of houses with basement: {number_of_outliers}'

Q1_without = houses_without_basement['price'].quantile(0.25)
Q3_without = houses_without_basement['price'].quantile(0.75)

IQR_without = Q3_without - Q1_without

lower_bound_without = Q1_without - 1.5 * IQR #bounds for outliers
upper_bound_without = Q3_without + 1.5 * IQR

outliers_without = houses_without_basement[(houses_without_basement['price'] < lower_bound_without) | (houses_without_basement['price'] > upper_bound_without)]

number_of_outliers_without = outliers_without.shape[0]
f'Numbers of outliers for group of houses with basement: {number_of_outliers_without}'

#Does the number of floors correlate with the price of a property?
plt.figure()
df_houseprice['floors'].hist(bins=50)

grouped_floors = df_houseprice.groupby('floors')['price'].mean()

f'Average prices of houses by number of floors are: {grouped_floors}'

plt.figure()
grouped_floors.plot(kind='bar')
plt.title('Average prices of houses by number of floors')
plt.xlabel('number of floors')
plt.ylabel('price')

corr_floors = df_houseprice['floors'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_plot_area}'


# Does the total living space of the house above ground level affect the price of the house?
plt.figure()
df_houseprice['sqft_above'].hist(bins=70)


plt.figure()
df_houseprice['sqft_above'].plot.box()

plt.figure()
plt.plot(df_houseprice['sqft_above'], df_houseprice['price'])
plt.title('Price by living space of the house above ground level')
plt.xlabel('living space above ground level')
plt.ylabel('price')
plt.show()

plt.figure()
plt.scatter(df_houseprice['sqft_above'], df_houseprice['price'], alpha=0.3)

corr_living_space_above = df_houseprice['sqft_above'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_living_space_above}'


#Does the living space of indoor living spaces for the nearest 15 neighbors affect the price
#of a house? Does a similar correlation occur for the area of lots for the nearest 15 neighbors?
plt.figure()
df_houseprice['sqft_living15'].hist(bins=70)

plt.figure()
df_houseprice['sqft_living15'].plot.box()

plt.figure()
plt.plot(df_houseprice['sqft_living15'], df_houseprice['price'])
plt.title('Price by living space of indoor living spaces for the nearest 15 neighbors')
plt.xlabel('indoor living spaces for the nearest 15 neighbors')
plt.ylabel('price')
plt.show()

plt.figure()
plt.scatter(df_houseprice['sqft_living15'], df_houseprice['price'], alpha=0.3)

corr_living_15 = df_houseprice['sqft_living15'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_living_15}'


plt.figure()
print(df_houseprice['sqft_living15'].hist(bins=30))

plt.figure()
df_houseprice['sqft_living15'].plot(kind='box')

#Does the plot area of the nearest 15 neighbors affect the price?
plt.figure()
df_houseprice['sqft_lot15'].hist(bins=90)

plt.figure()
df_houseprice['sqft_lot15'].plot.box()

plt.figure()
plt.plot(df_houseprice['sqft_lot15'], df_houseprice['price'])
plt.title('Price by area of lots for the nearest 15 neighbors?')
plt.xlabel('area of lots for the nearest 15 neighbors?')
plt.ylabel('price')
plt.show()

plt.figure()
plt.scatter(df_houseprice['sqft_lot15'], df_houseprice['price'], alpha=0.3)

corr_lot_15 = df_houseprice['sqft_lot15'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_lot_15}'


#Are houses with water views more expensive?
plt.figure()
price_with_waterfront = df_houseprice[df_houseprice['waterfront'] == 1]
plt.scatter(price_with_waterfront['waterfront'], price_with_waterfront['price'], alpha=0.3)

corr_waterfront = df_houseprice['waterfront'].corr(df_houseprice['price'])
corr_waterfront

#How does the view rating correlate with the price of the house?
plt.figure()
df_houseprice['view'].hist(bins=60)

plt.figure()
df_houseprice['view'].plot.box()

price_by_view = df_houseprice.groupby('view')['price'].mean()

plt.figure()
price_by_view.plot(kind='line')
plt.title('Average price by view rate')
plt.xlabel('rate of view')
plt.ylabel('price')

corr_view_rate = df_houseprice['view'].corr(df_houseprice['price'])
corr_view_rate

#Is the better condition of the house related to a higher selling price?
plt.figure()
df_houseprice['condition'].hist(bins=80)

plt.figure()
df_houseprice['condition'].plot.box()

price_by_cond = df_houseprice.groupby('condition')['price'].mean()

plt.figure()
price_by_cond.plot(kind='line')
plt.title('Average price by condition rate')
plt.xlabel('rate of condition')
plt.ylabel('price')

condition_3_and_4 = df_houseprice[df_houseprice['condition'].isin([3.0, 4.0])]

plt.figure()
condition_3_and_4.boxplot(column='price', by='condition')
plt.title('Box plot of price by condition rate 3 and 4')

corr_cond_rate = df_houseprice['condition'].corr(df_houseprice['price'])


#Is the better grade of the house related to a higher selling price?
plt.figure()
df_houseprice['grade'].hist(bins=90)

plt.figure()
df_houseprice['grade'].plot.box()

price_by_grade = df_houseprice.groupby('grade')['price'].mean()

plt.figure()
price_by_grade.plot(kind='line')
plt.title('Average price by grade rate')
plt.xlabel('rate of grade')
plt.ylabel('price')

corr_grade_rate = df_houseprice['grade'].corr(df_houseprice['price'])


df_houseprice.drop(columns=['date', 'id'], inplace=True)

corr = df_houseprice.corr()
print(corr)

plt.figure(figsize=(25, 17))
sns.heatmap(corr, annot=True)

df_houseprice.drop(columns='sqft_living', inplace=True)

df_houseprice.drop(columns='price', inplace=True)

df_houseprice.drop(columns=['zipcode', 'lat', 'long'], inplace=True)

corr2 = df_houseprice.corr()
corr2

plt.figure(figsize=(25, 17))
sns.heatmap(corr2, annot=True)


#Zapis danych do MongoDB
'''client = MongoClient('mongodb://localhost:27017/')
db = client['house_data_regression_database']
collection = db['house_data'] 

data = df_houseprice.to_dict('records')

collection.insert_many(data)

for doc in collection.find():
    print(doc)
    '''

client = MongoClient('mongodb://localhost:27017/')
db = client['test_db']
collection = db['test_data'] 

data = df_houseprice.to_dict('records')

collection.insert_many(data)

for doc in collection.find():
    print(doc)
    
    
    



