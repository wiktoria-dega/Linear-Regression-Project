import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pymongo import MongoClient
from house_price_geo_analysis import calc_manhattan_dist, add_distance_columns, analyze_dist_col, plot_dist_analysis
from locations_coords import LOCATIONS
from conv_calculation import conv_factor_kc
from datetime import datetime

df_houseprice = pd.read_csv(r"C:\Users\Wiktoria\Desktop\Python Basics\Projekt\Regresja liniowa\kc_house_data.csv")

pd.set_option('display.max_columns', None)

df_houseprice.info()

df_houseprice.describe()

df_houseprice.isna().any()
df_houseprice.isna().sum()

df_houseprice.drop(columns='id', inplace=True)

#Filling in missing data
#PRICE
plt.figure()
df_houseprice['price'].hist(bins=20)

df_houseprice['price'] = df_houseprice['price'].fillna(df_houseprice['price'].median())

#BATHROOMS
plt.figure()
df_houseprice['bathrooms'].hist(bins=30)

plt.figure()
df_houseprice['bathrooms'].plot(kind='box')
df_houseprice['bathrooms'].median()
df_houseprice['bathrooms'] = df_houseprice['bathrooms'].fillna(df_houseprice['bathrooms'].median())


#FLOORS
plt.figure()
df_houseprice['floors'].hist(bins=30)
df_houseprice['floors'].median()
df_houseprice['floors'].mode()[0]
df_houseprice['floors'] = df_houseprice['floors'].fillna(df_houseprice['floors'].mode()[0])


#LIVING SPACE ABOVE GROUND LEVEL
plt.figure()
df_houseprice['sqft_above'].hist(bins=60)
df_houseprice['sqft_above'] = df_houseprice['sqft_above'].fillna(df_houseprice['sqft_above'].median())


#BASEMENT
plt.figure()
df_houseprice['sqft_basement'].hist(bins=25)
df_houseprice['sqft_basement'] = df_houseprice['sqft_basement'].fillna(df_houseprice['sqft_basement'].mode()[0])


#YR OF BUILT
plt.figure()
df_houseprice['yr_built'].hist(bins=40)
df_houseprice['yr_built'] = df_houseprice['yr_built'].fillna(method = 'ffill')


#LIVING SPACE OF NEAREST 15
plt.figure()
df_houseprice['sqft_living15'].hist(bins=35)

df_houseprice['sqft_living15'].median()
df_houseprice['sqft_living15'].mean()
df_houseprice['sqft_living15'].mode()[0]

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
df_houseprice.isna().sum()

df_houseprice['price_by_sqft'] = df_houseprice['price'] / df_houseprice['sqft_living']
df_houseprice['price_by_sqft'].isnull().any()
df_houseprice['price_by_sqft'].describe()


col_to_convert = ['dist_to_downtown', 'dist_to_bellevue', 'dist_to_northwest_seattle']

df_houseprice = add_distance_columns(df_houseprice, LOCATIONS)
df_houseprice = analyze_dist_col(df_houseprice, col_to_convert, conv_factor_kc)
df_houseprice = plot_dist_analysis(df_houseprice)
#EDA

#TARGET
plt.figure()
df_houseprice['price'].hist(bins=50)
plt.title('Distribution of house price')
plt.xlabel('Price')
plt.ylabel('Count')

plt.figure()
df_houseprice['price'].plot(kind='box')
plt.title('Distribution of house price')
plt.xlabel('Price')
plt.ylabel('Count')

df_houseprice = df_houseprice[df_houseprice['price'] <= 4000000.0 ]

plt.figure()
df_houseprice['price'].hist(bins=50)
plt.title('Distribution of house price')
plt.xlabel('Price')
plt.ylabel('Count')

plt.figure()
df_houseprice['price'].plot(kind='box')
plt.title('Distribution of house price')
plt.xlabel('Price')
plt.ylabel('Count')

df_houseprice['price_log'] = np.log(df_houseprice['price'])

plt.figure()
df_houseprice['price_log'].hist(bins=40)
plt.title('Distribution of after logarithmized price')
plt.xlabel('Price Log')
plt.ylabel('Count')

plt.figure()
df_houseprice['price_log'].plot.box()


#What is the range of price?
df_houseprice['price'].min()
df_houseprice['price'].max()

#Does living space affect the price?
plt.figure()
df_houseprice['sqft_living'].hist(bins=30)
plt.title('Distribution of living space')
plt.xlabel('Living space')
plt.ylabel('Count')

plt.figure()
df_houseprice['sqft_living'].plot.box()
plt.title('Distribution of living space')
plt.xlabel('Living space')
  
df_houseprice = df_houseprice[df_houseprice['sqft_living'] <= 8000.0]

living_space_price = df_houseprice.groupby(df_houseprice['sqft_living'])['price'].mean()

plt.figure()
living_space_price.plot(kind='line')
plt.title('Average price by living space')
plt.xlabel('Living space')
plt.ylabel('Price')

plt.figure()
plt.scatter(df_houseprice['sqft_living'], df_houseprice['price'], alpha=0.2)
plt.title('Price by living space')
plt.xlabel('Living space')
plt.ylabel('Price')

corr_living_space = df_houseprice['sqft_living'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_living_space}'

#Does plot area affect the price?
plt.figure()
df_houseprice['sqft_lot'].hist(bins=30)
plt.title('Distribution of plot area')
plt.xlabel('Plot area')
plt.ylabel('Count')

plt.figure()
df_houseprice['sqft_lot'].plot.box()
plt.title('Distribution of plot area')
plt.xlabel('Plot area')
plt.ylabel('Count')

plot_area_price = df_houseprice.groupby(df_houseprice['sqft_lot'])['price'].mean()

plt.figure()
plot_area_price.plot(kind='line')
plt.title('Average price by plot area')
plt.xlabel('Plot area')
plt.ylabel('Price')

plt.figure()
plt.scatter(df_houseprice['sqft_lot'], df_houseprice['price'], alpha=0.2)
plt.title('Price by plot area')
plt.xlabel('plot area')
plt.ylabel('price')

corr_plot_area = df_houseprice['sqft_lot'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_plot_area}'

df_houseprice = df_houseprice[df_houseprice['sqft_lot'] <= 500000.0]

#How the number of bedrooms and bathrooms affect the price?
plt.figure()
df_houseprice['bedrooms'].hist(bins=80)

number_of_records_per_bedroom = df_houseprice.groupby('bedrooms').size()
number_of_records_per_bedroom

plt.figure()
number_of_records_per_bedroom.plot(kind='bar')
plt.ylim(0, 11000)
plt.title('Grouped Number of Bedrooms')
plt.xlabel('Number of bedrooms')
plt.ylabel('Count')

number_of_bedrooms = df_houseprice.groupby('bedrooms')['price'].mean()

plt.figure()
number_of_bedrooms.plot(kind='line')
plt.title('Average price by number of bedrooms')
plt.xlabel('Number of bedrooms')
plt.ylabel('Price')

plt.figure()
plt.scatter(df_houseprice['bedrooms'], df_houseprice['price'], alpha=0.2)
plt.title('Price by number of bedrooms')
plt.xlabel('Number of bedrooms')
plt.ylabel('Price')

plt.figure()
df_houseprice['bedrooms'].plot.box()
plt.title('Number of bedrooms')

corr_bedrooms_price = df_houseprice['bedrooms'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_bedrooms_price}'

df_houseprice = df_houseprice[df_houseprice['bedrooms'] <= 9.0 ]

plt.figure()
df_houseprice['bathrooms'].hist(bins=80)

number_of_records_per_bathroom = df_houseprice.groupby('bathrooms').size()

plt.figure()
number_of_records_per_bathroom.plot(kind='bar')
plt.ylim(0, 11000)
plt.title('Grouped number of bathrooms')
plt.xlabel('Number of bathroom')
plt.ylabel('Count')

plt.figure()
number_of_bathrooms = df_houseprice.groupby('bathrooms')['price'].mean()
number_of_bathrooms.plot(kind='line')
plt.title('Average price by number of bathrooms')
plt.xlabel('Number of bathrooms')
plt.ylabel('Price')

plt.figure()
plt.scatter(df_houseprice['bathrooms'], df_houseprice['price'], alpha=0.2)
plt.title('Price by number of bathrooms')
plt.xlabel('Number of bathrooms')
plt.ylabel('Price')

plt.figure()
df_houseprice['bathrooms'].plot.box()
plt.title('Number of bathrooms')

corr_bedrooms_price = df_houseprice['bathrooms'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_bedrooms_price}'

df_houseprice = df_houseprice[(df_houseprice['bathrooms'] <= 4.5) & (df_houseprice['bathrooms'] >= 0.75)]

#Does the year of built affect the price?
plt.figure()
df_houseprice['yr_built'].hist(bins=30)
plt.title('Distribution of years built')
plt.xlabel('Year of built')
plt.ylabel('Count')

plt.figure()
df_houseprice['yr_built'].plot.box()
plt.title('Distribution of years built')
plt.xlabel('Year of built')
plt.ylabel('Count')

plt.figure()
plt.scatter(df_houseprice['yr_built'], df_houseprice['price'], alpha=0.1) 
plt.title('Price by year of built')
plt.xlabel('Year of built')
plt.ylabel('Price of house')

#decade
df_houseprice['decade'] = (df_houseprice['yr_built'] // 10) * 10
decade_price_mean = df_houseprice.groupby('decade')['price'].mean()

plt.figure()
decade_price_mean.plot(kind='bar') 
plt.title('Average house price by decade of built')
plt.xlabel('Decade of built')
plt.ylabel('Average price')

plt.figure()
decade_price_mean.plot(kind='line')
plt.title('Average price by decade of built')
plt.xlabel('Decade of built')
plt.ylabel('Price')

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
plt.xlabel('Date')
plt.ylabel('Price')

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
plt.xlabel('Price')
plt.ylabel('Count')


plt.figure()
plt.hist(non_renovated_houses['price'], bins=30, alpha=0.5, label='Non Renovated')
plt.title('Prices for non renovated houses')
plt.xlabel('Price')
plt.ylabel('Count')

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
plt.xlabel('Renovation Status')
plt.ylabel('Price')

corr_renovated = df_houseprice['yr_renovated'].corr(df_houseprice['price'])
corr_renovated

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
plt.xlabel('Presence of a basement')
plt.ylabel('Price')

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

corr_basement = df_houseprice['sqft_basement'].corr(df_houseprice['price'])
corr_basement

std_basement = houses_with_basement['price'].std()
std_without_basement = houses_without_basement['price'].std()

cv_with_basement = std_basement / avg_price_with_basement
cv_with_basement

cv_without_basement = std_without_basement / avg_price_without_basement
cv_without_basement

#Does the number of floors correlate with the price of a property?
plt.figure()
df_houseprice['floors'].hist(bins=50)
plt.title('Distribution of floors')
plt.xlabel('Floors')
plt.ylabel('Count')

grouped_floors = df_houseprice.groupby('floors')['price'].mean()

f'Average prices of houses by number of floors are: {grouped_floors}'

plt.figure()
grouped_floors.plot(kind='bar')
plt.title('Average prices of houses by number of floors')
plt.xlabel('Number of floors')
plt.ylabel('Price')

corr_floors = df_houseprice['floors'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_plot_area}'

# Does the total living space of the house above ground level affect the price of the house?
plt.figure()
df_houseprice['sqft_above'].hist(bins=70)
plt.title('Distribution of total living space of the house above ground level')
plt.xlabel('Living space above ground level')
plt.ylabel('Count')

plt.figure()
df_houseprice['sqft_above'].plot.box()
plt.title('Distribution of total living space of the house above ground level')
plt.xlabel('Living space above ground level')
plt.ylabel('Count')

plt.figure()
plt.plot(df_houseprice['sqft_above'], df_houseprice['price'])
plt.title('Price by living space of the house above ground level')
plt.xlabel('Living space above ground level')
plt.ylabel('Price')

plt.figure()
plt.scatter(df_houseprice['sqft_above'], df_houseprice['price'], alpha=0.3)
plt.title('Price by living space of the house above ground level')
plt.xlabel('Living space above ground level')
plt.ylabel('Price')

corr_living_space_above = df_houseprice['sqft_above'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_living_space_above}'

df_houseprice = df_houseprice[df_houseprice['sqft_above'] <= 6000.0 ]

#Does the living space of indoor living spaces for the nearest 15 neighbors affect the price
#of a house? Does a similar correlation occur for the area of lots for the nearest 15 neighbors?
plt.figure()
df_houseprice['sqft_living15'].hist(bins=70)
plt.title('Distribution of indoor living spaces for the nearest 15 neighbors')
plt.xlabel('Indoor living spaces for the nearest 15 neighbors')
plt.ylabel('Count')


plt.figure()
df_houseprice['sqft_living15'].plot.box()
plt.title('Distribution of indoor living spaces for the nearest 15 neighbors')
plt.xlabel('Indoor living spaces for the nearest 15 neighbors')
plt.ylabel('Count')

plt.figure()
plt.plot(df_houseprice['sqft_living15'], df_houseprice['price'])
plt.title('Price by living space of indoor living spaces for the nearest 15 neighbors')
plt.xlabel('Indoor living spaces for the nearest 15 neighbors')
plt.ylabel('Price')

plt.figure()
plt.scatter(df_houseprice['sqft_living15'], df_houseprice['price'], alpha=0.3)

corr_living_15 = df_houseprice['sqft_living15'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_living_15}'

df_houseprice = df_houseprice[df_houseprice['sqft_living15'] <= 5000.0]

#Does the plot area of the nearest 15 neighbors affect the price?
plt.figure()
df_houseprice['sqft_lot15'].hist(bins=50)
plt.title('Distribution of plot area for the nearest 15 neighbors')
plt.xlabel('Plot area for the nearest 15 neighbors')
plt.ylabel('Count')

plt.figure()
df_houseprice['sqft_lot15'].plot.box()
plt.title('Distribution of plot area for the nearest 15 neighbors')
plt.xlabel('Plot area for the nearest 15 neighbors')
plt.ylabel('Count')

plt.figure()
plt.plot(df_houseprice['sqft_lot15'], df_houseprice['price'])
plt.title('Price by area of lots for the nearest 15 neighbors')
plt.xlabel('Plot area for the nearest 15 neighbors')
plt.ylabel('Price')

plt.figure()
plt.scatter(df_houseprice['sqft_lot15'], df_houseprice['price'], alpha=0.3)

corr_lot_15 = df_houseprice['sqft_lot15'].corr(df_houseprice['price'])
f'Correlation coefficient equals {corr_lot_15}'

df_houseprice = df_houseprice[df_houseprice['sqft_lot15'] <= 233500.0]

#Are houses with water views more expensive?
plt.figure()
df_houseprice['waterfront'].hist(bins=30)
plt.title('Distribution of waterfront (properties with and without sea view)')
plt.xlabel('Waterfront')
plt.ylabel('Count')

price_waterfront_stats = df_houseprice.groupby('waterfront')['price'].agg(['mean', 'median'])
price_waterfront_stats

corr_waterfront = df_houseprice['waterfront'].corr(df_houseprice['price'])
corr_waterfront

#How does the view rating correlate with the price of the house?
plt.figure()
df_houseprice['view'].hist(bins=60)
plt.title('Distribution of rates of view')
plt.xlabel('View')
plt.ylabel('Count')

df_houseprice['view'].value_counts()

price_by_view = df_houseprice.groupby('view')['price'].mean()

plt.figure()
price_by_view.plot(kind='line')
plt.title('Average price by view rate')
plt.xlabel('Rate of view')
plt.ylabel('Price')

price_view_stats = df_houseprice.groupby('view')['price'].agg(['mean', 'median',
                                                              'std', 'min', 'max'])
price_view_stats

plt.figure()
sns.boxplot(x='view', y='price', data=df_houseprice)
plt.title('Price by rates of view')
plt.xlabel('Rate of View')
plt.ylabel('Price')

corr_view_rate = df_houseprice['view'].corr(df_houseprice['price'])
corr_view_rate
'''
#upper_bound_view = 1.0

#df_houseprice = df_houseprice[df_houseprice['view'] <= upper_bound_view ]
'''

#Is the better condition of the house related to a higher selling price?
plt.figure()
df_houseprice['condition'].hist(bins=80)
plt.title('Distribution of condition')
plt.xlabel('Condition')
plt.ylabel('Count')

df_houseprice['condition'].value_counts()

plt.figure()
df_houseprice['condition'].plot.box()
plt.title('Distribution of condition')
plt.xlabel('Condition')
plt.ylabel('Count')

price_by_cond = df_houseprice.groupby('condition')['price'].mean()

plt.figure()
price_by_cond.plot(kind='line')
plt.title('Average price by condition rate')
plt.xlabel('Rate of condition')
plt.ylabel('Price')

price_cond_stats = df_houseprice.groupby('condition')['price'].agg(['mean', 'median',
                                                                    'std', 'min', 'max'])
price_cond_stats

df_houseprice = df_houseprice[df_houseprice['condition'] > 1.0]

condition_3_and_4 = df_houseprice[df_houseprice['condition'].isin([3.0, 4.0])]

plt.figure()
condition_3_and_4.boxplot(column='price', by='condition')
plt.title('Price for Condition Rate 3 and 4')
plt.xlabel('Rate of condition')
plt.ylabel('Price')

plt.figure()
sns.boxplot(x='condition', y='price', data=df_houseprice)
plt.title('Price by condition')
plt.xlabel('Condition')
plt.ylabel('Price')

corr_cond_rate = df_houseprice['condition'].corr(df_houseprice['price'])
corr_cond_rate

#Is the better grade of the house related to a higher selling price?
plt.figure()
df_houseprice['grade'].hist(bins=90)
plt.title('Distribution of grade')
plt.xlabel('Grade')
plt.ylabel('Count')

df_houseprice['grade'].value_counts()

plt.title('Distribution of grade')
plt.xlabel('Grade')
plt.ylabel('Count')

plt.figure()
df_houseprice['grade'].plot.box()

price_by_grade = df_houseprice.groupby('grade')['price'].mean()

plt.figure()
price_by_grade.plot(kind='line')
plt.title('Average price by grade rate')
plt.xlabel('Rate of grade')
plt.ylabel('Price')

price_grade_stats = df_houseprice.groupby('grade')['price'].agg(['mean', 'median',
                                                                    'std', 'min', 'max'])
price_grade_stats

plt.figure()
sns.boxplot(x='grade', y='price', data=df_houseprice)
plt.title('Price by grade')
plt.xlabel('Grade')
plt.ylabel('Price')

corr_grade_rate = df_houseprice['grade'].corr(df_houseprice['price'])
corr_grade_rate


df_houseprice = df_houseprice[(df_houseprice['grade'] <= 11.0) & (df_houseprice['grade'] > 4.0)]

#NEW COLUMNS
#number of all rooms
df_houseprice['number_of_rooms'] = df_houseprice['bedrooms'] + df_houseprice['bathrooms']
price_by_rooms = df_houseprice.groupby('number_of_rooms')['price'].mean()

plt.figure()
price_by_rooms.plot(kind='line')

df_houseprice['number_of_rooms'].corr(df_houseprice['price'])

#age of house
current_year = datetime.now().year

df_houseprice['age_of_house'] = current_year - df_houseprice['yr_built']
price_by_age = df_houseprice.groupby('age_of_house')['price'].mean()

plt.figure()
price_by_age.plot(kind='line')
df_houseprice['age_of_house'].corr(df_houseprice['price'])

#scale of view
df_houseprice['view_scale'] = df_houseprice['waterfront'] + df_houseprice['view']
price_by_view_sc = df_houseprice.groupby('view_scale')['price'].mean()

plt.figure()
price_by_rooms.plot(kind='line')

df_houseprice['view_scale'].corr(df_houseprice['price'])

#condition scale
df_houseprice['cond_scale'] = df_houseprice['condition'] * df_houseprice['grade']
price_by_cond_sc = df_houseprice.groupby('cond_scale')['price'].mean()

plt.figure()
price_by_cond_sc.plot(kind='line')

df_houseprice['cond_scale'].corr(df_houseprice['price'])

df_houseprice['dist_to_downtown_km'].corr(df_houseprice['price'])


df_houseprice = df_houseprice.drop(columns=['date', 'zipcode', 'lat', 'long',
                                            'yr_renovated', 'price', 'yr_built',
                                            'price_by_sqft', 'dist_to_downtown',
                                            'dist_to_bellevue', 'dist_to_northwest_seattle',
                                            'decade', 'dist_to_downtown_km_bins',
                                            'dist_to_bellevue_km_bins',
                                            'dist_to_northwest_seattle_km_bins'])

correlation = df_houseprice.corr()
correlation

plt.figure()
sns.heatmap(correlation, annot=True)
plt.title('Correlation Matrix')


df_houseprice = df_houseprice.drop(columns=['sqft_above', 'dist_to_downtown_km',
                                            'view_scale', 'bedrooms'])

client = MongoClient('mongodb://localhost:27017/')
db = client['test_nowe_kol']
collection = db['test_nowe'] 

data = df_houseprice.to_dict('records')

collection.insert_many(data)

for doc in collection.find():
    print(doc)
    
'''
client = MongoClient('mongodb://localhost:27017/')
db = client['house_data_regression_database']
collection = db['house_data'] 

data = df_houseprice.to_dict('records')

collection.insert_many(data)

for doc in collection.find():
    print(doc)
'''
'''
client = MongoClient('mongodb://localhost:27017/')
db = client['testowa_regresja']
collection = db['test'] 

data = df_houseprice.to_dict('records')

collection.insert_many(data)

for doc in collection.find():   
    print(doc)
    '''
    

