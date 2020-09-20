import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

#Import Data
data = pd.read_csv('covid-mobility.csv')
sp = pd.read_csv('SP500.csv')


#Covid Data Exploration
data.columns
columns = list(data.columns)
mobility = columns[7:]

#Subset US Data
data_us = data[data['country_region'] == 'United States']
data_us.shape

#Date Range
data_us.date.max
data_us.to_csv('data_us.csv')

#Number of Subregions in the Data
regions = data_us.sub_region_1.value_counts().index.tolist()
len(regions)

#Join SP500 Data with US Mobility Data by date
sp_close = sp.drop(columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1)
df = pd.merge(data_us, sp_close, left_on = 'date', right_on = 'Date', how = 'left')

#Drop Irrelevant Columns
df.drop(columns = ['Date', 'sub_region_2', 'iso_3166_2_code', 'census_fips_code', 'country_region_code', 'country_region'], axis = 1, inplace = True)

#Missing Data Imputation/ Row Removal
df = df[df.Close.notna()]
df = df[df.sub_region_1.notna()]
df.isna().sum()
var = df.iloc[:,2:]
list(var.columns)

df['month'] = pd.DatetimeIndex(df['date']).month
df.columns

#Imputation 
df['retail_and_recreation_percent_change_from_baseline']= df.groupby(['sub_region_1', 'month'])['retail_and_recreation_percent_change_from_baseline'].transform(lambda x: x.fillna(x.mean()))
df['grocery_and_pharmacy_percent_change_from_baseline']= df.groupby(['sub_region_1', 'month'])['grocery_and_pharmacy_percent_change_from_baseline'].transform(lambda x: x.fillna(x.mean()))
df['parks_percent_change_from_baseline']= df.groupby(['sub_region_1', 'month'])['parks_percent_change_from_baseline'].transform(lambda x: x.fillna(x.mean()))
df['transit_stations_percent_change_from_baseline']= df.groupby(['sub_region_1', 'month'])['transit_stations_percent_change_from_baseline'].transform(lambda x: x.fillna(x.mean()))
df['workplaces_percent_change_from_baseline']= df.groupby(['sub_region_1', 'month'])['workplaces_percent_change_from_baseline'].transform(lambda x: x.fillna(x.mean()))
df['residential_percent_change_from_baseline']= df.groupby(['sub_region_1', 'month'])['residential_percent_change_from_baseline'].transform(lambda x: x.fillna(x.mean()))

df.isna().sum()

#Save Cleaned Data
df.to_csv('data_clean.csv', index = False)


#Graph US regions
df_graph = pd.read_csv('data_clean.csv', index_col = 'date')

L = ['retail_and_recreation_percent_change_from_baseline',
'grocery_and_pharmacy_percent_change_from_baseline',
'parks_percent_change_from_baseline',
'transit_stations_percent_change_from_baseline',
'workplaces_percent_change_from_baseline',
'residential_percent_change_from_baseline']


ny = df_graph[df_graph['sub_region_1'] == "New York"].sort_index()
fl = df_graph[df_graph['sub_region_1'] == "Florida"].sort_index()
ca = df_graph[df_graph['sub_region_1'] == "California"].sort_index()
il = df_graph[df_graph['sub_region_1'] == "Illinois"].sort_index()


for x in L:
    a = ca[x].plot(color = 'orange')
    a.set_title(x)
    plt.show()

for x in L:
    a = il[x].plot()
    a.set_title(x)
    plt.show()

for x in L:
    a = fl[x].plot(color = 'green')
    a.set_title(x)
    plt.show()

for x in L:
    a = ny[x].plot(color = 'purple')
    a.set_title(x)
    plt.show()


