# COVID-Mobility-and-US-economy
Prediction model that uses COVID-19 Mobility Data provided by Google Big Data Query as features for S&amp;P500 Close price. 
This was a project made specifically for the 2020 Columbia Data Science Society Hackathon.

# Project Overview 

* Cleaned and merged two data sets to find underlying insights
* Created a model using GradientBoosting Regressor

## Code and Resources Used 
**Python Version:** 3.8 
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn 
* Mobility Impact Data found on Google Big Query : 
https://console.cloud.google.com/marketplace/product/bigquery-public-datasets/covid19-public-data-program?project=cdss-hackathon-test&folder=&organizationId=
* Financial Data for the S&amp;P500 Close price: https://finance.yahoo.com/quote/%5EGSPC?p=%5EGSPC


## Data Overview
Mobility Data:
* Data contains 2,000,000+ rows and 14 columns:
* Columns (features) are:
* country_region_code
* country_region
* sub_region_1
* sub_region_2
* iso_3166_2_code
* census_fips_code
* date
* retail_and_recreation_percent_change_from_baseline
* grocery_and_pharmacy_percent_change_from_baseline
* parks_percent_change_from_baseline
* transit_stations_percent_change_from_baseline
* workplaces_percent_change_from_baseline
* residential_percent_change_from_baseline

S&P500 Data:
* Data contains 145 Rows and 7 Columns:
* Columns (features) are:
* Open
* High
* Low
* Close
* Adj Close
* Volume

Target Variable: 
 S&amp;P500 Close price
 
## Data Cleaning:
* Used data visualization to safely remove outliers that were most likely erroneous 
*	Dropped irrelevant features such as Country
*	Data imputation applied with segmented data based on State and Month

## EDA
The expoloraty data analysis was done to better visualize and understand data set before undergoing the model building process.

![alt text](https://github.com/kevin7303/COVID-Mobility-and-US-economy/blob/master/sp500.png "SP500")
![alt text](https://github.com/kevin7303/COVID-Mobility-and-US-economy/blob/master/west.png "west")
![alt text](https://github.com/kevin7303/COVID-Mobility-and-US-economy/blob/master/midwest.png "midwest")
![alt text](https://github.com/kevin7303/COVID-Mobility-and-US-economy/blob/master/south.png "south")
![alt text](https://github.com/kevin7303/COVID-Mobility-and-US-economy/blob/master/north.png "north")


## Model Building 
I wanted to create a model that would use the Mobility Impact Data to help predict the US Economy using the  S&amp;P500 Close price as proxy.

**Evaluation Metric**

The specific metric used to evaluate the model was Root Mean Squared Error


**Steps Taken**

Performed One hot encoding on the categorical variables in order to accomodate Sklearn Decision trees treatment of categorical variables as continuous

Used an assortment of simple and complex classification models to create a robust model to predict Closing Price.
Started with base model to evaluate the performance of an unoptimzed and unfitted model on the problem and later used RandomSearchCV to optimize the hyper parameters.
All of these were done using a 5 fold cross validation.

Models Used:
* **Gradient Boosting Regression** 
* **Random Forest Classifier** 

## Model Tuning
Tuning was done on all of the functions above to improve RMSE through RandomizedSearchCv.


**Model Parameters

* **Gradient Boosting Regressionr** - verbose=1, n_estimators = 200, max_depth = 10, learning_rate = 0.05
* **Random Forest Classifier** - verbose=1, n_jobs=-1, n_estimators = 200, min_samples_split= 2, max_depth= 10


## **Results**
# The best algorithm was Gradient Boosting Regressor with a test data rmse: 116.455


