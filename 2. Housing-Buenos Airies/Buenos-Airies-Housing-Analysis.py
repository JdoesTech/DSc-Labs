import warnings
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
warnings.simplefilter(action="ignore", category=FutureWarning)

# Load and wrangle data
def wrangle(filepath):
    df= pd.read_csv(filepath)
    
    mask_District=df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_apt=df["property_type"]== "apartment"
    mask_price=df["price_aprox_usd"]<100000
    
    df= df[mask_District & mask_apt & mask_price]
    
    low, high=df["surface_covered_in_m2"].quantile([0.1,0.9])
    mask_area=df["surface_covered_in_m2"].between(low, high)
    df=df[mask_area]
    
    df[["lat", "lon"]]=df["lat-lon"].str.split(',', expand=True).astype(float)
    df["borough"]= df["place_with_parent_names"].str.split('|', expand=True)[1]
    
    df.drop(columns=[column for column in df.columns if df[column].isnull().mean()>0.5], inplace=True)
    df.drop(columns=["operation", "currency", "place_with_parent_names", "properati_url", "property_type"], inplace=True)
    df.drop(columns=["price", "price_aprox_local_currency", "price_per_m2" ], inplace=True)
    df.drop(columns=[ "lat-lon"], inplace=True)
    

    return df

#Load datasets
frame1 = wrangle("data/mexico-city-real-estate-1.csv")
files = glob("data/mexico-city-*.csv")
files=[f for f in files if "test" not in f]
files

#Combine datasets into one dataframe
df = pd.concat([wrangle(file) for file in files], ignore_index=True)
print(df.info())
df.head()

#Exploratory Analysis
#plot distribution of target variable
fig, ax = plt.subplots() 
ax.hist(df["price_aprox_usd"]) 
ax.set_xlabel("Price [$]")
ax.set_ylabel("Count")
ax.set_title("Distribution of Apartment Prices")

#plot price vs area
fig, ax = plt.subplots() 
ax.scatter(df["surface_covered_in_m2"], df["price_aprox_usd"]) 
ax.set_xlabel("Area [sq meters]")
ax.set_ylabel("Price [USD]")
ax.set_title("Mexico City: Price vs. Area")

#plot price vs location
fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    width=300,
    height=500,
    color="price_aprox_usd",
)
fig.show()

# Prepare training data
features=["surface_covered_in_m2", "lat", "lon", "borough"]
X_train = df[features]
y_train = df["price_aprox_usd"]
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

#Establish baseline MAE
y_mean = y_train.mean()
y_pred_baseline = [y_mean]*len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline).round(2)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)

#Build Ridge Regression model; encoded and imputed
model =make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()
)
model.fit(X_train, y_train)

#load test set
X_test = pd.read_csv("data/mexico-city-test-features.csv")
print(X_test.info())
X_test.head()

#predict on test set
y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()

#create coefficients and feature importance values
coefficients = model.named_steps["ridge"].coef_
features = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index=features).sort_values(key=abs, ascending=True)
print(feat_imp)

# Horizontal bar plot of feature importances
fig, ax = plt.subplots()
print(feat_imp.head(10))
feat_imp[-10:].plot(kind="barh", ax=ax) 
ax.set_xlabel("Importance [USD]") 
ax.set_ylabel("Feature")
ax.set_title("Feature Importances for Apartment Price")


