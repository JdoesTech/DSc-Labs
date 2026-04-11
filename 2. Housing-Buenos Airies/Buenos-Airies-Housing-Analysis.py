# Import libraries here
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)

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

frame1 = wrangle("data/mexico-city-real-estate-1.csv")

files = glob("data/mexico-city-*.csv")
files=[f for f in files if "test" not in f]
files

df = pd.concat([wrangle(file) for file in files], ignore_index=True)
print(df.info())
df.head()


fig, ax = plt.subplots() 
ax.hist(df["price_aprox_usd"]) 
ax.set_xlabel("Price [$]")
ax.set_ylabel("Count")
ax.set_title("Distribution of Apartment Prices")


fig, ax = plt.subplots() 
ax.scatter(df["surface_covered_in_m2"], df["price_aprox_usd"]) 
ax.set_xlabel("Area [sq meters]")
ax.set_ylabel("Price [USD]")
ax.set_title("Mexico City: Price vs. Area")