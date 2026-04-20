host = "I connected my script to the mongo server using this host address"

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pytz
from pprint import PrettyPrinter
import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

#For better formatting of MongoDB query results
pp = PrettyPrinter(indent=2)

#Connect to : MongoDB, collection in database
client = MongoClient(host=host, port=27017)
db = client["air-quality"]
dar = db["dar-es-salaam"]
#Get distinct sites and measurements
sites = dar.distinct("metadata.site")
sites

#Number of readings per site
result = dar.aggregate(
    [
        {"$group":{"_id":"$metadata.site", "count":{"$count": {}}}}
    ]
)
readings_per_site = list(result)
readings_per_site

#Get distinct measurements
dar.distinct("metadata.measurement")

result = dar.find_one({})
pp.pprint(result)

#Wrangle data from collection
def wrangle(collection):
    result = collection.find(
        {"metadata.site": 11, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )
    y= pd.DataFrame(result).set_index("timestamp")
    y.index= y.index.tz_localize("UTC").tz_convert("Africa/Dar_es_Salaam")
    y=y[y["P2"]<100]
    y=y["P2"].resample("1H").mean().fillna(method="ffill")
    return y
#Load and wrangle data
y = wrangle(dar)
print(type(y))

#Plot PM2.5 levels per day
fig, ax = plt.subplots(figsize=(15, 6))
y.plot(xlabel="Date" ,ylabel="PM2.5 Level" , title="Dar es Salaam PM2.5 Levels", ax=ax)

#Plot 7-day rolling average of PM2.5 levels per day
fig, ax = plt.subplots(figsize=(15, 6))
y.rolling(168).mean().plot(xlabel="Date",ylabel= "PM2.5 Level", title="Dar es Salaam PM2.5 Levels, 7-Day Rolling Average" ,ax=ax)

#Plot the AutoCorrelation Function plot
# This is done to determine where there is significant correlation between the PM2.5 levels at different lags 
# This helps in selecting the appropriate lag order for the AutoRegressive model.
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax=ax)
ax.set_xlabel("Lag [hours]")
ax.set_ylabel("Correlation Coefficient")
ax.set_title("Dar es Salaam PM2.5 Readings, ACF")

#Plot Partial AutoCorrelation Function plot
#This is done to determine the direct correlation between PM2.5 levels at different lags
fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax=ax)
ax.set_xlabel("Lag [hours]")
ax.set_ylabel("Correlation Coefficient")
ax.set_title("Dar es Salaam PM2.5 Readings, PACF")

#The difference between ACF and PACF:
# ACF shows the correlation between the time-series and its lags 
# PACF shows the correlation between the time-series and its lags after removing the effects of intermediate lags.


#Generate train and test sets
cutoff_test = int(len(y)*0.9)
y_train = y.iloc[:cutoff_test]
y_test = y.iloc[cutoff_test:]
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Baseline model: predict the mean of the training set for all time steps
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean]*len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print("Mean P2 Reading:", y_train_mean)
print("Baseline MAE:", mae_baseline)

#Evaluate AutoRegressive models with different lag orders (p) 
p_params = range(1, 31)

#Create empty list to store MAE values for each p
maes = []


for p in p_params:
    #build model and predict on respective training set
    model = AutoReg(y_train, lags=p).fit()
    y_pred = model.predict().dropna()
    #Get the MAE
    mae = mean_absolute_error(y_train.iloc[p:], y_pred)
    #Append the MAE to the list
    maes.append(mae)


mae_series = pd.Series(maes, name="mae", index=p_params)
mae_series.head()

#find the best performing model
best_p = mae_series.idxmin()
best_model = AutoReg(y_train, lags=best_p).fit()
print(type(best_model))
#Calculate the residuals of the best model on the training set
y_train_resid = best_model.resid
y_train_resid.name = "residuals"
y_train_resid.head()

#Plot the distribution of the residuals 
fig, ax = plt.subplots()
plt.hist(y_train_resid)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Best Model, Training Residuals")
plt.grid(True)
plt.show()

#Plot the ACF of the residuals to check for any remaining autocorrelation
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid, ax=ax)
ax.set_xlabel("Lag [hours]")
ax.set_ylabel("Correlation Coefficient")
ax.set_title("Dar es Salaam, Training Residuals ACF")

#Walk forward validation: 
# This is done to evaluate the model's performance while simulating a real-world scenario where predictions are made over time.
y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model=AutoReg(history, lags=26).fit()
    next_pred=model.forecast()
    y_pred_wfv= y_pred_wfv.append(next_pred)
    history= history.append(y_test[next_pred.index])
        
y_pred_wfv.name = "prediction"
y_pred_wfv.index.name = "timestamp"
y_pred_wfv.head()
#Calculate the MAE of the walk forward validation predictions
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (walk forward validation):", round(test_mae, 2))

#Plot the actual vs predicted PM2.5 levels for the test set
df_pred_test = pd.DataFrame(
    {"y_test": y_test, "y_pred_wfv": y_pred_wfv}
)
df_pred_test.index.name = "timestamp"
df_pred_test.columns = ["y_test", "y_pred_wfv"]
fig = px.line(df_pred_test, labels={"value": "PM2.5"})
fig.update_layout(
    title="Dar es Salaam, WFV Predictions",
    xaxis_title="Date",
    yaxis_title="PM2.5 Level",
)

