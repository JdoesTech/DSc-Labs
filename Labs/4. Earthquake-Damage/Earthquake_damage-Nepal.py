import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from category_encoders import OrdinalEncoder
from IPython.display import VimeoVideo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.validation import check_is_fitted

"""
%load_ext sql
%sql sqlite:///../nepal.sqlite
select distinct(district_id) FROM id_map limit(10)
select count(*) from id_map where district_id=1
select count(*) from id_map where district_id=3
select distinct(i.building_id) as b_id, b.* , d.damage_grade  from id_map as i
join building_structure as b on i.building_id = b.building_id 
join building_damage as d on i.building_id= d.building_id
where i.district_id=3
limit 5
"""


#Wrangle data from collection
def wrangle(dbpath):
    conn= sqlite3.connect(dbpath)
    query= """
        select distinct(i.building_id) as b_id, b.* , d.damage_grade
        from id_map as i
        join building_structure as b on i.building_id = b.building_id 
        join building_damage as d on i.building_id= d.building_id
        where i.district_id=3
    """

    df= pd.read_sql(query, conn, index_col="b_id")
    drop_cols=[col for col in df.columns if "post_eq" in col]
    
    df["damage_grade"]=df["damage_grade"].str[-1].astype(int)
    df["severe_damage"]=(df["damage_grade"]>3).astype(int)
    
    drop_cols.append("damage_grade")
    drop_cols.append("count_floors_pre_eq")
    drop_cols.append( "building_id")
    
    df.drop(columns=drop_cols, inplace=True)

    return df

#load and wrangle data
df = wrangle("../nepal.sqlite")
df.head()

#Check class balance of target variable
fig, ax = plt.subplots() 
df["severe_damage"].value_counts(normalize=True).plot(
    kind="bar",
    ax=ax  
)
ax.set_xlabel("Severe Damage")
ax.set_ylabel("Relative Frequency")
ax.set_title("Kavrepalanchok, Class Balance");

#Plot the relationship between plinth area and severe damage using a boxplot
fig, ax = plt.subplots() 
sns.boxplot(x="severe_damage", y="plinth_area_sq_ft", data=df, ax=ax)
ax.set_xlabel("Severe Damage")
ax.set_ylabel("Plinth Area [sq. ft.]")
ax.set_title("Kavrepalanchok, Plinth Area vs Building Damage")

#Plot the relationship between roof type and severe damage using a bar plot
roof_pivot = pd.pivot_table(df, index="roof_type", values="severe_damage", aggfunc=np.mean).sort_values(by="severe_damage")
roof_pivot

#Creating X and y values for modeling
coll=[col for col in df.columns if col != "severe_damage"]
X = df[coll]
y = df["severe_damage"]
print("X shape:", X.shape)
print("y shape:", y.shape)

#Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

#Baseline model
acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))

#Encoded Logistic Regression model
model_lr = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LogisticRegression(max_iter=1000)
)
model_lr.fit(X_train, y_train)

#Calculate the accuracy of the model
lr_train_acc = accuracy_score(y_train, model_lr.predict(X_train))
lr_val_acc = accuracy_score(y_val, model_lr.predict(X_val))
print("Logistic Regression, Training Accuracy Score:", lr_train_acc)
print("Logistic Regression, Validation Accuracy Score:", lr_val_acc)

# Ordinally Encoded Decision Tree model with default hyperparameters
depth_hyperparams = range(1, 16)
training_acc = []
validation_acc = []
for d in depth_hyperparams:
    model_dt = make_pipeline(
        OrdinalEncoder(),
        DecisionTreeClassifier(max_depth=d, random_state=42)
    )
    model_dt.fit(X_train, y_train)
    training_acc.append(model_dt.score(X_train, y_train))
    validation_acc.append(model_dt.score(X_val, y_val))
    
    
# Store validation accuracy scores as a Series 
submission = pd.Series(validation_acc, index=depth_hyperparams)
submission

# Plot of the training accuracy on the axes object
fig, ax = plt.subplots() t
ax.plot(depth_hyperparams, training_acc, label="training")
#  Plot of the validation accuracy on the same axes object
ax.plot(depth_hyperparams, validation_acc, label="validation") 
ax.set_xlabel("Max Depth")
ax.set_ylabel("Accuracy Score")
ax.set_title("Validation Curve, Decision Tree Model")
ax.legend()

#Best performing model
final_model_dt = make_pipeline(
    OrdinalEncoder(),
    DecisionTreeClassifier(max_depth=10, random_state=42)
)
final_model_dt.fit(X_train, y_train)

X_test = pd.read_csv("data/kavrepalanchok-test-features.csv", index_col="b_id")
X_test= X_test.loc[:, X_train.columns]
y_test_pred = final_model_dt.predict(X_test)
y_test_pred[:5]

#Create a series of the importance of individual features in prediction
features=X_test.columns
importance= final_model_dt.named_steps["decisiontreeclassifier"].feature_importances_
feat_imp= pd.Series(importance, index=features).sort_values(ascending=True)
feat_imp.head()

#Plot the feature importance
fig, ax = plt.subplots() 
feat_imp.plot(kind="barh", ax=ax)
ax.set_xlabel("Gini Importance")
ax.set_ylabel("Feature")
ax.set_title("Kavrepalanchok Decision Tree, Feature Importance")
fig.tight_layout()



