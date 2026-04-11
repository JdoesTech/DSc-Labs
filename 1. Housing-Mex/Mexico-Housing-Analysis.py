#import packages
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

#load datasaet
df1 = pd.read_csv("data/brasil-real-estate-1.csv")
df1.head()

#drop null values
df1.dropna(inplace=True)
#split location data into latitude and longitude
df1[["lat", "lon"]] = df1["lat-lon"].str.split(",", expand=True).astype(float)
#extract state name
df1["state"] = df1["place_with_parent_names"].str.split("|", expand=True)[2]
#Modify price column into number format
df1["price_usd"] = df1["price_usd"].str.replace(r"[\$,]", "", regex=True).astype(float)
#drop unnecessary columns
df1.drop(columns=["lat-lon", "place_with_parent_names"], inplace=True)
#display cleaned dataset
df1.head()

#load second dataset
df2 = pd.read_csv("data/brasil-real-estate-2.csv")
df2.info()
df2.head()

#convert price column to USD
df2["price_usd"] = df2["price_brl"]/3.19

#drop unnecessary columns and null values
df2=df2.drop("price_brl", axis=1)
df2.dropna(inplace=True)

#concatenate the two datasets
df = pd.concat([df1, df2])
print("df shape:", df.shape)

#visualize the distribution of house prices
fig = px.scatter_mapbox(
    df,
    lat=df["lat"], 
    lon=df["lon"], 
    center={"lat": -14.2, "lon": -51.9},  # Map will be centered on Brazil
    width=600,
    height=600,
    hover_data=["price_usd"],  # Display price when hovering mouse over house
)
fig.update_layout(mapbox_style="open-street-map")
fig.show()

# Display summary statistics for area and price
summary_stats = df[["area_m2","price_usd"]].describe()
summary_stats


#Plot histogram of house prices
fig, ax = plt.subplots()

ax.hist(df["price_usd"][:20000], bins=10)
ax.set_xlabel("Price [USD]")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Home Prices")

#Plot box plot of house sizes
fig, ax = plt.subplots()

ax.boxplot(df["area_m2"], vert=False)
ax.set_xlabel("Area [sq meters]")
ax.set_title("Distribution of Home Sizes")

#Calculate mean price by region
mean_price_by_region = df.groupby("region")["price_usd"].mean().sort_values(ascending=True)
mean_price_by_region

#Plot bar chart of mean price by region
fig, ax = plt.subplots()

mean_price_by_region.plot(kind="bar", ax=ax)
ax.set_xlabel("Region")
ax.set_ylabel("Mean Price [USD]")
ax.set_title("Mean Home Price by Region")

#Filter dataset for South region
df_south = df[df["region"]=="South"]
df_south.head()

#Calculate number of homes for sale by state in South region
homes_by_state = df_south["state"].value_counts()
homes_by_state


# Subset dataset for Rio Grande do Sul state
df_south_rgs = df_south[df_south["state"]=="Rio Grande do Sul"]

#Plot scatter plot of price vs. area for Rio Grande do Sul
fig, ax = plt.subplots()

ax.scatter(df_south_rgs["area_m2"], df_south_rgs["price_usd"])
ax.set_xlabel("Area [sq meters]")
ax.set_ylabel("Price [USD]")
ax.set_title("Rio Grande do Sul: Price vs. Area")

#Calculate correlation between area and price for each state in South region
south_states_corr = {}

for state, value in df_south.groupby("state"):
    corr= value["area_m2"].corr(value["price_usd"])
    south_states_corr[state]=corr.round(10)

south_states_corr