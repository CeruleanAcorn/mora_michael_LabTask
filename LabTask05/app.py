import streamlit as st
import json
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("business Location Explorer")
st.write("Learning how to use streamlit")

def load_data(path="business_locations.geojson"):
    with open(path) as f:
        geojson = json.load(f)
    rows = [] 
    for feat in geojson["features"]:
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        rows.append({**props, "lon":lon, "lat":lat})
    return pd.DataFrame(rows)


df = load_data()
# st.dataframe(df.head(200))

with st.expander("Look at Data:"):
    st.dataframe(df.head(20))
    st.write(f"{len(df)} locations, {df["Neighborhood"].nunique()} neighborhoods.")

st.sidebar.header("1. Select Features")

# st.write(df.columns)

NUMERIC_COLS = ["Floor_Area_sqm", "Daily_Foot_Traffic", "Community_Impact_Score", "Annual_Revenue_k"]
selected_features = st.sidebar.multiselect("Features to be used in models", options=NUMERIC_COLS, default=NUMERIC_COLS)

if len(selected_features) < 2:
    st.warning("Pick at least two features to continue")
    st.stop()


X = df[NUMERIC_COLS].to_numpy()
X_scaled = StandardScaler().fit_transform(X)

st.sidebar.header("2. Clustering")
algo = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN"])
if algo == "KMeans":
    k = st.sidebar.slider("Number of Clusters", 2, 10)
elif algo == "DBSCAN":
    #st.write("STUDENTS, ADD DB SCAN parameters")
    # Instead of a slider, using originally recommended params from lab.
    theeps = 0.22
    k = 5

labels = []
if algo == "KMeans":
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(X_scaled)
elif algo == "DBSCAN":
    #st.write("STUDENTS, ADD DB SCAN")
    #st.write("DBSCAN RESULTS BELOW")
    model = DBSCAN(eps=theeps, min_samples = k) 
    labels = model.fit_predict(X_scaled)

# Catch error if no labels do not continue
if len(labels) == 0:
    st.warning("There is no clustering labels")
    st.stop()

#We can add the output from st.write(labels) to a df!

df["cluster"] = pd.Categorical(labels.astype(str))
n_clusters_found = df["cluster"].nunique()
st.metric("Number of Clusters", n_clusters_found)

# st.dataframe(df.head())
# st.write(X_scaled)

map_tab, dr_tab = st.tabs(["Map", "Dimensionality Reduction"])

with map_tab:
    st.write("MAP")
    fig = px.scatter_map(df, lat="lat", lon="lon", color="cluster", zoom=10, height=550, map_style="carto-darkmatter")
    st.plotly_chart(fig, width="stretch")

with dr_tab:
    reducer = PCA(n_components = 2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    df["dim_1"] = embedding[:,0]
    df["dim_2"] = embedding[:,1]

    fig_dr = px.scatter(
        df,
        x='dim_1',
        y='dim_2',
        color="cluster",
        height = 550
    )

    st.plotly_chart(fig_dr, width="stretch")
 # st.write(embedding)

# In folder with environment, run this in terminal to start the app.
# streamlit run app.py

# python3 -m venv venv

#todo: Add DBSCAN, If it's on DBSCAN add Epsilon