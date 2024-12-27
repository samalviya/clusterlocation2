import pandas as pd
import numpy as np
import folium
from geopy.distance import geodesic
import streamlit as st   
from streamlit_folium import folium_static
from sklearn.cluster import DBSCAN
import io

# Load the CSV data (make sure to upload the correct file)
st.title("Enhanced GPS Clustering Tool")
st.subheader("Upload your GPS data and visualize clusters on the map")
st.markdown("### Instructions:")
st.markdown("1. Upload a CSV file containing GPS coordinates in a column named 'GPS' (formatted as 'latitude,longitude').")
st.markdown("2. Choose clustering algorithm and parameters.")
st.markdown("3. Optionally, display the center of each cluster on the map.")

# Sidebar settings
with st.sidebar:
    st.header("Upload Data & Settings")
    uploaded_file = st.file_uploader("Upload CSV file with GPS Coordinates", type=["csv"])
    group_size = st.slider("Group Size", min_value=5, max_value=50, value=20, step=1)
    st.markdown("---")
    st.markdown("### Clustering Settings")
    eps = st.number_input("Epsilon (Max Distance for DBSCAN, in meters)", min_value=10, max_value=5000, step=10, value=500)
    min_samples = st.slider("Minimum Samples for DBSCAN", min_value=1, max_value=20, value=5, step=1)

# Function to calculate distance between two GPS coordinates
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).meters  # Distance in meters

# Function to group shops into groups based on proximity
def group_shops(coords, group_size):
    remaining_shops = set(range(len(coords)))  # Set of all shop indices
    groups = []  # To store the resulting groups
    
    while remaining_shops:
        group = []  # Current group
        current_shop = remaining_shops.pop()  # Start with a random shop
        group.append(current_shop)
        
        # Find nearest neighbors until we have a full group
        while len(group) < group_size and remaining_shops:
            distances = [(i, calculate_distance(coords[current_shop], coords[i])) for i in remaining_shops]
            distances.sort(key=lambda x: x[1])  # Sort by distance
            nearest_shop = distances[0][0]  # Nearest shop
            group.append(nearest_shop)
            remaining_shops.remove(nearest_shop)
            current_shop = nearest_shop  # Set the last added shop as the current shop
        
        # Append the group of shops to the groups list
        groups.append(group)
    
    return groups

# Function to apply DBSCAN clustering
def apply_dbscan(coords, eps, min_samples):
    # Convert coordinates to the correct format for DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="haversine")
    # Convert degrees to radians for DBSCAN
    coords_rad = np.radians(coords)
    model.fit(coords_rad)
    return model.labels_

# Main logic
if uploaded_file is not None:
    try:
        # Read uploaded file
        data = pd.read_csv(uploaded_file)
        if 'GPS' not in data.columns:
            st.error("The CSV must contain a 'GPS' column.")
            st.stop()

        # Extract GPS coordinates and convert them to tuples
        coordinates = data['GPS'].apply(lambda x: tuple(map(float, x.split(','))))
        coords = np.array(list(coordinates))

        # Apply DBSCAN clustering
        cluster_labels = apply_dbscan(coords, eps / 6371000, min_samples)  # Convert eps to radians for DBSCAN
        data['DBSCAN_Cluster'] = cluster_labels

        # Group shops into groups based on proximity
        groups = group_shops(coords, group_size)
        group_labels = [-1] * len(coords)  # Initialize with -1 (not assigned)
        for group_id, group in enumerate(groups):
            for shop_id in group:
                group_labels[shop_id] = group_id  # Assign each shop to its group

        data['Proximity_Group'] = group_labels

        # Map creation
        map_center = [data['GPS'].apply(lambda x: float(x.split(',')[0])).mean(), 
                      data['GPS'].apply(lambda x: float(x.split(',')[1])).mean()]
        m = folium.Map(location=map_center, zoom_start=15)

        # Plot clusters on the map
        for _, row in data.iterrows():
            lat, lon = map(float, row['GPS'].split(','))
            group = row['Proximity_Group']
            cluster = row['DBSCAN_Cluster']
            color = f'#{hash(str(group)) & 0xFFFFFF:06x}'  # Random color for each group
            folium.CircleMarker(
                location=[lat, lon],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"Shop ID: {row['ID']}, Group: {group}, DBSCAN Cluster: {cluster}",
            ).add_to(m)

        # Add DBSCAN clusters as layers
        for cluster in np.unique(data['DBSCAN_Cluster']):
            if cluster != -1:  # Skip noise points (DBSCAN cluster = -1)
                cluster_data = data[data['DBSCAN_Cluster'] == cluster]
                for _, row in cluster_data.iterrows():
                    lat, lon = map(float, row['GPS'].split(','))
                    color = f'#{hash(str(cluster)) & 0xFFFFFF:06x}'
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=7,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.6,
                        popup=f"Shop ID: {row['ID']}, Group: {row['Proximity_Group']}, Cluster: {cluster}",
                    ).add_to(m)

        # Show map
        folium_static(m)

        # Visualize the output data in a table
        st.subheader("Processed Data")
        st.dataframe(data)

        # Prepare the output CSV for download
        output_csv = data.to_csv(index=False)
        st.download_button(
            label="Download Processed Data as CSV",
            data=output_csv,
            file_name="processed_shop_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please upload a CSV file to proceed.")
