
import osmnx as ox
import networkx as nx
import pandas as pd
from itertools import product

# Load your POI file (make sure path is correct)
poi_df = pd.read_csv("C:/Users/Dell/Desktop/itinerary-drl/data/poi_it.csv")  # Change path if needed

# Build a bike graph around Verona
center_point = (poi_df['latitude'].mean(), poi_df['longitude'].mean())
G = ox.graph_from_point(center_point, dist=3000, network_type='bike')

# Map each POI to nearest OSM node
poi_df['osmid'] = poi_df.apply(lambda row: ox.distance.nearest_nodes(G, row['longitude'], row['latitude']), axis=1)

# Constants
bike_speed_m_per_min = 250  # 15 km/h

# Generate all pairs (excluding self)
pairs = [(i, j) for i in range(len(poi_df)) for j in range(len(poi_df)) if i != j]
results = []

for i, j in pairs:
    start = poi_df.iloc[i]
    end = poi_df.iloc[j]

    try:
        length = nx.shortest_path_length(G, start['osmid'], end['osmid'], weight='length')
        time_min = round(length / bike_speed_m_per_min, 2)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        time_min = 999  # Large penalty if no path

    results.append({
        "poi_start": start["id"],
        "poi_dest": end["id"],
        "time_travel": time_min
    })

# Save to CSV
df_bike_times = pd.DataFrame(results)
df_bike_times.to_csv("C:/Users/Dell/Desktop/itinerary-drl/data/bike_poi_time_travel.csv", index=False)

print("✅ Done! Saved as 'bike_poi_time_travel.csv'")
