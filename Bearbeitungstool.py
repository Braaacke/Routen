
import streamlit as st
import pandas as pd
import networkx as nx
import osmnx as ox
import pickle
from datetime import timedelta
from networkx.algorithms.approximation.traveling_salesman import greedy_tsp
from streamlit_leaflet import st_leaflet, Map, Marker, Icon, LayerGroup

@st.cache_resource
def get_graph():
    with open("munster_graph.pickle", "rb") as f:
        return pickle.load(f)

def tsp_solve_route(graph, stops_df):
    if len(stops_df) <= 2:
        return stops_df
    coords = list(zip(stops_df["lat"], stops_df["lon"]))
    nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
    G = nx.complete_graph(len(nodes))
    for i in G.nodes:
        for j in G.nodes:
            if i != j:
                try:
                    length = nx.shortest_path_length(graph, nodes[i], nodes[j], weight='length')
                    G[i][j]['weight'] = length
                except:
                    G[i][j]['weight'] = float('inf')
    tsp_path = greedy_tsp(G)
    return stops_df.iloc[tsp_path].reset_index(drop=True)

st.set_page_config(layout="wide")

with st.sidebar:
    st.title("Interaktives Tool zur Routenbearbeitung")

    if "base_addresses" not in st.session_state:
        base_addresses = pd.read_csv("cleaned_addresses.csv").reset_index(drop=True)
        team_df = pd.read_excel("routes_optimized.xlsx", sheet_name=None)
        temp_assignments = []
        for sheet, df in team_df.items():
            if sheet != "Übersicht" and "Adresse" in df.columns:
                team = int(sheet.split("_")[1])
                for addr in df["Adresse"]:
                    temp_assignments.append((addr, team))
        assignments_df = pd.DataFrame(temp_assignments, columns=["Wahlraum-A", "team"])
        base_addresses = base_addresses.merge(assignments_df, on="Wahlraum-A", how="left")
        st.session_state.base_addresses = base_addresses.copy()
        st.session_state.new_assignments = base_addresses.copy()

    addresses_df = st.session_state.base_addresses.copy()

    markers = []
    for _, row in addresses_df.iterrows():
        popup_text = f"{row.get('Wahlraum-B', '')}<br>{row.get('Wahlraum-A', '')}<br>{row.get('num_rooms', '')} Wahlräume"
        icon = Icon(iconUrl="https://cdn-icons-png.flaticon.com/512/252/252025.png", iconSize=[20, 20])
        markers.append(
            Marker(location=[row["lat"], row["lon"]], popup=popup_text, icon=icon)
        )
    marker_layer = LayerGroup(markers=markers)

    m = Map(center=[addresses_df["lat"].mean(), addresses_df["lon"].mean()], zoom=12, layers=[marker_layer])
    result = st_leaflet(m, height=600, key="map")

    clicked = result.get("last_clicked")
    if clicked:
        st.write(f"Geklickt: {clicked['lat']:.5f}, {clicked['lng']:.5f}")
        nearest_idx = ((addresses_df["lat"] - clicked["lat"])**2 + (addresses_df["lon"] - clicked["lng"])**2).idxmin()
        nearest_address = addresses_df.loc[nearest_idx, "Wahlraum-A"]
        st.write(f"Ausgewählter Stop: {nearest_address}")
        if "selected_stops" not in st.session_state:
            st.session_state.selected_stops = []
        if nearest_address not in st.session_state.selected_stops:
            st.session_state.selected_stops.append(nearest_address)

    st.markdown("### Ausgewählte Stops:")
    st.write(st.session_state.get("selected_stops", []))

    existing_teams = sorted([int(t) for t in st.session_state.new_assignments["team"].dropna().unique()])
    selected_team = st.selectbox("Ziel-Team auswählen", options=[None] + existing_teams)

    if st.button("Zuweisung übernehmen") and selected_team is not None and st.session_state.get("selected_stops"):
        graph = get_graph()
        for addr in st.session_state.selected_stops:
            current_team = st.session_state.new_assignments.loc[st.session_state.new_assignments["Wahlraum-A"] == addr, "team"].values[0]
            idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"] == addr].index[0]
            st.session_state.new_assignments.at[idx, "team"] = selected_team

        for team_id in [current_team, selected_team]:
            team_rows = st.session_state.new_assignments[st.session_state.new_assignments["team"] == team_id]
            optimized_rows = tsp_solve_route(graph, team_rows)
            st.session_state.new_assignments.loc[optimized_rows.index, "tsp_order"] = range(len(optimized_rows))

        st.session_state.selected_stops.clear()
        st.rerun()
