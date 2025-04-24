import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import osmnx as ox
import networkx as nx
import pickle
from networkx.algorithms.approximation.traveling_salesman import greedy_tsp
from datetime import timedelta
import io
from streamlit_folium import folium_static

# Funktionen für TSP
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

# Funktion zur Stop-Auswahl
def select_stop_on_map(df_addresses):
    # Initialisierung der ausgewählten Stops
    if 'selected_stops' not in st.session_state:
        st.session_state.selected_stops = []

    # Streamlit Map-Integration
    m = folium.Map(location=[df_addresses["lat"].mean(), df_addresses["lon"].mean()], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers for each address in the dataframe
    for _, row in df_addresses.iterrows():
        marker = folium.Marker(location=[row['lat'], row['lon']], popup=row['Wahlraum-A'])
        marker.add_to(marker_cluster)
        
        # Capture the stop when clicked (update the selected_stops list in session_state)
        marker.add_child(folium.Popup(f"Click to select {row['Wahlraum-A']}"))

    # Display the map with interactive stop selection
    folium_static(m)

    # Rückgabe der aktuell ausgewählten Stops
    return st.session_state.selected_stops  # Diese Liste wird dynamisch aktualisiert


st.set_page_config(layout="wide")

with st.sidebar:
    st.title("Interaktives Tool zur Routenbearbeitung")

    # Daten laden
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

    # Interaktive Stop-Auswahl auf der Karte
    selected_stops = select_stop_on_map(addresses_df)

    # Anzeige der aktuell ausgewählten Stops
    st.write("### Ausgewählte Stops:")
    st.write(f"Anzahl ausgewählter Stops: {len(selected_stops)}")
    selected_stop_names = [addresses_df.loc[stop, "Wahlraum-A"] for stop in selected_stops]
    st.write(selected_stop_names)

    # Bestehende Teams ermitteln
    existing_teams = sorted([int(t) for t in st.session_state.new_assignments["team"].dropna().unique()])
    selected_team = st.selectbox("Ziel-Team auswählen", options=[None] + existing_teams)

    if st.button("Zuweisung übernehmen") and selected_team is not None and selected_stops:
        graph = get_graph()

        # Stopps mit dem ausgewählten Team aktualisieren
        for addr in selected_stops:
            current_team = st.session_state.new_assignments.loc[st.session_state.new_assignments["Wahlraum-A"] == addr, "team"].values[0]
            idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"] == addr].index[0]
            st.session_state.new_assignments.at[idx, "team"] = selected_team

        for team_id in [current_team, selected_team]:
            team_rows = st.session_state.new_assignments[st.session_state.new_assignments["team"] == team_id]
            optimized_rows = tsp_solve_route(graph, team_rows)
            st.session_state.new_assignments.loc[optimized_rows.index, "tsp_order"] = range(len(optimized_rows))

        st.rerun()

    if st.button("Neues Team erstellen"):
        st.session_state.show_new_team_form = True

    if st.session_state.get("show_new_team_form"):
        max_team = max([int(t) for t in st.session_state.new_assignments["team"].dropna().unique()]) if len(st.session_state.new_assignments["team"].dropna()) > 0 else 0
        new_team = max_team + 1
        with st.form(key="neues_team_form", clear_on_submit=True):
            st.markdown(f"### Stop(s) für Team {new_team} auswählen")
            new_team_stops = st.multiselect("Stop(s) auswählen", options=st.session_state.new_assignments["Wahlraum-A"].dropna().tolist(), key="form_team_selection")
            submitted = st.form_submit_button("Stop(s) zuweisen und Team erstellen")
            if submitted and new_team_stops:
                for addr in new_team_stops:
                    idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"] == addr].index[0]
                    st.session_state.new_assignments.at[idx, "team"] = new_team

                team_rows = st.session_state.new_assignments[st.session_state.new_assignments["team"] == new_team]
                optimized_rows = tsp_solve_route(get_graph(), team_rows)
                st.session_state.new_assignments.loc[optimized_rows.index, "tsp_order"] = range(len(optimized_rows))
                st.success(f"Team {new_team} wurde erstellt und die ausgewählten Stop(s) wurden zugewiesen.")
                st.session_state.show_new_team_form = False
                st.rerun()

# Interaktive Karte vorbereiten
graph = get_graph() 
m = folium.Map(
    location=[addresses_df["lat"].mean(), addresses_df["lon"].mean()],
    zoom_start=12  # <-- korrekt statt 'zoom'
)

# Routen zeichnen
color_list = ["#FF00FF", "#00FFFF", "#00FF00", "#FF0000", "#FFA500", "#FFFF00", "#00CED1", "#DA70D6", "#FF69B4", "#8A2BE2"]
for i, team_id in enumerate(sorted(st.session_state.new_assignments["team"].dropna().unique())):
    team_rows = st.session_state.new_assignments[st.session_state.new_assignments["team"] == team_id]
    if "tsp_order" in team_rows.columns:
        team_rows = team_rows.sort_values("tsp_order")
    coords = team_rows[["lat", "lon"]].values.tolist()
    if len(coords) > 1:
        route_coords = []
        try:
            nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
            for u, v in zip(nodes[:-1], nodes[1:]):
                try:
                    path = nx.shortest_path(graph, u, v, weight="length")
                    segment = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in path]
                    route_coords.extend(segment)
                except Exception as e:
                    st.warning(f"Routing für Segment {u} → {v} in Team {team_id} fehlgeschlagen: {e}")
                    continue
            folium.PolyLine(route_coords, color=color_list[i % len(color_list)], weight=8, opacity=0.9,
                            tooltip=f"Team {int(team_id)}").add_to(m)
        except Exception as e:
            st.warning(f"Routenaufbau für Team {team_id} fehlgeschlagen: {e}")
            continue

# Marker hinzufügen
marker_cluster = MarkerCluster()
for _, row in addresses_df.dropna(subset=["lat", "lon"]).iterrows():
    wahlraum_b = row.get("Wahlraum-B", "")
    wahlraum_a = row.get("Wahlraum-A", "")
    num_rooms = row.get("num_rooms", "")
    popup = f"<b>{wahlraum_b}</b><br>{wahlraum_a}<br>Anzahl Räume: {num_rooms}"
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=folium.Popup(popup, max_width=300)
    ).add_to(marker_cluster)
marker_cluster.add_to(m)

# Karte anzeigen
folium_static(m)
