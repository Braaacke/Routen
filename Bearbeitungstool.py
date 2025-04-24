"""Interaktives Routenbearbeitungstool mit Zoom-abhängiger Markersichtbarkeit und TSP-Optimierung bei Zuweisung"""

import streamlit as st
import pandas as pd
import leafmap.foliumap as leafmap
import folium
from folium.plugins import MarkerCluster
import networkx as nx
import osmnx as ox
import pickle
from networkx.algorithms.approximation.traveling_salesman import greedy_tsp
from urllib.parse import quote_plus
from datetime import timedelta
import io

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

st.set_page_config(layout="wide")

with st.sidebar:
    st.title("Interaktives Tool zur Routenbearbeitung")

    # Standardmäßiges Laden nur einmal beim Start (initialisieren, wenn nicht im Session State)
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

    # Optionaler manueller Import zur Überschreibung der Zuordnung
    uploaded_file = st.file_uploader("Importiere alternative Zuweisung (Excel-Datei)", type=["xlsx"])
    if uploaded_file:
        imported_team_df = pd.read_excel(uploaded_file, sheet_name=None)
        imported_assignments = []
        for sheet, df in imported_team_df.items():
            if sheet != "Übersicht" and "Adresse" in df.columns:
                team = int(sheet.split("_")[1])
                for addr in df["Adresse"]:
                    imported_assignments.append((addr, team))
        assignments_df = pd.DataFrame(imported_assignments, columns=["Wahlraum-A", "team"])
        if "team" in addresses_df.columns:
            addresses_df = addresses_df.drop(columns="team")
        addresses_df = addresses_df.merge(assignments_df, on="Wahlraum-A", how="left")
        st.success("Import erfolgreich – aktuelle Zuweisung wurde überschrieben.")

        # Routen nach Import neu berechnen
        graph = get_graph()
        for team_id in assignments_df["team"].dropna().unique():
            team_rows = addresses_df[addresses_df["team"] == team_id]
            optimized_rows = tsp_solve_route(graph, team_rows)
            addresses_df.loc[optimized_rows.index, "tsp_order"] = range(len(optimized_rows))
        st.session_state.new_assignments = addresses_df.copy()

        with st.expander("📋 Vorschau der importierten Zuweisung"):
            st.dataframe(assignments_df)

    # Fallback falls addresses_df nicht definiert ist
    try:
        addresses_df = addresses_df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Fehler beim Zurücksetzen des Index von addresses_df: {e}")

    if "new_assignments" not in st.session_state:
        try:
            st.session_state.new_assignments = addresses_df.copy()
        except Exception as e:
            st.error(f"Fehler beim Kopieren von addresses_df in den Session State: {e}")

    if isinstance(addresses_df, pd.DataFrame) and "Wahlraum-A" in addresses_df.columns:
        selected_indices = st.multiselect("Stops auswählen (nach Adresse)", options=addresses_df["Wahlraum-A"].dropna().tolist())
    else:
        selected_indices = []
        st.warning("Daten konnten nicht geladen werden oder 'Wahlraum-A' fehlt.")
    # Bestehende Teams erst nach allen möglichen Änderungen ermitteln
    existing_teams = sorted([int(t) for t in st.session_state.new_assignments["team"].dropna().unique()])
    selected_team = st.selectbox("Ziel-Team auswählen", options=[None] + existing_teams)

    if st.button("Zuweisung übernehmen") and selected_team is not None and selected_indices:
        graph = get_graph()

        for addr in selected_indices:
            current_team = st.session_state.new_assignments.loc[st.session_state.new_assignments["Wahlraum-A"] == addr, "team"].values[0]
            idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"] == addr].index[0]
            st.session_state.new_assignments.at[idx, "team"] = selected_team

        for team_id in [current_team, selected_team]:
            team_rows = st.session_state.new_assignments[st.session_state.new_assignments["team"] == team_id]
            optimized_rows = tsp_solve_route(get_graph(), team_rows)
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

m = leafmap.Map(center=[addresses_df["lat"].mean(), addresses_df["lon"].mean()], zoom=12)
graph = get_graph()

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

marker_cluster = MarkerCluster()

# Für jedes Stop in der cleaned_addresses.csv, füge Marker mit Popups hinzu
for _, row in addresses_df.dropna(subset=["lat", "lon"]).iterrows():
    # Extrahiere die relevanten Spalten für das Popup
    wahlraum_b = row.get("Wahlraum-B", "Keine Wahlraum-B-Daten")
    wahlraum_a = row.get("Wahlraum-A", "Keine Wahlraum-A-Daten")
    num_rooms = row.get("num_rooms", "Keine Raumanzahl")

    # Popup-Inhalt formatieren
    popup_content = f"""
    {wahlraum_b}<br>
    {wahlraum_a}<br>
    Anzahl Räume: {num_rooms}
    """
    # Erstelle ein HTML Popup mit max-width und max-height
    popup_html = f"""
    <div style="max-width: 500px; max-height: 500px; overflow:auto;">
        {popup_content}
    </div>
    """    
    # Marker für jeden Stop erstellen und zum Marker-Cluster hinzufügen
    marker = folium.Marker(location=[row["lat"], row["lon"]], popup=popup_content)
    marker.add_to(marker_cluster)

# Füge das Marker Cluster zur Karte hinzu
marker_cluster.add_to(m)

# Füge Marker Cluster zur Karte hinzu
marker_cluster.add_to(m)
m.to_streamlit(height=700)

if st.button("Zuordnung exportieren"):
    overview = []
    team_sheets = {}
    for team in sorted(st.session_state.new_assignments["team"].dropna().unique()):
        stops = st.session_state.new_assignments[st.session_state.new_assignments["team"] == team]
        if "tsp_order" in stops.columns:
            stops = stops.sort_values("tsp_order")
        rooms = stops["num_rooms"].sum()
        travel_km = 0
        travel_min = 0
        coords = stops[["lat", "lon"]].values.tolist()
        if len(coords) > 1:
            try:
                nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
                for u, v in zip(nodes[:-1], nodes[1:]):
                    length = nx.shortest_path_length(graph, u, v, weight="length")
                    travel_km += length / 1000
                    travel_min += length / 1000 * 2
            except:
                pass
        service_min = int(rooms * 10)
        time_total = service_min + travel_min
        gmaps_link = "https://www.google.com/maps/dir/" + "/".join([f"{lat},{lon}" for lat, lon in coords])
        overview.append({
            "Kontrollbezirk": team,
            "Anzahl Wahllokale": len(stops),
            "Anzahl Stimmbezirke": rooms,
            "Wegstrecke (km)": round(travel_km, 1),
            "Fahrtzeit (min)": int(travel_min),
            "Kontrollzeit (min)": service_min,
            "Gesamtzeit": str(timedelta(minutes=int(time_total))),
            "Google-Link": gmaps_link
        })
        rows = []
        for i, row in stops.iterrows():
            address_coords = f"{row['lat']},{row['lon']}"
            rows.append({
                "Reihenfolge": i,
                "Adresse": row["Wahlraum-A"],
                "Stimmbezirke": row["rooms"],
                "Anzahl Stimmbezirke": row["num_rooms"],
                "Google-Link": f"https://www.google.com/maps/search/?api=1&query={quote_plus(address_coords)}"
            })
        team_sheets[f"Team_{team}"] = pd.DataFrame(rows)

    overview_df = pd.DataFrame(overview)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="Übersicht", index=False)
        for sheet_name, df in team_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)

    st.download_button(
        label="📥 Excel-Datei herunterladen",
        data=output,
        file_name="routen_zuweisung_aktualisiert.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
