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

    # Initiales Laden
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

    # Arbeits-Daten setzen (nach Import oder initial)
    addresses_df = st.session_state.new_assignments.copy()

    # Manueller Import alternative Zuweisung
    uploaded_file = st.file_uploader("Importiere alternative Zuweisung (Excel-Datei)", type=["xlsx"])
    if uploaded_file:
        imported = pd.read_excel(uploaded_file, sheet_name=None)
        imported_list = []
        for sheet, df in imported.items():
            if sheet != "Übersicht" and "Adresse" in df.columns:
                team = int(sheet.split("_")[1])
                for addr in df["Adresse"]:
                    imported_list.append((addr, team))
        assignments_df = pd.DataFrame(imported_list, columns=["Wahlraum-A", "team"] )
        # Merge auf new_assignments
        df_merge = pd.merge(
            st.session_state.base_addresses.drop(columns=["team"]),
            assignments_df,
            on="Wahlraum-A",
            how="left"
        )
        st.session_state.new_assignments = df_merge.copy()
        st.success("Import erfolgreich – aktuelle Zuweisung wurde überschrieben.")
        st.experimental_rerun()

    # Auswahl für manuelle Zuordnung
    stops = addresses_df["Wahlraum-A"].dropna().tolist()
    selected = st.multiselect("Stops auswählen (nach Adresse)", options=stops)
    teams = sorted([int(t) for t in addresses_df["team"].dropna().unique()])
    target = st.selectbox("Ziel-Team auswählen", options=[None] + teams)
    if st.button("Zuweisung übernehmen") and target and selected:
        for addr in selected:
            idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"] == addr].index[0]
            st.session_state.new_assignments.at[idx, "team"] = target
        # Routen neu berechnen
        for tid in set([target]):
            rows = st.session_state.new_assignments[st.session_state.new_assignments["team"] == tid]
            opt = tsp_solve_route(get_graph(), rows)
            st.session_state.new_assignments.loc[opt.index, "tsp_order"] = range(len(opt))
        st.experimental_rerun()

    # Neues Team erstellen
    if st.button("Neues Team erstellen"):
        st.session_state.show_new_team_form = True
    if st.session_state.get("show_new_team_form"):
        max_team = int(addresses_df["team"].max()) if addresses_df["team"].notnull().any() else 0
        new_team = max_team + 1
        with st.form(key="new_team_form", clear_on_submit=True):
            st.markdown(f"### Stop(s) für Team {new_team} auswählen")
            sel = st.multiselect("Stop(s)", options=st.session_state.new_assignments["Wahlraum-A"].dropna().tolist())
            if st.form_submit_button("Zuweisen und Team erstellen") and sel:
                for addr in sel:
                    idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"] == addr].index[0]
                    st.session_state.new_assignments.at[idx, "team"] = new_team
                opt = tsp_solve_route(get_graph(), st.session_state.new_assignments[st.session_state.new_assignments["team"] == new_team])
                st.session_state.new_assignments.loc[opt.index, "tsp_order"] = range(len(opt))
                st.success(f"Team {new_team} erstellt.")
                st.session_state.show_new_team_form = False
                st.experimental_rerun()

# Karte zeichnen aus new_assignments
m = leafmap.Map(center=[
    st.session_state.new_assignments["lat"].mean(),
    st.session_state.new_assignments["lon"].mean()
], zoom=12)

# Routenlinien
graph = get_graph()
color_list = ["#FF00FF", "#00FFFF", "#00FF00", "#FF0000", "#FFA500", "#FFFF00",
              "#00CED1", "#DA70D6", "#FF69B4", "#8A2BE2"]
for i, tid in enumerate(sorted(st.session_state.new_assignments["team"].dropna().unique())):
    df = st.session_state.new_assignments[st.session_state.new_assignments["team"] == tid]
    if "tsp_order" in df.columns:
        df = df.sort_values("tsp_order")
    coords = df[["lat", "lon"]].values.tolist()
    if len(coords) > 1:
        line = []
        nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
        for u, v in zip(nodes[:-1], nodes[1:]):
            try:
                path = nx.shortest_path(graph, u, v, weight="length")
                seg = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in path]
                line.extend(seg)
            except:
                continue
        folium.PolyLine(line, color=color_list[i % len(color_list)], weight=8, opacity=0.9,
                        tooltip=f"Team {int(tid)}").add_to(m)

# Marker Cluster mit Team-Anzeige
marker_cluster = MarkerCluster()
for _, row in st.session_state.new_assignments.dropna(subset=["lat", "lon"]).iterrows():
    team_id = row.get("team", None)
    color = color_list[sorted(st.session_state.new_assignments["team"].dropna().unique()).index(team_id) % len(color_list)] if pd.notnull(team_id) else "#000000"
    popup = f"<b>Team:</b> {int(team_id) if pd.notnull(team_id) else 'n/a'}<br>" + \
            f"<b>{row.get('Wahlraum-B')}</b><br><b>{row.get('Wahlraum-A')}</b><br><b>Anzahl Räume:</b> {row.get('num_rooms')}"
    popup_html = f"<div style='max-width:500px; max-height:500px; overflow:auto;'>{popup}</div>"
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=6,
                        color=color, fill=True, fill_opacity=0.7,
                        popup=folium.Popup(popup_html, max_width=500)).add_to(marker_cluster)
marker_cluster.add_to(m)

m.to_streamlit(height=700)

# Export
if st.button("Zuordnung exportieren"):
    overview = []
    team_sheets = {}
    for idx, tid in enumerate(sorted(st.session_state.new_assignments["team"].dropna().unique()), start=1):
        df = st.session_state.new_assignments[st.session_state.new_assignments["team"] == tid]
        if "tsp_order" in df.columns:
            df = df.sort_values("tsp_order")
        rooms = df["num_rooms"].sum()
        travel_km = travel_min = 0
        coords = df[["lat", "lon"]].values.tolist()
        if len(coords) > 1:
            nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
            for u, v in zip(nodes[:-1], nodes[1:]):
                try:
                    length = nx.shortest_path_length(graph, u, v, weight="length")
                    travel_km += length / 1000
                    travel_min += length / 1000 * 2
                except:
                    pass
        service = int(rooms * 10)
        total = service + travel_min
        gmaps = "https://www.google.com/maps/dir/" + "/".join([f"{lat},{lon}" for lat, lon in coords])
        overview.append({
            "Kontrollbezirk": idx,
            "Anzahl Wahllokale": len(df),
            "Anzahl Stimmbezirke": rooms,
            "Wegstrecke (km)": round(travel_km, 1),
            "Fahrtzeit (min)": int(travel_min),
            "Kontrollzeit (min)": service,
            "Gesamtzeit": str(timedelta(minutes=int(total))),
            "Google-Link": gmaps
        })
        rows = []
        for j, (_, row) in enumerate(df.iterrows(), start=1):
            coords_str = f"{row['lat']},{row['lon']}"
            rows.append({
                "Bezirk": j,
                "Adresse": row['Wahlraum-A'],
                "Stimmbezirke": row.get('rooms'),
                "Anzahl Stimmbezirke": row.get('num_rooms'),
                "Google-Link": f"https://www.google.com/maps/search/?api=1&query={quote_plus(coords_str)}"
            })
        team_sheets[f"Bezirk_{idx}"] = pd.DataFrame(rows)
    overview_df = pd.DataFrame(overview)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="Übersicht", index=False)
        for name, df in team_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    out.seek(0)
    st.download_button("📥 Excel herunterladen", data=out, file_name="routen_zuweisung_aktualisiert.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
