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

# Sidebar-Logik
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
        st.session_state.needs_rerun = False

    # Arbeits-Daten setzen
    df_assign = st.session_state.new_assignments.copy()

    # Manueller Import
    uploaded_file = st.file_uploader("Importiere alternative Zuweisung (Excel-Datei)", type=["xlsx"])
    if uploaded_file:
        imported = pd.read_excel(uploaded_file, sheet_name=None)
        imported_list = []
        for sheet, df in imported.items():
            if sheet != "Übersicht" and "Adresse" in df.columns:
                team = int(sheet.split("_")[1])
                for addr in df["Adresse"]:
                    imported_list.append((addr, team))
        assignments_df = pd.DataFrame(imported_list, columns=["Wahlraum-A", "team"])
        merged = pd.merge(
            st.session_state.base_addresses.drop(columns=["team"]),
            assignments_df,
            on="Wahlraum-A",
            how="left"
        )
        st.session_state.new_assignments = merged.copy()
        st.session_state.needs_rerun = True
        st.success("Import erfolgreich – Karte wird aktualisiert.")

    # Manuelle Zuweisung
    stops = df_assign["Wahlraum-A"].dropna().tolist()
    selected = st.multiselect("Stops auswählen", options=stops)
    teams = sorted([int(t) for t in df_assign["team"].dropna().unique()])
    target = st.selectbox("Ziel-Team auswählen", options=[None] + teams)
    if st.button("Zuweisung übernehmen") and target and selected:
        for addr in selected:
            idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"] == addr].index[0]
            st.session_state.new_assignments.at[idx, "team"] = target
        # Routen neu berechnen
        rows = st.session_state.new_assignments[st.session_state.new_assignments["team"] == target]
        opt = tsp_solve_route(get_graph(), rows)
        st.session_state.new_assignments.loc[opt.index, "tsp_order"] = range(len(opt))
        st.session_state.needs_rerun = True
        st.success("Zuweisung übernommen – Karte wird aktualisiert.")

    # Neues Team erstellen
    if st.button("Neues Team erstellen"):
        st.session_state.show_new_team_form = True
    if st.session_state.get("show_new_team_form"):
        max_team = int(df_assign["team"].max()) if df_assign["team"].notnull().any() else 0
        new_team = max_team + 1
        with st.form(key="new_team_form", clear_on_submit=True):
            st.markdown(f"### Stop(s) für Team {new_team} auswählen")
            sel = st.multiselect("Stop(s) auswählen", options=st.session_state.new_assignments["Wahlraum-A"].dropna().tolist())
            if st.form_submit_button("Zuweisen und Team erstellen") and sel:
                for addr in sel:
                    idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"] == addr].index[0]
                    st.session_state.new_assignments.at[idx, "team"] = new_team
                opt = tsp_solve_route(get_graph(), st.session_state.new_assignments[st.session_state.new_assignments["team"] == new_team])
                st.session_state.new_assignments.loc[opt.index, "tsp_order"] = range(len(opt))
                st.session_state.show_new_team_form = False
                st.session_state.needs_rerun = True
                st.success(f"Team {new_team} erstellt – Karte wird aktualisiert.")

    # Bei Bedarf neu laden
    if st.session_state.get("needs_rerun"):
        st.experimental_rerun()

# Karte zeichnen aus new_assignments
m = leafmap.Map(center=[
    st.session_state.new_assignments["lat"].mean(),
    st.session_state.new_assignments["lon"].mean()
], zoom=12)

# Routenlinien zeichnen
graph = get_graph()
color_list = ["#FF00FF", "#00FFFF", "#00FF00", "#FF0000", "#FFA500",
              "#FFFF00", "#00CED1", "#DA70D6", "#FF69B4", "#8A2BE2"]
for i, tid in enumerate(sorted(st.session_state.new_assignments["team"].dropna().unique())):
    df_team = st.session_state.new_assignments[st.session_state.new_assignments["team"] == tid]
    if "tsp_order" in df_team.columns:
        df_team = df_team.sort_values("tsp_order")
    coords = df_team[["lat", "lon"]].values.tolist()
    if len(coords) > 1:
        path = []
        nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
        for u, v in zip(nodes[:-1], nodes[1:]):
            try:
                p = nx.shortest_path(graph, u, v, weight="length")
                seg = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in p]
                path.extend(seg)
            except:
                continue
        folium.PolyLine(path, color=color_list[i % len(color_list)], weight=8, opacity=0.9,
                        tooltip=f"Team {int(tid)}").add_to(m)

# Marker Cluster mit Team-Anzeige
marker_cluster = MarkerCluster()
for _, row in st.session_state.new_assignments.dropna(subset=["lat", "lon"]).iterrows():
    team_id = row.get("team")
    if pd.notnull(team_id):
        idx = sorted(st.session_state.new_assignments["team"].dropna().unique()).index(team_id)
        color = color_list[idx % len(color_list)]
    else:
        color = "#000000"
    popup_html = f"<div style='max-width:500px; max-height:500px; overflow:auto;'><b>Team:</b> {int(team_id) if pd.notnull(team_id) else 'n/a'}<br>"
    popup_html += f"<b>{row.get('Wahlraum-B', '')}</b><br><b>{row.get('Wahlraum-A', '')}</b><br>"
    popup_html += f"<b>Anzahl Räume:</b> {row.get('num_rooms', '')}</div>"
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
        df_team = st.session_state.new_assignments[st.session_state.new_assignments["team"] == tid]
        if "tsp_order" in df_team.columns:
            df_team = df_team.sort_values("tsp_order")
        rooms = df_team["num_rooms"].sum()
        travel_km = travel_min = 0
        coords = df_team[["lat", "lon"]].values.tolist()
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
        gmaps_link = "https://www.google.com/maps/dir/" + "/".join([f"{lat},{lon}" for lat, lon in coords])
        overview.append({
            "Kontrollbezirk": idx,
            "Anzahl Wahllokale": len(df_team),
            "Anzahl Stimmbezirke": rooms,
            "Wegstrecke (km)": round(travel_km, 1),
            "Fahrtzeit (min)": int(travel_min),
            "Kontrollzeit (min)": service,
            "Gesamtzeit": str(timedelta(minutes=int(total))),
            "Google-Link": gmaps_link
        })
        rows = []
        for j, (_, row) in enumerate(df_team.iterrows(), start=1):
            coords_str = f"{row['lat']},{row['lon']}"
            rows.append({
                "Bezirk": j,
                "Adresse": row['Wahlraum-A'],
                "Stimmbezirke": row.get('rooms', ''),
                "Anzahl Stimmbezirke": row.get('num_rooms', ''),
                "Google-Link": f"https://www.google.com/maps/search/?api=1&query={quote_plus(coords_str)}"
            })
        team_sheets[f"Bezirk_{idx}"] = pd.DataFrame(rows)
    overview_df = pd.DataFrame(overview)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="Übersicht", index=False)
        for name, df_sheet in team_sheets.items():
            df_sheet.to_excel(writer, sheet_name=name, index=False)
    output.seek(0)
    st.download_button("📥 Excel herunterladen", data=output,
                       file_name="routen_zuweisung_aktualisiert.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
