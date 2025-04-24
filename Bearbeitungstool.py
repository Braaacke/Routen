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

    addresses_df = st.session_state.base_addresses.copy()

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
        if "team" in addresses_df.columns:
            addresses_df = addresses_df.drop(columns="team")
        addresses_df = addresses_df.merge(assignments_df, on="Wahlraum-A", how="left")
        st.success("Import erfolgreich – aktuelle Zuweisung wurde überschrieben.")

        # Neu berechnen
        graph = get_graph()
        for tid in assignments_df["team"].dropna().unique():
            team_rows = addresses_df[addresses_df["team"]==tid]
            optimized = tsp_solve_route(graph, team_rows)
            addresses_df.loc[optimized.index, "tsp_order"] = range(len(optimized))
        st.session_state.new_assignments = addresses_df.copy()

        with st.expander("📋 Vorschau der importierten Zuweisung"):
            st.dataframe(assignments_df)

    # Indizes zurücksetzen
    try:
        addresses_df = addresses_df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Fehler beim Zurücksetzen: {e}")

    if "new_assignments" not in st.session_state:
        st.session_state.new_assignments = addresses_df.copy()

    # Auswahl für manuelle Zuordnung
    stops = addresses_df["Wahlraum-A"].dropna().tolist() if isinstance(addresses_df, pd.DataFrame) else []
    selected = st.multiselect("Stops auswählen (nach Adresse)", options=stops)
    teams = sorted([int(t) for t in st.session_state.new_assignments["team"].dropna().unique()])
    target = st.selectbox("Ziel-Team auswählen", options=[None]+teams)
    if st.button("Zuweisung übernehmen") and target and selected:
        for addr in selected:
            idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"]==addr].index[0]
            st.session_state.new_assignments.at[idx, "team"] = target
        # Neu optimieren für betroffene Teams
        for tid in set([target]):
            rows = st.session_state.new_assignments[st.session_state.new_assignments["team"]==tid]
            optimized = tsp_solve_route(get_graph(), rows)
            st.session_state.new_assignments.loc[optimized.index, "tsp_order"] = range(len(optimized))
        st.rerun()

    # Neues Team
    if st.button("Neues Team erstellen"):
        st.session_state.show_new_team_form = True
    if st.session_state.get("show_new_team_form"):
        max_team = max([int(t) for t in st.session_state.new_assignments["team"].dropna().unique()]) if st.session_state.new_assignments["team"].notnull().any() else 0
        new_team = max_team+1
        with st.form(key="new_team_form", clear_on_submit=True):
            st.markdown(f"### Stop(s) für Team {new_team} auswählen")
            sel = st.multiselect("Stop(s)", options=st.session_state.new_assignments["Wahlraum-A"].dropna().tolist(), key="form_sel")
            if st.form_submit_button("Zuweisen und Team erstellen") and sel:
                for addr in sel:
                    idx = st.session_state.new_assignments[st.session_state.new_assignments["Wahlraum-A"]==addr].index[0]
                    st.session_state.new_assignments.at[idx, "team"] = new_team
                opt = tsp_solve_route(get_graph(), st.session_state.new_assignments[st.session_state.new_assignments["team"]==new_team])
                st.session_state.new_assignments.loc[opt.index, "tsp_order"] = range(len(opt))
                st.success(f"Team {new_team} erstellt.")
                st.session_state.show_new_team_form=False
                st.rerun()

# Karte zeichnen
m = leafmap.Map(center=[st.session_state.base_addresses["lat"].mean(), st.session_state.base_addresses["lon"].mean()], zoom=12)
# Routenlinien
graph = get_graph()
color_list = ["#FF00FF", "#00FFFF", "#00FF00", "#FF0000", "#FFA500", "#FFFF00", "#00CED1", "#DA70D6", "#FF69B4", "#8A2BE2"]
for i, tid in enumerate(sorted(st.session_state.new_assignments["team"].dropna().unique())):
    df = st.session_state.new_assignments[st.session_state.new_assignments["team"]==tid]
    if "tsp_order" in df.columns:
        df = df.sort_values("tsp_order")
    coords = df[["lat","lon"]].values.tolist()
    if len(coords)>1:
        path_coords=[]
        nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
        for u,v in zip(nodes[:-1],nodes[1:]):
            try:
                path=nx.shortest_path(graph,u,v,weight="length")
                seg=[(graph.nodes[n]["y"],graph.nodes[n]["x"]) for n in path]
                path_coords.extend(seg)
            except:
                continue
        folium.PolyLine(path_coords, color=color_list[i%len(color_list)], weight=8, opacity=0.9, tooltip=f"Team {int(tid)}").add_to(m)

# Marker Cluster mit Team-Anzeige
marker_cluster = MarkerCluster()
unique_teams = sorted(st.session_state.new_assignments["team"].dropna().unique())
for _, row in st.session_state.new_assignments.dropna(subset=["lat","lon"]).iterrows():
    wahlraum_b = row.get("Wahlraum-B", "Keine Daten")
    wahlraum_a = row.get("Wahlraum-A", "Keine Daten")
    num_rooms = row.get("num_rooms", "n/a")
    team_id = row.get("team", None)
    popup_content = f"""
    <b>Team:</b> {int(team_id) if pd.notnull(team_id) else 'nicht zugewiesen'}<br>
    <b>{wahlraum_b}</b><br>
    <b>{wahlraum_a}</b><br>
    <b>Anzahl Räume:</b> {num_rooms}
    """
    popup_html = f"""
    <div style="max-width:500px; max-height:500px; overflow:auto;">{popup_content}</div>
    """
    if team_id in unique_teams:
        color = color_list[unique_teams.index(team_id)%len(color_list)]
    else:
        color = "#000000"
    marker = folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=500)
    )
    marker.add_to(marker_cluster)
marker_cluster.add_to(m)

m.to_streamlit(height=700)

# Export
if st.button("Zuordnung exportieren"):
    overview=[]
    team_sheets={}
    for idx, tid in enumerate(sorted(st.session_state.new_assignments["team"].dropna().unique()), start=1):
        df=st.session_state.new_assignments[st.session_state.new_assignments["team"]==tid]
        if "tsp_order" in df.columns:
            df=df.sort_values("tsp_order")
        rooms=df["num_rooms"].sum()
        travel_km=0; travel_min=0
        coords=df[["lat","lon"]].values.tolist()
        if len(coords)>1:
            nodes=[ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
            for u,v in zip(nodes[:-1],nodes[1:]):
                try:
                    length=nx.shortest_path_length(graph,u,v,weight="length")
                    travel_km+=length/1000; travel_min+=length/1000*2
                except:
                    pass
        service= int(rooms*10)
        total=service+travel_min
        gmaps="https://www.google.com/maps/dir/"+"/".join([f"{lat},{lon}" for lat, lon in coords])
        overview.append({
            "Kontrollbezirk": idx,
            "Anzahl Wahllokale": len(df),
            "Anzahl Stimmbezirke": rooms,
            "Wegstrecke (km)": round(travel_km,1),
            "Fahrtzeit (min)": int(travel_min),
            "Kontrollzeit (min)": service,
            "Gesamtzeit": str(timedelta(minutes=int(total))),
            "Google-Link": gmaps
        })
        rows=[]
        for j, (_, row) in enumerate(df.iterrows(), start=1):
            coords_str=f"{row['lat']},{row['lon']}"
            rows.append({
                "Bezirk": j,
                "Adresse": row["Wahlraum-A"],
                "Stimmbezirke": row.get("rooms","n/a"),
                "Anzahl Stimmbezirke": row.get("num_rooms","n/a"),
                "Google-Link": f"https://www.google.com/maps/search/?api=1&query={quote_plus(coords_str)}"
            })
        team_sheets[f"Bezirk_{idx}"]=pd.DataFrame(rows)
    overview_df=pd.DataFrame(overview)
    output=io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="Übersicht", index=False)
        for name, df in team_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    output.seek(0)
    st.download_button("📥 Excel herunterladen", data=output, file_name="routen_zuweisung_aktualisiert.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
