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
from openpyxl.utils import get_column_letter

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

# Seite konfigurieren
st.set_page_config(layout="wide")

# Basisdaten laden
if "base_addresses" not in st.session_state:
    base = pd.read_csv("cleaned_addresses.csv").reset_index(drop=True)
    initial_sheets = pd.read_excel("routes_optimized.xlsx", sheet_name=None)
    temp = []
    for name, df in initial_sheets.items():
        if name != "Übersicht" and "Adresse" in df.columns:
            team = int(name.split("_")[1])
            for addr in df["Adresse"]:
                temp.append((addr, team))
    assign_df = pd.DataFrame(temp, columns=["Wahlraum-A", "team"])
    merged = base.merge(assign_df, on="Wahlraum-A", how="left")
    st.session_state.base_addresses = merged.copy()
    st.session_state.new_assignments = merged.copy()

# Arbeitsdaten kopieren und latlong aufsplitten
df_assign = st.session_state.new_assignments.copy()
if 'latlong' in df_assign.columns and ('lat' not in df_assign.columns or 'lon' not in df_assign.columns):
    df_assign[['lat', 'lon']] = df_assign['latlong'].str.split(',', expand=True).astype(float)

# Excel-Export vorbereiten
output = io.BytesIO()
export_graph = get_graph()
overview = []
sheets = {}
for idx, t in enumerate(sorted(df_assign["team"].dropna().unique()), start=1):
    df_t = df_assign[df_assign["team"] == t]
    if "tsp_order" in df_t.columns:
        df_t = df_t.sort_values("tsp_order")
    rooms = df_t["num_rooms"].sum()
    travel_km, travel_min = 0.0, 0.0
    coords = df_t[["lat","lon"]].values.tolist()
    if len(coords) > 1:
        nodes = [ox.distance.nearest_nodes(export_graph, X=lon, Y=lat) for lat, lon in coords]
        for u, v in zip(nodes[:-1], nodes[1:]):
            try:
                length = nx.shortest_path_length(export_graph, u, v, weight='length')
                travel_km += length / 1000.0
                travel_min += (length / 1000.0) * 2.0
            except:
                pass
    ctrl_time = int(rooms * 10)
    total_time = travel_min + ctrl_time
    overview.append({
        "Kontrollbezirk": idx,
        "Anzahl Wahllokale": len(df_t),
        "Anzahl Stimmbezirke": rooms,
        "Wegstrecke (km)": round(travel_km, 1),
        "Fahrtzeit (min)": int(travel_min),
        "Kontrollzeit (min)": ctrl_time,
        "Gesamtzeit": str(timedelta(minutes=int(total_time))),
        "Google-Link": "https://www.google.com/maps/dir/" + "/".join([f"{lat},{lon}" for lat, lon in coords])
    })
    detail = []
    for j, (_, r) in enumerate(df_t.iterrows(), start=1):
        coord = f"{r['lat']},{r['lon']}"
        detail.append({
            "Bezirk": j,
            "Adresse": r['Wahlraum-A'],
            "Stimmbezirke": r.get('rooms', ''),
            "Anzahl Stimmbezirke": r.get('num_rooms', ''),
            "Google-Link": f"https://www.google.com/maps/search/?api=1&query={quote_plus(coord)}"
        })
    sheets[f"Bezirk_{idx}"] = pd.DataFrame(detail)
# Excel schreiben und Spaltenbreiten anpassen
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    pd.DataFrame(overview).to_excel(writer, sheet_name='Übersicht', index=False)
    for name, df_s in sheets.items():
        df_s.to_excel(writer, sheet_name=name, index=False)
    for ws in writer.sheets.values():
        for col_cells in ws.columns:
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
            col_letter = get_column_letter(col_cells[0].column)
            ws.column_dimensions[col_letter].width = max_length + 2
output.seek(0)

# Sidebar Controls + Export
with st.sidebar:
    st.title('Bearbeitung Kontrollbezirke')
    # Suchfeld
    opts = df_assign.dropna(subset=['Wahlraum-B','Wahlraum-A'])
    addrs = opts.apply(lambda r: f"{r['Wahlraum-B']} - {r['Wahlraum-A']}", axis=1).tolist()
    st.selectbox(
        'Wahllokal oder Adresse suchen',
        options=[''] + addrs,
        index=0,
        format_func=lambda x: 'Wahllokal oder Adresse suchen' if x == '' else x,
        key='search_selection'
    )
    # Datei-Upload und Zuweisungs-Logik bleibt unverändert...
    st.download_button(
        label='Kontrollbezirke herunterladen',
        data=output,
        file_name='routen_zuweisung.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# Map initialisieren und anzeigen
search_sel = st.session_state.get('search_selection', '')
if search_sel:
    addr = search_sel.split(' - ',1)[1]
    row = df_assign[df_assign['Wahlraum-A']==addr]
    center = [row.iloc[0]['lat'], row.iloc[0]['lon']] if not row.empty else [df_assign['lat'].mean(), df_assign['lon'].mean()]
    zoom = 17
else:
    center = [df_assign['lat'].mean(), df_assign['lon'].mean()]
    zoom = 10
m = leafmap.Map(center=center, zoom=zoom)
# Routenlinien
colors = ['#FF00FF','#00FFFF','#00FF00','#FF0000','#FFA500','#FFFF00','#00CED1','#DA70D6','#FF69B4','#8A2BE2']
for i, t in enumerate(sorted(df_assign['team'].dropna().unique())):
    df_t = df_assign[df_assign['team']==t]
    if 'tsp_order' in df_t.columns:
        df_t = df_t.sort_values('tsp_order')
    pts = df_t[['lat','lon']].values.tolist()
    if len(pts)>1:
        path=[]
        nodes=[ox.distance.nearest_nodes(export_graph,X=lon,Y=lat) for lat,lon in pts]
        for u,v in zip(nodes[:-1],nodes[1:]):
            try:
                p=nx.shortest_path(export_graph,u,v,weight='length')
                path.extend([(export_graph.nodes[n]['y'],export_graph.nodes[n]['x']) for n in p])
            except:
                pass
        folium.PolyLine(path,color=colors[i%len(colors)],weight=6,opacity=0.8,tooltip=f"Kontrollbezirk {int(t)}").add_to(m)
# MarkerCluster
cluster = MarkerCluster(disableClusteringAtZoom=13)
for _,r in df_assign.dropna(subset=['lat','lon']).iterrows():
    popup = folium.Popup(html=f"<b>Kontrollbezirk:</b> {int(r['team'])}<br>{r['Wahlraum-B']}<br>{r['Wahlraum-A']}<br>Anzahl Räume: {r['num_rooms']}")
    cluster.add_child(folium.Marker(location=[r['lat'],r['lon']],popup=popup))
cluster.add_to(m)
# Full extent bei keiner Suche\ nif not search_sel:
    m.fit_bounds(df_assign[['lat','lon']].values.tolist())
# Karte anzeigen
m.to_streamlit(use_container_width=True, height=700)
