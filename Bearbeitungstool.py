"""Interaktives Routenbearbeitungstool mit Zoom-abhängiger Markersichtbarkeit und TSP-Optimierung bei Zuweisung"""

import streamlit as st
import pandas as pd
import leafmap.foliumap as leafmap
import folium
from folium.plugins import MarkerCluster, Search
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

# Seite konfigurieren
st.set_page_config(layout="wide")

# Basisdaten laden
if "base_addresses" not in st.session_state:
    base = pd.read_csv("cleaned_addresses.csv").reset_index(drop=True)
    sheets = pd.read_excel("routes_optimized.xlsx", sheet_name=None)
    temp = []
    for name, df in sheets.items():
        if name != "Übersicht" and "Adresse" in df.columns:
            team = int(name.split("_")[1])
            for addr in df["Adresse"]:
                temp.append((addr, team))
    assign_df = pd.DataFrame(temp, columns=["Wahlraum-A", "team"])
    merged = base.merge(assign_df, on="Wahlraum-A", how="left")
    st.session_state.base_addresses = merged.copy()
    st.session_state.new_assignments = merged.copy()

# Arbeitsdaten kopieren
df_assign = st.session_state.new_assignments.copy()

# Excel-Export vorbereiten
output = io.BytesIO()
overview = []
sheets = {}
for idx, t in enumerate(sorted(df_assign["team"].dropna().unique()), start=1):
    df_t = df_assign[df_assign["team"] == t]
    if "tsp_order" in df_t.columns:
        df_t = df_t.sort_values("tsp_order")
    rooms = df_t["num_rooms"].sum()
    km = mn = 0
    pts = df_t[["lat","lon"]].values.tolist()
    if len(pts) > 1:
        nodes = [ox.distance.nearest_nodes(get_graph(), X=lon, Y=lat) for lat, lon in pts]
        for u, v in zip(nodes[:-1], nodes[1:]):
            try:
                l = nx.shortest_path_length(graph, u, v, weight="length")
                km += l/1000
                mn += l/1000*2
            except:
                pass
    svc = int(rooms*10)
    total = svc + mn
    overview.append({
        "Kontrollbezirk": idx,
        "Anzahl Wahllokale": len(df_t),
        "Anzahl Stimmbezirke": rooms,
        "Wegstrecke (km)": round(km,1),
        "Fahrtzeit (min)": int(mn),
        "Kontrollzeit (min)": svc,
        "Gesamtzeit": str(timedelta(minutes=int(total))),
        "Google-Link": "https://www.google.com/maps/dir/" + "/".join([f"{lat},{lon}" for lat, lon in pts])
    })
    detail = []
    for j, (_, r) in enumerate(df_t.iterrows(), start=1):
        coord = f"{r['lat']},{r['lon']}"
        detail.append({
            "Bezirk": j,
            "Adresse": r['Wahlraum-A'],
            "Stimmbezirke": r.get('rooms',''),
            "Anzahl Stimmbezirke": r.get('num_rooms',''),
            "Google-Link": f"https://www.google.com/maps/search/?api=1&query={quote_plus(coord)}"
        })
    sheets[f"Bezirk_{idx}"] = pd.DataFrame(detail)
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    pd.DataFrame(overview).to_excel(writer, sheet_name="Übersicht", index=False)
    for name, df_s in sheets.items():
        df_s.to_excel(writer, sheet_name=name, index=False)
output.seek(0)

# Sidebar Controls + Export
with st.sidebar:
    st.title("Bearbeitung Kontrollbezirke")
    uploaded = st.file_uploader("Alternative Zuweisung importieren", type=["xlsx"])
    if uploaded:
        imp = pd.read_excel(uploaded, sheet_name=None)
        temp = []
        for name, df in imp.items():
            if name != "Übersicht" and "Adresse" in df.columns:
                team = int(name.split("_")[1])
                for addr in df["Adresse"]:
                    temp.append((addr, team))
        assigns = pd.DataFrame(temp, columns=["Wahlraum-A","team"])
        st.session_state.new_assignments = (
            st.session_state.base_addresses.drop(columns=["team"]) 
            .merge(assigns, on="Wahlraum-A", how="left")
        )
        st.success("Import erfolgreich.")
        
    df_opts = st.session_state.new_assignments.dropna(subset=["Wahlraum-B","Wahlraum-A"])
    addrs = df_opts.apply(lambda r: f"{r['Wahlraum-B']} - {r['Wahlraum-A']}", axis=1).tolist()
    sel = st.multiselect("Wahllokal wählen", options=addrs, placeholder="Auswählen")
    teams = sorted(st.session_state.new_assignments["team"].dropna().astype(int).unique())
    tgt = st.selectbox("Kontrollbezirk wählen", options=[None]+teams, format_func=lambda x: 'Auswählen' if x is None else str(x))
    if st.button("Zuweisung übernehmen") and tgt is not None and sel:
        for label in sel:
            addr = label.split(" - ",1)[1]
            idx = st.session_state.new_assignments.index[st.session_state.new_assignments["Wahlraum-A"]==addr][0]
            st.session_state.new_assignments.at[idx,"team"] = tgt
        df_t = st.session_state.new_assignments[st.session_state.new_assignments["team"]==tgt]
        opt = tsp_solve_route(get_graph(), df_t)
        st.session_state.new_assignments.loc[opt.index,"tsp_order"] = range(len(opt))
        st.success("Zuweisung gesetzt.")
        
    if st.button("Neuen Kontrollbezirk erstellen"):
        max_t = int(st.session_state.new_assignments["team"].max(skipna=True) or 0)+1
        sel2 = st.multiselect(f"Stops für Team {max_t}", options=addrs, placeholder="Auswählen", key="new_team_sel")
        if st.button("Erstellen und zuweisen") and sel2:
            for a in sel2:
                idx = st.session_state.new_assignments.index[st.session_state.new_assignments["Wahlraum-A"]==a.split(' - ',1)[1]][0]
                st.session_state.new_assignments.at[idx,"team"]=max_t
            df_nt = st.session_state.new_assignments[st.session_state.new_assignments["team"]==max_t]
            opt2 = tsp_solve_route(get_graph(), df_nt)
            st.session_state.new_assignments.loc[opt2.index,"tsp_order"]=range(len(opt2))
            st.success(f"Kontrollbezirk {max_t} erstellt.")
            
    if st.button("Routen berechnen"):
        graph = get_graph()
        for tid in sorted(st.session_state.new_assignments["team"].dropna().unique()):
            team_rows = st.session_state.new_assignments[st.session_state.new_assignments["team"]==tid]
            opt = tsp_solve_route(graph, team_rows)
            st.session_state.new_assignments.loc[opt.index,"tsp_order"]=range(len(opt))
        st.success("Routen neu berechnet.")
        
    st.download_button(
        label="Kontrollbezirke herunterladen",
        data=output,
        file_name="routen_zuweisung.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Karte
# Lade Graph für Routing
graph = get_graph()
m = leafmap.Map(center=[df_assign["lat"].mean(),df_assign["lon"].mean()], zoom=10)
col = ["#FF00FF","#00FFFF","#00FF00","#FF0000","#FFA500","#FFFF00","#00CED1","#DA70D6","#FF69B4","#8A2BE2"]
for i,t in enumerate(sorted(df_assign["team"].dropna().unique())):
    df_t = df_assign[df_assign["team"]==t]
    if "tsp_order" in df_t.columns:
        df_t=df_t.sort_values("tsp_order")
    pts = df_t[["lat","lon"]].values.tolist()
    if len(pts)>1:
        path=[]
        nodes=[ox.distance.nearest_nodes(graph,X=lon,Y=lat) for lat,lon in pts]
        for u,v in zip(nodes[:-1],nodes[1:]):
            try:
                p=nx.shortest_path(graph,u,v,weight="length")
                path.extend([(graph.nodes[n]["y"],graph.nodes[n]["x"]) for n in p])
            except:
                pass
        folium.PolyLine(path,color=col[i%len(col)],weight=6,opacity=0.8,tooltip=f"Team {int(t)}").add_to(m)
mc = MarkerCluster(disableClusteringAtZoom=13)
for _,r in df_assign.dropna(subset=["lat","lon"]).iterrows():
    html=f"<div style='max-width:200px'><b>Team:</b> {int(r['team']) if pd.notnull(r['team']) else 'n/a'}<br>{r['Wahlraum-B']}<br>{r['Wahlraum-A']}<br>Anzahl Räume: {r['num_rooms']}</div>"
    mc.add_child(folium.Marker(location=[r['lat'],r['lon']],popup=html))
mc.add_to(m)
# Suchfeld für Marker
Search(
    layer=mc,
    search_label='popup',
    placeholder='Suchen...',
    collapsed=False,
    position='topleft'
).add_to(m)
m.fit_bounds(df_assign[['lat','lon']].values.tolist())
m.to_streamlit(use_container_width=True,height=700)
