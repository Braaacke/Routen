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

# Seite konfigurieren
st.set_page_config(layout="wide")

# Initiales Laden der Basisdaten
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

# Sidebar für Zuweisungen
with st.sidebar:
    st.title("Routenbearbeitung")

    uploaded = st.file_uploader("Alternative Zuweisung importieren", type=["xlsx"])
    if uploaded:
        imp = pd.read_excel(uploaded, sheet_name=None)
        temp = []
        for name, df in imp.items():
            if name != "Übersicht" and "Adresse" in df.columns:
                team = int(name.split("_")[1])
                for addr in df["Adresse"]:
                    temp.append((addr, team))
        assigns = pd.DataFrame(temp, columns=["Wahlraum-A", "team"])
        st.session_state.new_assignments = (
            st.session_state.base_addresses.drop(columns=["team"]) 
            .merge(assigns, on="Wahlraum-A", how="left")
        )
        st.success("Import erfolgreich.")

    # Manuelle Zuweisung
    addrs = st.session_state.new_assignments["Wahlraum-A"].dropna().tolist()
    sel = st.multiselect("Stops wählen", options=addrs)
    teams = sorted(st.session_state.new_assignments["team"].dropna().astype(int).unique())
    tgt = st.selectbox("Team auswählen", options=[None] + teams)
    if st.button("Zuweisung übernehmen") and tgt and sel:
        for a in sel:
            idx = st.session_state.new_assignments.index[st.session_state.new_assignments["Wahlraum-A"] == a][0]
            st.session_state.new_assignments.at[idx, "team"] = tgt
        df_t = st.session_state.new_assignments.loc[st.session_state.new_assignments["team"] == tgt]
        opt = tsp_solve_route(get_graph(), df_t)
        st.session_state.new_assignments.loc[opt.index, "tsp_order"] = range(len(opt))
        st.success("Zuweisung gesetzt.")

    # Neues Team
    if st.button("Neues Team erstellen"):
        max_t = int(st.session_state.new_assignments["team"].max(skipna=True) or 0) + 1
        sel2 = st.multiselect(f"Stops für Team {max_t}", options=addrs, key="new_team_sel")
        if st.button("Erstellen und zuweisen") and sel2:
            for a in sel2:
                idx = st.session_state.new_assignments.index[st.session_state.new_assignments["Wahlraum-A"] == a][0]
                st.session_state.new_assignments.at[idx, "team"] = max_t
            df_nt = st.session_state.new_assignments.loc[st.session_state.new_assignments["team"] == max_t]
            opt2 = tsp_solve_route(get_graph(), df_nt)
            st.session_state.new_assignments.loc[opt2.index, "tsp_order"] = range(len(opt2))
            st.success(f"Team {max_t} erstellt.")

# Arbeits-Daten setzen für Map und Export
df_assign = st.session_state.new_assignments.copy()

# Karte darstellen
m = leafmap.Map(center=[df_assign["lat"].mean(), df_assign["lon"].mean()], zoom=12)
# Routen
col = ["#FF00FF","#00FFFF","#00FF00","#FF0000","#FFA500","#FFFF00","#00CED1","#DA70D6","#FF69B4","#8A2BE2"]
for i, t in enumerate(sorted(df_assign["team"].dropna().unique())):
    df_t = df_assign[df_assign["team"] == t]
    if "tsp_order" in df_t.columns:
        df_t = df_t.sort_values("tsp_order")
    pts = df_t[["lat","lon"]].values.tolist()
    if len(pts) > 1:
        path = []
        nodes = [ox.distance.nearest_nodes(get_graph(), X=lon, Y=lat) for lat, lon in pts]
        for u,v in zip(nodes[:-1], nodes[1:]):
            try:
                p = nx.shortest_path(get_graph(), u, v, weight="length")
                path.extend([(get_graph().nodes[n]["y"], get_graph().nodes[n]["x"]) for n in p])
            except:
                pass
        folium.PolyLine(path, color=col[i % len(col)], weight=6, opacity=0.8, tooltip=f"Team {int(t)}").add_to(m)
# Marker
mc = MarkerCluster()
for _, r in df_assign.dropna(subset=["lat","lon"]).iterrows():
    html = f"<div style='max-width:200px'><b>Team:</b> {int(r['team']) if pd.notnull(r['team']) else 'n/a'}<br>"
    html += f"{r['Wahlraum-B']}<br>{r['Wahlraum-A']}<br>Anzahl Räume: {r['num_rooms']}</div>"
    marker = folium.Marker(location=[r['lat'], r['lon']], popup=html)
    mc.add_child(marker)
mc.add_to(m)

m.to_streamlit(height=700)

# Export als Excel
if st.button("Zuordnung exportieren"):
    overview=[]; sheets={}
    for idx, t in enumerate(sorted(df_assign["team"].dropna().unique()), start=1):
        df_t = df_assign[df_assign["team"] == t]
        df_t = df_t.sort_values("tsp_order") if "tsp_order" in df_t else df_t
        rooms = df_t["num_rooms"].sum()
        km = mn = 0
        pts = df_t[["lat","lon"]].values.tolist()
        if len(pts)>1:
            nodes = [ox.distance.nearest_nodes(get_graph(), X=lon, Y=lat) for lat, lon in pts]
            for u,v in zip(nodes[:-1], nodes[1:]):
                try:
                    l = nx.shortest_path_length(get_graph(), u, v, weight="length")
                    km += l/1000; mn += l/1000*2
                except:
                    pass
        service = int(rooms*10); total = service+mn
        overview.append({
            "Kontrollbezirk":idx,
            "Anzahl Wahllokale":len(df_t),
            "Anzahl Stimmbezirke":rooms,
            "Wegstrecke (km)":round(km,1),
            "Fahrtzeit (min)":int(mn),
            "Kontrollzeit (min)":service,
            "Gesamtzeit":str(timedelta(minutes=int(total))),
            "Google-Link":"https://www.google.com/maps/dir/"+"/".join([f"{lat},{lon}" for lat, lon in pts])
        })
        rows=[]
        for j,(_,r2) in enumerate(df_t.iterrows(), start=1):
            cr = f"{r2['lat']},{r2['lon']}"
            rows.append({
                "Bezirk":j,
                "Adresse":r2['Wahlraum-А'],
                "Stimmbezirke":r2['rooms'],
                "Anzahl Stimmbezirke":r2['num_rooms'],
                "Google-Link":f"https://www.google.com/maps/search/?api=1&query={quote_plus(cr)}"
            })
        sheets[f"Bezirk_{idx}"] = pd.DataFrame(rows)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(overview).to_excel(w, sheet_name="Übersicht", index=False)
        for name, df_s in sheets.items(): df_s.to_excel(w, sheet_name=name, index=False)
    out.seek(0)
    st.download_button("📥 Excel herunterladen", data=out, file_name="routen_zuweisung.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
