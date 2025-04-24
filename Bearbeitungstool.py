import streamlit as st
import pandas as pd
import leafmap.foliumap as leafmap
import folium
from folium.plugins import MarkerCluster
import networkx as nx
import osmnx as ox
import pickle
from networkx.algorithms.approximation import greedy_tsp, christofides
import random
import math
from datetime import timedelta
from urllib.parse import quote_plus
import io

# Laden des Graphen für TSP-Routing
@st.cache_resource
def get_graph():
    with open("munster_graph.pickle", "rb") as f:
        return pickle.load(f)

# 2-Opt Optimierung
```python
def two_opt(G, weight='weight'):
    n = G.number_of_nodes()
    tour = greedy_tsp(G, weight=weight)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i+1, n):
                if j - i == 1:
                    continue
                a, b = tour[i-1], tour[i]
                c, d = tour[j-1], tour[j % n]
                if G[a][c][weight] + G[b][d][weight] < G[a][b][weight] + G[c][d][weight]:
                    tour[i:j] = list(reversed(tour[i:j]))
                    improved = True
    return tour
```
# Simulated Annealing Optimierung
def simulated_annealing(G, weight='weight', initial_temp=10000, cooling_rate=0.995, stopping_temp=1e-3, max_iter=10000):
    def tour_length(tour):
        return sum(G[tour[i]][tour[(i+1) % len(tour)]][weight] for i in range(len(tour)))
    n = G.number_of_nodes()
    current_tour = greedy_tsp(G, weight=weight)
    best_tour = current_tour.copy()
    current_len = tour_length(current_tour)
    best_len = current_len
    T = initial_temp
    iteration = 0
    while T > stopping_temp and iteration < max_iter:
        i, j = sorted(random.sample(range(n), 2))
        if i == j:
            continue
        new_tour = current_tour.copy()
        new_tour[i:j] = reversed(new_tour[i:j])
        new_len = tour_length(new_tour)
        delta = new_len - current_len
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_tour = list(new_tour)
            current_len = new_len
            if current_len < best_len:
                best_tour = current_tour.copy()
                best_len = current_len
        T *= cooling_rate
        iteration += 1
    return best_tour

# Generalisierte TSP-Löser-Funktion
def tsp_solve_route(graph, stops_df, method="Greedy"):
    if len(stops_df) <= 2:
        return stops_df.copy().reset_index(drop=True)
    coords = list(zip(stops_df["lat"], stops_df["lon"]))
    nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
    H = nx.complete_graph(len(nodes))
    for i in H.nodes():
        for j in H.nodes():
            if i == j:
                continue
            try:
                length = nx.shortest_path_length(graph, nodes[i], nodes[j], weight='length')
            except:
                length = float('inf')
            H[i][j]['weight'] = length
    if method == "Greedy":
        path = greedy_tsp(H, weight='weight')
    elif method == "Christofides":
        path = christofides(H, weight='weight')
    elif method == "2-Opt":
        path = two_opt(H, weight='weight')
    elif method == "Simulated Annealing":
        path = simulated_annealing(H, weight='weight')
    else:
        path = greedy_tsp(H, weight='weight')
    return stops_df.iloc[path].reset_index(drop=True)

# Batch-Optimierung für alle oder ausgewählte Teams
def optimize_routes(algo, target, selected_team=None):
    graph = get_graph()
    df = st.session_state.new_assignments
    teams = ([selected_team] if target == "Ausgewähltes Team" and selected_team is not None else sorted(df.team.dropna().unique()))
    for team in teams:
        stops = df[df.team == team]
        optimized = tsp_solve_route(graph, stops, method=algo)
        st.session_state.new_assignments.loc[optimized.index, "tsp_order"] = range(len(optimized))
    st.success(f"Routen für {target.lower()} mit '{algo}' optimiert.")
    st.rerun()

# Streamlit UI
st.set_page_config(layout="wide")
with st.sidebar:
    st.title("Interaktives Tool zur Routenbearbeitung")

    # Basisdaten laden
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

    # Optionaler Import zur Überschreibung
    uploaded_file = st.file_uploader("Importiere alternative Zuweisung (Excel-Datei)", type=["xlsx"])
    if uploaded_file:
        imported_team_df = pd.read_excel(uploaded_file, sheet_name=None)
        imported = []
        for sheet, df in imported_team_df.items():
            if sheet != "Übersicht" and "Adresse" in df.columns:
                team = int(sheet.split("_")[1])
                for addr in df["Adresse"]:
                    imported.append((addr, team))
        assignments_df = pd.DataFrame(imported, columns=["Wahlraum-A", "team"])
        addresses_df = addresses_df.drop(columns=["team"], errors='ignore')
        addresses_df = addresses_df.merge(assignments_df, on="Wahlraum-A", how="left")
        st.session_state.new_assignments = addresses_df.copy()
        st.success("Import erfolgreich – aktuelle Zuweisung wurde überschrieben.")
        with st.expander("📋 Vorschau der importierten Zuweisung"):
            st.dataframe(assignments_df)

    addresses_df = st.session_state.new_assignments.reset_index(drop=True)

    # Stopps auswählen und Team
    selected_indices = st.multiselect("Stops auswählen (nach Adresse)", options=addresses_df['Wahlraum-A'].tolist())
    existing_teams = sorted([int(t) for t in addresses_df.team.dropna().unique()])
    selected_team = st.selectbox("Ziel-Team auswählen", options=[None]+existing_teams)

    # Algorithmus und Target
    algo = st.selectbox("Optimierungs-Algorithmus wählen", ("Greedy","2-Opt","Simulated Annealing","Christofides"))
    target = st.radio("Zu optimierende Route", ("Alle Teams","Ausgewähltes Team"))
    if st.button("Routen optimieren"):
        optimize_routes(algo, target, selected_team)

    # Übernehmen oder neues Team
    if st.button("Zuweisung übernehmen") and selected_team is not None and selected_indices:
        graph = get_graph()
        for addr in selected_indices:
            idx = addresses_df[addresses_df['Wahlraum-A']==addr].index[0]
            st.session_state.new_assignments.at[idx,'team'] = selected_team
        for team_id in set([selected_team]):
            rows = st.session_state.new_assignments[st.session_state.new_assignments.team==team_id]
            opt = tsp_solve_route(graph, rows)
            st.session_state.new_assignments.loc[opt.index,'tsp_order'] = range(len(opt))
        st.rerun()
    if st.button("Neues Team erstellen"):
        st.session_state.show_new_team_form = True
    if st.session_state.get("show_new_team_form"):
        max_team = max(existing_teams)+1 if existing_teams else 1
        with st.form("neues_team"):  
            stops = st.multiselect("Stops für Team {max_team}", options=addresses_df['Wahlraum-A'].tolist())
            if st.form_submit_button("Team erstellen") and stops:
                for addr in stops:
                    idx = addresses_df[addresses_df['Wahlraum-A']==addr].index[0]
                    st.session_state.new_assignments.at[idx,'team']=max_team
                st.session_state.show_new_team_form=False
                st.rerun()

# Karte
addresses_df = st.session_state.new_assignments
m = leafmap.Map(center=[addresses_df['lat'].mean(),addresses_df['lon'].mean()],zoom=12)
for team_id in sorted(addresses_df.team.dropna().unique()):
    team_rows=addresses_df[addresses_df.team==team_id]
    if 'tsp_order' in team_rows.columns:
        team_rows=team_rows.sort_values('tsp_order')
    coords=team_rows[['lat','lon']].values.tolist()
    if len(coords)>1:
        nodes=[ox.distance.nearest_nodes(get_graph(),X=lon,Y=lat) for lat,lon in coords]
        path=[]
        for u,v in zip(nodes[:-1],nodes[1:]):
            path+=[(get_graph().nodes[n]['y'],get_graph().nodes[n]['x']) for n in nx.shortest_path(get_graph(),u,v,weight='length')]
        folium.PolyLine(path,weight=6,opacity=0.8).add_to(m)
marker_cluster=MarkerCluster()
for _,row in addresses_df.dropna(subset=['lat','lon']).iterrows():
    popup=f"<b>{row['Wahlraum-A']}</b><br>Anz Räume:{row.get('num_rooms','')}"
    folium.Marker([row['lat'],row['lon']],popup=popup).add_to(marker_cluster)
marker_cluster.add_to(m)
m.to_streamlit(height=700)

# Export
if st.button('Zuordnung exportieren'):
    overview=[]
    sheets={}
    for t in sorted(addresses_df.team.dropna().unique()):
        s=addresses_df[addresses_df.team==t]
        if 'tsp_order' in s.columns: s=s.sort_values('tsp_order')
        coords=s[['lat','lon']].values.tolist()
        rows=[]
        for i,row in s.iterrows():
            rows.append({'Reihenfolge':i,'Adresse':row['Wahlraum-A']})
        sheets[f'Team_{t}']=pd.DataFrame(rows)
    out=io.BytesIO()
    writer=pd.ExcelWriter(out,engine='openpyxl')
    pd.DataFrame([{'Team':t,'Stops':len(sheets[f'Team_{t}'])} for t in sheets]).to_excel(writer,'Übersicht',index=False)
    for name,df in sheets.items(): df.to_excel(writer,name,index=False)
    writer.save();out.seek(0)
    st.download_button('📥 Excel',data=out,file_name='zuweisung.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
