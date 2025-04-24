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

# Initialisiere Session State
if "action_log" not in st.session_state:
    st.session_state.action_log = []
if "show_map" not in st.session_state:
    st.session_state.show_map = True  # initial map display
if "show_new_team_form" not in st.session_state:
    st.session_state.show_new_team_form = False

# Laden des Graphen für TSP-Routing
@st.cache_resource
def get_graph():
    with open("munster_graph.pickle", "rb") as f:
        return pickle.load(f)

# 2-Opt Optimierung
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
        new_tour = current_tour.copy()
        new_tour[i:j] = list(reversed(new_tour[i:j]))
        new_len = tour_length(new_tour)
        delta = new_len - current_len
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_tour = new_tour
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
        return stops_df.reset_index(drop=True)
    coords = list(zip(stops_df["lat"], stops_df["lon"]))
    nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
    H = nx.complete_graph(len(nodes))
    for i in H.nodes():
        for j in H.nodes():
            if i == j: continue
            try:
                H[i][j]['weight'] = nx.shortest_path_length(graph, nodes[i], nodes[j], weight='length')
            except:
                H[i][j]['weight'] = float('inf')
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

# Batch-Optimierung
def optimize_routes(algo, target, selected_team=None):
    graph = get_graph()
    df = st.session_state.new_assignments
    teams = ([selected_team] if target == "Ausgewähltes Team" and selected_team else sorted(df.team.dropna().unique()))
    for team in teams:
        stops = df[df.team == team]
        optimized = tsp_solve_route(graph, stops, method=algo)
        st.session_state.new_assignments.loc[optimized.index, 'tsp_order'] = range(len(optimized))
    st.session_state.action_log.append(f"Routen optimiert mit '{algo}' für {target}.")
    st.session_state.show_map = True
    st.experimental_rerun()

# Generiere Excel Bytes
def _generate_excel_bytes():
    graph = get_graph()
    overview = []
    team_sheets = {}
    for team in sorted(st.session_state.new_assignments.team.dropna().unique()):
        stops = st.session_state.new_assignments[st.session_state.new_assignments.team == team]
        if 'tsp_order' in stops.columns:
            stops = stops.sort_values('tsp_order')
        rooms = stops.get('num_rooms', pd.Series()).sum()
        travel_km = travel_min = 0
        coords = stops[['lat','lon']].values.tolist()
        # compute nearest nodes once for performance
        nodes = []
        for lat, lon in coords:
            try:
                node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
            except Exception:
                node = None
            nodes.append(node)
        # sum up travel distances
        for u, v in zip(nodes[:-1], nodes[1:]):
            if u is None or v is None:
                continue
            try:
                length = nx.shortest_path_length(graph, u, v, weight='length')
                travel_km += length / 1000
                travel_min += length / 1000 * 2
            except Exception:
                pass
            try:
                length = nx.shortest_path_length(graph, u, v, weight='length')
                travel_km += length/1000
                travel_min += length/1000*2
            except:
                pass
        service_min = int(rooms*10)
        total_min = service_min + travel_min
        gmaps_link = "https://www.google.com/maps/dir/" + "/".join([f"{lat},{lon}" for lat,lon in coords])
        overview.append({
            'Kontrollbezirk': team,
            'Anzahl Wahllokale': len(stops),
            'Anzahl Stimmbezirke': int(rooms),
            'Wegstrecke (km)': round(travel_km,1),
            'Fahrtzeit (min)': int(travel_min),
            'Kontrollzeit (min)': service_min,
            'Gesamtzeit': str(timedelta(minutes=int(total_min))),
            'Google-Link': gmaps_link
        })
        rows=[]
        for i,row in stops.iterrows():
            coord_str=f"{row['lat']},{row['lon']}"
            rows.append({
                'Reihenfolge':i,'Adresse':row['Wahlraum-A'],
                'Stimmbezirke':row.get('rooms',''),
                'Anzahl Stimmbezirke':row.get('num_rooms',''),
                'Google-Link':f"https://www.google.com/maps/search/?api=1&query={quote_plus(coord_str)}"
            })
        team_sheets[f'Team_{team}'] = pd.DataFrame(rows)
    buf=io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_over = pd.DataFrame(overview)
        df_over.to_excel(writer, sheet_name='Übersicht', index=False)
        from openpyxl.utils import get_column_letter
        ws=writer.sheets['Übersicht']
        for idx,col in enumerate(df_over.columns,1):
            width=max(df_over[col].astype(str).map(len).max(),len(col))+2
            ws.column_dimensions[get_column_letter(idx)].width=width
        for name,df_ in team_sheets.items():
            df_.to_excel(writer, sheet_name=name, index=False)
            ws2=writer.sheets[name]
            for idx,col in enumerate(df_.columns,1):
                width=max(df_[col].astype(str).map(len).max(),len(col))+2
                ws2.column_dimensions[get_column_letter(idx)].width=width
    buf.seek(0)
    return buf.getvalue()

# Streamlit UI setup
st.set_page_config(layout='wide')

with st.sidebar:
    st.title('Interaktives Tool zur Routenbearbeitung')

    # Load base addresses once
    if 'base_addresses' not in st.session_state:
        base_addresses = pd.read_csv('cleaned_addresses.csv')
        team_df = pd.read_excel('routes_optimized.xlsx', sheet_name=None)
        assigns=[]
        for sheet,df in team_df.items():
            if sheet!='Übersicht' and 'Adresse' in df:
                t=int(sheet.split('_')[1])
                assigns += [(addr,t) for addr in df['Adresse']]
        st.session_state.base_addresses = base_addresses.merge(pd.DataFrame(assigns,columns=['Wahlraum-A','team']), on='Wahlraum-A', how='left')
        st.session_state.new_assignments = st.session_state.base_addresses.copy()

    addresses_df = st.session_state.new_assignments.reset_index(drop=True)
    # Precompute node IDs to avoid repeated nearest_nodes calls
    if 'node_id' not in addresses_df.columns:
        g = get_graph()
        addresses_df['node_id'] = addresses_df.apply(lambda r: ox.distance.nearest_nodes(g, X=r['lon'], Y=r['lat']), axis=1)
        st.session_state.new_assignments = addresses_df.copy()


    # Auswahl & Buttons
    selected = st.multiselect('Stops auswählen',options=addresses_df['Wahlraum-A'])
    teams = sorted(addresses_df.team.dropna().unique())
    sel_team = st.selectbox('Team wählen',[None]+teams)
    algo = st.selectbox('Algorithmus',('Greedy','2-Opt','Simulated Annealing','Christofides'))
    target = st.radio('Ziel',('Alle Teams','Ausgewähltes Team'))

    if st.button('Routen optimieren'):
        optimize_routes(algo,target,sel_team)
    if st.button('Zuweisung übernehmen') and sel_team and selected:
        g=get_graph()
        for addr in selected:
            idx=addresses_df[addresses_df['Wahlraum-A']==addr].index[0]
            st.session_state.new_assignments.at[idx,'team']=sel_team
        st.session_state.action_log.append(f'Zuweisung übernommen: {len(selected)}→Team{sel_team}')
        st.session_state.show_map=True
        st.experimental_rerun()
    if st.button('Neues Team erstellen'):
        st.session_state.show_new_team_form=True
    if st.session_state.show_new_team_form:
        max_t = max(teams)+1 if teams else 1
        with st.form('newteam'): stops2=st.multiselect(f'Stops für Team{max_t}',options=addresses_df['Wahlraum-A'])
        if st.form_submit_button('Erstellen') and stops2:
            for addr in stops2:
                idx=addresses_df[addresses_df['Wahlraum-A']==addr].index[0]
                st.session_state.new_assignments.at[idx,'team']=max_t
            st.session_state.action_log.append(f'Team{max_t} erstellt mit{len(stops2)}Stops')
            st.session_state.show_map=True
    # Export-Download
    excel_bytes=_generate_excel_bytes()
    st.download_button('📥 Export & Download Excel',data=excel_bytes,file_name='zuweisung.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    st.markdown('---')
    st.subheader('Aktionen-Log')
    for e in st.session_state.action_log:
        st.write(f'- {e}')

# Karte nur wenn show_map
if st.session_state.show_map:
    dfm = st.session_state.new_assignments
    m = leafmap.Map(center=[dfm.lat.mean(), dfm.lon.mean()], zoom=12)
    cols=["#FF0000","#00FF00","#0000FF","#FFA500","#800080","#008080","#FFD700","#FF1493","#40E0D0","#A52A2A"]
    g=get_graph()
    for i,t in enumerate(sorted(dfm.team.dropna().unique())):
        rows=dfm[dfm.team==t]
        if 'tsp_order' in rows: rows=rows.sort_values('tsp_order')
        # use precomputed node_ids instead of recomputing
            nodes = rows['node_id'].tolist()
            # build path segments
            path = []
            for u, v in zip(nodes[:-1], nodes[1:]):
                if u is None or v is None:
                    continue
                # shortest path between nodes
                segment = nx.shortest_path(g, u, v, weight='length')
                path.extend([(g.nodes[n]['y'], g.nodes[n]['x']) for n in segment])
            folium.PolyLine(path, color=cols[i%len(cols)], weight=6, opacity=0.8, tooltip=f'Route{t}').add_to(m) path+=[(g.nodes[n]['y'],g.nodes[n]['x']) for n in nx.shortest_path(g,u,v,weight='length')]
            folium.PolyLine(path,color=cols[i%len(cols)],weight=6,opacity=0.8,tooltip=f'Route{t}').add_to(m)
    mc=MarkerCluster()
    for _,r in dfm.dropna(subset=['lat','lon']).iterrows():
        html=f"<div style='font-weight:bold;'><b>{r['Wahlraum-B']}</b><br><b>{r['Wahlraum-A']}</b><br><b>AnzRäume:</b>{r.get('num_rooms','')}</div>"
        folium.Marker([r.lat,r.lon],popup=folium.Popup(html,max_width=0)).add_to(mc)
    mc.add_to(m)
    m.to_streamlit(height=700)
