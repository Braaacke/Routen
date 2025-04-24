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

# Load graph for routing
@st.cache_resource
def get_graph():
    with open("munster_graph.pickle", "rb") as f:
        return pickle.load(f)

# Initialize session state
if "action_log" not in st.session_state:
    st.session_state.action_log = []
if "show_map" not in st.session_state:
    st.session_state.show_map = True
if "show_new_team_form" not in st.session_state:
    st.session_state.show_new_team_form = False

# Load base data and precompute node IDs (once)
if 'new_assignments' not in st.session_state:
    # Load addresses and assignments
    base_addresses = pd.read_csv('cleaned_addresses.csv')
    routes = pd.read_excel('routes_optimized.xlsx', sheet_name=None)
    assigns = []
    for sheet_name, df0 in routes.items():
        # Skip summary sheet and ensure 'Adresse' column exists
        if sheet_name == 'Übersicht' or 'Adresse' not in df0.columns:
            continue
        # Extract team ID and extend assigns list
        team_id = int(sheet_name.split('_')[1])
        assigns.extend([(addr, team_id) for addr in df0['Adresse']])
    df_full = base_addresses.merge(
        pd.DataFrame(assigns, columns=['Wahlraum-A','team']),
        on='Wahlraum-A', how='left'
    )
    # Compute nearest node IDs
    graph = get_graph()
    node_ids = []
    for _, row in df_full.iterrows():
        try:
            node_ids.append(ox.distance.nearest_nodes(graph, X=row['lon'], Y=row['lat']))
        except Exception:
            node_ids.append(None)
    df_full['node_id'] = node_ids
    st.session_state.new_assignments = df_full.copy()

# TSP heuristics

def two_opt(G, weight='weight'):
    n = G.number_of_nodes()
    tour = greedy_tsp(G, weight=weight)
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j - i == 1:
                    continue
                a, b = tour[i-1], tour[i]
                c, d = tour[j-1], tour[j % n]
                if G[a][c][weight] + G[b][d][weight] < G[a][b][weight] + G[c][d][weight]:
                    tour[i:j] = list(reversed(tour[i:j]))
                    improved = True
    return tour


def simulated_annealing(G, weight='weight', initial_temp=10000, cooling_rate=0.995,
                        stopping_temp=1e-3, max_iter=10000):
    def tour_length(tour):
        return sum(G[tour[i]][tour[(i+1)%len(tour)]][weight] for i in range(len(tour)))
    n = G.number_of_nodes()
    current = greedy_tsp(G, weight=weight)
    best, current_len = current.copy(), tour_length(current)
    best_len = current_len
    T = initial_temp
    for _ in range(max_iter):
        if T < stopping_temp:
            break
        i, j = sorted(random.sample(range(n), 2))
        new = current.copy()
        new[i:j] = list(reversed(new[i:j]))
        new_len = tour_length(new)
        delta = new_len - current_len
        if delta < 0 or random.random() < math.exp(-delta / T):
            current, current_len = new, new_len
            if new_len < best_len:
                best, best_len = new.copy(), new_len
        T *= cooling_rate
    return best


def tsp_solve_route(graph, stops_df, method="Greedy"):
    if len(stops_df) <= 2:
        return stops_df.reset_index(drop=True)
    node_ids = stops_df['node_id'].tolist()
    H = nx.complete_graph(len(node_ids))
    for i in H.nodes:
        for j in H.nodes:
            if i == j:
                continue
            ni, nj = node_ids[i], node_ids[j]
            if ni is None or nj is None:
                w = float('inf')
            else:
                try:
                    w = nx.shortest_path_length(graph, ni, nj, weight='length')
                except:
                    w = float('inf')
            H[i][j]['weight'] = w
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

# Batch optimization handler

def optimize_routes(algo, target, selected_team=None):
    graph = get_graph()
    df = st.session_state.new_assignments
    # Ensure node_id column exists
    if 'node_id' not in df.columns:
        g = get_graph()
        node_list = []
        for _, row in df.iterrows():
            try:
                nid = ox.distance.nearest_nodes(g, X=row['lon'], Y=row['lat'])
            except:
                nid = None
            node_list.append(nid)
        df['node_id'] = node_list
        st.session_state.new_assignments = df.copy()
    teams = ([selected_team] if target == 'Ausgewähltes Team' and selected_team else
             df['team'].dropna().unique())
    for t in teams:
        subset = df[df['team'] == t]
        optimized = tsp_solve_route(graph, subset, method=algo)
        st.session_state.new_assignments.loc[optimized.index, 'tsp_order'] = range(len(optimized))
    st.session_state.action_log.append(f"Routen optimiert ('{algo}', {target})")
    st.session_state.show_map = True
    st.experimental_rerun()

# Generate Excel report bytes

def _generate_excel_bytes():
    # Ensure node_id column exists before export
    df_export = st.session_state.new_assignments
    if 'node_id' not in df_export.columns:
        g = get_graph()
        node_list = []
        for _, row in df_export.iterrows():
            try:
                nid = ox.distance.nearest_nodes(g, X=row['lon'], Y=row['lat'])
            except:
                nid = None
            node_list.append(nid)
        df_export['node_id'] = node_list
        st.session_state.new_assignments = df_export.copy()
    # Begin export using updated DataFrame
    graph = get_graph()
    df = st.session_state.new_assignments
    # Ensure node_id exists in case new_assignments was modified
    df = st.session_state.new_assignments
    if 'node_id' not in df.columns:
        g = get_graph()
        node_list = []
        for _, row in df.iterrows():
            try:
                nid = ox.distance.nearest_nodes(g, X=row['lon'], Y=row['lat'])
            except:
                nid = None
            node_list.append(nid)
        df['node_id'] = node_list
        st.session_state.new_assignments = df.copy()
    graph = get_graph()
    graph = get_graph()
    df = st.session_state.new_assignments
    overview = []
    team_sheets = {}
    for t in sorted(df['team'].dropna().unique()):
        stops = df[df['team'] == t].copy()
        if 'tsp_order' in stops:
            stops.sort_values('tsp_order', inplace=True)
        rooms = int(stops.get('num_rooms', 0).sum())
        travel_km = travel_min = 0
        node_ids = stops['node_id'].tolist()
        for u, v in zip(node_ids[:-1], node_ids[1:]):
            if u is not None and v is not None:
                try:
                    dist = nx.shortest_path_length(graph, u, v, weight='length')
                except:
                    dist = 0
            else:
                dist = 0
            travel_km += dist / 1000
            travel_min += dist / 1000 * 2
        service_min = rooms * 10
        total_min = int(service_min + travel_min)
        link = "https://www.google.com/maps/dir/" + "/".join(
            f"{lat},{lon}" for lat, lon in stops[['lat','lon']].values
        )
        overview.append({
            'Kontrollbezirk': t,
            'Stops': len(stops),
            'Stimmbezirke': rooms,
            'Wegstrecke (km)': round(travel_km, 1),
            'Fahrtzeit (min)': int(travel_min),
            'Kontrollzeit (min)': service_min,
            'Gesamtzeit': str(timedelta(minutes=total_min)),
            'Google-Link': link
        })
        rows = []
        for idx, row in stops.iterrows():
            coord = f"{row['lat']},{row['lon']}"
            rows.append({
                'Reihenfolge': idx,
                'Adresse': row['Wahlraum-A'],
                'Stimmbezirke': row.get('rooms', ''),
                'Anzahl Stimmbezirke': row.get('num_rooms', ''),
                'Link': f"https://www.google.com/maps/search/?api=1&query={quote_plus(coord)}"
            })
        team_sheets[f'Team_{t}'] = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_over = pd.DataFrame(overview)
        df_over.to_excel(writer, sheet_name='Übersicht', index=False)
        from openpyxl.utils import get_column_letter
        ws0 = writer.sheets['Übersicht']
        for i, col in enumerate(df_over.columns, 1):
            max_len = max(df_over[col].astype(str).map(len).max(), len(col))
            ws0.column_dimensions[get_column_letter(i)].width = max_len + 2
        for name, df_sheet in team_sheets.items():
            df_sheet.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for i, col in enumerate(df_sheet.columns, 1):
                max_len = max(df_sheet[col].astype(str).map(len).max(), len(col))
                ws.column_dimensions[get_column_letter(i)].width = max_len + 2
    buf.seek(0)
    return buf.getvalue()

# Streamlit UI setup
st.set_page_config(layout='wide')
with st.sidebar:
    st.title('Routenbearbeitung')
    df = st.session_state.new_assignments
    stops = st.multiselect('Stops auswählen', options=df['Wahlraum-A'])
    teams = sorted(df['team'].dropna().unique())
    sel_team = st.selectbox('Team auswählen', [None] + teams)
    algo = st.selectbox('Algorithmus', ['Greedy', '2-Opt', 'Simulated Annealing', 'Christofides'])
    target = st.radio('Zu optimierende Route', ['Alle Teams', 'Ausgewähltes Team'])
    if st.button('Routen optimieren'):
        optimize_routes(algo, target, sel_team)
    if st.button('Zuweisung übernehmen') and sel_team and stops:
        for addr in stops:
            idx = df[df['Wahlraum-A'] == addr].index[0]
            st.session_state.new_assignments.at[idx, 'team'] = sel_team
        st.session_state.action_log.append(f'Zuweisung übernommen: {len(stops)} → Team {sel_team}')
        st.session_state.show_map = True
        st.experimental_rerun()
    if st.button('Neues Team erstellen'):
        st.session_state.show_new_team_form = True
    if st.session_state.show_new_team_form:
        next_team = max(teams) + 1 if teams else 1
        with st.form('new_team_form'):
            selected_new = st.multiselect(f'Stops für Team {next_team}', options=df['Wahlraum-A'])
            if st.form_submit_button('Team erstellen') and selected_new:
                for addr in selected_new:
                    idx = df[df['Wahlraum-A'] == addr].index[0]
                    st.session_state.new_assignments.at[idx, 'team'] = next_team
                st.session_state.action_log.append(f'Team {next_team} erstellt mit {len(selected_new)} Stops')
                st.session_state.show_map = True
    excel_bytes = _generate_excel_bytes()
    st.download_button(
        '📥 Export & Download Excel',
        data=excel_bytes,
        file_name='routen_zuweisung.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    st.markdown('---')
    st.subheader('Aktionen-Log')
    for entry in st.session_state.action_log:
        st.write(f'- {entry}')

if st.session_state.show_map:
    dfm = st.session_state.new_assignments
    graph = get_graph()
    m = leafmap.Map(center=[dfm['lat'].mean(), dfm['lon'].mean()], zoom=12)
    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFA500", "#800080"]
    for i, team_id in enumerate(sorted(dfm['team'].dropna().unique())):
    subset = dfm[dfm['team'] == team_id]
    if 'tsp_order' in subset.columns:
        subset = subset.sort_values('tsp_order')
    node_ids = subset['node_id'].tolist()
    path = []
    for u, v in zip(node_ids[:-1], node_ids[1:]):
        if u is None or v is None:
            continue
        try:
            seg = nx.shortest_path(graph, u, v, weight='length')
        except:
            continue
        path.extend([(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in seg])
    coords = list(zip(subset['lat'], subset['lon']))
    if path:
        folium.PolyLine(
            path,
            color=colors[i % len(colors)],
            weight=6,
            opacity=0.8,
            tooltip=f'Route {team_id}'
        ).add_to(m)
    elif len(coords) > 1:
        folium.PolyLine(
            coords,
            color=colors[i % len(colors)],
            weight=6,
            opacity=0.8,
            tooltip=f'Route {team_id}'
        ).add_to(m)
marker_cluster = MarkerCluster()()
    for _, row in dfm.dropna(subset=['lat', 'lon']).iterrows():
        popup_html = (
            f"<div style='font-weight:bold;'>"
            f"<b>{row['Wahlraum-B']}</b><br>"
            f"<b>{row['Wahlraum-A']}</b><br>"
            f"<b>Anzahl Räume:</b> {row.get('num_rooms', '')}"
            f"</div>"
        )
        folium.Marker(
            [row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=0)
        ).add_to(marker_cluster)
    marker_cluster.add_to(m)
    m.to_streamlit(height=700)
