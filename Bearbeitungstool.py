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

# LOAD GRAPH FOR TSP
@st.cache_resource
def get_graph():
    with open("munster_graph.pickle", "rb") as f:
        return pickle.load(f)

# INITIALIZE SESSION STATE
if "action_log" not in st.session_state:
    st.session_state.action_log = []
if "show_map" not in st.session_state:
    st.session_state.show_map = True
if "show_new_team_form" not in st.session_state:
    st.session_state.show_new_team_form = False

# STREAMLIT LOAD BASE DATA AND PRECOMPUTE (only once)
if 'new_assignments' not in st.session_state:
    base_addresses = pd.read_csv('cleaned_addresses.csv')
    team_df = pd.read_excel('routes_optimized.xlsx', sheet_name=None)
    assigns = []
    for sh, df0 in team_df.items():
        if sh != 'Übersicht' and 'Adresse' in df0:
            tid = int(sh.split('_')[1])
            assigns += [(a, tid) for a in df0['Adresse']]
    df_full = base_addresses.merge(
        pd.DataFrame(assigns, columns=['Wahlraum-A','team']), on='Wahlraum-A', how='left')
    # precompute node IDs once
    g0 = get_graph()
    node_list = []
    for _, r in df_full.iterrows():
        try:
            nid = ox.distance.nearest_nodes(g0, X=r['lon'], Y=r['lat'])
        except:
            nid = None
        node_list.append(nid)
    df_full['node_id'] = node_list
    st.session_state.new_assignments = df_full.copy()

# TSP HEURISTICS

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
    nodes = stops_df['node_id'].tolist()
    H = nx.complete_graph(len(nodes))
    for i in H.nodes:
        for j in H.nodes:
            if i == j:
                continue
            ni, nj = nodes[i], nodes[j]
            try:
                dist = nx.shortest_path_length(graph, ni, nj, weight='length')
            except:
                dist = float('inf')
            H[i][j]['weight'] = dist
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

# BATCH OPTIMIZATION HANDLER

def optimize_routes(algo, target, selected_team=None):
    graph = get_graph()
    df = st.session_state.new_assignments
    teams = ([selected_team] if target == 'Ausgewähltes Team' and selected_team else
             df['team'].dropna().unique())
    for t in teams:
        subset = df[df['team'] == t]
        optimized = tsp_solve_route(graph, subset, method=algo)
        st.session_state.new_assignments.loc[optimized.index, 'tsp_order'] = range(len(optimized))
    st.session_state.action_log.append(f"Routen optimiert ('{algo}', {target})")
    st.session_state.show_map = True
    st.experimental_rerun()

# GENERATE EXCEL BYTES

def _generate_excel_bytes():
    graph = get_graph()
    df = st.session_state.new_assignments
    overview, team_sheets = [], {}
    for t in sorted(df['team'].dropna().unique()):
        stops = df[df['team'] == t].copy()
        if 'tsp_order' in stops:
            stops.sort_values('tsp_order', inplace=True)
        rooms = int(stops.get('num_rooms', 0).sum())
        km = mn = 0
        nodes = stops['node_id'].tolist()
        for u, v in zip(nodes[:-1], nodes[1:]):
            if u and v:
                try:
                    d = nx.shortest_path_length(graph, u, v, weight='length')
                except:
                    d = 0
            else:
                d = 0
            km += d / 1000
            mn += d / 1000 * 2
        service = rooms * 10
        total = int(service + mn)
        link = "https://www.google.com/maps/dir/" + "/".join([
            f"{lat},{lon}" for lat, lon in stops[['lat','lon']].values
        ])
        overview.append({
            'Kontrollbezirk': t,
            'Stops': len(stops),
            'Stimmbezirke': rooms,
            'Wegstrecke (km)': round(km, 1),
            'Fahrtzeit (min)': int(mn),
            'Kontrollzeit (min)': service,
            'Gesamtzeit': str(timedelta(minutes=total)),
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
        df_over.to_excel(writer, 'Übersicht', index=False)
        from openpyxl.utils import get_column_letter
        ws0 = writer.sheets['Übersicht']
        for i, col in enumerate(df_over.columns, 1):
            ws0.column_dimensions[get_column_letter(i)].width = max(
                df_over[col].astype(str).map(len).max(), len(col)
            ) + 2
        for name, tf in team_sheets.items():
            tf.to_excel(writer, name, index=False)
            ws = writer.sheets[name]
            for i, col in enumerate(tf.columns, 1):
                ws.column_dimensions[get_column_letter(i)].width = max(
                    tf[col].astype(str).map(len).max(), len(col)
                ) + 2
    buf.seek(0)
    return buf.getvalue()

# STREAMLIT UI
st.set_page_config(layout='wide')
with st.sidebar:
    st.title('Routenbearbeitung')
    df = st.session_state.new_assignments
    stops = st.multiselect('Stops', options=df['Wahlraum-A'])
    teams = sorted(df['team'].dropna().unique())
    sel = st.selectbox('Team', [None] + teams)
    algo = st.selectbox('Algo', ['Greedy', '2-Opt', 'Simulated Annealing', 'Christofides'])
    tgt = st.radio('Ziel', ['Alle Teams', 'Ausgewähltes Team'])
    if st.button('Optimieren'):
        optimize_routes(algo, tgt, sel)
    if st.button('Zuweisung übernehmen') and sel and stops:
        for a in stops:
            idx = df[df['Wahlraum-A'] == a].index[0]
            st.session_state.new_assignments.at[idx, 'team'] = sel
        st.session_state.action_log.append(f'Zuweisung {len(stops)} → Team {sel}')
        st.session_state.show_map = True
        st.experimental_rerun()
    if st.button('Neues Team'):
        st.session_state.show_new_team_form = True
    if st.session_state.show_new_team_form:
        nt = max(teams) + 1 if teams else 1
        form = st.form('nt')
        sel2 = form.multiselect(f'Stops für Team {nt}', df['Wahlraum-A'])
        if form.form_submit_button('Erstellen') and sel2:
            for a in sel2:
                i = df[df['Wahlraum-A'] == a].index[0]
                st.session_state.new_assignments.at[i, 'team'] = nt
            st.session_state.action_log.append(f'Team {nt} erstellt ({len(sel2)} Stops)')
            st.session_state.show_map = True
    excel = _generate_excel_bytes()
    st.download_button(
        '📥 Export Excel',
        data=excel,
        file_name='zuweisung.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    st.markdown('---')
    st.subheader('Log')
    for e in st.session_state.action_log:
        st.write(f'- {e}')

if st.session_state.show_map:
    dfm = st.session_state.new_assignments
    g = get_graph()
    m = leafmap.Map(center=[dfm['lat'].mean(), dfm['lon'].mean()], zoom=12)
    cols = ["#FF0000", "#00FF00", "#0000FF", "#FFA500"]
    for i, t in enumerate(sorted(dfm['team'].dropna().unique())):
        r = dfm[dfm['team'] == t]
        if 'tsp_order' in r:
            r = r.sort_values('tsp_order')
        nds = r['node_id'].tolist()
        path = []
        for u, v in zip(nds[:-1], nds[1:]):
            try:
                seg = nx.shortest_path(g, u, v, weight='length')
            except:
                continue
            path += [(g.nodes[n]['y'], g.nodes[n]['x']) for n in seg]
        folium.PolyLine(
            path,
            color=cols[i % len(cols)],
            weight=6,
            opacity=0.8,
            tooltip=f'Route {t}'
        ).add_to(m)
    mc = MarkerCluster()
    for _, r in dfm.dropna(subset=['lat', 'lon']).iterrows():
        html = (
            f"<div style='font-weight:bold;'>"
            f"<b>{r['Wahlraum-B']}</b><br>"
            f"<b>{r['Wahlraum-A']}</b><br>"
            f"<b>Anzahl Räume:</b> {r.get('num_rooms', '')}"
            f"</div>"
        )
        folium.Marker(
            [r['lat'], r['lon']],
            popup=folium.Popup(html, max_width=0)
        ).add_to(mc)
    mc.add_to(m)
    m.to_streamlit(height=700)
