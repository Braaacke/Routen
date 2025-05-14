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
from math import radians, sin, cos, sqrt, asin

# Haversine fallback
def haversine(lon1, lat1, lon2, lat2):
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * 6371000 * asin(sqrt(a))

# Caching graph
@st.cache_data
def load_graph():
    with open("munster_graph.pickle","rb") as f:
        return pickle.load(f)

# Solve TSP

def solve_tsp(graph, df):
    if len(df) <= 2:
        return df
    coords = list(zip(df.lat, df.lon))
    nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
    G = nx.complete_graph(len(nodes))
    for i in G.nodes:
        for j in G.nodes:
            if i != j:
                try:
                    w = nx.shortest_path_length(graph, nodes[i], nodes[j], weight='length')
                except:
                    w = float('inf')
                G[i][j]['weight'] = w
    order = greedy_tsp(G)
    return df.iloc[order].reset_index(drop=True)

# Export helper
def make_export(df, routing_method, central_addr, central_coord):
    buf = io.BytesIO()
    graph = load_graph()
    teams = df.team.dropna().astype(int).unique()
    centers = {t: (df[df.team == t].lat.mean(), df[df.team == t].lon.mean()) for t in teams}
    sorted_teams = sorted(centers, key=lambda t: (-centers[t][0], centers[t][1]))
    overview = []
    sheets = {}
    for idx, t in enumerate(sorted_teams, start=1):
        grp = df[df.team == t]
        # Für sternförmig: append central point als Halt und lösen TSP inklusive Zentrale
        if routing_method == 'Sternförmig':
            # zentrale als zusätzlicher Stop
            central_row = pd.DataFrame([{ 'Wahlraum-A': central_addr, 'lat': central_coord[0], 'lon': central_coord[1], 'rooms': '', 'num_rooms': 0, 'team': t }])
            grp_ext = pd.concat([grp, central_row], ignore_index=True)
            ordered = solve_tsp(graph, grp_ext)
        else:
            ordered = grp.sort_values('tsp_order') if 'tsp_order' in grp else grp
        # Berechne Übersichtsdaten auf ordered
        rooms = int(ordered.num_rooms.sum())
        km = mn = 0.0
        pts = list(zip(ordered.lat, ordered.lon))
        if len(pts) > 1:
            nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in pts]
            for u, v in zip(nodes[:-1], nodes[1:]):
                try:
                    d = nx.shortest_path_length(graph, u, v, weight='length')
                except:
                    d = haversine(pts[nodes.index(u)][1], pts[nodes.index(u)][0], pts[nodes.index(v)][1], pts[nodes.index(v)][0])
                km += d / 1000.0
                mn += (d / 1000.0) * 2.0
        overview.append({
            'Kontrollbezirk': idx,
            'Anzahl Wahllokale': len(grp),
            'Anzahl Stimmbezirke': rooms,
            'Wegstrecke (km)': round(km, 1),
            'Fahrtzeit (min)': int(mn),
            'Kontrollzeit (min)': rooms * 10,
            'Gesamtzeit': str(timedelta(minutes=int(mn + rooms * 10))),
            'Google-Link': 'https://www.google.com/maps/dir/' + '/'.join(f"{lat},{lon}" for lat, lon in pts)
        })
        # Detailblatt
        detail = []
        for j, (_, r) in enumerate(ordered.iterrows(), start=1):
            coord = f"{r.lat},{r.lon}"
            detail.append({
                'Bezirk': j,
                'Adresse': r['Wahlraum-A'],
                'Stimmbezirke': r.get('rooms', ''),
                'Anzahl Stimmbezirke': r.get('num_rooms', ''),
                'Google-Link': f"https://www.google.com/maps/search/?api=1&query={quote_plus(coord)}"
            })
        sheets[f"Bezirk_{idx}"] = pd.DataFrame(detail)
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        pd.DataFrame(overview).to_excel(writer, sheet_name='Übersicht', index=False)
        for name, df_s in sheets.items():
            df_s.to_excel(writer, sheet_name=name, index=False)
        for ws in writer.sheets.values():
            for col_cells in ws.columns:
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                col_letter = get_column_letter(col_cells[0].column)
                ws.column_dimensions[col_letter].width = max_length + 2
    buf.seek(0)
    return buf

# Init
st.set_page_config(layout='wide')
# Default routing method
if 'routing_method' not in st.session_state:
    st.session_state.routing_method = 'Dezentral'
if 'show_new' not in st.session_state:
    st.session_state.show_new = False
if 'new_assignments' not in st.session_state:
    base = pd.read_csv('cleaned_addresses.csv')
    xls = pd.read_excel('routes_optimized.xlsx', sheet_name=None)
    tmp = []
    for name, df in xls.items():
        parts = name.split('_')
        if name != 'Übersicht' and len(parts) > 1 and parts[1].isdigit():
            team_id = int(parts[1])
            for addr in df['Adresse']:
                tmp.append((addr, team_id))
    st.session_state.base_addresses = base.merge(
        pd.DataFrame(tmp, columns=['Wahlraum-A', 'team']),
        on='Wahlraum-A', how='left'
    )
    st.session_state.new_assignments = st.session_state.base_addresses.copy()
assign = st.session_state.new_assignments
if 'latlong' in assign and ('lat' not in assign or 'lon' not in assign):
    assign[['lat', 'lon']] = assign.latlong.str.split(',', expand=True).astype(float)
if 'latlong' in assign and ('lat' not in assign or 'lon' not in assign):
    assign[['lat', 'lon']] = assign.latlong.str.split(',', expand=True).astype(float)

# Zentrale für sternförmige Routen
central_addr = 'Prinzipalmarkt 8'
central_row = st.session_state.get('new_assignments', pd.DataFrame())
central_row = central_row[central_row['Wahlraum-A'] == central_addr]
if not central_row.empty:
    central_coord = (central_row.lat.iloc[0], central_row.lon.iloc[0])
else:
    df_tmp = st.session_state.get('new_assignments', pd.DataFrame())
    central_coord = (df_tmp.lat.mean(), df_tmp.lon.mean())

# Sidebar
with st.sidebar:
    st.title('Bearbeitung Kontrollbezirke')
    # Routing-Methode wählen
    st.session_state.routing_method = st.radio(
        'Routing-Methode', ['Dezentral', 'Sternförmig'],
        index=['Dezentral', 'Sternförmig'].index(st.session_state.get('routing_method', 'Dezentral'))
    )
    opts = assign.dropna(subset=['Wahlraum-B', 'Wahlraum-A'])
    labels = opts.apply(lambda r: f"{r['Wahlraum-B']} - {r['Wahlraum-A']}", axis=1).tolist()
    st.selectbox('Wahllokal oder Adresse suchen', options=[''] + labels, key='search')

    # Import alternative Zuweisung
    file = st.file_uploader('Import alternative Zuweisung', type=['xlsx'])
    if file:
        imp = pd.read_excel(file, sheet_name=None)
        tmp = []
        for n, df in imp.items():
            parts = n.split('_')
            if n != 'Übersicht' and len(parts) > 1 and parts[1].isdigit():
                team_id = int(parts[1])
                for a in df['Adresse']:
                    tmp.append((a, team_id))
        st.session_state.new_assignments = (
            st.session_state.base_addresses.drop(columns='team')
            .merge(pd.DataFrame(tmp, columns=['Wahlraum-A', 'team']), on='Wahlraum-A', how='left')
        )
        st.success('Import erfolgreich')

    # Manuelle Zuweisung
    sel = st.multiselect('Wahllokal wählen', options=labels, placeholder='Auswählen')
    teams = sorted(assign.team.dropna().astype(int).unique())
    tgt = st.selectbox('Kontrollbezirk wählen', options=[None] + teams, key='tgt', placeholder='Auswählen')
    if st.button('Zuweisung übernehmen') and tgt and sel:
        for l in sel:
            a = l.split(' - ', 1)[1]
            idx = assign.index[assign['Wahlraum-A'] == a][0]
            assign.at[idx, 'team'] = tgt
        g = load_graph()
        df_sel = assign[assign.team == tgt]
        opt = solve_tsp(g, df_sel)
        assign.loc[opt.index, 'tsp_order'] = range(len(opt))
        st.success('Zuweisung gesetzt')

    # Neuer Kontrollbezirk
    if not st.session_state.show_new:
        if st.button('Neuen Kontrollbezirk erstellen'):
            st.session_state.show_new = True
    else:
        max_t = int(assign.team.max(skipna=True) or 0) + 1
        with st.form('new_district'):
            st.markdown(f"### Kontrollbezirk {max_t} erstellen")
            sel2 = st.multiselect('Stops auswählen', options=labels, key='new_sel')
            if st.form_submit_button('Erstellen und zuweisen'):
                if sel2:
                    g = load_graph()
                    for l in sel2:
                        a = l.split(' - ', 1)[1]
                        idx = assign.index[assign['Wahlraum-A'] == a][0]
                        assign.at[idx, 'team'] = max_t
                    df_nt = assign[assign.team == max_t]
                    opt2 = solve_tsp(g, df_nt)
                    assign.loc[opt2.index, 'tsp_order'] = range(len(opt2))
                    st.success(f'Kontrollbezirk {max_t} erstellt')
                    st.session_state.show_new = False
                else:
                    st.warning('Bitte mindestens ein Wahllokal auswählen')

    # Routen neu berechnen
    if st.button('Routen berechnen'):
        g = load_graph()
        for t in assign['team'].dropna().astype(int).unique():
            df_t = assign[assign['team'] == t]
            opt = solve_tsp(g, df_t)
            assign.loc[opt.index, 'tsp_order'] = range(len(opt))
        st.success('Routen berechnet')

    # Download-Button mit korrekter Parameterliste
    st.download_button(
        label='Herunterladen',
        data=make_export(
            assign,
            st.session_state.routing_method,
            central_addr,
            central_coord
        ),
        file_name='routen.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# Map
search = st.session_state.get('search', '')
# Zentrale für sternförmige Routen
central_addr = 'Prinzipalmarkt 8'
central_row = assign[assign['Wahlraum-A'] == central_addr]
if not central_row.empty:
    central_coord = (central_row.lat.iloc[0], central_row.lon.iloc[0])
else:
    central_coord = (assign.lat.mean(), assign.lon.mean())

if search:
    addr = search.split(' - ', 1)[1]
    r = assign[assign['Wahlraum-A'] == addr]
    center = [r.lat.iloc[0], r.lon.iloc[0]] if not r.empty else [assign.lat.mean(), assign.lon.mean()]
    z = 17
else:
    center = [assign.lat.mean(), assign.lon.mean()]
    z = 10
m = leafmap.Map(center=center, zoom=z)
G = load_graph()
cols = ['#FF00FF', '#00FFFF', '#00FF00', '#FF0000', '#FFA500', '#FFFF00', '#00CED1', '#DA70D6', '#FF69B4', '#8A2BE2']
for i, t in enumerate(sorted(assign.team.dropna().astype(int).unique())):
    df_t = assign[assign.team == t]
    method = st.session_state.get('routing_method', 'Dezentral')
    if method == 'Dezentral':
        # wie bisher: TSP-Route
        if 'tsp_order' in df_t:
            df_t = df_t.sort_values('tsp_order')
        pts = list(zip(df_t.lat, df_t.lon))
        if len(pts) > 1:
            path = []
            nodes = [ox.distance.nearest_nodes(G, X=lon, Y=lat) for lat, lon in pts]
            for u, v in zip(nodes[:-1], nodes[1:]):
                try:
                    p = nx.shortest_path(G, u, v, weight='length')
                    path.extend([(G.nodes[n]['y'], G.nodes[n]['x']) for n in p])
                except:
                    pass
            folium.PolyLine(path, color=cols[i % len(cols)], weight=6, opacity=0.8,
                            tooltip=f"Kontrollbezirk {int(t)}").add_to(m)
    else:
        # sternförmige Routen: jede Route endet am zentralen Punkt
        for _, r_stop in df_t.iterrows():
            try:
                n1 = ox.distance.nearest_nodes(G, X=r_stop.lon, Y=r_stop.lat)
                n2 = ox.distance.nearest_nodes(G, X=central_coord[1], Y=central_coord[0])
                p = nx.shortest_path(G, n1, n2, weight='length')
                seg = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in p]
                folium.PolyLine(seg, color=cols[i % len(cols)], weight=3, opacity=0.6,
                                tooltip=f"Kontrollbezirk {int(t)} sternförmig").add_to(m)
            except:
                continue
        # sternförmige Routen: jede Route endet am zentralen Punkt
        for _, r_stop in df_t.iterrows():
            try:
                n1 = ox.distance.nearest_nodes(G, X=r_stop.lon, Y=r_stop.lat)
                n2 = ox.distance.nearest_nodes(G, X=central_coord[1], Y=central_coord[0])
                p = nx.shortest_path(G, n1, n2, weight='length')
                seg = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in p]
                folium.PolyLine(seg, color=cols[i % len(cols)], weight=3, opacity=0.6,
                                tooltip=f"Kontrollbezirk {int(t)} sternförmig").add_to(m)
            except:
                continue

cluster = MarkerCluster(disableClusteringAtZoom=13)
for _, r in assign.dropna(subset=['lat','lon']).iterrows():
    html = f"<div style='white-space: nowrap;'><b>{r['Wahlraum-B']}</b><br>{r['Wahlraum-A']}<br>Anzahl Räume: {r['num_rooms']}</div>"
    cluster.add_child(folium.Marker(location=[r.lat, r.lon], popup=folium.Popup(html, max_width=300)))
cluster.add_to(m)
if not search:
    m.fit_bounds(assign[['lat','lon']].values.tolist())
m.to_streamlit(use_container_width=True, height=700)
