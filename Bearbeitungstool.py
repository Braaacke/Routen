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
from openpyxl.styles import PatternFill
from math import radians, sin, cos, sqrt, asin

# Haversine-Distanz (Fallback)
def haversine(lon1, lat1, lon2, lat2):
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * 6371000 * asin(sqrt(a))

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

# Export-Funktion, erst beim Download aufrufen
def make_export(df_assign):
    output = io.BytesIO()
    graph = get_graph()
    overview = []
    sheets = {}
    # Sortiere Kontrollbezirke geografisch (Nordwest nach Südost)
    teams = df_assign['team'].dropna().astype(int).unique()
    centers = {t: (
        df_assign[df_assign['team']==t]['lat'].mean(),
        df_assign[df_assign['team']==t]['lon'].mean()
    ) for t in teams}
    sorted_teams = sorted(centers.keys(), key=lambda t: (-centers[t][0], centers[t][1]))
    # Detail- und Übersichtsdaten sammeln
    for idx, t in enumerate(sorted_teams, start=1):
        df_t = df_assign[df_assign['team'] == t]
        if 'tsp_order' in df_t.columns:
            df_t = df_t.sort_values('tsp_order')
        rooms = df_t.get('num_rooms', pd.Series()).sum()
        travel_km = travel_min = 0.0
        coords = df_t[['lat','lon']].values.tolist()
        if len(coords) > 1:
            for (lat1, lon1), (lat2, lon2) in zip(coords[:-1], coords[1:]):
                try:
                    n1 = ox.distance.nearest_nodes(graph, X=lon1, Y=lat1)
                    n2 = ox.distance.nearest_nodes(graph, X=lon2, Y=lat2)
                    length = nx.shortest_path_length(graph, n1, n2, weight='length')
                except:
                    length = haversine(lon1, lat1, lon2, lat2)
                travel_km += length / 1000.0
                travel_min += (length / 1000.0) * 2.0
        ctrl_time = int(rooms * 10)
        total_time = travel_min + ctrl_time
        overview.append({
            'Kontrollbezirk': idx,
            'Anzahl Wahllokale': len(df_t),
            'Anzahl Stimmbezirke': rooms,
            'Wegstrecke (km)': round(travel_km,1),
            'Fahrtzeit (min)': int(travel_min),
            'Kontrollzeit (min)': ctrl_time,
            'Gesamtzeit': str(timedelta(minutes=int(total_time))),
            'Google-Link': 'https://www.google.com/maps/dir/' + '/'.join([f"{lat},{lon}" for lat,lon in coords])
        })
        detail = []
        for j, (_, r) in enumerate(df_t.iterrows(), start=1):
            coord = f"{r['lat']},{r['lon']}"
            detail.append({
                'Nr.': j,
                'Wahllokal': r.get('Wahlraum-B',''),
                'Adresse': r['Wahlraum-A'],
                'Stimmbezirke': r.get('rooms',''),
                'Anzahl Stimmbezirke': r.get('num_rooms',''),
            })
        sheets[idx] = pd.DataFrame(detail)
    # Schreibe Excel mit Übersicht, Gesamtübersicht (gruppiert) und Details
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Übersicht
        pd.DataFrame(overview).to_excel(writer, sheet_name='Übersicht', index=False)
                # Gesamtübersicht: jeweils Block pro Kontrollbezirk
        ws = writer.book.create_sheet('Gesamtübersicht')
        row_cursor = 1
        # Definiere Spaltenanzahl für Merge
        num_cols = len(sheets[next(iter(sheets))].columns)
        for kb, df_s in sheets.items():
            # Überschrift 'Kontrollbezirk X' über alle Spalten mergen
            start_col = 1
            end_col = num_cols
            ws.merge_cells(start_row=row_cursor, start_column=start_col, end_row=row_cursor, end_column=end_col)
            cell = ws.cell(row=row_cursor, column=1, value=f"Kontrollbezirk {kb}")
            cell.font = cell.font.copy(bold=True)
cell.fill = PatternFill(fill_type='solid', start_color='DDDDDD')
            row_cursor += 1
            # Spaltenüberschriften
            for col_idx, col in enumerate(df_s.columns, start=1):
                hdr = ws.cell(row=row_cursor, column=col_idx, value=col)
hdr.font = hdr.font.copy(bold=True)
hdr.fill = PatternFill(fill_type='solid', start_color='EEEEEE')
                hdr.font = hdr.font.copy(bold=True)
            row_cursor += 1
            # Datenzeilen
            for _, r in df_s.iterrows():
                for col_idx, val in enumerate(r, start=1):
                    ws.cell(row=row_cursor, column=col_idx, value=val)
                row_cursor += 1
            row_cursor += 1  # Leerzeile
        # Detail-Sheets
        for kb, df_s in sheets.items():
            name = f"Bezirk_{kb}"
            df_s.to_excel(writer, sheet_name=name, index=False)
        # Spaltenbreiten anpassen
        for ws in writer.sheets.values():
            for col_cells in ws.columns:
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                col_letter = get_column_letter(col_cells[0].column)
                ws.column_dimensions[col_letter].width = max_length + 2
    output.seek(0)
    return output

# Streamlit-Seite konfigurieren
st.set_page_config(layout='wide')
if 'show_new' not in st.session_state:
    st.session_state.show_new = False
# Basisdaten laden
if 'base_addresses' not in st.session_state:
    base = pd.read_csv('cleaned_addresses.csv').reset_index(drop=True)
    initial = pd.read_excel('routes_optimized.xlsx', sheet_name=None)
    tmp = []
    for name, df in initial.items():
        if name != 'Übersicht' and 'Adresse' in df.columns:
            team = int(name.split('_')[1])
            for addr in df['Adresse']:
                tmp.append((addr, team))
    assign_df = pd.DataFrame(tmp, columns=['Wahlraum-A','team'])
    merged = base.merge(assign_df, on='Wahlraum-A', how='left')
    st.session_state.base_addresses = merged.copy()
    st.session_state.new_assignments = merged.copy()
# Arbeitsdaten
df_assign = st.session_state.new_assignments.copy()
if 'latlong' in df_assign.columns and ('lat' not in df_assign.columns or 'lon' not in df_assign.columns):
    df_assign[['lat','lon']] = df_assign['latlong'].str.split(',',expand=True).astype(float)
# Sidebar
with st.sidebar:
    st.title('Bearbeitung Kontrollbezirke')
    opts = df_assign.dropna(subset=['Wahlraum-B','Wahlraum-A'])
    addrs_search = opts.apply(lambda r: f"{r['Wahlraum-B']} - {r['Wahlraum-A']}", axis=1).tolist()
    st.selectbox(
        'Wahllokal oder Adresse suchen',
        options=[''] + addrs_search,
        index=0,
        format_func=lambda x: 'Wahllokal oder Adresse suchen' if x=='' else x,
        key='search_selection'
    )
    uploaded = st.file_uploader('Alternative Zuweisung importieren', type=['xlsx'])
    if uploaded:
        imp = pd.read_excel(uploaded, sheet_name=None)
        temp = []
        for sheet, df in imp.items():
            if sheet != 'Übersicht' and 'Adresse' in df.columns:
                team_id = int(sheet.split('_')[1])
                for addr in df['Adresse']:
                    temp.append((addr, team_id))
        assigns = pd.DataFrame(temp, columns=['Wahlraum-A','team'])
        st.session_state.new_assignments = (
            st.session_state.base_addresses.drop(columns=['team'])
            .merge(assigns, on='Wahlraum-A', how='left')
        )
        st.success('Import erfolgreich.')
    opts_assign = st.session_state.new_assignments.dropna(subset=['Wahlraum-B','Wahlraum-A'])
    addrs_assign = opts_assign.apply(lambda r: f"{r['Wahlraum-B']} - {r['Wahlraum-A']}", axis=1).tolist()
    sel = st.multiselect('Wahllokal wählen', options=addrs_assign, placeholder='Auswählen')
    teams = sorted(st.session_state.new_assignments['team'].dropna().astype(int).unique())
    tgt = st.selectbox('Kontrollbezirk wählen', options=[None] + teams, placeholder='Auswählen')
    if st.button('Zuweisung übernehmen') and tgt and sel:
        graph = get_graph()
        for label in sel:
            addr = label.split(' - ',1)[1]
            idx = st.session_state.new_assignments.index[st.session_state.new_assignments['Wahlraum-A']==addr][0]
            st.session_state.new_assignments.at[idx,'team'] = tgt
        for team_id in {tgt}:
            df_team = st.session_state.new_assignments[st.session_state.new_assignments['team']==team_id]
            opt = tsp_solve_route(graph, df_team)
            st.session_state.new_assignments.loc[opt.index,'tsp_order'] = range(len(opt))
        st.success('Zuweisung gesetzt.')
    if not st.session_state.show_new:
        if st.button('Neuen Kontrollbezirk erstellen'):
            st.session_state.show_new = True
            st.experimental_rerun()
    else:
        max_t = int(st.session_state.new_assignments['team'].max(skipna=True) or 0) + 1
        with st.form('new_district_form'):
            st.markdown(f"### Neuen Kontrollbezirk {max_t} erstellen")
            sel2 = st.multiselect(f"Stops für Kontrollbezirk {max_t}", options=addrs_assign)
            if st.form_submit_button('Erstellen und zuweisen') and sel2:
                graph = get_graph()
                for label in sel2:
                    addr = label.split(' - ',1)[1]
                    idx = st.session_state.new_assignments.index[st.session_state.new_assignments['Wahlraum-A']==addr][0]
                    st.session_state.new_assignments.at[idx,'team'] = max_t
                df_nt = st.session_state.new_assignments[st.session_state.new_assignments['team']==max_t]
                opt2 = tsp_solve_route(graph, df_nt)
                st.session_state.new_assignments.loc[opt2.index,'tsp_order'] = range(len(opt2))
                st.success(f'Kontrollbezirk {max_t} erstellt.')
                st.session_state.show_new = False
    if st.button('Routen berechnen', key='recalc_routes'):
        graph = get_graph()
        for team_id in sorted(st.session_state.new_assignments['team'].dropna().astype(int).unique()):
            df_team = st.session_state.new_assignments[st.session_state.new_assignments['team']==team_id]
            opt = tsp_solve_route(graph, df_team)
            st.session_state.new_assignments.loc[opt.index,'tsp_order'] = range(len(opt))
        st.success('Routen neu berechnet.')
    export_buf = make_export(st.session_state.new_assignments)
    st.download_button(
        'Kontrollbezirke herunterladen',
        data=export_buf,
        file_name='routen_zuweisung.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
# Map anzeigen
def draw_map(df_assign):
    search_sel = st.session_state.get('search_selection','')
    if search_sel:
        addr = search_sel.split(' - ',1)[1]
        row = df_assign[df_assign['Wahlraum-A']==addr]
        center = [row.iloc[0]['lat'], row.iloc[0]['lon']] if not row.empty else [df_assign['lat'].mean(), df_assign['lon'].mean()]
        zoom = 17
    else:
        center = [df_assign['lat'].mean(), df_assign['lon'].mean()]
        zoom = 10
    m = leafmap.Map(center=center, zoom=zoom)
    graph = get_graph()
    colors = ['#FF00FF','#00FFFF','#00FF00','#FF0000','#FFA500','#FFFF00','#00CED1','#DA70D6','#FF69B4','#8A2BE2']
    for i, t in enumerate(sorted(df_assign['team'].dropna().unique())):
        df_t = df_assign[df_assign['team']==t]
        if 'tsp_order' in df_t.columns:
            df_t = df_t.sort_values('tsp_order')
        pts = df_t[['lat','lon']].values.tolist()
        if len(pts)>1:
            path=[]
            nodes=[ox.distance.nearest_nodes(graph,X=lon,Y=lat) for lat,lon in pts]
            for u,v in zip(nodes[:-1],nodes[1:]):
                try:
                    p=nx.shortest_path(graph,u,v,weight='length')
                    path.extend([(graph.nodes[n]['y'],graph.nodes[n]['x']) for n in p])
                except:
                    pass
            folium.PolyLine(path,color=colors[i%len(colors)],weight=6,opacity=0.8,
                            tooltip=f"Kontrollbezirk {int(t)}").add_to(m)
        cluster = MarkerCluster(disableClusteringAtZoom=13)
    for _, r in df_assign.dropna(subset=['lat','lon']).iterrows():
        popup_html = (
            f"<div style='white-space: nowrap;'>"
            f"<b>{r['Wahlraum-B']}</b><br>{r['Wahlraum-A']}<br>Anzahl Räume: {r['num_rooms']}"
            "</div>"
        )
        popup = folium.Popup(popup_html, max_width=300)
        marker = folium.Marker(location=[r['lat'], r['lon']], popup=popup)
        cluster.add_child(marker)
    cluster.add_to(m)
    if not st.session_state.get('search_selection',''):
        m.fit_bounds(df_assign[['lat','lon']].values.tolist())
    m.to_streamlit(use_container_width=True, height=700)

draw_map(df_assign)
