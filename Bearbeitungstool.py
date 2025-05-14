"""""Interaktives Routenbearbeitungstool mit Zoom-abhängiger Markersichtbarkeit und TSP-Optimierung bei Zuweisung"""

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
from openpyxl.styles import PatternFill, Border, Side
from math import radians, sin, cos, sqrt, asin
import re

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
    teams = df_assign['team'].dropna().astype(int).unique()
    centers = {t: (
        df_assign[df_assign['team']==t]['lat'].mean(),
        df_assign[df_assign['team']==t]['lon'].mean()
    ) for t in teams}
    sorted_teams = sorted(centers.keys(), key=lambda t: (-centers[t][0], centers[t][1]))
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
            rooms_str = r.get('rooms','')
            numbers = re.findall(r'\b(\d+)\b', rooms_str)
            stimmbez = ', '.join(numbers)
            detail.append({
               'Nr.': j,
               'Wahllokal': r.get('Wahlraum-B',''),
               'Adresse': r['Wahlraum-A'],
               'Stimmbezirke': stimmbez,
            })
        sheets[idx] = pd.DataFrame(detail)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(overview).to_excel(writer, sheet_name='Übersicht', index=False)
        ws = writer.book.create_sheet('Gesamtübersicht')
        row_cursor = 1
        num_cols = 4
        data_fill = PatternFill(fill_type='solid', start_color='F2F2F2')
        thick = Side(style='thick')
        thin = Side(style='thin')
        for kb, df_s in sheets.items():
            title_cell = ws.cell(row=row_cursor, column=1, value=f"Kontrollbezirk {kb}")
            title_cell.font = title_cell.font.copy(bold=True)
            title_cell.fill = data_fill
            title_cell.border = Border(top=thick, left=thick, right=thin)
            headers = ['Wahllokal', 'Adresse', 'Stimmbezirk']
            for idx_h, hdr_text in enumerate(headers, start=2):
                hdr = ws.cell(row=row_cursor, column=idx_h, value=hdr_text)
                hdr.font = hdr.font.copy(bold=True)
                hdr.fill = data_fill
                left = thin
                right = thick if idx_h==num_cols else thin
                hdr.border = Border(top=thick, left=left, right=right)
            row_cursor += 1
            for j, r in enumerate(df_s.itertuples(), start=1):
                vals = [j, r.Wahllokal, r.Adresse, r.Stimmbezirke]
                for col_idx, val in enumerate(vals, start=1):
                    c = ws.cell(row=row_cursor, column=col_idx, value=val)
                    c.fill = data_fill
                    left = thick if col_idx==1 else thin
                    right = thick if col_idx==num_cols else thin
                    c.border = Border(left=left, right=right)
                row_cursor += 1
            for col in range(1, num_cols+1):
                c = ws.cell(row=row_cursor-1, column=col)
                left = thick if col==1 else thin
                right = thick if col==num_cols else thin
                c.border = Border(bottom=thick, left=left, right=right)
            row_cursor += 1
        for kb, df_s in sheets.items():
            df_s.to_excel(writer, sheet_name=f"Bezirk_{kb}", index=False)
        for ws in writer.sheets.values():
            for col_cells in ws.columns:
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                col_letter = get_column_letter(col_cells[0].column)
                ws.column_dimensions[col_letter].width = max_length + 2
    output.seek(0)
    return output

st.set_page_config(layout='wide')
if 'show_new' not in st.session_state:
    st.session_state.show_new = False
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

df_assign = st.session_state.new_assignments.copy()
if 'latlong' in df_assign.columns and ('lat' not in df_assign.columns or 'lon' not in df_assign.columns):
    df_assign[['lat','lon']] = df_assign['latlong'].str.split(',',expand=True).astype(float)
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
        st.success('Import
""
