"""Interaktives Routenbearbeitungstool mit Zoom-abhängiger Markersichtbarkeit, TSP-Optimierung, Excel- und PDF-Export (mit OSM-Hintergrund)"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import leafmap.foliumap as leafmap
import folium
from folium.plugins import MarkerCluster
import networkx as nx
import osmnx as ox
import pickle
import io
import re
from math import radians, sin, cos, sqrt, asin
from datetime import timedelta
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, LineString
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Border, Side
from networkx.algorithms.approximation.traveling_salesman import greedy_tsp

# -- Konfiguration --------------------------------------------------------------
FORMAT_SIZES = {
    "A4": (8.27, 11.69),
    "A3": (11.69, 16.54),
    "A2": (16.54, 23.39),
    "A1": (23.39, 33.11),
    "A0": (33.11, 46.81),
}
DEFAULT_ZOOMS = {fmt: zoom for fmt, zoom in zip(FORMAT_SIZES.keys(), [16,16,15,14,13])}

# -- Utility-Funktionen --------------------------------------------------------
@st.cache_resource
def get_graph():
    with open("munster_graph.pickle", "rb") as f:
        return pickle.load(f)

def haversine(lon1, lat1, lon2, lat2):
    dlon, dlat = radians(lon2-lon1), radians(lat2-lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*6371000*asin(sqrt(a))

# TSP-Löser
@st.cache_resource
def tsp_solve_route(graph, stops_df):
    if len(stops_df) <= 2:
        return stops_df
    coords = list(zip(stops_df['lat'], stops_df['lon']))
    nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat,lon in coords]
    G = nx.complete_graph(len(nodes))
    for i,j in G.edges():
        try:
            G[i][j]['weight'] = nx.shortest_path_length(graph, nodes[i], nodes[j], weight='length')
        except:
            G[i][j]['weight'] = float('inf')
    path = greedy_tsp(G)
    return stops_df.iloc[path].reset_index(drop=True)

# Excel-Export

def make_export(df_assign):
    output = io.BytesIO()
    graph = get_graph()
    overview, sheets = [], {}
    teams = sorted(df_assign['team'].dropna().astype(int).unique())
    for idx, t in enumerate(teams, start=1):
        df_t = df_assign[df_assign['team']==t]
        if 'tsp_order' in df_t:
            df_t = df_t.sort_values('tsp_order')
        # berechne Metriken...
        # (Übersicht und Detailblätter analog aufgebaut)
        # ...
        sheets[idx] = pd.DataFrame()  # placeholder
    # schreibe nach Excel
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(overview).to_excel(writer, sheet_name='Übersicht', index=False)
        # weitere Sheets...
    output.seek(0)
    return output

# PDF-Export mit OSM-Hintergrund

def export_routes_pdf_osm(df_assign, filename='routen_uebersicht.pdf', figsize=(8.27,11.69), dpi=300, zoom=15):
    graph = get_graph()
    # Punkte & Linien vorbereiten
    pts_gdf = gpd.GeoDataFrame(df_assign, geometry=gpd.points_from_xy(df_assign['lon'], df_assign['lat']), crs='EPSG:4326').to_crs(epsg=3857)
    line_features = []
    for t in sorted(df_assign['team'].dropna().astype(int).unique()):
        df_t = df_assign[df_assign['team']==t]
        if 'tsp_order' in df_t:
            df_t = df_t.sort_values('tsp_order')
        coords = []
        nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lon,lat in zip(df_t['lon'],df_t['lat'])]
        for u,v in zip(nodes[:-1],nodes[1:]):
            try:
                path = nx.shortest_path(graph, u, v, weight='length')
                coords += [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in path]
            except:
                pass
        if coords:
            line_features.append({'team':t, 'geometry':LineString(coords)})
    gdf_lines = gpd.GeoDataFrame(line_features, crs='EPSG:4326').to_crs(epsg=3857)
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    if not pts_gdf.empty:
        minx,miny,maxx,maxy = pts_gdf.total_bounds
        buf=0.05
        ax.set_xlim(minx-(maxx-minx)*buf, maxx+(maxx-minx)*buf)
        ax.set_ylim(miny-(maxy-miny)*buf, maxy+(maxy-miny)*buf)
    # OSM Basemap
    prov = ctx.providers.OpenStreetMap.Mapnik.copy()
    prov['tile_scale']=2
    ctx.add_basemap(ax, crs=pts_gdf.crs.to_string(), source=prov, zoom=zoom)
    # Routen
    colors=['magenta','cyan','lime','red','orange','yellow','turquoise','purple','pink','blue']
    for idx,row in gdf_lines.iterrows():
        xs,ys=row.geometry.xy
        ax.plot(xs,ys,color=colors[idx%len(colors)],linewidth=2,zorder=5)
        mid=row.geometry.interpolate(0.5,normalized=True)
        ax.text(mid.x, mid.y, str(int(row['team'])), fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2',fc='white',alpha=0.8,lw=0), zorder=6)
    # Punkte
    pts_gdf.plot(ax=ax, color='k', markersize=10, zorder=7)
    # Districts: load from shapefile directly
    try:
        dist = gpd.read_file("stadtbezirk.shp")
        # filter Ebene 2 if present
        if 'layer' in dist.columns:
            dist = dist[dist['layer'] == 2]
        # ensure name column
        if 'name' not in dist.columns and 'Stadtteil' in dist.columns:
            dist['name'] = dist['Stadtteil']
        dist = dist.to_crs(epsg=3857)
        dist.boundary.plot(ax=ax, linewidth=0.8, edgecolor='gray', zorder=2)
        for _, drow in dist.dropna(subset=['name']).iterrows():
            pt = drow.geometry.representative_point()
            ax.annotate(
                drow['name'],
                xy=(pt.x, pt.y), xycoords='data',
                xytext=(5,5), textcoords='offset points',
                fontsize=6, color='gray', ha='center', va='center', zorder=3,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, lw=0)
            )
    except Exception as e:
        print(f"District shapefile load error: {e}")
    # finalize plot
    ax.set_axis_off()
    ax.set_title('Routenübersicht Kontrollbezirke',fontsize=18,fontweight='bold',pad=15)
    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight',dpi=dpi)
    plt.close(fig)
    return filename

# Streamlit App ---------------------------------------------------------------
st.set_page_config(layout='wide')
# Basisadressen laden
if 'base_addresses' not in st.session_state:
    base=pd.read_csv('cleaned_addresses.csv')
    xls=pd.read_excel('routes_optimized.xlsx',sheet_name=None)
    tmp=[]
    for name,df in xls.items():
        m=re.match(r'Bezirk_(\d+)',name)
        if m and 'Adresse' in df:
            team_id=int(m.group(1))
            tmp+=[(addr,team_id) for addr in df['Adresse']]
    st.session_state['base_addresses']=pd.DataFrame(tmp,columns=['Wahlraum-A','team'])
    st.session_state['new_assignments']=st.session_state['base_addresses'].merge(pd.read_csv('cleaned_addresses.csv'),on='Wahlraum-A',how='right')
# Datenframe
df_assign=st.session_state['new_assignments']
# Sidebar
with st.sidebar:
    st.title('Bearbeitung Kontrollbezirke')
    # Suchfeld
    search=st.text_input('Suche Wahllokal oder Adresse')
    # Excel Export
    if st.button('Excel-Export'):
        buf=make_export(df_assign)
        st.download_button('Herunterladen',data=buf,file_name='routen.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    # PDF Export
    if st.button('PDF-Export A3 600dpi'):
        pdf=export_routes_pdf_osm(df_assign,figsize=FORMAT_SIZES['A3'],dpi=600,zoom=DEFAULT_ZOOMS['A3'])
        with open(pdf,'rb') as f:
            st.download_button('Herunterladen PDF',data=f,file_name='routen.pdf',mime='application/pdf')
# Karte rendern
draw_map(df_assign)
