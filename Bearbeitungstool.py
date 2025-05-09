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
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*6371000*asin(sqrt(a))(lon1, lat1, lon2, lat2):
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*6371000*asin(sqrt(a))

# Caching graph
@st.cache_data
def load_graph():
    with open("munster_graph.pickle","rb") as f:
        return pickle.load(f)():
    with open("munster_graph.pickle","rb") as f:
        return pickle.load(f)

def solve_tsp(graph,df):
    if len(df)<=2: return df
    coords=list(zip(df.lat,df.lon))
    nodes=[ox.distance.nearest_nodes(graph,X=lon,Y=lat) for lat,lon in coords]
    G=nx.complete_graph(len(nodes))
    for i in G.nodes:
        for j in G.nodes:
            if i!=j:
                try: w=nx.shortest_path_length(graph,nodes[i],nodes[j],weight='length')
                except: w=float('inf')
                G[i][j]['weight']=w
    order=greedy_tsp(G)
    return df.iloc[order].reset_index(drop=True)

# Export helper
def make_export(df):
    buf=io.BytesIO()
    graph=load_graph()
    teams=df.team.dropna().astype(int).unique()
    centers={t:(df[df.team==t].lat.mean(),df[df.team==t].lon.mean()) for t in teams}
    sorted_teams=sorted(centers,key=lambda t:(-centers[t][0],centers[t][1]))
    overview=[]; sheets={}
    for idx,t in enumerate(sorted_teams,1):
        grp=df[df.team==t]
        if 'tsp_order' in grp: grp=grp.sort_values('tsp_order')
        rooms=int(grp.num_rooms.sum())
        km=mn=0.0
        pts=list(zip(grp.lat,grp.lon))
        if len(pts)>1:
            for (la,lo),(la2,lo2) in zip(pts,pts[1:]):
                try:
                    n1=ox.distance.nearest_nodes(graph,X=lo,Y=la)
                    n2=ox.distance.nearest_nodes(graph,X=lo2,Y=la2)
                    d=nx.shortest_path_length(graph,n1,n2,weight='length')
                except:
                    d=haversine(lo,la,lo2,la2)
                km+=d/1000; mn+=d/1000*2
        overview.append({
            'Kontrollbezirk':idx,'Anzahl Wahllokale':len(grp),'Anzahl Stimmbezirke':rooms,
            'Wegstrecke (km)':round(km,1),'Fahrtzeit (min)':int(mn),'Kontrollzeit (min)':rooms*10,
            'Gesamtzeit':str(timedelta(minutes=int(mn+rooms*10))),
            'Google-Link':'https://www.google.com/maps/dir/'+'/'.join(f"{la},{lo}" for la,lo in pts)
        })
        detail=[]
        for j,(_,r) in enumerate(grp.iterrows(),1):
            coord=f"{r.lat},{r.lon}"
            detail.append({'Bezirk':j,'Adresse':r['Wahlraum-A'],'Stimmbezirke':r.get('rooms',''),
                           'Anzahl Stimmbezirke':r.get('num_rooms',''),
                           'Google-Link':f"https://www.google.com/maps/search/?api=1&query={quote_plus(coord)}"})
        sheets[f"Bezirk_{idx}"]=pd.DataFrame(detail)
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        pd.DataFrame(overview).to_excel(w,'Übersicht',index=False)
        for name,df_s in sheets.items(): df_s.to_excel(w,name,index=False)
        for ws in w.sheets.values():
            for col in ws.columns:
                ml=max(len(str(c.value)) for c in col)
                ws.column_dimensions[get_column_letter(col[0].column)].width=ml+2
    buf.seek(0)
    return buf

# Init
st.set_page_config(layout='wide')
if 'show_new' not in st.session_state: st.session_state.show_new=False
if 'new_assignments' not in st.session_state:
    base=pd.read_csv('cleaned_addresses.csv'); xls=pd.read_excel('routes_optimized.xlsx',sheet_name=None)
    tmp=[(addr,int(name.split('_')[1])) for name,df in xls.items() if name!='Übersicht' for addr in df['Adresse']]
    st.session_state.base_addresses=base.merge(pd.DataFrame(tmp,columns=['Wahlraum-A','team']),on='Wahlraum-A',how='left')
    st.session_state.new_assignments=st.session_state.base_addresses.copy()
assign=st.session_state.new_assignments
if 'latlong' in assign and ('lat' not in assign or 'lon' not in assign):
    assign[['lat','lon']]=assign.latlong.str.split(',',expand=True).astype(float)

# Sidebar
with st.sidebar:
    st.title('Bearbeitung Kontrollbezirke')
    opts=assign.dropna(subset=['Wahlraum-B','Wahlraum-A'])
    labels=opts.apply(lambda r:f"{r['Wahlraum-B']} - {r['Wahlraum-A']}",axis=1)
    st.selectbox('Wahllokal oder Adresse suchen',options=['']+labels.tolist(),key='search')
    if file:=st.file_uploader('Import alternative Zuweisung',type=['xlsx']):
        imp=pd.read_excel(file,sheet_name=None)
        tmp=[(a,int(n.split('_')[1])) for n,df in imp.items() if n!='Übersicht' for a in df['Adresse']]
        st.session_state.new_assignments=st.session_state.base_addresses.drop(columns='team').merge(pd.DataFrame(tmp,columns=['Wahlraum-A','team']),on='Wahlraum-A',how='left')
        st.success('Import erfolgreich')
    sel=st.multiselect('Wahllokal wählen',options=labels.tolist(),placeholder='Auswählen')
    teams=sorted(assign.team.dropna().astype(int).unique())
    tgt=st.selectbox('Kontrollbezirk wählen',options=[None]+teams,key='tgt',placeholder='Auswählen')
    if st.button('Zuweisung übernehmen') and tgt and sel:
        for l in sel:
            a=l.split(' - ',1)[1]
            idx=assign.index[assign['Wahlraum-A']==a][0]
            assign.at[idx,'team']=tgt
        g=load_graph(); df=assign[assign.team==tgt]
        opt=solve_tsp(g,df); assign.loc[opt.index,'tsp_order']=range(len(opt))
        st.success('Zuweisung gesetzt'); st.experimental_rerun()
    if not st.session_state.show_new:
        if st.button('Neuen Kontrollbezirk erstellen'):
            st.session_state.show_new=True; st.experimental_rerun()
    else:
        max_t=int(assign.team.max(skipna=True) or 0)+1
        with st.form('new'): 
            st.markdown(f"### Kontrollbezirk {max_t}")
            sel2=st.multiselect('Stops auswählen',labels.tolist(),key='new_sel')
            if st.form_submit_button('Erstellen'):
                if sel2:
                    g=load_graph()
                    for l in sel2:
                        a=l.split(' - ',1)[1]; idx=assign.index[assign['Wahlraum-A']==a][0]; assign.at[idx,'team']=max_t
                    opt2=solve_tsp(g,assign[assign.team==max_t]); assign.loc[opt2.index,'tsp_order']=range(len(opt2))
                    st.success(f'Bezirk {max_t} erstellt'); st.session_state.show_new=False; st.experimental_rerun()
                else: st.warning('Wählen bitte')
    if st.button('Routen berechnen'):
        g=load_graph(); [assign.loc[solve_tsp(g,assign[assign.team==t]).index,'tsp_order']=range(len(solve_tsp(g,assign[assign.team==t]))) for t in assign.team.dropna().astype(int).unique()]
        st.success('Routen berechnet'); st.experimental_rerun()
    st.download_button('Herunterladen',make_export(assign),'routen.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Karte
search=st.session_state.get('search','')
if search:
    addr=search.split(' - ',1)[1]; r=assign[assign['Wahlraum-A']==addr]
    center=[r.lat.iloc[0],r.lon.iloc[0]] if not r.empty else [assign.lat.mean(),assign.lon.mean()]; z=17
else: center=[assign.lat.mean(),assign.lon.mean()]; z=10
m=leafmap.Map(center=center,zoom=z)
G=load_graph()
cols=['#FF00FF','#00FFFF','#00FF00','#FF0000','#FFA500','#FFFF00','#00CED1','#DA70D6','#FF69B4','#8A2BE2']
for i,t in enumerate(sorted(assign.team.dropna().astype(int).unique())):
    df_t=assign[assign.team==t]
    if 'tsp_order'in df_t: df_t=df_t.sort_values('tsp_order')
    pts=list(zip(df_t.lat,df_t.lon))
    if len(pts)>1:
        path=[]; nodes=[ox.distance.nearest_nodes(G,X=lo,Y=la) for la,lo in pts]
        for u,v in zip(nodes[:-1],nodes[1:]):
            try: p=nx.shortest_path(G,u,v,weight='length'); path.extend([(G.nodes[n]['y'],G.nodes[n]['x']) for n in p])
            except: pass
        folium.PolyLine(path,color=cols[i%len(cols)],weight=6,opacity=0.8,tooltip=f"Kontrollbezirk {int(t)}").add_to(m)
cluster=MarkerCluster(disableClusteringAtZoom=13)
for _,r in assign.dropna(subset=['lat','lon']).iterrows():
    html=f"<div style='white-space: nowrap;'><b>{r['Wahlraum-B']}</b><br>{r['Wahlraum-A']}<br>Anzahl Räume: {r['num_rooms']}</div>"
    cluster.add_child(folium.Marker(location=[r.lat,r.lon],popup=folium.Popup(html,max_width=300)))
cluster.add_to(m)
if not search: m.fit_bounds(assign[['lat','lon']].values.tolist())
m.to_streamlit(use_container_width=True,height=700)
