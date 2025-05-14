"""Interaktives Routenbearbeitungstool mit Zoom-abhängiger Markersichtbarkeit"""

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import io

@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = df.columns.str.lower()
    return df

# Zeichnet eine Folium-Karte mit Marker-Cluster
def create_map(df: pd.DataFrame, zoom_start=12):
    m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=zoom_start)
    cluster = MarkerCluster().add_to(m)
    for _, r in df.iterrows():
        popup_html = (
            f"<div style='white-space: nowrap;'>"
            f"<b>{r.get('wahlraum-b', '')}</b><br>{r.get('wahlraum-a', '')}<br>Räume: {r.get('num_rooms', '')}"
            "</div>"
        )
        folium.Marker(location=[r['lat'], r['lon']], popup=folium.Popup(popup_html, max_width=300))
    return m

def main():
    st.title("Routen-Bearbeitungstool")

    uploaded_file = st.file_uploader("Excel-Datei hochladen", type=['xlsx'])
    if not uploaded_file:
        st.info("Bitte laden Sie eine Excel-Datei hoch.")
        return

    df = load_excel(uploaded_file)

    # Spalten umbenennen
    alt = st.multiselect("Spalten umbenennen (Alt)", df.columns)
    neu = st.text_input("Neue Spaltennamen (Komma-getrennt)")
    if alt and neu:
        new_names = [n.strip() for n in neu.split(',')]
        if len(new_names) == len(alt):
            df.rename(columns=dict(zip(alt, new_names)), inplace=True)

    st.dataframe(df)

    lat_col = st.selectbox("Spalte für Breite (lat)", df.columns)
    lon_col = st.selectbox("Spalte für Länge (lon)", df.columns)
    name_col = st.selectbox("Spalte für Namen", df.columns)

    df['lat'] = df[lat_col].astype(float)
    df['lon'] = df[lon_col].astype(float)
    df['name'] = df[name_col].astype(str)

    if 'latlong' in df.columns:
        df[['lat', 'lon']] = (
            df['latlong']
            .str.split(',', expand=True)
            .iloc[:, :2]
            .astype(float)
        )

    zoom = st.slider("Kartenzoom", 1, 18, 12)
    m = create_map(df, zoom_start=zoom)
    st_folium(m, width=700, height=500)

    # Team-ID aus Namen extrahieren
    if 'name' in df:
        choice = st.selectbox("Punkte zur Route hinzufügen", df['name'].tolist())
        if '_' in choice:
            part = choice.split('_', 1)[1]
            if part.isdigit():
                st.write(f"Team-ID: {int(part)}")
            else:
                st.warning(f"Ungültiges Team-Format: {choice}")
        else:
            st.warning(f"Kein Unterstrich in Name: {choice}")

    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("Tabelle herunterladen", data=buf.getvalue(), file_name="bearbeitet.xlsx")

if __name__ == '__main__':
    main()
