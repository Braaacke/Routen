
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

st.set_page_config(layout="wide")

st.title("🗺 Interaktive Stop-Auswahl per Kartenklick (pydeck)")

# Beispiel-Daten laden
df = pd.read_csv("cleaned_addresses.csv")

# Session-State für Auswahl initialisieren
if "selected" not in st.session_state:
    st.session_state.selected = []

# Kartenmittelpunkt
midpoint = (np.average(df["lat"]), np.average(df["lon"]))

# pydeck Layer mit Marker
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_radius=100,
    get_fill_color=[0, 128, 255, 160],
    pickable=True,
)

# pydeck Deck definieren
r = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=midpoint[0],
        longitude=midpoint[1],
        zoom=11,
        pitch=0,
    ),
    layers=[layer],
    tooltip={"text": "{Wahlraum-A}\n{Wahlraum-B}"},
)

# Zeige die Karte
clicked = st.pydeck_chart(r)

# Eventuelle Koordinaten aus dem Klick abfangen
event = st.get_last_clicked()
if event and "lng" in event and "lat" in event:
    st.write(f"📍 Geklickt: {event['lat']:.5f}, {event['lng']:.5f}")

    # Nächstgelegenen Punkt ermitteln
    df["dist"] = np.sqrt((df["lat"] - event["lat"])**2 + (df["lon"] - event["lng"])**2)
    nearest = df.sort_values("dist").iloc[0]
    selected_stop = nearest["Wahlraum-A"]

    if selected_stop not in st.session_state.selected:
        st.session_state.selected.append(selected_stop)

# Anzeige der ausgewählten Stops
st.subheader("📝 Ausgewählte Stops")
st.write(st.session_state.selected)
