if st.button("Zuordnung exportieren"):
    # Graph einmal laden
    graph = get_graph()
    overview = []
    team_sheets = {}
    for team in sorted(st.session_state.new_assignments["team"].dropna().unique()):
        stops = st.session_state.new_assignments[st.session_state.new_assignments["team"] == team]
        if "tsp_order" in stops.columns:
            stops = stops.sort_values("tsp_order")
        rooms = stops.get("num_rooms", pd.Series()).sum()
        travel_km = 0
        travel_min = 0
        coords = stops[["lat", "lon"]].values.tolist()
        if len(coords) > 1:
            try:
                nodes = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in coords]
                for u, v in zip(nodes[:-1], nodes[1:]):
                    length = nx.shortest_path_length(graph, u, v, weight="length")
                    travel_km += length / 1000
                    travel_min += length / 1000 * 2
            except Exception:
                pass
        service_min = int(rooms * 10)
        time_total = service_min + travel_min
        gmaps_link = "https://www.google.com/maps/dir/" + "/".join([f"{lat},{lon}" for lat, lon in coords])
        overview.append({
            "Kontrollbezirk": team,
            "Anzahl Wahllokale": len(stops),
            "Anzahl Stimmbezirke": int(rooms),
            "Wegstrecke (km)": round(travel_km, 1),
            "Fahrtzeit (min)": int(travel_min),
            "Kontrollzeit (min)": service_min,
            "Gesamtzeit": str(timedelta(minutes=int(time_total))),
            "Google-Link": gmaps_link
        })
        rows = []
        for idx, row in stops.iterrows():
            coord_str = f"{row['lat']},{row['lon']}"
            rows.append({
                "Reihenfolge": idx,
                "Adresse": row["Wahlraum-A"],
                "Stimmbezirke": row.get('rooms', ''),
                "Anzahl Stimmbezirke": row.get('num_rooms', ''),
                "Google-Link": f"https://www.google.com/maps/search/?api=1&query={quote_plus(coord_str)}"
            })
        team_sheets[f"Team_{team}"] = pd.DataFrame(rows)

    overview_df = pd.DataFrame(overview)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="Übersicht", index=False)
        # Auto-adjust column widths for Übersicht
        ws_over = writer.sheets["Übersicht"]
        for idx, col in enumerate(overview_df.columns, 1):
            max_len = max(overview_df[col].astype(str).map(len).max(), len(col))
            from openpyxl.utils import get_column_letter
            ws_over.column_dimensions[get_column_letter(idx)].width = max_len + 2
        for sheet_name, df in team_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            # Auto-adjust column widths for each team sheet
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns, 1):
                max_len = max(df[col].astype(str).map(len).max(), len(col))
                from openpyxl.utils import get_column_letter
                ws.column_dimensions[get_column_letter(idx)].width = max_len + 2
    output.seek(0)

    st.download_button(
        label="📥 Excel-Datei herunterladen",
        data=output,
        file_name="routen_zuweisung_aktualisiert.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
