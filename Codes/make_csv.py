# Makes two CSV files QGIS can open with "Add Delimited Text Layer" (geometry is in the WKT column).
import os, json
import pandas as pd

OUT = r"C:\sar_project\3_verification"
HERE = os.path.dirname(os.path.abspath(__file__))

# 1) raw JSON centroids (points) from the inventory
inv = pd.read_csv(os.path.join(HERE, "inventory_argentina_brazil.csv"))
pts = pd.DataFrame({
    "WKT": [f"POINT({lo} {la})" for lo, la in zip(inv.centroid_longitude_wgs84, inv.centroid_latitude_wgs84)],
    "field_shared_name": inv.field_shared_name, "country": inv.country_folder,
    "crop": inv.crop, "year": inv.year,
    "lat": inv.centroid_latitude_wgs84, "lon": inv.centroid_longitude_wgs84})
pts.to_csv(os.path.join(OUT, "raw_json_centroids.csv"), index=False)
print("points written:", len(pts), "->", os.path.join(OUT, "raw_json_centroids.csv"))

# 2) SAR tiles (polygons) from the file make_footprints.py already created
g = json.load(open(os.path.join(OUT, "sar_tiles.geojson")))
rows = []
for f in g["features"]:
    ring = ", ".join(f"{x} {y}" for x, y in f["geometry"]["coordinates"][0])
    p = f["properties"]
    rows.append({"WKT": f"POLYGON(({ring}))", "field": p.get("field"), "country": p.get("country"),
                 "result": p.get("result"), "reasons": p.get("reasons"),
                 "n_dates": p.get("n_dates"), "valid_fraction": p.get("valid_pixel_fraction"),
                 "VV_dB": p.get("tile_mean_VV_dB"), "VH_dB": p.get("tile_mean_VH_dB")})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "sar_tiles.csv"), index=False)
print("tiles written:", len(rows), "->", os.path.join(OUT, "sar_tiles.csv"))