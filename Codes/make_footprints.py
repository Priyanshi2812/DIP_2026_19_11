# Builds ONE file with the footprint of every SAR tile (+ PASS/FAIL), so QGIS can show ALL fields at once.
import os, glob, json
import pandas as pd, rasterio
from rasterio.warp import transform

B = r"C:\sar_project"
rep = pd.read_csv(os.path.join(B, "3_verification", "verification_report.csv")).drop_duplicates("field").set_index("field")

def clean(v):
    if pd.isna(v): return None
    return v.item() if hasattr(v, "item") else v

feats = []
for path in sorted(glob.glob(os.path.join(B, "2_coregistered", "*.tif"))):
    name = os.path.splitext(os.path.basename(path))[0]
    with rasterio.open(path) as src:
        b = src.bounds
        xs = [b.left, b.right, b.right, b.left, b.left]
        ys = [b.bottom, b.bottom, b.top, b.top, b.bottom]
        lon, lat = transform(src.crs, "EPSG:4326", xs, ys)
    props = {"field": name, "country": name.split("_")[0]}
    if name in rep.index:
        props.update({k: clean(v) for k, v in rep.loc[name].to_dict().items()})
    feats.append({"type": "Feature", "properties": props,
                  "geometry": {"type": "Polygon", "coordinates": [[[x, y] for x, y in zip(lon, lat)]]}})

out = os.path.join(B, "3_verification", "sar_tiles.geojson")
json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
print("tiles written:", len(feats), "->", out)