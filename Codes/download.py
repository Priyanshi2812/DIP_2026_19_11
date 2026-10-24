# ONE SCRIPT FOR EVERYTHING (TA steps 4-6):
#   1) download Sentinel-1 VV+VH GeoTIFF per field (sowing -> harvest, raw JSON dates) to your laptop
#   2) co-register the dates inside each file
#   3) verify every file against the raw JSON centroids and write reports
# Safe to stop (Ctrl+C) and run again: finished work is skipped.
import os, sys, subprocess, importlib, time, math, collections, glob, warnings, json, zipfile, random

# ---------------- SETTINGS (only change these) ----------------
LIMIT     = None      # None = ALL 1302 fields (set to 5 for a quick test)
BASE_DIR  = r"C:\sar_project"          # everything is saved here
ZIP_PATH  = ""         # optional: path to the raw YieldSAT zip, e.g. r"D:\yieldsat_raw.zip" (enables raw-JSON check). Leave "" to skip.
TARGET_MB = 5.5        # aim for 5-10 MB per file
PROJECT   = "dip-sar-project"
# --------------------------------------------------------------

# auto-install missing libraries
for mod, pipname in [("ee", "earthengine-api"), ("requests", "requests"), ("pandas", "pandas"),
                     ("numpy", "numpy"), ("scipy", "scipy"), ("skimage", "scikit-image"),
                     ("rasterio", "rasterio")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        print("installing", pipname, "...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pipname])

import requests, numpy as np, pandas as pd, rasterio, ee
from rasterio.warp import transform
from scipy.ndimage import shift as nd_shift, gaussian_filter
from skimage.registration import phase_cross_correlation
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.path.join(HERE, "inventory_argentina_brazil.csv")
RAW_DIR   = os.path.join(BASE_DIR, "1_sar_downloaded")
COREG_DIR = os.path.join(BASE_DIR, "2_coregistered")
REPORT_DIR = os.path.join(BASE_DIR, "3_verification")
for d in (RAW_DIR, COREG_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

inv_all = pd.read_csv(INVENTORY)
inv = inv_all.head(LIMIT) if LIMIT else inv_all
print(f"inventory rows: {len(inv_all)} | processing: {len(inv)}")

try:
    ee.Initialize(project=PROJECT)
except Exception:
    print("First-time sign-in: a browser window will open. Choose your Google account and click Allow.")
    ee.Authenticate()
    ee.Initialize(project=PROJECT)

# =========================== PHASE 1: DOWNLOAD ===========================
LOG1 = os.path.join(RAW_DIR, "download_log.csv")

def log1(row):
    pd.DataFrame([row]).to_csv(LOG1, mode="a", header=not os.path.exists(LOG1), index=False)

def get_url(name, lon, lat, epsg, start, end):
    centre = ee.Geometry.Point([lon, lat])
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(centre)
           .filterDate(start, ee.Date(end).advance(1, "day"))
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    orbits = col.aggregate_array("relativeOrbitNumber_start").getInfo()
    if not orbits:
        return None, 0, None, None, None
    orbit = collections.Counter(orbits).most_common(1)[0][0]
    col = (col.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)).sort("system:time_start")
              .map(lambda i: i.set("d", i.date().format("YYYYMMdd"))))
    dates = sorted(set(col.aggregate_array("d").getInfo()))   # unique dates (two slices on one day are merged)
    n = len(dates)
    npix = TARGET_MB * 1e6 / (4 * 2 * n)
    half = min(max(5 * math.sqrt(npix), 300), 3000)
    region = centre.buffer(half).bounds()
    names = [p + d for d in dates for p in ("VV_", "VH_")]
    per_date = ee.ImageCollection.fromImages(
        [col.filter(ee.Filter.eq("d", d)).select(["VV", "VH"]).mosaic() for d in dates])
    img = per_date.toBands().rename(names).clip(region).toFloat()
    url = img.getDownloadURL({"name": name, "region": region, "scale": 10,
                              "crs": f"EPSG:{int(epsg)}", "format": "GEO_TIFF"})
    return url, n, orbit, dates[0], dates[-1]

print("\n=== PHASE 1/3: downloading SAR files ===")
total = len(inv)
for idx, (_, r) in enumerate(inv.iterrows(), 1):
    name = r["field_shared_name"]
    if os.path.exists(os.path.join(RAW_DIR, name + ".tif")):
        continue
    for attempt in range(1, 4):
        try:
            url, n, orbit, d0, d1 = get_url(name, r["centroid_longitude_wgs84"], r["centroid_latitude_wgs84"],
                                            r["projected_crs_epsg"], r["sar_start"], r["sar_end"])
            if url is None:
                log1({"field": name, "status": "no S1 coverage", "n_dates": 0, "MB": 0})
                print(f"[{idx}/{total}] no Sentinel-1 coverage: {name}")
                break
            tmp = os.path.join(RAW_DIR, name + ".part")
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(tmp, os.path.join(RAW_DIR, name + ".tif"))
            mb = os.path.getsize(os.path.join(RAW_DIR, name + ".tif")) / 1e6
            log1({"field": name, "status": "ok", "n_dates": n, "MB": round(mb, 2),
                  "relative_orbit": orbit, "first_date": d0, "last_date": d1})
            print(f"[{idx}/{total}] downloaded {name}  dates={n}  {mb:.1f} MB")
            break
        except Exception as e:
            print(f"[{idx}/{total}] retry {attempt} {name}: {e}")
            time.sleep(10 * attempt)
    else:
        log1({"field": name, "status": "failed", "n_dates": 0, "MB": 0})

# ========================== PHASE 2: CO-REGISTER ==========================
LOG2 = os.path.join(COREG_DIR, "coregistration_log.csv")

def prep(a):
    a = np.array(a, dtype="float64")
    a[~np.isfinite(a)] = np.nan                      # inf / -inf / nodata -> NaN
    m = np.nanmean(a) if np.isfinite(a).any() else 0.0
    a = np.where(np.isnan(a), m, a)                  # fill gaps with the mean
    a = np.clip(a, -60, 20)                          # sane dB range
    a = gaussian_filter(a, 2)
    return a - a.mean()

def apply_shift(b, sh):
    b = np.array(b, dtype="float64")
    nanmask = ~np.isfinite(b)
    out = nd_shift(np.where(nanmask, 0.0, b), sh, order=1, mode="nearest")
    if nanmask.any():
        out[nd_shift(nanmask.astype("float32"), sh, order=0, mode="nearest") > 0.5] = np.nan
    return out

print("\n=== PHASE 2/3: co-registration ===")
files = sorted(glob.glob(os.path.join(RAW_DIR, "*.tif")))
for n_f, path in enumerate(files, 1):
    base = os.path.basename(path)
    if os.path.exists(os.path.join(COREG_DIR, base)):
        continue
    try:
        with rasterio.open(path) as src:
            data = src.read().astype("float32"); prof = src.profile.copy(); desc = list(src.descriptions)
        nb = data.shape[0]
        if nb < 4 or nb % 2:
            raise ValueError(f"unexpected band count {nb}")
        vv_idx = list(range(0, nb, 2))
        ref = prep(np.nanmedian(data[vv_idx], axis=0))
        out = data.copy(); rows = []
        for i in vv_idx:
            try:
                sh, _, _ = phase_cross_correlation(ref, prep(data[i]), upsample_factor=10, normalization=None)
                mag = float(np.hypot(sh[0], sh[1])); applied = bool(np.isfinite(mag)) and mag <= 5
            except Exception:
                sh, mag, applied = (0.0, 0.0), 0.0, False   # could not measure: leave this date unchanged
            if applied:
                out[i] = apply_shift(data[i], sh); out[i + 1] = apply_shift(data[i + 1], sh)
            rows.append({"file": base, "date_band": desc[i] if desc[i] else f"band{i+1}",
                         "shift_row_px": round(float(sh[0]), 2), "shift_col_px": round(float(sh[1]), 2),
                         "shift_m": round(mag * 10, 1), "applied": applied})
        prof.update(dtype="float32", compress="deflate")
        with rasterio.open(os.path.join(COREG_DIR, base), "w", **prof) as dst:
            dst.write(out)
            if any(desc):
                dst.descriptions = tuple(d if d else "" for d in desc)
        pd.DataFrame(rows).to_csv(LOG2, mode="a", header=not os.path.exists(LOG2), index=False)
        print(f"[{n_f}/{len(files)}] co-registered {base}  max shift {max(r['shift_m'] for r in rows)} m")
    except Exception as e:
        print(f"[{n_f}/{len(files)}] FAILED {base}: {e}")

# =========================== PHASE 3: VERIFY ===========================
print("\n=== PHASE 3/3: verification against raw JSON centroids ===")
invi = inv_all.set_index("field_shared_name")
rows = []
for path in sorted(glob.glob(os.path.join(COREG_DIR, "*.tif"))):
    name = os.path.splitext(os.path.basename(path))[0]
    r = {"field": name, "size_MB": round(os.path.getsize(path) / 1e6, 2)}
    if name not in invi.index:
        r.update(result="FAIL", reasons="not in inventory"); rows.append(r); continue
    f = invi.loc[name]; problems = []
    try:
        with rasterio.open(path) as src:
            data = src.read().astype("float32"); desc = list(src.descriptions)
            epsg = src.crs.to_epsg() if src.crs else None
            r["epsg_tif"], r["epsg_json"] = epsg, int(f.projected_crs_epsg)
            if epsg != int(f.projected_crs_epsg): problems.append("CRS differs")
            x, y = transform("EPSG:4326", src.crs, [f.centroid_longitude_wgs84], [f.centroid_latitude_wgs84])
            x, y = x[0], y[0]; b = src.bounds
            inside = (b.left < x < b.right) and (b.bottom < y < b.top)
            r["centroid_inside_raster"] = inside
            r["centroid_offset_from_tile_centre_m"] = round(float(np.hypot(x - (b.left + b.right) / 2, y - (b.bottom + b.top) / 2)), 1)
            if not inside: problems.append("raw centroid outside raster")
            elif r["centroid_offset_from_tile_centre_m"] > 100: problems.append("centroid far from tile centre")
            r["pixel_size_m"] = round(abs(src.res[0]), 2)
            r["n_dates"] = data.shape[0] // 2
            valid = np.isfinite(data) & (data != 0)
            r["valid_pixel_fraction"] = round(float(valid.mean()), 3)
            if valid.mean() < 0.9: problems.append("<90% valid pixels")
            vv = np.where(valid[0::2], data[0::2], np.nan); vh = np.where(valid[1::2], data[1::2], np.nan)
            r["tile_mean_VV_dB"] = round(float(np.nanmean(vv)), 2); r["tile_mean_VH_dB"] = round(float(np.nanmean(vh)), 2)
            if not (-25 <= r["tile_mean_VV_dB"] <= 0): problems.append("VV out of -25..0 dB")
            if not (-35 <= r["tile_mean_VH_dB"] <= -5): problems.append("VH out of -35..-5 dB")
            if inside:
                row, col = src.index(x, y); r0, c0 = max(row - 1, 0), max(col - 1, 0)
                r["centroid_3x3_VV_dB"] = round(float(np.nanmean(vv[:, r0:row + 2, c0:col + 2])), 2)
                r["centroid_3x3_VH_dB"] = round(float(np.nanmean(vh[:, r0:row + 2, c0:col + 2])), 2)
        dates = [d.split("_")[1] for d in desc[0::2] if d and "_" in d]
        if dates:
            r["first_date"], r["last_date"] = min(dates), max(dates)
            if min(dates) < f.sar_start.replace("-", "") or max(dates) > f.sar_end.replace("-", ""):
                problems.append("SAR dates outside sowing-harvest")
        r["season_sowing"], r["season_harvest"] = f.sar_start, f.sar_end
        r["size_5_to_10MB"] = 5 <= r["size_MB"] <= 10
    except Exception as e:
        problems.append(f"read error: {e}")
    r["result"] = "PASS" if not problems else "FAIL"; r["reasons"] = "; ".join(problems)
    rows.append(r)
rep = pd.DataFrame(rows)
rep.to_csv(os.path.join(REPORT_DIR, "verification_report.csv"), index=False)
print(rep["result"].value_counts().to_string() if len(rep) else "no files to verify")

# optional: direct comparison with the raw JSON files inside the zip
if ZIP_PATH and os.path.exists(ZIP_PATH) and len(rep):
    def find_key(obj, key):
        if isinstance(obj, dict):
            if key in obj: return obj[key]
            for v in obj.values():
                res = find_key(v, key)
                if res is not None: return res
        elif isinstance(obj, list):
            for v in obj:
                res = find_key(v, key)
                if res is not None: return res
        return None
    random.seed(1)
    have = [n for n in rep["field"] if n in invi.index]
    pick = random.sample(have, min(20, len(have)))
    zf = zipfile.ZipFile(ZIP_PATH)
    jsons = {n.split("/")[-1]: n for n in zf.namelist() if n.lower().endswith(".json")}
    jr = []
    for name in pick:
        f = invi.loc[name]; jn = jsons.get(os.path.basename(f.path))
        if jn is None:
            jr.append({"field": name, "result": "JSON not found in zip"}); continue
        j = json.loads(zf.read(jn).decode("utf-8"))
        lat = float(find_key(j, "centroid_latitude_wgs84")); lon = float(find_key(j, "centroid_longitude_wgs84"))
        sow, har = find_key(j, "seeding_date"), find_key(j, "harvesting_date")
        with rasterio.open(os.path.join(COREG_DIR, name + ".tif")) as src:
            x, y = transform("EPSG:4326", src.crs, [lon], [lat]); b = src.bounds
            inside = (b.left < x[0] < b.right) and (b.bottom < y[0] < b.top)
            off = float(np.hypot(x[0] - (b.left + b.right) / 2, y[0] - (b.bottom + b.top) / 2))
        ok = inside and off < 100 and sow == f.seeding_date and har == f.harvesting_date
        jr.append({"field": name, "json_lat": lat, "json_lon": lon, "json_seeding": sow, "json_harvest": har,
                   "raw_centroid_inside_SAR_tile": inside, "offset_from_tile_centre_m": round(off, 1),
                   "result": "MATCH" if ok else "CHECK"})
    pd.DataFrame(jr).to_csv(os.path.join(REPORT_DIR, "json_vs_sar_check.csv"), index=False)
    print("raw JSON check:", pd.DataFrame(jr)["result"].value_counts().to_dict())

print("\nALL DONE. Open this folder:", BASE_DIR)
print("  1_sar_downloaded  = SAR .tif files + download_log.csv")
print("  2_coregistered    = aligned .tif files + coregistration_log.csv")
print("  3_verification    = verification_report.csv (+ json_vs_sar_check.csv)")
if LIMIT:
    print(f"\nThis was a TEST on {LIMIT} fields. Set LIMIT = None at the top and run again for all fields.")