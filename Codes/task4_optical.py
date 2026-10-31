# TASK 4 (optical) in one run, for the SAME 1,206 fields as SAR: Sentinel-2 cloud mask, NDVI / NDRE / NDWI, season bins,
# s2_ features, optical-only models, NDVI boxplot, and (if sar_features.csv exists) the SAR vs optical vs FUSED comparison.
# Needs internet + Earth Engine login for the first stage. Safe to stop (Ctrl+C) and run again: it continues where it stopped.
# Needs in C:\sar_project: data\master_table.csv, data\folds.csv, evaluation.py, task2_3_sar.py (for the model list)
import os, sys, subprocess, importlib, time
from concurrent.futures import ThreadPoolExecutor, as_completed
for mod, p in [("numpy","numpy"),("pandas","pandas"),("sklearn","scikit-learn"),("matplotlib","matplotlib"),("scipy","scipy"),("ee","earthengine-api")]:
    try: importlib.import_module(mod)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install",p,"--default-timeout=600","--retries","10"])
import numpy as np, pandas as pd

BASE = r"C:\Users\Hp\OneDrive\Documents\sar_project"
PROJECT = "dip-sar-project"
WINDOW_HALF_M = 55      # indices averaged inside a 110 m x 110 m square (about 11 x 11 pixels of 10 m) centred on the raw centroid = the SAME window as SAR
MASK_MODE = "cloudscore"  # "cloudscore" = Cloud Score+ on S2_HARMONIZED (TOA, works for all years 2016-2023)
                          # "scl"        = Scene Classification Layer on S2_SR_HARMONIZED (surface reflectance; global coverage only from about Dec 2018, so earlier fields get no data)
CLOUD_KEEP = 0.60       # Cloud Score+ threshold: keep pixels that are at least 60% clear
SCL_KEEP = [4, 5, 6, 7] # SCL classes KEPT: 4 vegetation, 5 bare soil, 6 water, 7 unclassified. REMOVED: 0 no data, 1 saturated, 2 dark/shadow, 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 thin cirrus, 11 snow
N_BINS = 6
WORKERS = 5
LIMIT = None            # set e.g. 30 for a quick test
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, BASE)
FEAT = os.path.join(BASE, "features"); RES = os.path.join(BASE, "results")
os.makedirs(FEAT, exist_ok=True); os.makedirs(RES, exist_ok=True)
RAW = os.path.join(FEAT, f"s2_raw_observations_{MASK_MODE}_{2*WINDOW_HALF_M}m.csv")   # file name carries the settings, so different settings never mix
LOG = []
def say(s=""): print(s); LOG.append(s)

# ---------------- stage 1: Earth Engine download of per-image index values ----------------
def fetch_one(r):
    import ee
    pt = ee.Geometry.Point([float(r.lon), float(r.lat)]); region = pt.buffer(WINDOW_HALF_M).bounds()
    end = (pd.to_datetime(r.harvest) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if MASK_MODE == "scl":
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt).filterDate(str(r.sowing)[:10], end)
               .map(lambda im: im.updateMask(im.select("SCL").remap(SCL_KEEP, [1] * len(SCL_KEEP), 0).eq(1))))
    else:
        cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(pt).filterDate(str(r.sowing)[:10], end)
               .linkCollection(cs, ["cs_cdf"]).map(lambda im: im.updateMask(im.select("cs_cdf").gte(CLOUD_KEEP))))
    def per_img(im):
        idx = ee.Image.cat([im.normalizedDifference(["B8", "B4"]).rename("ndvi"),
                            im.normalizedDifference(["B8", "B5"]).rename("ndre"),
                            im.normalizedDifference(["B3", "B8"]).rename("ndwi")])
        v = idx.reduceRegion(ee.Reducer.mean(), region, 10, maxPixels=1e6)
        return ee.Feature(None, {"date": im.date().format("YYYY-MM-dd"), "ndvi": v.get("ndvi"), "ndre": v.get("ndre"), "ndwi": v.get("ndwi")})
    feats = col.map(per_img).filter(ee.Filter.notNull(["ndvi"])).getInfo()["features"]
    return [(r.field_id, f["properties"]["date"], f["properties"]["ndvi"], f["properties"]["ndre"], f["properties"]["ndwi"]) for f in feats]

def with_retry(r, tries=3):
    for k in range(tries):
        try: return fetch_one(r)
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(3 * (k + 1))

def download(todo):
    import ee
    try: ee.Initialize(project=PROJECT)
    except Exception:
        ee.Authenticate(); ee.Initialize(project=PROJECT)
    first, n, t0 = not os.path.exists(RAW), 0, time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(with_retry, r): r.field_id for r in todo.itertuples()}
        for fu in as_completed(futs):
            name = futs[fu]
            try: rows = fu.result()
            except Exception as e:
                print("failed", name, str(e)[:90]); continue
            if not rows: rows = [(name, "", np.nan, np.nan, np.nan)]      # marks "no clear image" so it is not retried
            pd.DataFrame(rows, columns=["field_id","date","ndvi","ndre","ndwi"]).to_csv(RAW, mode="a", header=first, index=False)
            first = False; n += 1
            if n % 25 == 0: print(f"[{n}/{len(todo)}] {(time.time()-t0)/60:.1f} min")

# ---------------- stage 2: season bins and s2_ features ----------------
IDX = ("ndvi", "ndre", "ndwi")
def bins_and_features(raw, m):
    ts_rows, feat_rows = [], []
    meta = m.set_index("field_id")
    for fid, g in raw.dropna(subset=["date", "ndvi"]).groupby("field_id"):
        if fid not in meta.index: continue
        sow, har = pd.to_datetime(meta.loc[fid, "sowing"]), pd.to_datetime(meta.loc[fid, "harvest"])
        d = pd.to_datetime(g.date); frac = np.clip(((d - sow).dt.days / max((har - sow).days, 1)).values.astype(float), 0, 1)
        if len(g) < 2: continue
        idx = np.minimum((frac * N_BINS).astype(int), N_BINS - 1); centers = (np.arange(N_BINS) + 0.5) / N_BINS
        f = {"field_id": fid, "s2_n_images": len(g)}
        for k in IDX:
            v = g[k].values.astype(float)
            b = np.array([v[idx == j].mean() if (idx == j).any() else np.nan for j in range(N_BINS)])
            empty = np.isnan(b); b = np.interp(centers, centers[~empty], b[~empty])
            for j in range(N_BINS):
                ts_rows.append(dict(field_id=fid, index=k, bin=j + 1, value=b[j], interpolated=bool(empty[j])))
                f[f"s2_{k}_bin{j+1}"] = b[j]
            f.update({f"s2_{k}_mean": v.mean(), f"s2_{k}_std": v.std(), f"s2_{k}_max": v.max(), f"s2_{k}_min": v.min(),
                      f"s2_{k}_amp": v.max() - v.min(), f"s2_{k}_peak_frac": float(frac[int(np.argmax(v))])})
        feat_rows.append(f)
    return pd.DataFrame(ts_rows), pd.DataFrame(feat_rows)

def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from evaluation import evaluate_models
    from task2_3_sar import models
    m = pd.read_csv(os.path.join(BASE, "data", "master_table.csv"))
    if LIMIT: m = m.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), max(LIMIT // 3, 1)), random_state=0))
    done = set(pd.read_csv(RAW).field_id) if os.path.exists(RAW) else set()
    todo = m[~m.field_id.isin(done)]
    say(f"FIELDS: {len(m)}; optical data already downloaded for {len(done)}; to download now: {len(todo)}")
    if len(todo): download(todo)
    raw = pd.read_csv(RAW)
    ts, ft = bins_and_features(raw, m)
    ts.to_csv(os.path.join(FEAT, "s2_timeseries.csv"), index=False); ft.to_csv(os.path.join(FEAT, "s2_features.csv"), index=False)
    mask_txt = (f"Scene Classification Layer, kept classes {SCL_KEEP} (surface reflectance, S2_SR_HARMONIZED)" if MASK_MODE == "scl"
                else f"Cloud Score+ >= {CLOUD_KEEP} (top-of-atmosphere, S2_HARMONIZED)")
    say(f"CLOUD MASK: {mask_txt}; bands B3, B4, B5, B8; mean inside a {2*WINDOW_HALF_M} m square window (about 11 x 11 pixels) around the raw centroid; {N_BINS} season bins")
    ni = int(ts.interpolated.sum()) if len(ts) else 0
    say(f"INTERPOLATED ENTRIES: {ni} of {len(ts)} field-index-bin rows ({ni/max(len(ts),1)*100:.1f}%); fields with fewer than 2 clear images are left out: {len(m)-len(ft)}")
    say(f"FIELDS WITH OPTICAL FEATURES: {len(ft)} of {len(m)} (no clear Sentinel-2 image in the window for {len(m)-len(ft)})")
    say(f"images per field: median {ft.s2_n_images.median():.0f}, min {ft.s2_n_images.min()}, max {ft.s2_n_images.max()}")
    mm = ft.merge(m[["field_id", "label", "country"]], on="field_id")
    say("\nOPTICAL RESULTS BY CROP (field season statistics)")
    for c, g in mm.groupby("label"):
        say(f"{c:8s} n={len(g):4d}  NDVI mean {g.s2_ndvi_mean.mean():.3f} +/- {g.s2_ndvi_mean.std():.3f}   NDVI max {g.s2_ndvi_max.mean():.3f}   "
            f"NDRE mean {g.s2_ndre_mean.mean():.3f}   NDWI mean {g.s2_ndwi_mean.mean():.3f}   NDVI peak at {g.s2_ndvi_peak_frac.mean()*100:.0f}% of season")
    crops = sorted(mm.label.unique()); col = {"corn": "#d4a017", "soybean": "#2e7d32", "wheat": "#1565c0"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].boxplot([mm.loc[mm.label == c, "s2_ndvi_mean"] for c in crops]); ax[0].set_xticklabels(crops); ax[0].set_title("Mean NDVI per field")
    ax[1].boxplot([mm.loc[mm.label == c, "s2_ndvi_max"] for c in crops]); ax[1].set_xticklabels(crops); ax[1].set_title("Peak NDVI per field")
    for c in crops:
        cols = [f"s2_ndvi_bin{j}" for j in range(1, N_BINS + 1)]; mu = mm.loc[mm.label == c, cols].mean().values; sd = mm.loc[mm.label == c, cols].std().values
        x = (np.arange(N_BINS) + 0.5) / N_BINS * 100
        ax[2].plot(x, mu, label=c, color=col.get(c)); ax[2].fill_between(x, mu - sd, mu + sd, alpha=0.12, color=col.get(c))
    ax[2].set_title("Average NDVI through the season"); ax[2].set_xlabel("% of season (sowing to harvest)"); ax[2].legend()
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_s2_ndvi_by_class.png"), dpi=160); plt.close(fig)

    cols_s2 = [c for c in ft.columns if c.startswith("s2_") and c != "s2_n_images"]
    say(f"\nOPTICAL FEATURES: {len(cols_s2)} (NDVI, NDRE, NDWI: season mean/std/max/min/amplitude/peak position + 6-bin profile)")
    say("OPTICAL-ONLY MODELS (5-fold spatial folds, same folds.csv):")
    mods = models()
    res = evaluate_models(ft, cols_s2, mods, "s2")
    for _, r in res.iterrows(): say(f"{r.model:14s} accuracy {r.accuracy:.3f}  macro-F1 {r.macro_f1:.3f}  weighted-F1 {r.weighted_f1:.3f}  (n={r.n_fields})")
    best = res.sort_values("macro_f1").iloc[-1]
    say(f"BEST OPTICAL MODEL: {best.model}, macro-F1 {best.macro_f1:.3f}"); say(pd.read_csv(os.path.join(RES, f"confusion_s2_{best.model}.csv"), index_col=0).to_string())

    # ---------------- stage 3: fusion, only if the SAR features exist ----------------
    sp = os.path.join(FEAT, "sar_features.csv")
    if os.path.exists(sp):
        from sklearn.metrics import f1_score
        from scipy import stats
        sar = pd.read_csv(sp); cols_sar = [c for c in sar.columns if c.startswith("sar_") and c != "sar_n_dates"]
        both = sar.merge(ft, on="field_id"); say(f"\n=== FUSION: SAR vs OPTICAL vs FUSED on the {len(both)} fields that have both ===")
        out = {}
        for tag, cols in (("cmp_sar", cols_sar), ("cmp_s2", cols_s2), ("cmp_fused", cols_sar + cols_s2)):
            out[tag] = evaluate_models(both, cols, mods, tag)
        say("\nsummary (macro-F1):")
        for mn in mods: say(f"{mn:14s} SAR {out['cmp_sar'].set_index('model').macro_f1[mn]:.3f}   optical {out['cmp_s2'].set_index('model').macro_f1[mn]:.3f}   FUSED {out['cmp_fused'].set_index('model').macro_f1[mn]:.3f}")
        mn = "RandomForest"; P = {t: pd.read_csv(os.path.join(RES, f"pred_{t}_{mn}.csv")) for t in out}; y = P["cmp_fused"].label.values
        say(f"\nEVIDENCE ({mn}, every field predicted once by the spatial folds):")
        labs = sorted(set(y))
        for t, nm in (("cmp_sar", "SAR only"), ("cmp_s2", "optical only"), ("cmp_fused", "FUSED")):
            f = f1_score(y, P[t].pred, average=None, labels=labs); say(f"per-crop F1 {nm:13s} " + "  ".join(f"{l}={v:.3f}" for l, v in zip(labs, f)))
        for t, nm in (("cmp_sar", "SAR only"), ("cmp_s2", "optical only")):
            a_ok, b_ok = P["cmp_fused"].pred.values == y, P[t].pred.values == y
            b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
            p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
            say(f"fused right & {nm} wrong: {b} fields; {nm} right & fused wrong: {c} fields; McNemar p = {p:.4g}")
        say("fused model information share: see feature importances below")
        mdl = mods[mn]().fit(both[cols_sar + cols_s2], both.merge(m[['field_id','label']], on='field_id').label)
        imp = pd.Series(mdl[-1].feature_importances_, index=cols_sar + cols_s2)
        say(f"share from optical features {imp[cols_s2].sum():.2f}, from SAR features {imp[cols_sar].sum():.2f}; top 6: " + ", ".join(f"{k} ({v:.3f})" for k, v in imp.sort_values(ascending=False).head(6).items()))
    else:
        say("\n(sar_features.csv not found, so fusion was skipped)")
    open(os.path.join(RES, "optical_fusion_report.txt"), "w", encoding="utf-8").write("\n".join(LOG)); print("\nsaved in", BASE)

if __name__ == "__main__":
    main()
