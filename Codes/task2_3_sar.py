# TASKS 2 + 3 in one run: SAR time series per field (linear-power averaging, season bins), SAR features, SAR-only models.
# Needs (C:\sar_project): 2_coregistered\*.tif, data\master_table.csv, data\folds.csv, evaluation.py (all from Task 1).
# Outputs: features\sar_timeseries.csv, features\sar_features.csv, results\results_sar.csv, results\confusion_sar_<model>.csv, results\sar_report.txt
# Set LIMIT = 20 for a quick test first, then LIMIT = None for everything.
import os, sys, subprocess, importlib, warnings
for mod, p in [("numpy","numpy"),("pandas","pandas"),("rasterio","rasterio"),("sklearn","scikit-learn")]:
    try: importlib.import_module(mod)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install",p])
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

BASE = r"C:\Users\Hp\OneDrive\Documents\sar_project"
LIMIT = None
N_BINS = 6
WIN = 11                     # fallback: 11 x 11 pixel window (110 m) around the raw centroid
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, BASE)
for d in ("features", "results"): os.makedirs(os.path.join(BASE, d), exist_ok=True)
LOG = []
def say(s=""): print(s); LOG.append(s)

def read_tif(path):
    import rasterio
    with rasterio.open(path) as src:
        desc = []
        for i in range(1, src.count + 1):      # band name from description, or from any band tag
            d = src.descriptions[i - 1] or ""
            desc.append(d + " " + " ".join(str(v) for v in src.tags(i).values()))
        return src.read().astype("float32"), desc, str(src.crs)

_LOG = None
def first_last(field_id):
    """first/last Sentinel-1 date of a field from download_log.csv (used only if band names carry no date)"""
    global _LOG
    if _LOG is None:
        p = os.path.join(BASE, "1_sar_downloaded", "download_log.csv")
        _LOG = pd.read_csv(p).drop_duplicates("field", keep="last").set_index("field") if os.path.exists(p) else pd.DataFrame()
    try:
        r = _LOG.loc[field_id]; return pd.to_datetime(str(int(r.first_date)), format="%Y%m%d"), pd.to_datetime(str(int(r.last_date)), format="%Y%m%d")
    except Exception:
        return None

def parse_dates(desc, nb, sow, har, field_id=""):
    """1) YYYYMMDD found in band names/tags; 2) else spread evenly between the field's first and last S1 date; 3) else sowing..harvest"""
    import re
    found = [re.search(r"(20\d{6})", x) for x in desc[0::2]]
    if all(found) and len(found) == nb // 2:
        return pd.to_datetime([f.group(1) for f in found], format="%Y%m%d"), "names"
    fl = first_last(field_id)
    if fl is not None:
        return pd.to_datetime(np.linspace(fl[0].value, fl[1].value, nb // 2)), "log"
    return pd.to_datetime(np.linspace(sow.value, har.value, nb // 2)), "even"

def field_series(data):
    """per date: window mean in LINEAR power -> dB, and the pixel std (dB). Returns vv, vh, vv_sd, vh_sd."""
    nb, h, w = data.shape
    r0, c0 = max(h // 2 - WIN // 2, 0), max(w // 2 - WIN // 2, 0)
    win = data[:, r0:r0 + WIN, c0:c0 + WIN].copy()
    win[~np.isfinite(win)] = np.nan; win[win == 0] = np.nan
    out = []
    for b in (win[0::2], win[1::2]):
        lin = 10 ** (b / 10.0)
        m = np.nanmean(lin.reshape(len(lin), -1), axis=1)
        out.append(10 * np.log10(m)); out.append(np.nanstd(b.reshape(len(b), -1), axis=1))
    return out[0], out[2], out[1], out[3]

def to_bins(frac, vv, vh, vv_sd, vh_sd):
    """average inside 6 fixed season bins (linear power for the means), then interpolate empty bins"""
    edges = np.linspace(0, 1, N_BINS + 1); idx = np.minimum((frac * N_BINS).astype(int), N_BINS - 1)
    res = {k: np.full(N_BINS, np.nan) for k in ("vv", "vh", "vv_sd", "vh_sd")}
    for k in range(N_BINS):
        s = idx == k
        if s.any():
            res["vv"][k] = 10 * np.log10(np.nanmean(10 ** (vv[s] / 10))); res["vh"][k] = 10 * np.log10(np.nanmean(10 ** (vh[s] / 10)))
            res["vv_sd"][k] = np.nanmean(vv_sd[s]); res["vh_sd"][k] = np.nanmean(vh_sd[s])
    empty = np.isnan(res["vv"]); centers = (edges[:-1] + edges[1:]) / 2
    if empty.all(): return None
    for k in res:
        ok = np.isfinite(res[k]); res[k] = np.interp(centers, centers[ok], res[k][ok])
    return res, empty

def feature_row(frac, vv, vh):
    """season statistics from the per-date series (VV, VH in dB)"""
    lv, lh = 10 ** (vv / 10), 10 ** (vh / 10)
    series = {"vv": vv, "vh": vh, "vh_minus_vv": vh - vv, "vh_div_vv": lh / lv, "rvi": 4 * lh / (lv + lh)}
    f = {}
    for n, s in series.items():
        f[f"sar_{n}_mean"], f[f"sar_{n}_std"] = np.nanmean(s), np.nanstd(s)
        f[f"sar_{n}_max"], f[f"sar_{n}_min"] = np.nanmax(s), np.nanmin(s)
        f[f"sar_{n}_amp"] = f[f"sar_{n}_max"] - f[f"sar_{n}_min"]
        f[f"sar_{n}_peak_frac"] = float(frac[int(np.nanargmax(s))])
    return f

def build_tables():
    m = pd.read_csv(os.path.join(BASE, "data", "master_table.csv"))
    if LIMIT: m = m.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), max(LIMIT // 3, 1)), random_state=0)).reset_index(drop=True)
    ts_rows, feat_rows, bad, src_count, crs_set = [], [], [], {}, set()
    for i, r in enumerate(m.itertuples(), 1):
        path = os.path.join(BASE, "2_coregistered", r.field_id + ".tif")
        if not os.path.exists(path): bad.append((r.field_id, "no co-registered file")); continue
        try:
            data, desc, crs = read_tif(path); crs_set.add(crs)
            sow, har = pd.to_datetime(r.sowing), pd.to_datetime(r.harvest)
            dates, how = parse_dates(desc, data.shape[0], sow, har, r.field_id); src_count[how] = src_count.get(how, 0) + 1
            frac = np.clip(((dates - sow).days / max((har - sow).days, 1)).values.astype(float), 0, 1)
            vv, vh, vv_sd, vh_sd = field_series(data)
            keep = np.isfinite(vv) & np.isfinite(vh)
            if keep.sum() < 2: bad.append((r.field_id, "fewer than 2 valid dates")); continue
            frac, vv, vh, vv_sd, vh_sd = frac[keep], vv[keep], vh[keep], vv_sd[keep], vh_sd[keep]
            b, empty = to_bins(frac, vv, vh, vv_sd, vh_sd)
        except Exception as e:
            bad.append((r.field_id, str(e)[:60])); continue
        for k in range(N_BINS):
            ts_rows.append(dict(field_id=r.field_id, bin=k + 1, vv_mean_db=b["vv"][k], vv_std_db=b["vv_sd"][k],
                                vh_mean_db=b["vh"][k], vh_std_db=b["vh_sd"][k], interpolated=bool(empty[k])))
        f = {"field_id": r.field_id, "sar_n_dates": int(keep.sum())}
        f.update(feature_row(frac, vv, vh))
        for k in range(N_BINS): f[f"sar_vv_bin{k+1}"], f[f"sar_vh_bin{k+1}"] = b["vv"][k], b["vh"][k]
        feat_rows.append(f)
        if i % 100 == 0: print(f"[{i}/{len(m)}]")
    ts, ft = pd.DataFrame(ts_rows), pd.DataFrame(feat_rows)
    ts.to_csv(os.path.join(BASE, "features", "sar_timeseries.csv"), index=False)
    ft.to_csv(os.path.join(BASE, "features", "sar_features.csv"), index=False)
    say(f"SAMPLE FILE CHECK: CRS seen = {sorted(crs_set)[:3]}, band order VV,VH pairs per date, first field has {ft.sar_n_dates.iloc[0] if len(ft) else 0} dates")
    say(f"MASK: fixed {WIN}x{WIN} window around the raw centroid (yield masks are not in the downloaded SAR tiles)")
    say(f"FIELDS WITH SAR TABLES: {len(ft)} of {len(m)}; skipped: {len(bad)}; how the dates were found {src_count} (names = real dates from band names; log = spread between the field's first and last Sentinel-1 date; even = sowing to harvest)")
    for b_ in bad[:10]: say(f"  skipped {b_[0]}: {b_[1]}")
    return ft

def models():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.base import BaseEstimator, ClassifierMixin
    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = None
    class XGB(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.le_ = LabelEncoder().fit(y); self.classes_ = self.le_.classes_
            self.m_ = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=0)
            self.m_.fit(X, self.le_.transform(y)); return self
        def predict(self, X): return self.le_.inverse_transform(self.m_.predict(X))
    imp = lambda: SimpleImputer(strategy="median")
    out = {"LogReg": lambda: make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
            "RandomForest": lambda: make_pipeline(imp(), RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=0)),
            "SVM_RBF": lambda: make_pipeline(imp(), StandardScaler(), SVC(kernel="rbf", C=3, class_weight="balanced"))}
    if XGBClassifier is not None:
        out["XGBoost"] = lambda: make_pipeline(imp(), XGB())
    else:
        say("XGBoost is not installed, so it is skipped (pip install xgboost to add it later)")
    return out

def main():
    ft = build_tables()
    if LIMIT:
        say(f"\nTEST MODE (LIMIT = {LIMIT}): tables were written, models skipped because a handful of fields is not enough to train on.")
        say("Set LIMIT = None and run again for the real results.")
        print(ft[["field_id","sar_n_dates","sar_vv_mean","sar_vh_mean","sar_rvi_mean"]].head(8).to_string(index=False)); return
    from evaluation import evaluate_models
    cols = [c for c in ft.columns if c.startswith("sar_") and c != "sar_n_dates"]
    say(f"\nSAR FEATURES: {len(cols)} (season stats of VV, VH, VH-VV, VH/VV, RVI + 6-bin VV and VH profile)")
    say("MODELS (5-fold, spatial blocks from folds.csv):")
    res = evaluate_models(ft, cols, models(), "sar")
    for _, r in res.iterrows(): say(f"{r.model:14s} accuracy {r.accuracy:.3f}  macro-F1 {r.macro_f1:.3f}  weighted-F1 {r.weighted_f1:.3f}  (n={r.n_fields})")
    best = res.sort_values("macro_f1").iloc[-1]
    say(f"\nBEST SAR MODEL (by macro-F1): {best.model}  macro-F1 {best.macro_f1:.3f}")
    say("Confusion matrix of the best model (rows = true, columns = predicted):")
    say(pd.read_csv(os.path.join(BASE, "results", f"confusion_sar_{best.model}.csv"), index_col=0).to_string())
    open(os.path.join(BASE, "results", "sar_report.txt"), "w", encoding="utf-8").write("\n".join(LOG))
    print("\nsaved in", BASE)

if __name__ == "__main__":
    main()
