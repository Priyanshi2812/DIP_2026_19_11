# PRIYANSHI'S WEEK 7 PART: SAR analysis. One run gives everything on the list:
#  SAR-only confusion matrix, per-crop precision/recall/F1, corn/soybean/wheat misclassification analysis,
#  top-10 SAR features, feature-importance figure, and the contribution of VV, VH, VH-VV, VH/VV and RVI.
# Needs (C:\sar_project): features\sar_features.csv, data\master_table.csv, data\folds.csv, evaluation.py (latest), task2_3_sar.py
# Run:  python sar_analysis.py        Output: results\sar_analysis.txt + sar_*.png + sar_*.csv
import os, sys, subprocess, importlib
for mod, p in [("numpy","numpy"),("pandas","pandas"),("sklearn","scikit-learn"),("scipy","scipy"),("matplotlib","matplotlib")]:
    try: importlib.import_module(mod)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install",p,"--default-timeout=600","--retries","10"])
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score, classification_report)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

BASE = r"C:\sar_project"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, BASE)
RES = os.path.join(BASE, "results"); os.makedirs(RES, exist_ok=True)
LOG = []
def say(s=""): print(s); LOG.append(s)

def family(c):
    """which radar quantity a feature comes from"""
    if c.startswith("sar_vh_minus_vv"): return "VH-VV"
    if c.startswith("sar_vh_div_vv"):   return "VH/VV"
    if c.startswith("sar_rvi"):         return "RVI"
    if c.startswith("sar_vh"):          return "VH"
    if c.startswith("sar_vv"):          return "VV"
    return "other"
FAMS = ["VV", "VH", "VH-VV", "VH/VV", "RVI"]

def cv_macro_f1(d, cols, folds, y, n_trees=200):
    pred = np.empty(len(d), dtype=object)
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        m = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=n_trees, class_weight="balanced", n_jobs=-1, random_state=0))
        pred[te] = m.fit(d.loc[tr, cols], y[tr]).predict(d.loc[te, cols])
    return f1_score(y, pred, average="macro")

def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from evaluation import evaluate_models
    from task2_3_sar import models
    mods = models()
    m = pd.read_csv(os.path.join(BASE, "data", "master_table.csv")); f = pd.read_csv(os.path.join(BASE, "data", "folds.csv"))
    ft = pd.read_csv(os.path.join(BASE, "features", "sar_features.csv"))
    cols = [c for c in ft.columns if c.startswith("sar_") and c != "sar_n_dates"]
    d = m[["field_id", "country", "label", "year"]].merge(f[["field_id", "fold"]], on="field_id").merge(ft, on="field_id").reset_index(drop=True)
    y = d.label.values; folds = d.fold.values; labs = sorted(set(y))
    say(f"SAR-ONLY ANALYSIS: {len(d)} fields, {len(cols)} SAR features, 5-fold spatial (10 km block) cross-validation from folds.csv")

    # ---- 1. models and best model (re-saves the prediction files) ----
    say("\n1) MODEL SCORES (all SAR-only)")
    res = evaluate_models(ft.merge(d[["field_id"]], on="field_id"), cols, mods, "sar")
    best = res.sort_values("macro_f1").iloc[-1].model
    say(f"best model by macro-F1: {best}")
    P = pd.read_csv(os.path.join(RES, f"pred_sar_{best}.csv")).merge(d[["field_id", "country", "year"]], on="field_id")
    yt, yp = P.label.values, P.pred.values

    # ---- 2. confusion matrix + per-crop precision / recall / F1 ----
    cm = confusion_matrix(yt, yp, labels=labs)
    say(f"\n2) CONFUSION MATRIX of {best} (rows = true crop, columns = predicted)")
    say(pd.DataFrame(cm, index=labs, columns=labs).to_string())
    pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=labs, zero_division=0)
    met = pd.DataFrame({"crop": labs, "precision": pr, "recall": rc, "f1": f1, "n_fields": sup})
    met.loc[len(met)] = ["macro avg", pr.mean(), rc.mean(), f1.mean(), sup.sum()]
    wp, wr, wf, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    met.loc[len(met)] = ["weighted avg", wp, wr, wf, sup.sum()]
    met.loc[len(met)] = ["accuracy", np.nan, np.nan, accuracy_score(yt, yp), sup.sum()]
    met.to_csv(os.path.join(RES, "sar_per_crop_metrics.csv"), index=False)
    say("\nPER-CROP PRECISION / RECALL / F1"); say(met.round(3).to_string(index=False))
    say("(precision = of the fields called this crop, how many really are; recall = of the real fields of this crop, how many were found)")

    # ---- 3. misclassification analysis ----
    say("\n3) MISCLASSIFICATIONS")
    for i, t in enumerate(labs):
        tot = cm[i].sum(); errs = [(labs[j], cm[i, j]) for j in range(len(labs)) if j != i and cm[i, j] > 0]
        say(f"true {t} ({tot} fields): {cm[i,i]} correct ({cm[i,i]/tot*100:.1f}%); wrong: " + ", ".join(f"{n} called {o} ({n/tot*100:.1f}%)" for o, n in errs))
    tot_err = int((yt != yp).sum()); ci, si = labs.index("corn"), labs.index("soybean")
    cs_err = int(cm[ci, si] + cm[si, ci])
    say(f"total errors {tot_err} of {len(yt)}; corn<->soybean confusions {cs_err} = {cs_err/max(tot_err,1)*100:.0f}% of all errors")
    say("error rate by country: " + ", ".join(f"{c}={(P.loc[P.country==c,'label']!=P.loc[P.country==c,'pred']).mean()*100:.1f}% (n={int((P.country==c).sum())})" for c in sorted(P.country.unique())))
    yr = P.assign(err=(P.label != P.pred)).groupby("year").err.agg(["mean", "size"])
    say("error rate by season year: " + ", ".join(f"{int(k)}={v['mean']*100:.0f}% (n={int(v['size'])})" for k, v in yr.iterrows()))
    for c in sorted(P.country.unique()):
        sub = P[P.country == c]; cmc = confusion_matrix(sub.label, sub.pred, labels=labs)
        say(f"{c} confusion (rows true, columns predicted):\n" + pd.DataFrame(cmc, index=labs, columns=labs).to_string())

    # ---- 4. what the radar features look like per crop (why the errors happen) ----
    say("\n4) SEASON-MEAN RADAR VALUES BY CROP (mean +/- sd over fields, dB except ratios)")
    key = {"VV": "sar_vv_mean", "VH": "sar_vh_mean", "VH-VV": "sar_vh_minus_vv_mean", "VH/VV": "sar_vh_div_vv_mean", "RVI": "sar_rvi_mean"}
    for fam, c in key.items():
        say(f"{fam:6s} " + "   ".join(f"{l}: {d.loc[d.label==l, c].mean():.3f} +/- {d.loc[d.label==l, c].std():.3f}" for l in labs))
    say("\nHOW WELL EACH QUANTITY SEPARATES TWO CROPS ALONE (AUC from the season-mean value; 0.5 = no separation, 1.0 = perfect)")
    pairs = [("corn", "soybean"), ("corn", "wheat"), ("soybean", "wheat")]
    for fam, c in key.items():
        out = []
        for a, b in pairs:
            xa, xb = d.loc[d.label == a, c].dropna(), d.loc[d.label == b, c].dropna()
            auc = stats.mannwhitneyu(xa, xb).statistic / (len(xa) * len(xb)); out.append(f"{a}-{b} {max(auc, 1-auc):.2f}")
        say(f"{fam:6s} " + "   ".join(out))

    # ---- 5. feature importance (top 10) ----
    say("\n5) FEATURE IMPORTANCE (Random Forest; permutation importance measured on held-out spatial folds = drop in macro-F1 when a feature is shuffled)")
    rows = []
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)).fit(d.loc[tr, cols], y[tr])
        pi = permutation_importance(rf, d.loc[te, cols], y[te], scoring="f1_macro", n_repeats=5, random_state=0)
        rows.append(pi.importances_mean)
    perm = pd.Series(np.mean(rows, axis=0), index=cols)
    rf_all = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=0)).fit(d[cols], y)
    imp = pd.Series(rf_all[-1].feature_importances_, index=cols)
    top = pd.DataFrame({"feature": cols, "family": [family(c) for c in cols], "permutation_drop_macroF1": perm.values, "rf_importance": imp.values}).sort_values("permutation_drop_macroF1", ascending=False)
    top.to_csv(os.path.join(RES, "sar_feature_importance_all.csv"), index=False); top.head(10).to_csv(os.path.join(RES, "sar_top10_features.csv"), index=False)
    say("TOP 10 SAR FEATURES"); say(top.head(10).round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4.8)); t10 = top.head(10).iloc[::-1]
    colors = {"VV": "#1565c0", "VH": "#2e7d32", "VH-VV": "#d4a017", "VH/VV": "#8e24aa", "RVI": "#c62828"}
    ax.barh(t10.feature, t10.permutation_drop_macroF1, color=[colors.get(x, "grey") for x in t10.family])
    ax.set_xlabel("drop in macro-F1 when the feature is shuffled (held-out folds)"); ax.set_title("Top 10 SAR features")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], labels=list(colors), title="radar quantity", loc="lower right")
    ax.grid(axis="x", alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(RES, "sar_feature_importance.png"), dpi=160); plt.close(fig)

    # ---- 6. contribution of VV, VH, VH-VV, VH/VV, RVI ----
    say("\n6) CONTRIBUTION OF VV, VH, VH-VV, VH/VV AND RVI")
    fam_of = pd.Series({c: family(c) for c in cols}); share = imp.groupby(fam_of).sum() / imp.sum()
    base = cv_macro_f1(d, cols, folds, y)
    say(f"all {len(cols)} features: macro-F1 {base:.3f} (Random Forest, 200 trees)")
    abl = []
    for fam in FAMS:
        mine = [c for c in cols if fam_of[c] == fam]; rest = [c for c in cols if fam_of[c] != fam]
        only, without = cv_macro_f1(d, mine, folds, y), cv_macro_f1(d, rest, folds, y)
        abl.append(dict(family=fam, n_features=len(mine), importance_share=float(share.get(fam, 0)), only_macroF1=only, without_macroF1=without, drop_when_removed=base - without))
        print("  done", fam)
    abl = pd.DataFrame(abl); abl.to_csv(os.path.join(RES, "sar_family_contribution.csv"), index=False)
    for _, r in abl.iterrows():
        say(f"{r.family:6s} {int(r.n_features):2d} features | share of Random Forest importance {r.importance_share*100:4.1f}% | alone macro-F1 {r.only_macroF1:.3f} | without it macro-F1 {r.without_macroF1:.3f} (change {-r.drop_when_removed:+.3f})")
    say("Note: the five families overlap (VH-VV, VH/VV and RVI are all built from VV and VH), so removing one changes little when the others carry the same information.")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(abl.family, abl.importance_share * 100, color=[colors[x] for x in abl.family]); ax[0].set_ylabel("% of Random Forest importance"); ax[0].set_title("Share of importance")
    w = 0.38; x = np.arange(len(abl))
    ax[1].bar(x - w/2, abl.only_macroF1, w, label="family alone", color="#90a4ae"); ax[1].bar(x + w/2, abl.without_macroF1, w, label="all features except family", color="#455a64")
    ax[1].axhline(base, color="red", ls="--", label=f"all features ({base:.3f})"); ax[1].set_xticks(x); ax[1].set_xticklabels(abl.family); ax[1].set_ylim(0, 1); ax[1].set_title("macro-F1"); ax[1].legend(fontsize=8)
    for a in ax: a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "sar_family_contribution.png"), dpi=160); plt.close(fig)

    # ---- figures: confusion matrix and season profile ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2)); cmn = cm / cm.sum(axis=1, keepdims=True)
    for a, M, tt in ((ax[0], cm, "counts"), (ax[1], cmn, "share of each true crop")):
        a.imshow(cmn, vmin=0, vmax=1, cmap="Blues"); a.set_xticks(range(len(labs))); a.set_xticklabels(labs); a.set_yticks(range(len(labs))); a.set_yticklabels(labs)
        a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"SAR-only {best}: {tt}")
        for i in range(len(labs)):
            for j in range(len(labs)): a.text(j, i, (f"{M[i,j]}" if M is cm else f"{M[i,j]:.0%}"), ha="center", va="center", color="white" if cmn[i, j] > .5 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "sar_confusion_matrix.png"), dpi=160); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharex=True); xs = (np.arange(6) + .5) / 6 * 100; cc = {"corn": "#d4a017", "soybean": "#2e7d32", "wheat": "#1565c0"}
    for a, pol in zip(ax, ("vv", "vh")):
        for l in labs:
            B = d.loc[d.label == l, [f"sar_{pol}_bin{j}" for j in range(1, 7)]]; mu, sd = B.mean().values, B.std().values
            a.plot(xs, mu, label=l, color=cc.get(l)); a.fill_between(xs, mu - sd, mu + sd, alpha=.12, color=cc.get(l))
        a.set_title(f"{pol.upper()} (dB) through the season"); a.set_xlabel("% of season (sowing to harvest)"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "sar_season_profile.png"), dpi=160); plt.close(fig)

    open(os.path.join(RES, "sar_analysis.txt"), "w", encoding="utf-8").write("\n".join(LOG)); print("\nsaved in", RES)

if __name__ == "__main__":
    main()
