# Prints the final numbers for the report and saves them to C:\sar_project\summary.txt
import os, pandas as pd
B = r"C:\sar_project"
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def country(s): return s.str.split("_").str[0]

d = pd.read_csv(os.path.join(B, "1_sar_downloaded", "download_log.csv"))
d = d.drop_duplicates("field", keep="last")          # latest status per field
d["country"] = country(d["field"])
p("=== DOWNLOAD ===")
p("fields in inventory: 1302")
p(pd.crosstab(d["country"], d["status"]).to_string())
ok = d[d.status == "ok"]
p("files downloaded:", len(ok), "| MB min/median/max:", ok.MB.min(), ok.MB.median(), ok.MB.max())
p("files within 5-10 MB:", int(ok.MB.between(5, 10).sum()), "of", len(ok))
p("S1 dates per field min/median/max:", ok.n_dates.min(), ok.n_dates.median(), ok.n_dates.max())

c = pd.read_csv(os.path.join(B, "2_coregistered", "coregistration_log.csv"))
p("\n=== CO-REGISTRATION ===")
p("fields co-registered:", c.file.nunique(), "| dates measured:", len(c))
a = c[c.applied == True]
p("dates corrected (|shift| <= 5 px):", len(a), f"({100*len(a)/len(c):.1f}%)")
p("dates left unchanged (unreliable, >5 px):", len(c) - len(a))
p("corrected shifts in metres: median", a.shift_m.median(), "| 95th pct", a.shift_m.quantile(.95), "| max", a.shift_m.max())

v = pd.read_csv(os.path.join(B, "3_verification", "verification_report.csv"))
v["country"] = country(v["field"])
p("\n=== VERIFICATION AGAINST RAW JSON CENTROIDS ===")
p(pd.crosstab(v["country"], v["result"]).to_string())
p("raw centroid inside SAR tile:", int(v.centroid_inside_raster.sum()), "of", len(v))
p("centroid offset from tile centre (m): median", v.centroid_offset_from_tile_centre_m.median(), "| max", v.centroid_offset_from_tile_centre_m.max())
p("valid pixel fraction median:", v.valid_pixel_fraction.median())
p("mean VV dB (median over fields):", v.tile_mean_VV_dB.median(), "| mean VH dB:", v.tile_mean_VH_dB.median())
f = v[v.result == "FAIL"]
if len(f):
    p("\nFAIL reasons:"); p(f.reasons.str.split("; ").explode().value_counts().to_string())
    p("first FAIL fields:", ", ".join(f.field.head(5)))
jp = os.path.join(B, "3_verification", "json_vs_sar_check.csv")
if os.path.exists(jp):
    j = pd.read_csv(jp); p("\nraw JSON direct check:", j.result.value_counts().to_dict())
open(os.path.join(B, "summary.txt"), "w").write("\n".join(out))
print("\nsaved:", os.path.join(B, "summary.txt"))