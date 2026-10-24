ZIP_PATH = r"C:\Users\User\Downloads\Raw.zip"  # change this (see Step 2)

import zipfile, json, csv, collections

COUNTRIES = ["argentina", "brazil", "germany", "uruguay"]
MAX_BYTES = 5_000_000

def flatten(o, prefix, out):
    if isinstance(o, dict):
        for k, v in o.items():
            flatten(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(o, list):
        if len(o) <= 3 and all(not isinstance(x, (dict, list)) for x in o):
            out[prefix] = str(o)
    else:
        out[prefix] = o

records, counts = [], collections.Counter()
with zipfile.ZipFile(ZIP_PATH) as z:
    entries = [i for i in z.infolist() if i.filename.lower().endswith(".json") and not i.is_dir()]
    print("JSON files in zip:", len(entries))
    for i in entries:
        name = i.filename.replace("\\", "/")
        low = name.lower()
        country = next((c for c in COUNTRIES if c in low), "other")
        counts[country] += 1
        if i.file_size > MAX_BYTES:
            continue
        try:
            d = json.loads(z.read(i).decode("utf-8"))
        except Exception:
            continue
        flat = {"path": name, "country_folder": country, "size": i.file_size}
        flatten(d, "", flat)
        records.append(flat)

print("JSON per country folder:", dict(counts))
keycount = collections.Counter(k for r in records for k in r)
print("Most common keys:")
for k, n in keycount.most_common(40):
    print("  ", n, k)
print("Keys that look date-related:",
      [k for k in keycount if any(s in k.lower() for s in ["sow", "harvest", "plant", "date", "season"])])

cols = ["path", "country_folder", "size"] + [k for k, _ in keycount.most_common(60) if k not in ("path", "country_folder", "size")]
with open("inventory_yieldsat.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(records)
print("Wrote inventory_yieldsat.csv with", len(records), "rows")
if records:
    print("EXAMPLE:", json.dumps(records[0], default=str)[:1500])