import json, os, subprocess, sys, csv
sys.path.insert(0, "/app/tools")
import numpy as np
import mesh_to_model as M2M
from survey_parts import write_stl

picks = [p.replace("\\", "/") for p in json.load(open("outputs/ar_models/_weldments.json"))]
seen, uniq = set(), []
for p in picks:
    b = os.path.basename(p)
    if b not in seen:
        seen.add(b); uniq.append(p)
os.makedirs("outputs/ar_models/_wsx", exist_ok=True)
rows = []
for k, src in enumerate(uniq[:8]):
    base = "w%02d" % k
    try:
        tris = M2M.load_stl(src)
        pts = tris.reshape(-1, 3)
        longest = float((pts.max(0) - pts.min(0)).max())
        if longest < 1e-6: continue
        stl = "outputs/ar_models/_wsx/%s.stl" % base
        js  = "outputs/ar_models/_wsx/%s.json" % base
        write_stl((tris - pts.mean(0)) * (432.0 / longest), stl)
        m = M2M.build(stl, angle_deg=25.0, name=base)
        json.dump(m, open(js, "w"))
    except Exception as e:
        print("  %s skip (%s)" % (base, str(e)[:70])); continue
    cmd = [sys.executable, "tools/stereo_orientation_sweep.py", "--mesh", stl,
           "--fit", "outputs/ar_fits/turn90", "--rig", "outputs/ar_captures/turn90",
           "--yaw", "0:120:60", "--baseline", "60"]
    pr = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in pr.stdout.splitlines() if "chosen correctly" in l]
    res = line[0].split("===")[1].strip() if line else "no result"
    print("%-5s %-46s %s" % (base, os.path.basename(src)[:46], res))
    rows.append({"part": base, "source": os.path.basename(src), "edges": m["summary"]["edges"],
                 "result": res})
if rows:
    with open("outputs/ar_fits/stereo_multi.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
