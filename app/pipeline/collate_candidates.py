"""
Collate STL meshes for formed-plate candidate solids into a viewable gallery.

Selects the rows flagged as "filed as plate but not flat" (the formed-plate /
misfiled-section suspects), dedupes by geometry, meshes one representative solid
per distinct geometry, and writes:

    /app/outputs/stl/_formed_candidates/<file>.stl
    /app/outputs/stl/_formed_candidates/manifest.csv
    /app/outputs/stl/_formed_candidates/gallery.html

Because the folder lives under the app's ``/outputs/stl/`` static mount, open:

    http://localhost:8000/outputs/stl/_formed_candidates/gallery.html

The gallery is a single Three.js viewer with a clickable list (one shared WebGL
context — avoids the per-card context limit), showing each candidate's measured
features so they can be judged and labelled.

Run inside the container::

    docker exec cad-automation-api python -m app.pipeline.collate_candidates
"""
from __future__ import annotations

import csv
import faulthandler
import json
import re
import sys
from pathlib import Path

faulthandler.enable()

try:
    import structlog
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
except Exception:
    pass
import logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
from app.config import settings  # noqa: E402

_HEX_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_")
_SAFE_RE = re.compile(r"[^\w\-]+")


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _safe(s: str, n: int = 48) -> str:
    return (_SAFE_RE.sub("_", str(s)).strip("_") or "x")[:n]


def _resolve_step(sidecar_stem: str):
    upd = Path(settings.UPLOAD_DIR)
    for ext in (".step", ".stp", ".STEP", ".STP"):
        c = upd / f"{sidecar_stem}{ext}"
        if c.exists():
            return c
    return None


def _round_robin_by_job(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """Interleave rows across jobs so coverage is spread, not dominated by one job."""
    if n <= 0 or d.empty:
        return d.iloc[0:0]
    groups = {job: list(idx) for job, idx in d.groupby("job").groups.items()}
    order = []
    while len(order) < n and any(groups.values()):
        for job in list(groups):
            if groups[job]:
                order.append(groups[job].pop(0))
                if len(order) >= n:
                    break
    return d.loc[order]


def _load_labelled(df: pd.DataFrame):
    """Return (labelled (job,ref,sidx) keys, labelled fingerprint_keys) from
    verified.csv — so the gallery can skip what's already been classified."""
    keys = set()
    path = Path(__file__).parent / "data" / "labels" / "verified.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].lstrip().startswith("#") or row[0] == "job":
                    continue
                keys.add((row[0], row[1], str(row[2])))
    dd = df.copy()
    dd["key"] = list(zip(dd["job"], dd["ref_id"], dd["solid_index"].astype(str)))
    fps = set(dd[dd["key"].isin(keys)]["fingerprint_key"])
    return keys, fps


def _select_candidates(df: pd.DataFrame, max_n: int = 320,
                       fill_gate: float = 0.65) -> pd.DataFrame:
    """Stratified set of UNLABELLED distinct geometries for labelling.

    One representative per distinct geometry (fingerprint), spread across jobs,
    EXCLUDING anything already classified in verified.csv (by exact part and by
    shape — labelling one instance covers all copies).  Prioritises the
    classification-ambiguous zone (thin-walled) with a smaller flat-plate sample.
    """
    labelled_keys, labelled_fps = _load_labelled(df)

    d = df.copy()
    d["fill"] = pd.to_numeric(d["fill_ratio"], errors="coerce")
    d = d[d["features_ok"].fillna(False).astype(bool) & d["fill"].notna()]
    # Drop merged/unseparated STEP exports: jobs that parsed to <=2 distinct
    # refs are a single whole-assembly lump (no part structure).
    ref_per_job = df.groupby("job")["ref_id"].nunique()
    degenerate = set(ref_per_job[ref_per_job <= 2].index)
    d = d[~d["job"].isin(degenerate)]
    # Skip already-labelled parts and any other instance of a labelled shape.
    d["key"] = list(zip(d["job"], d["ref_id"], d["solid_index"].astype(str)))
    d = d[~d["key"].isin(labelled_keys) & ~d["fingerprint_key"].isin(labelled_fps)]
    _log(f"excluding {len(labelled_keys)} labelled parts "
         f"({len(labelled_fps)} shapes) — showing only unlabelled")
    # one representative per distinct geometry
    d = d.sort_values("fill").drop_duplicates("fingerprint_key", keep="first")

    thin = d[d["fill"] < fill_gate]          # the confusion zone
    flat = d[d["fill"] >= fill_gate]         # clearer plates (for balance)
    n_thin = int(max_n * 0.8)
    pick_thin = _round_robin_by_job(thin, n_thin)
    pick_flat = _round_robin_by_job(flat, max_n - len(pick_thin))
    out = pd.concat([pick_thin, pick_flat])
    return out


def _sidecar_stem_for_job(job: str, analysis_dir: Path):
    """Find a sidecar whose job name matches and whose STEP exists."""
    for p in sorted(analysis_dir.glob("*.json")):
        if _HEX_PREFIX_RE.sub("", p.stem) == job and _resolve_step(p.stem):
            return p.stem
    return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Collate labelling gallery.")
    ap.add_argument("--max", type=int, default=320, help="max geometries to mesh")
    ap.add_argument("--fill-gate", type=float, default=0.65,
                    help="fill_ratio below this = thin-walled (ambiguous) zone")
    ap.add_argument("--gallery-only", action="store_true",
                    help="rebuild gallery.html from existing manifest.csv (no meshing)")
    args = ap.parse_args()

    out_dir = Path(settings.STL_OUTPUT_DIR) / "_formed_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_csv = Path(settings.OUTPUT_DIR) / "ml" / "dataset.csv"
    analysis_dir = Path(settings.ANALYSIS_OUTPUT_DIR)

    if args.gallery_only:
        man = out_dir / "manifest.csv"
        with man.open(newline="", encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
        for r in manifest:  # csv reads all as str; coerce numerics for the JS
            for k in ("t_eff", "t_eff_thin_ratio", "fill_ratio", "dim_long",
                      "dim_mid", "dim_thin", "n_holes", "thk_max_over_teff",
                      "n_convex_bends", "developed_ratio", "solid_index"):
                if r.get(k) not in (None, ""):
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
        (out_dir / "gallery.html").write_text(
            _GALLERY.replace("/*__ITEMS__*/", json.dumps(manifest)), encoding="utf-8")
        _log(f"Rebuilt gallery.html from {len(manifest)} manifest rows")
        _log("Open: http://localhost:8000/outputs/stl/_formed_candidates/gallery.html")
        return

    df = pd.read_csv(data_csv)
    cand = _select_candidates(df, max_n=args.max, fill_gate=args.fill_gate)
    _log(f"{len(cand)} candidate geometries selected "
         f"(across {cand['job'].nunique()} jobs)")

    from app.services.cnc_shape_analyser import _read_xcaf, _get_shape, _iter_solids
    from app.services.stl_generator import STLGenerator
    from app.pipeline.feature_extract import extract_solid_features

    # group by job → resolve STEP → open once
    manifest = []
    by_job = {}
    for _, row in cand.iterrows():
        by_job.setdefault(row["job"], []).append(row)

    for job, rows in by_job.items():
        stem = _sidecar_stem_for_job(job, analysis_dir)
        step = _resolve_step(stem) if stem else None
        if step is None:
            _log(f"SKIP job {job}: no STEP upload")
            continue
        _log(f"--- {job}: {len(rows)} candidates (STEP {step.name})")
        doc, _st = _read_xcaf(str(step))
        gen = STLGenerator(str(step), str(out_dir))
        for row in rows:
            ref_id = row["ref_id"]
            sidx = int(row["solid_index"])
            try:
                shape = _get_shape(doc, ref_id)
                solids = list(_iter_solids(shape))
                solid = solids[sidx] if 0 <= sidx < len(solids) else shape
            except Exception as e:
                _log(f"  {ref_id}: shape error {e}")
                continue
            fname = f"{_safe(job,24)}__{_safe(row['part_name'],28)}__{_safe(ref_id)}_{sidx}.stl"
            if (out_dir / fname).exists():
                _log(f"  skip mesh (exists) {fname}")
            else:
                try:
                    gen._generate_stl(solid, out_dir / fname)
                except Exception as e:
                    _log(f"  {ref_id}: mesh failed {e}")
                    continue
            # Compute section features inline (holes/thickness) — the solid is
            # already open, so this is far cheaper than a full-dataset --section
            # export and gives the gallery the discriminators while labelling.
            try:
                f = extract_solid_features(solid, section=True)
            except Exception as e:
                _log(f"  {ref_id}: feature calc failed {e}")
                f = {}
            manifest.append({
                "file": fname,
                "job": job,
                "part_name": row["part_name"],
                "ref_id": ref_id,
                "solid_index": sidx,
                "t_eff": f.get("t_eff", row.get("t_eff")),
                "t_eff_thin_ratio": f.get("t_eff_thin_ratio", row.get("t_eff_thin_ratio")),
                "fill_ratio": f.get("fill_ratio", row.get("fill_ratio")),
                "dim_long": f.get("dim_long", row.get("dim_long")),
                "dim_mid": f.get("dim_mid", row.get("dim_mid")),
                "dim_thin": f.get("dim_thin", row.get("dim_thin")),
                "rule_type": row.get("rule_type"),
                "n_holes": f.get("n_holes"),
                "thk_max_over_teff": f.get("thk_max_over_teff"),
                "n_convex_bends": f.get("n_convex_bends"),
                "developed_ratio": f.get("developed_ratio", row.get("developed_ratio")),
            })
            _log(f"  ok {fname}")

    # manifest.csv
    if manifest:
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)
    # gallery.html
    (out_dir / "gallery.html").write_text(_GALLERY.replace(
        "/*__ITEMS__*/", json.dumps(manifest)), encoding="utf-8")

    _log(f"\nDONE: {len(manifest)} STLs -> {out_dir}")
    _log("Open: http://localhost:8000/outputs/stl/_formed_candidates/gallery.html")


_GALLERY = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Formed-plate candidates</title>
<style>
  body{margin:0;font-family:system-ui,sans-serif;display:flex;height:100vh;background:#1a1a1a;color:#ddd}
  #list{width:430px;overflow:auto;border-right:1px solid #333;flex:none}
  #bar{position:sticky;top:0;background:#161616;padding:8px 10px;border-bottom:1px solid #333;z-index:2}
  #bar h2{font-size:14px;margin:0 0 6px;color:#9cf}
  #bar button{font-size:12px;padding:4px 8px;margin-right:6px;background:#2d3a4a;color:#cde;border:1px solid #456;border-radius:4px;cursor:pointer}
  #bar button:hover{background:#37485c}
  #count{font-size:12px;color:#9a9}
  .it{padding:8px 10px;border-bottom:1px solid #2a2a2a;font-size:12px;border-left:5px solid transparent}
  .it.sel{background:#23303f}
  .it b{color:#fff;cursor:pointer}.it small{color:#8a8a8a;display:block;margin:2px 0 5px}
  .it.L-section{border-left-color:#e6a13c}.it.L-plate{border-left-color:#4caf50}
  .it.L-formed{border-left-color:#e3496b}.it.L-bought{border-left-color:#8a7bd8}
  .it.L-bent{border-left-color:#46c2c2}.it.L-excl{border-left-color:#777}
  .btns label{display:inline-block;font-size:11px;padding:2px 6px;margin:1px 3px 1px 0;border:1px solid #444;border-radius:4px;cursor:pointer;color:#bbb}
  .btns input{display:none}
  .btns input:checked+span{font-weight:bold}
  .btns label:has(input:checked){background:#37485c;border-color:#69c;color:#fff}
  #view{flex:1;position:relative}#cv{width:100%;height:100%;display:block}
  #cap{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.6);padding:8px 12px;border-radius:6px;font-size:13px;max-width:60%}
  .tag{display:inline-block;background:#333;border-radius:4px;padding:1px 6px;margin-right:6px;font-size:11px}
  #modal{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;align-items:center;justify-content:center;z-index:9}
  #modal div{background:#222;padding:16px;border-radius:8px;width:70%;max-width:760px}
  #modal textarea{width:100%;height:300px;background:#111;color:#cde;font-family:monospace;font-size:12px;border:1px solid #444}
  #hint{position:absolute;bottom:10px;left:10px;font-size:11px;color:#888;background:rgba(0,0,0,.5);padding:4px 8px;border-radius:4px}
</style>
<script type="importmap">{"imports":{
 "three":"https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js",
 "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/"}}</script>
</head><body>
<div id="list">
  <div id="bar"><h2>Classify candidates</h2>
    <button id="exp">Export CSV</button><button id="clr">Clear all</button>
    <span id="count"></span><br><span id="acc" style="font-size:12px"></span>
  </div>
</div>
<div id="view"><canvas id="cv"></canvas><div id="cap">Select an item &rarr;</div>
  <div id="hint">keys: 1=Section 2=Plate 3=Formed 4=Bent 5=Bought-out 6=Excluded &middot; drag to rotate</div></div>
<div id="modal"><div>
  <p>Copy this and paste it back to Claude (or Save As <code>verified.csv</code>):</p>
  <textarea id="csv" readonly></textarea>
  <p><button id="dl">Download</button> <button id="close">Close</button></p>
</div></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
const ITEMS=/*__ITEMS__*/;
const CLASSES=[['section','Section'],['plate','Plate'],['formed','Formed plate'],['bent','Bent section'],['bought','Bought-out'],['excl','Excluded']];
const CAT={section:'SECTION',plate:'PLATE',formed:'FORMED_PLATE',bent:'BENT_SECTION',bought:'BOUGHT_OUT',excl:'EXCLUDE'};
// Preview of the co-designed rule decision tree, applied to the features already
// in the gallery. Cannot yet predict BENT_SECTION / BOUGHT_OUT / EXCLUDE
// (no feature) — those will count as misses, which is the point (shows gaps).
function predict(it){
  const num=v=>(v===''||v==null)?null:parseFloat(v);
  // Known catalogue products are confidently bought-out (e.g. Unistrut), checked
  // before geometry. Keep in sync with app/pipeline/catalogue_products.py.
  if(it.part_name){const nm=String(it.part_name).toLowerCase();
    if(nm.includes('unist') && !nm.includes('plate')) return 'BOUGHT_OUT';}
  const holes=num(it.n_holes), thk=num(it.thk_max_over_teff), tthin=num(it.t_eff_thin_ratio), nb=num(it.n_convex_bends);
  if(nb!=null && nb>=5) return 'BENT_SECTION';            // curved tube => many bend faces
  if(holes!=null && holes>=1) return 'SECTION';          // hollow box (RHS/SHS/CHS)
  if(tthin!=null && tthin>=0.45) return 'PLATE';          // flat-ish plate (gauge ~ thinnest dim)
  if(nb!=null && nb>=1) return 'FORMED_PLATE';            // convex bend (R>=gauge) => formed
  if(it.rule_type==='section') return 'SECTION';          // matched a standard section in the library
  if(thk!=null && thk>=1.5) return 'SECTION';             // open profile w/ distinct flanges (I/UC/PFC)
  return 'FORMED_PLATE';                                  // thin uniform open wall
}
const KEY='fp_labels_v1';
let labels=JSON.parse(localStorage.getItem(KEY)||'{}');
function save(){localStorage.setItem(KEY,JSON.stringify(labels));refreshCount();}
function refreshCount(){
  document.getElementById('count').textContent=Object.keys(labels).length+' / '+ITEMS.length+' labelled';
  // live rule-preview accuracy: predicted vs your label, over labelled items
  let n=0,ok=0;
  ITEMS.forEach(it=>{const c=labels[it.file];if(!c)return;const p=predict(it);if(p==null)return;
    n++; if(p===CAT[c])ok++;});
  const a=document.getElementById('acc');
  a.innerHTML = n? `rules match your label: <b>${ok}/${n} = ${(100*ok/n).toFixed(0)}%</b> <span style="color:#888">(preview — no bent/BO/excl yet)</span>` : '';
}

const cv=document.getElementById('cv'), view=document.getElementById('view');
const r=new THREE.WebGLRenderer({canvas:cv,antialias:true});r.setPixelRatio(devicePixelRatio);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x1a1a1a);
const cam=new THREE.PerspectiveCamera(45,1,0.1,1e6);
const ctr=new OrbitControls(cam,cv);ctr.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x444444,1.2));
const dl=new THREE.DirectionalLight(0xffffff,1.0);dl.position.set(1,1,1);scene.add(dl);
let mesh=null,curIdx=-1; const loader=new STLLoader();
function size(){const w=view.clientWidth,h=view.clientHeight;r.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();}
addEventListener('resize',size);size();
function load(i){
  const it=ITEMS[i];curIdx=i;
  document.querySelectorAll('.it').forEach(e=>e.classList.remove('sel'));
  const el=document.getElementById('it'+i);if(el){el.classList.add('sel');el.scrollIntoView({block:'nearest'});}
  if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();mesh=null;}
  loader.load(it.file,g=>{
    g.computeVertexNormals();g.computeBoundingBox();
    const c=new THREE.Vector3();g.boundingBox.getCenter(c);g.translate(-c.x,-c.y,-c.z);
    const s=g.boundingBox.getSize(new THREE.Vector3());const d=Math.max(s.x,s.y,s.z)||1;
    mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0x8fb3d9,metalness:.1,roughness:.7,side:THREE.DoubleSide}));
    scene.add(mesh);
    cam.position.set(d*1.4,d*1.1,d*1.6);cam.near=d/100;cam.far=d*100;cam.updateProjectionMatrix();
    ctr.target.set(0,0,0);ctr.update();
    document.getElementById('cap').innerHTML=
      `<b>${it.part_name}</b> &mdash; ${it.job}<br>`+
      `<span class=tag>t_eff ${(+it.t_eff).toFixed(1)}mm</span>`+
      `<span class=tag>t/thin ${(+it.t_eff_thin_ratio).toFixed(2)}</span>`+
      `<span class=tag>fill ${(+it.fill_ratio).toFixed(2)}</span>`+
      (it.n_holes!=null&&it.n_holes!==''?`<span class=tag>holes ${it.n_holes}</span>`:'')+
      (it.thk_max_over_teff!=null&&it.thk_max_over_teff!==''?`<span class=tag>thk/teff ${(+it.thk_max_over_teff).toFixed(2)}</span>`:'')+
      `<span class=tag>${(+it.dim_long).toFixed(0)}&times;${(+it.dim_mid).toFixed(0)}&times;${(+it.dim_thin).toFixed(0)}</span>`+
      (predict(it)?`<br><span class=tag style="background:#3a3a5a">rules predict: <b>${predict(it)}</b></span>`:'');
  },undefined,()=>{document.getElementById('cap').textContent='load error: '+it.file;});
}
function setLabel(i,cls){
  const it=ITEMS[i];
  if(cls)labels[it.file]=cls; else delete labels[it.file];
  const el=document.getElementById('it'+i);
  el.className='it'+(i===curIdx?' sel':'')+(cls?' L-'+cls:'');
  const inp=el.querySelector(`input[value="${cls}"]`);if(inp)inp.checked=true;
  const p=predict(it),mk=document.getElementById('m'+i);
  if(mk) mk.innerHTML = (cls&&p)? (p===CAT[cls]?'<span style="color:#5c5">&#10003;</span>':'<span style="color:#e66">&#10007;</span>') : '';
  save();
}
const list=document.getElementById('list');
ITEMS.forEach((it,i)=>{
  const cur=labels[it.file];
  const el=document.createElement('div');el.id='it'+i;
  el.className='it'+(cur?' L-'+cur:'');
  const radios=CLASSES.map(([v,lbl])=>
    `<label><input type=radio name=c${i} value="${v}" ${cur===v?'checked':''}><span>${lbl}</span></label>`).join('');
  el.innerHTML=`<b id=b${i}>${it.part_name}</b> <span id=m${i}></span><small>${it.job} &middot; pred ${predict(it)||'-'} &middot; fill ${(+it.fill_ratio).toFixed(2)}</small><div class=btns>${radios}</div>`;
  list.appendChild(el);
  if(cur){const p=predict(it),mk=el.querySelector('#m'+i);
    if(mk&&p) mk.innerHTML=(p===CAT[cur]?'<span style="color:#5c5">&#10003;</span>':'<span style="color:#e66">&#10007;</span>');}
  el.querySelector('#b'+i).onclick=()=>load(i);
  el.querySelectorAll('input').forEach(inp=>inp.onchange=()=>{load(i);setLabel(i,inp.value);});
});
addEventListener('keydown',e=>{
  if(curIdx<0)return;const m={'1':'section','2':'plate','3':'formed','4':'bent','5':'bought','6':'excl'};
  if(m[e.key]){setLabel(curIdx,m[e.key]);if(curIdx<ITEMS.length-1)load(curIdx+1);}
});
function buildCSV(){
  let out='job,ref_id,solid_index,category,designation,issue,note\n';
  ITEMS.forEach(it=>{const c=labels[it.file];if(!c)return;
    const note=(c==='formed')?'gallery: formed plate':(c==='bent')?'gallery: bent section':(c==='excl')?'gallery: artifact/excluded':'gallery';
    const issue=(c==='formed'||c==='bent')?'1':'';
    out+=[JSON.stringify(it.job),it.ref_id,it.solid_index,CAT[c],'',issue,note].join(',')+'\n';});
  return out;
}
document.getElementById('exp').onclick=()=>{document.getElementById('csv').value=buildCSV();
  document.getElementById('modal').style.display='flex';
  try{navigator.clipboard.writeText(buildCSV());}catch(e){}};
document.getElementById('dl').onclick=()=>{const b=new Blob([buildCSV()],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='verified_labels.csv';a.click();};
document.getElementById('close').onclick=()=>document.getElementById('modal').style.display='none';
document.getElementById('clr').onclick=()=>{if(confirm('Clear all labels?')){labels={};save();location.reload();}};
refreshCount();
(function loop(){requestAnimationFrame(loop);ctr.update();r.render(scene,cam);})();
if(ITEMS.length)load(0);
</script></body></html>"""


if __name__ == "__main__":
    main()
