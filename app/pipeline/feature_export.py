"""
ML feature/label dataset exporter.

Walks every analysis sidecar in ``ANALYSIS_OUTPUT_DIR``, re-opens the matching
STEP upload, and emits **one row per solid** to ``outputs/ml/dataset.csv``
(and ``.parquet`` if pyarrow is available).

Each row pairs the rule-independent measured feature vector
(:func:`app.pipeline.feature_extract.extract_solid_features`) with a *hybrid*
label assembled from three sources, in priority order:

    verified  >  name  >  rule

  - **verified** — manual ground truth from ``data/labels/verified.csv``
  - **name**     — Tekla designation parsed from the part name
                   (:func:`app.pipeline.classification.classify_by_name`)
  - **rule**     — the existing classifier's verdict from ``cnc_analysis``

Rule columns are kept *separately* (``rule_*``) so the trainer can compute an
agreement matrix between ML, names, and rules.  A ``fingerprint_key`` column lets
the trainer split by geometry (no identical solid in both train and test), and
``job`` lets it split leave-one-job-out.

Run inside the container::

    docker exec cad-automation-api python -m app.pipeline.feature_export --all

Follows the OCC worker pattern: structlog/logging to stderr, project root on
sys.path, geometry work in a 64 MB-stack thread.
"""
from __future__ import annotations

import csv
import faulthandler
import gc
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

faulthandler.enable()

try:  # keep diagnostic logging off stdout
    import structlog
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
except Exception:
    pass

import logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.pipeline.feature_extract import FEATURE_KEYS, extract_solid_features  # noqa: E402
from app.pipeline.classification import classify_by_name  # noqa: E402

_HEX_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_")
_SECTION_LIBRARY = Path(__file__).parent / "data" / "Shape_classifier_info.json"
_VERIFIED_CSV = Path(__file__).parent / "data" / "labels" / "verified.csv"

# Identity + provenance columns written before the feature columns.
_ID_COLS = [
    "job", "sidecar", "ref_id", "solid_index", "part_name", "node_type",
    "n_solids", "instance_count", "fingerprint_key",
]
_LABEL_COLS = [
    "label_source", "y_type", "y_category", "y_designation",
    "name_category", "name_designation",
    "rule_type", "rule_category", "rule_designation", "rule_profile_type",
    "rule_match_score", "rule_confidence",
    "verified_issue", "verified_note",
    # human CNC/Bought-out/Exclude routing label from this sidecar's own
    # project_state.classifications (aligned to the same XCAF parse → ref_ids match)
    "cnc_class",
]


def _job_name(sidecar_stem: str) -> str:
    """Strip the 8-hex upload prefix to get the human job name."""
    return _HEX_PREFIX_RE.sub("", sidecar_stem)


def _resolve_step(sidecar_stem: str) -> Optional[Path]:
    """Find the STEP upload matching a sidecar stem, trying common extensions."""
    upload_dir = Path(settings.UPLOAD_DIR)
    for ext in (".step", ".stp", ".STEP", ".STP"):
        cand = upload_dir / f"{sidecar_stem}{ext}"
        if cand.exists():
            return cand
    return None


def _load_verified() -> Dict[tuple, Dict[str, str]]:
    """Load manual overrides keyed by (job, ref_id, solid_index)."""
    out: Dict[tuple, Dict[str, str]] = {}
    if not _VERIFIED_CSV.exists():
        return out
    with _VERIFIED_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            rec = {header[i]: (row[i] if i < len(row) else "") for i in range(len(header))}
            key = (rec.get("job", "").strip(),
                   rec.get("ref_id", "").strip(),
                   str(rec.get("solid_index", "0")).strip())
            out[key] = rec
    return out


def _collect_leaf_parts(nodes: list, parent_name: Optional[str],
                        out: Dict[str, Dict]) -> None:
    """Walk the assembly tree, collecting leaf parts keyed by ref_id."""
    for node in nodes:
        if node.get("node_type") == "assembly":
            _collect_leaf_parts(node.get("children", []), node.get("name"), out)
        else:
            ref_id = node.get("ref_id") or node.get("id")
            if not ref_id:
                continue
            rec = out.setdefault(ref_id, {
                "name": node.get("name", ""),
                "node_type": node.get("node_type", ""),
                "parent_name": parent_name,
                "instance_count": 0,
            })
            rec["instance_count"] += 1


def _cnc_class_by_ref(data: dict) -> Dict[str, str]:
    """Map ref_id -> human routing class (postprocess/bought-out/exclude) from
    this sidecar's own project_state.classifications.

    Classifications are keyed by node_id (instances, or ``ref:sN`` solid nodes);
    we resolve each to its ref_id via the tree and take the majority class.
    Because this reads the SAME sidecar the features come from, ref_ids align.
    """
    cls = (data.get("project_state") or {}).get("classifications") or {}
    if not cls:
        return {}
    id2ref: Dict[str, str] = {}

    def walk(nodes):
        for n in nodes:
            rid = n.get("ref_id") or n.get("id")
            if n.get("id"):
                id2ref[n["id"]] = rid
            walk(n.get("children", []))

    walk((data.get("analysis") or {}).get("assembly_tree", []))
    by_ref: Dict[str, list] = {}
    for node_id, c in cls.items():
        base = node_id.split(":s")[0]  # strip synthetic solid suffix
        rid = id2ref.get(node_id) or id2ref.get(base) or base
        by_ref.setdefault(rid, []).append(c)
    return {rid: max(set(cs), key=cs.count) for rid, cs in by_ref.items()}


def _instance_count_from_consolidation(consolidation: dict) -> Dict[str, int]:
    """Map ref_id -> total instance count from consolidation part_groups."""
    counts: Dict[str, int] = {}
    for grp in (consolidation or {}).get("part_groups", []) or []:
        total = grp.get("total_count")
        for rid in grp.get("ref_ids", []):
            if total is not None:
                counts[rid] = total
    return counts


def _rule_for_solid(cnc_entry: Optional[dict], solid_index: int) -> Dict[str, Any]:
    """Extract the rule verdict for a specific solid from a cnc_analysis entry."""
    if not isinstance(cnc_entry, dict):
        return {}
    if cnc_entry.get("type") == "multi_solid":
        solids = cnc_entry.get("solids") or []
        entry = solids[solid_index] if 0 <= solid_index < len(solids) else {}
    else:
        entry = cnc_entry
    if not isinstance(entry, dict):
        return {}
    return {
        "rule_type": entry.get("type"),
        "rule_category": entry.get("category"),
        "rule_designation": entry.get("designation"),
        "rule_profile_type": entry.get("profile_type"),
        "rule_match_score": entry.get("match_score"),
        "rule_confidence": entry.get("confidence"),
    }


def _fingerprint_key(feats: Dict[str, Any]) -> str:
    """Stable geometry key so identical solids never split across train/test."""
    parts = [feats.get(k) for k in (
        "volume_mm3", "surface_area_mm2", "dim_thin", "dim_mid", "dim_long",
        "inertia0", "inertia1", "inertia2", "chirality")]
    return "|".join("" if p is None else str(p) for p in parts)


def _name_label(part_name: str) -> Dict[str, Any]:
    try:
        m = classify_by_name(part_name, str(_SECTION_LIBRARY))
    except Exception:
        m = None
    if not m:
        return {"name_category": None, "name_designation": None}
    return {"name_category": m.get("Category"), "name_designation": m.get("Designation")}


def _resolve_label(name_lbl: Dict, rule: Dict, verified: Optional[Dict]) -> Dict[str, Any]:
    """Choose final y_* by priority verified > name > rule."""
    if verified and (verified.get("category") or verified.get("designation")):
        cat = (verified.get("category") or "").strip() or None
        des = (verified.get("designation") or "").strip() or None
        y_type = "plate" if (cat or "").upper() == "PLATE" else "section"
        return {"label_source": "verified", "y_type": y_type,
                "y_category": cat, "y_designation": des}
    if name_lbl.get("name_category"):
        return {"label_source": "name", "y_type": "section",
                "y_category": name_lbl["name_category"],
                "y_designation": name_lbl["name_designation"]}
    rt = rule.get("rule_type")
    if rt in ("section", "plate"):
        return {"label_source": "rule", "y_type": rt,
                "y_category": rule.get("rule_category") or ("PLATE" if rt == "plate" else None),
                "y_designation": rule.get("rule_designation")}
    return {"label_source": "unknown", "y_type": rt or "unknown",
            "y_category": None, "y_designation": None}


_COLUMNS = (_ID_COLS + FEATURE_KEYS + ["align_ok", "features_ok", "feat_mode"]
            + _LABEL_COLS)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _process_sidecar(sidecar: Path, verified: Dict[tuple, Dict],
                     writer: "csv.DictWriter", csv_file, fast: bool,
                     section: bool = False) -> Dict[str, int]:
    """Stream one row per solid for a single sidecar straight to the CSV writer.

    Writing incrementally (and flushing per sidecar) means a crash or timeout
    only loses the job in flight — every completed job is already on disk.
    """
    stats = {"refs": 0, "solids": 0}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _log(f"SKIP {sidecar.name}: unreadable ({e})")
        return stats

    analysis = data.get("analysis") or {}
    tree = analysis.get("assembly_tree")
    if not tree:
        _log(f"SKIP {sidecar.name}: no assembly_tree")
        return stats

    stem = sidecar.stem
    job = _job_name(stem)
    step_path = _resolve_step(stem)
    if step_path is None:
        _log(f"SKIP {sidecar.name}: no matching STEP upload")
        return stats

    parts: Dict[str, Dict] = {}
    _collect_leaf_parts(tree, None, parts)
    inst_counts = _instance_count_from_consolidation(data.get("consolidation") or {})
    cnc = data.get("cnc_analysis") or {}
    cnc_class_map = _cnc_class_by_ref(data)

    # Open the STEP file once and reuse for all refs (OCC memory discipline).
    from app.services.cnc_shape_analyser import _read_xcaf, _get_shape, _iter_solids
    from app.pipeline.geometry_utils import count_solids_in_shape

    _log(f"--- {job}: {len(parts)} refs (STEP {step_path.name})")
    doc, _shape_tool = _read_xcaf(str(step_path))

    for n_ref, (ref_id, info) in enumerate(parts.items(), 1):
        try:
            shape = _get_shape(doc, ref_id)
        except Exception as e:
            _log(f"  ref {ref_id}: shape lookup failed ({e})")
            continue
        try:
            n_solids = count_solids_in_shape(shape)
        except Exception:
            n_solids = 0

        solids = list(_iter_solids(shape)) if n_solids != 1 else [shape]
        if not solids:
            solids = [shape]
        cnc_entry = cnc.get(ref_id)
        name_lbl = _name_label(info.get("name", ""))
        inst = inst_counts.get(ref_id, info.get("instance_count", 1))

        for solid_index, solid in enumerate(solids):
            feats = extract_solid_features(solid, fast=fast, section=section)
            rule = _rule_for_solid(cnc_entry, solid_index)
            vkey = (job, ref_id, str(solid_index))
            vrec = verified.get(vkey)
            label = _resolve_label(name_lbl, rule, vrec)

            row: Dict[str, Any] = {
                "job": job,
                "sidecar": sidecar.name,
                "ref_id": ref_id,
                "solid_index": solid_index,
                "part_name": info.get("name", ""),
                "node_type": info.get("node_type", ""),
                "n_solids": n_solids,
                "instance_count": inst,
                "fingerprint_key": _fingerprint_key(feats),
                **{k: feats.get(k) for k in FEATURE_KEYS},
                "align_ok": feats.get("align_ok"),
                "features_ok": feats.get("features_ok"),
                "feat_mode": feats.get("feat_mode"),
                **name_lbl,
                **rule,
                "verified_issue": (vrec or {}).get("issue", ""),
                "verified_note": (vrec or {}).get("note", ""),
                "cnc_class": cnc_class_map.get(ref_id, ""),
                **label,
            }
            writer.writerow(row)
            stats["solids"] += 1
        stats["refs"] += 1
        if n_ref % 50 == 0:
            _log(f"    .. {n_ref}/{len(parts)} refs")
            gc.collect()

    csv_file.flush()
    _log(f"    done {job}: {stats['refs']} refs, {stats['solids']} solids")
    return stats


def _dedupe_sidecars(sidecars: List[Path]) -> List[Path]:
    """Keep one sidecar per job name, preferring the one with cnc_analysis.

    Many jobs are re-uploaded under different hex prefixes; re-parsing each
    large XCAF file is the dominant cost, so we process one per job.  Geometry
    is identical across re-uploads, so the trainer would dedupe them anyway.
    """
    best: Dict[str, Path] = {}
    scored: Dict[str, tuple] = {}
    for p in sidecars:
        job = _job_name(p.stem)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_class = len((d.get("project_state") or {}).get("classifications") or {})
        # Tuple score, compared lexicographically: a usable STEP first (else the
        # job is silently lost), then the most human classifications (CNC/BO
        # training labels), then richer geometry sections.
        score = (
            1 if _resolve_step(p.stem) else 0,
            n_class,
            1 if d.get("cnc_analysis") else 0,
            1 if d.get("consolidation") else 0,
        )
        if job not in best or score > scored[job]:
            best[job] = p
            scored[job] = score
    return sorted(best.values())


def _run(out_dir: Path, fast: bool, dedupe: bool, section: bool) -> int:
    verified = _load_verified()
    analysis_dir = Path(settings.ANALYSIS_OUTPUT_DIR)
    sidecars = sorted(
        p for p in analysis_dir.glob("*.json")
        if not p.stem.endswith("portal-tree") and not p.stem.endswith("portal-tree.min")
    )
    _log(f"Found {len(sidecars)} sidecars in {analysis_dir}")
    if dedupe:
        sidecars = _dedupe_sidecars(sidecars)
        _log(f"After job-dedupe: {len(sidecars)} sidecars")
    _log(f"Mode: {'fast (fingerprint)' if fast else 'slow (alignment+slice)'}"
         f"{' + section raster' if section else ''}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dataset.csv"
    total = 0
    n_jobs = 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for i, sidecar in enumerate(sidecars, 1):
            _log(f"[{i}/{len(sidecars)}]")
            stats = _process_sidecar(sidecar, verified, writer, f, fast, section)
            total += stats["solids"]
            if stats["solids"]:
                n_jobs += 1

    _log(f"\nWrote {total} rows -> {csv_path}")
    # Best-effort parquet for faster trainer loads.
    try:
        import pandas as pd
        pq_path = out_dir / "dataset.parquet"
        pd.read_csv(csv_path).to_parquet(pq_path, index=False)
        _log(f"Wrote {pq_path}")
    except Exception as e:
        _log(f"(parquet skipped: {e})")

    _log(f"\nDONE: {total} solid rows from {n_jobs} jobs.")
    return 0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Export ML feature/label dataset.")
    ap.add_argument("--all", action="store_true", help="process all sidecars")
    ap.add_argument("--slow", action="store_true",
                    help="use alignment + 21-slice cross-section (accurate, slow)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="process every sidecar, including job re-uploads")
    ap.add_argument("--section", action="store_true",
                    help="also run the cross-section raster (holes/thickness) on "
                         "thin-walled candidates — slower")
    args = ap.parse_args()

    out_dir = Path(settings.OUTPUT_DIR) / "ml"
    rc_holder = [1]

    def runner():
        rc_holder[0] = _run(out_dir, fast=not args.slow, dedupe=not args.no_dedupe,
                            section=args.section)

    try:
        threading.stack_size(64 * 1024 * 1024)
    except OSError:
        pass
    t = threading.Thread(target=runner)
    t.start()
    t.join()
    sys.exit(rc_holder[0])


if __name__ == "__main__":
    main()
