"""BOM with weights and bounding boxes for a stage2 project analysis.

    python scripts/bom_10364.py <analysis.json> <out_dir>

MASS IS RECOMPUTED FROM VOLUME, NOT READ FROM THE FILE. The stored ``mass_kg`` is correct for
sections (implied density 7.85e-6 kg/mm^3 exactly) and wrong for every plate: on 10364 the 22
plates carry 8.4 kg between them where the geometry says 184.8 kg, and a 452 x 452 x 30 base
plate is recorded as 2.313 kg against an actual 48.2 kg. The error is not a constant factor,
so it cannot be corrected by scaling - it has to be recomputed.

Volume is trustworthy: it reproduces L x W x T on every plate checked, and it agrees with the
fingerprint's own volume term. So volume x density is the honest figure, and the file reports
both so the discrepancy is visible rather than quietly repaired.

Quantities come from ``consolidation.part_groups`` (total_count), which is the deduplicated
count of identical parts - including mirrored ones, reported separately because a mirrored part
is a different piece to fabricate even though it weighs the same.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

DENSITY = 7.85e-6          # kg/mm^3, steel. Grade is S355 on this job.


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    d = json.loads(src.read_text(encoding="utf-8"))

    ca = d.get("cnc_analysis", {})
    names = d.get("cnc_member_names", {})
    parents = d.get("cnc_parent_names", {})
    groups = d.get("consolidation", {}).get("part_groups", [])

    rows, unanalysed = [], 0
    for g in groups:
        refs = g.get("ref_ids") or []
        rec = next((ca[r] for r in refs if r in ca), None)
        qty = int(g.get("total_count") or 0)
        name = next((names[r] for r in refs if names.get(r)), "")
        parent = next((parents[r] for r in refs if parents.get(r)), "")

        if rec is None:
            # A group nothing measured. Reported with a blank weight rather than dropped:
            # a BOM that silently omits what it could not size understates the job.
            unanalysed += 1
            rows.append({"name": g.get("canonical_name", ""), "member": name, "parent": parent,
                         "type": "", "qty": qty, "mirrored": g.get("mirrored_count") or 0,
                         "length_mm": "", "width_mm": "", "thickness_mm": "",
                         "volume_mm3": "", "mass_each_kg": "", "mass_total_kg": "",
                         "note": "not analysed - no dimensions or weight"})
            continue

        dims = rec.get("dims") or {}
        vol = float(rec.get("volume_mm3") or 0.0)
        each = vol * DENSITY
        stored = float(rec.get("mass_kg") or 0.0)
        note = ""
        if stored and abs(stored - each) > max(0.05, 0.02 * each):
            note = f"stored mass was {stored:.3f} kg - recomputed from volume"
        rows.append({
            "name": g.get("canonical_name", ""), "member": name, "parent": parent,
            "type": rec.get("type") or "", "qty": qty,
            "mirrored": g.get("mirrored_count") or 0,
            "length_mm": round(float(dims.get("L") or 0.0), 1),
            "width_mm": round(float(dims.get("W") or 0.0), 1),
            "thickness_mm": round(float(dims.get("T") or 0.0), 1),
            "volume_mm3": round(vol, 0),
            "mass_each_kg": round(each, 2),
            "mass_total_kg": round(each * qty, 2),
            "note": note,
        })

    rows.sort(key=lambda r: -(r["mass_total_kg"] or 0))
    cols = ["name", "member", "parent", "type", "qty", "mirrored", "length_mm", "width_mm",
            "thickness_mm", "volume_mm3", "mass_each_kg", "mass_total_kg", "note"]
    out = out_dir / "10364_bom.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    tot = sum(r["mass_total_kg"] or 0 for r in rows)
    pieces = sum(r["qty"] for r in rows)
    stored_tot = sum(float(v.get("mass_kg") or 0.0) for v in ca.values())
    by_type: dict[str, list] = {}
    for r in rows:
        t = r["type"] or "(not analysed)"
        b = by_type.setdefault(t, [0, 0, 0.0])
        b[0] += 1; b[1] += r["qty"]; b[2] += (r["mass_total_kg"] or 0)

    print(f"{len(rows)} groups, {pieces} pieces -> {out}")
    print()
    print(f"  {'type':<16}{'groups':>7}{'pieces':>8}{'kg':>11}")
    for t, (n, q, m) in sorted(by_type.items(), key=lambda x: -x[1][2]):
        print(f"  {t:<16}{n:>7}{q:>8}{m:>11,.1f}")
    print(f"  {'TOTAL':<16}{len(rows):>7}{pieces:>8}{tot:>11,.1f}   ({tot/1000:.2f} t)")
    print()
    print(f"  stored masses in the file summed to {stored_tot:,.1f} kg for the analysed parts;")
    print(f"  recomputed from volume they are {sum(float(v.get('volume_mm3') or 0)*DENSITY for v in ca.values()):,.1f} kg.")
    if unanalysed:
        print(f"  {unanalysed} group(s) carry no dimensions or weight - listed with blanks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
