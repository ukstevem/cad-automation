# Weld tagging, identification and IFC carriage

How a weld found by geometry gets a number, keeps it, and travels to the people and systems that
need it. Written 2026-09-10, against schemas and standards checked at the time — see
**Verification** at the end for what was confirmed and what is still recollection.

Scope is deliberately narrow: **identification and carriage**. The broader standards landscape —
what exists for welding data at all, and why there is no DSTV equivalent — is in the vault at
`reference/welding-data-standards.md`.

---

## 1. What a weld identifier has to do

No standard prescribes the *format* of a weld number. **EN 1090-2** instead specifies what the
**weld map** must record, which constrains the identifier more usefully than a format would.

For execution class **EXC3/EXC4** the weld map records:

| field | source |
|---|---|
| weld joint ID | **us** |
| joint type and geometry | us (partly) — see §4 |
| applicable WPS reference | engineering |
| welder ID and qualification standard | the shop, at weld time |
| NDT method and acceptance class | QA |
| inspection status | QA |
| repair records | QA |

**EXC2** requires a streamlined subset. The execution class is set per contract, so the exporter
should not assume EXC3 — it should emit what it knows and leave the rest addressable.

The identifier is the **join key** for every one of those columns. That gives it two hard
properties, neither of which is about formatting:

- **Unique across a job.** Not per part, not per detection run. Two different parts in one
  fabrication must not both contain a `W001`.
- **Stable over time.** The number must still resolve to the same physical weld when someone pulls
  the record in three years. Re-running the detector with a different tolerance must not renumber
  anything.

Everything in §2 follows from those two lines.

---

## 2. The identification scheme

```
<piece-mark>-W<nnn>
B041-W003
```

**Piece mark prefix** gives job-wide uniqueness without a central allocator. The piece mark already
identifies the part uniquely within a job, so namespacing by it inherits that guarantee for free.
A flat sequence across a whole job would need a registry and would still collide whenever two
assemblies are analysed independently.

**The ordinal follows geometry, not iteration order.** Joints are sorted by centroid — lexicographic
on (x, y, z), rounded to `--sort-tol` (1 mm by default), with the solid-id pair breaking ties. X
leads because it is the extrusion axis by convention here, so the numbers run along the member the
way a welder walks it. Sorting by **length** instead — what the projection stage used to do —
reshuffles everything below any weld that crosses a filter.

**But a geometric sort is not sufficient, and this is the part that caught us out.** An ordinal is
a *rank*, so it shifts whenever the set changes. The first implementation numbered the joints that
survived `--min-length` and failed its own stability test at **0 of 50**: dropping fourteen short
welds moved every number above them. Two further rules make it hold.

*Number the whole set, filter afterwards.* A display threshold then cannot renumber anything. The
sequence gains gaps where welds were filtered out, which is correct — weld maps gain gaps at every
revision, and a gap is honest where a renumber is not.

*Carry previous assignments forward by identity.* Geometric order still shifts if the **detector**
finds a joint it previously missed, because a new weld inserts into the middle. Real fabrication
does not renumber for that: it keeps the numbers already issued and allocates new ones at the end.
`--carry-forward <previous.json>` does exactly that.

**Number once, at joint level.** A "joint" is the pair of solids being joined, not a fragment of a
weld perimeter. An operator inspects *the cleat to the rail*, and a drawing specifies it that way,
so that is the unit that carries a number and eventually a WPS reference. The detector's path
fragments are geometry, not identity.

### Measured behaviour

| test | result |
|---|---|
| re-run at `--min-length` 10 → 200 (64 → 50 joints) | **50 of 50** kept `Name` and `GlobalId` |
| detector finds 3 joints it previously missed | **61 of 61** kept their number; new ones issued W062–W064 |

Numbering lives in `extract`, the stage that knows the piece mark. `project` and every downstream
consumer **reads** the number and never mints one.

---

## 3. What this replaced

`weld_locate.py` used to assign weld numbers **twice, independently, and the two disagreed**:
`extract` numbered path *fragments* in detector order, and `project` discarded those and renumbered
*joints* by descending length. The same physical weld carried a different number depending on which
file you read, and the numbers moved whenever `--min-length` changed.

It also mistook fragments for welds. The detector already returns **one connection per joint** —
the boolean intersection's several path fragments are gathered inside it — so flattening them
turned 64 joints into 269 "welds" and stacked eight labels on one T-joint.

Fixed in **cad-automation-dwd**.

---

## 4. What geometry can and cannot tell us

Measured on the real test assembly (`outputs/welds/mainframe.json`, node `0:1:1:1:1`):

```
64 welded joints  ->  19 245 mm of weld
joint length: min 124 mm, median 260 mm, max 896 mm
joint type detected: 21 of 64 joints (33%) — the rest carry no Type1
```

That 33% matters. (An earlier draft said 39%, counting *fragments* rather than joints; the joint
figure is the meaningful one, since a joint is what carries a number and a type.) Connection detection currently labels some contacts `t-joint` and leaves the
majority unclassified, so **joint type is not reliably available from geometry today**. Anything
downstream that needs `Type1`/`Type2` has to tolerate its absence.

More fundamentally, there is a hard boundary here that no amount of detection work crosses:

| we can derive from geometry | we cannot — it is an engineering decision |
|---|---|
| weld position (path, centroid) | throat thickness, leg length |
| weld length | penetration depth |
| which two solids are joined | welding process |
| joint type *(partly, 33%)* | intermittent vs continuous, pitch |
| surface form (planar/curved) | quality level, NDT requirement |

The right-hand column comes from the **WPS**, not from the model. The exporter's job is to carry
the left-hand column faithfully and to leave the right-hand column addressable by identifier — not
to guess it.

---

## 5. IFC carriage

A weld is an **`IfcFastener`** with `PredefinedType = WELD`. Verified from the schema in our own
container (ifcopenshell 0.8.5): `IfcFastenerTypeEnum` = `GLUE, MORTAR, WELD, USERDEFINED,
NOTDEFINED`, identical across IFC4, IFC4X3 and IFC4X3_ADD2.

`IfcMechanicalFastener` is a **sibling**, not a subtype, and covers bolts — `BOLT`, `RIVET`,
`SHEARCONNECTOR` and so on. `WELD` is correctly absent from it. Do not reach for it.

### 5.1 Where the identifier goes

**`Pset_FastenerWeld` carries no weld identifier.** This surprised us and is worth stating plainly,
because it is the one thing the whole traceability chain hangs on. The mapping is:

| what | where |
|---|---|
| weld number (`B041-W003`) | `IfcFastener.Name` (and/or `.Tag`) |
| WPS reference | `IfcClassificationReference` |
| geometry and process data | `Pset_FastenerWeld` |
| the weld path itself | the fastener's own representation |

### 5.2 `Pset_FastenerWeld` field by field

Applies to `IfcFastener/WELD` in IFC4; IFC4X3_ADD2 extends it to `IfcFastenerType/WELD` as well, so
shared properties can sit on the type object.

| property | type | ISO basis | can we populate it? |
|---|---|---|---|
| `Type1`, `Type2` | IfcLabel | ISO 2553 seam type | partly — 33% today |
| `Surface1`, `Surface2` | IfcLabel | plane / curved / hollow | in principle, but `contact_faces` is null today |
| `Process` | IfcInteger | **ISO 4063** process number | no — from WPS |
| `ProcessName` | IfcLabel | text alternative | no — from WPS |
| `a` | IfcPositiveLengthMeasure | nominal throat thickness | no — engineering |
| `c` | IfcPositiveLengthMeasure | weld width | no — engineering |
| `d` | IfcPositiveLengthMeasure | weld diameter (spot/plug) | no — engineering |
| `e` | IfcPositiveLengthMeasure | spacing between weld elements | no — engineering |
| `l` | IfcPositiveLengthMeasure | length of one weld element | see gotcha below |
| `n` | IfcCountMeasure | number of weld elements | no — engineering |
| `s` | IfcPositiveLengthMeasure | deep-penetration throat thickness | no — engineering |
| `z` | IfcPositiveLengthMeasure | leg length | no — engineering |
| `Intermittent` | IfcBoolean | — | no — engineering |
| `Staggered` | IfcBoolean | — | no — engineering |

So of sixteen properties, our pipeline populates exactly **one** today — `Type1`, on the 33% of
joints where the detector names a type. `Surface1`/`Surface2` are derivable in principle but the
detector returns `contact_faces: null`, so they are not available yet. That is not a shortfall — it is the
correct division of labour. The Pset is designed to hold a *specified* weld; we are supplying the
*detected* geometry that a specification gets attached to.

### 5.3 Gotcha: `l` is not total weld length

`l` is the length of **one weld element**, not the length of the joint. For a continuous weld those
coincide; for an intermittent weld `l` is the length of each stitch, `n` how many, and `e` the
spacing. Our `length_mm` is total joint length, which equals `l` **only** when the weld is
continuous — and we do not know whether it is, because that is an engineering decision.

Writing our measured length into `l` unconditionally would silently misrepresent every intermittent
weld. Either emit `l` only alongside `Intermittent = FALSE`, or carry the measured length in a
custom property and leave `l` to the specification.

### 5.4 Gotcha: IFC4 and IFC4X3 name these properties differently

IFC4X3_ADD2 renamed the single-letter measures to explicit names. **The property names are not
interchangeable between schema versions**, so an exporter must know which it is targeting:

| IFC4 | IFC4X3_ADD2 |
|---|---|
| `a` | `NominalThroatThickness` |
| `c` | `WeldWidth` |
| `d` | `WeldDiameter` |
| `e` | `WeldElementSpacing` |
| `l` | `WeldElementLength` |
| `n` | `NumberOfWeldElements` |
| `s` | `DeepPenetrationThroatThickness` |
| `z` | `WeldLegLength` |

`Type1`, `Type2`, `Surface1`, `Surface2`, `Process`, `ProcessName`, `Intermittent` and `Staggered`
keep their names in both.

The rename is also the most authoritative statement of what the ISO 2553 letters mean — it is
buildingSMART spelling out the abbreviations — which is where the "ISO basis" column in §5.2 comes
from.

---

## 6. The sidecar

`weld_locate.py extract` writes a sidecar shaped like the IFC payload, so the eventual export is a
serialisation step rather than a re-modelling one — and so the schema questions get asked now,
while they are cheap.

```json
{
  "schema": "IFC4",
  "generator": "cad-automation weld_locate",
  "generated": "2026-09-10T06:58:17+00:00",
  "project": "10370",
  "steel_grade": "S355",
  "piece_mark": { "value": "MAINFRAME", "derived_from": "the part name carried on the solids" },
  "source": { "analysis": "...", "node": "0:1:1:1:1", "scope": "within-part" },
  "placement": { "frame": "model", "units": "mm", "to_project": null, "note": "..." },
  "summary": { "weld_count": 64, "total_length_mm": 19244.7 },
  "welds": [{
    "GlobalId": "1c2cBjcPpv4Uw7e7fsKJ44",
    "Name": "MAINFRAME-W025",
    "Tag": "MAINFRAME-W025",
    "PredefinedType": "WELD",
    "Description": "solid 0 to solid 1",
    "ConnectedTo": ["0:1:1:1:1:s0", "0:1:1:1:1:s1"],
    "Pset_FastenerWeld":     { "Type1": "t-joint" },
    "Pset_PSS_WeldGeometry": { "MeasuredLengthMm": 896.0, "SegmentCount": 16,
                               "CentroidMm": [-156.95, 2133.33, 190.26] },
    "Representation": { "type": "Polyline", "segments": [[[...]]] }
  }]
}
```

Four decisions in that are worth defending.

**Two Psets, deliberately.** `Pset_FastenerWeld` holds what somebody *specified*;
`Pset_PSS_WeldGeometry` holds what we *measured*. That makes §4's boundary structural instead of a
paragraph in a document, and it keeps the measured length out of `l`, where it would assert
something false about any intermittent weld (§5.3).

**Only what we know.** No null `Process`, no null throat thickness. A missing property says "not
specified yet"; a null says "specified as nothing". Only the second is a lie. In practice
`Pset_FastenerWeld` carries `Type1` alone, on 33% of joints.

**`GlobalId` seeded from identity, never from position in a list.** The first version included the
ordinal in the seed and so inherited every renumber — a GlobalId that changes is not an identifier,
it is a serial number for the run. It is now `sha1(project | piece mark | solid pair)` rendered in
IFC's own 22-character alphabet. Verified on all 64: 22 characters, correct alphabet, leading
character in 0–3, no collisions.

**The schema version is declared, not assumed.** IFC4X3 renamed the single-letter measures (§5.4),
so a reader has to know which naming applies. `IFC4` is the target because **IFC2X3 cannot carry
this at all** — it has `IfcFastener` but no `IfcFastenerTypeEnum` and no `Pset_FastenerWeld`, so a
weld there needs `ObjectType` text and a custom Pset. Verified against both schemas via
ifcopenshell.

Still to resolve: `placement.to_project` is null. The paths are in *model* coordinates — the same
frame the AR pose maps from — and an IFC export must place them in the project coordinate system
explicitly rather than assume the two agree.

## 7. Open questions

- **Joint type coverage.** 61% of fragments have no detected type. Worth knowing whether that is a
  detection gap or genuinely ambiguous geometry before deciding how much effort it deserves.
- **Piece mark provenance.** Resolved for now — derived from the part name on the solids
  (`MAINFRAME`) and recorded in the output, with `--piece-mark` to override. But it is a CAD part
  name, not a fabrication piece mark, and on a real job those should agree explicitly.
- **`Surface1`/`Surface2`.** The detector returns `contact_faces: null`, so the one other property
  geometry could fill is unavailable. Worth finding out whether that is cheap to populate.
- **`placement.to_project`.** Null today. An IFC export needs the model → project transform stated,
  not assumed.
- **Execution class.** The exporter should probably take EXC as a parameter rather than assume one.
- **ISO 4063 values.** Common structural process numbers (111 MMA, 135 MAG, 136 FCAW, 141 TIG) are
  **from recollection and unverified** — check against the document before embedding them anywhere.

---

## 8. Verification

Confirmed 2026-09-09/10:

- `IfcFastenerTypeEnum` and the full `Pset_FastenerWeld` property list with types — read from the
  schema via ifcopenshell 0.8.5 **in our own container**, for IFC4 and IFC4X3_ADD2, not from
  documentation.
- The IFC4 → IFC4X3 property rename (§5.4) — same source, by direct comparison.
- EN 1090-2 weld map contents by execution class — from the sources below.
- Fragment/joint counts and joint-type coverage (§4) — computed from `outputs/welds/mainframe.json`.

Not verified: the ISO 4063 process numbers in §7, and the precise EN 1090-2 clause numbers (the
requirement is confirmed, the clause references are not).

## References

- ISO 2553:2019 — https://www.iso.org/standard/72740.html
- ISO 4063:2023 — https://www.iso.org/standard/75108.html
- `IfcFastener` — https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/HTML/lexical/IfcFastener.htm
- `Pset_FastenerWeld` — https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/HTML/lexical/Pset_FastenerWeld.htm
- EN 1090-2 overview — https://www.axisinspection.com/en-1090-2-2018/
- Weld map documentation — https://www.therness.com/blog/weld-map-documentation-iso-3834-asme-compliance-guide/
- Vault: `reference/welding-data-standards.md` — the wider standards landscape
