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

## 2. The identification scheme (PROPOSED — not yet implemented)

```
<piece-mark>-W<nnn>
B041-W003
```

**Piece mark prefix** gives job-wide uniqueness without a central allocator. The piece mark already
identifies the part uniquely within a job, so namespacing by it inherits that guarantee for free.
A flat sequence across a whole job would need a registry and would still collide whenever two
assemblies are analysed independently.

**The ordinal is derived from geometry, not from iteration order.** Sort joints by their centroid
in model coordinates — a deterministic lexicographic sort on (x, y, z), rounded to a tolerance
coarse enough to be robust to solver noise but fine enough to be unambiguous. This is the property
that makes the number survive a re-run: the centroid of a joint does not change when the detector's
`--min-length` changes, so the joints that survive both runs keep their numbers.

Sorting by **length** — which is what the projection stage does today — fails this outright. Add
one weld, or filter one out, and every number below it shifts.

**Number once, at joint level.** A "joint" is the pair of solids being joined, not a fragment of a
weld perimeter. An operator inspects *the cleat to the rail*, and a drawing specifies it that way,
so that is the unit that carries a number and eventually a WPS reference. The detector's path
fragments are geometry, not identity.

### Consequences for the pipeline

Numbering belongs in `extract`, which is the stage that has the model and the piece mark.
`project` and every downstream consumer **reads** the number and never mints one.

---

## 3. Current state — and a defect

`tools/weld_locate.py` currently assigns weld numbers **twice, independently, and the two
disagree**:

- `extract` ([tools/weld_locate.py:82](../tools/weld_locate.py#L82)) numbers path **fragments**
  `W001…Wnnn` in detector order.
- `project` ([tools/weld_locate.py:183](../tools/weld_locate.py#L183)) discards those, groups
  fragments by the solid pair they join, and renumbers **joints** `W001…Wnnn` sorted by descending
  total length.

So the same physical weld carries a different number depending on which stage you read, the numbers
shift when `--min-length` changes, and there is no piece-mark namespace so they collide across a
job.

Tracked as **cad-automation-dwd** (P1). The grouping logic in `project` is correct and should move
into `extract`; what has to change is *when* and *how* the number is issued.

---

## 4. What geometry can and cannot tell us

Measured on the real test assembly (`outputs/welds/mainframe.json`, node `0:1:1:1:1`):

```
269 path fragments  ->  64 joints  ->  18 596 mm of weld
joint length: min 124 mm, median 260 mm, max 875 mm
joint type detected: 106 of 269 fragments (39%) — the rest are null
```

That 39% matters. Connection detection currently labels some contacts `t-joint` and leaves the
majority unclassified, so **joint type is not reliably available from geometry today**. Anything
downstream that needs `Type1`/`Type2` has to tolerate its absence.

More fundamentally, there is a hard boundary here that no amount of detection work crosses:

| we can derive from geometry | we cannot — it is an engineering decision |
|---|---|
| weld position (path, centroid) | throat thickness, leg length |
| weld length | penetration depth |
| which two solids are joined | welding process |
| joint type *(partly, 39%)* | intermittent vs continuous, pitch |
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
| `Type1`, `Type2` | IfcLabel | ISO 2553 seam type | partly — 39% today |
| `Surface1`, `Surface2` | IfcLabel | plane / curved / hollow | yes, from face geometry |
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

So of seventeen properties, our pipeline can honestly populate **three** (`Surface1`, `Surface2`,
and `Type1`/`Type2` when detected) plus a qualified `l`. That is not a shortfall — it is the
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

## 6. The sidecar (current prototype)

`tools/weld_locate.py extract` writes a sidecar that is the prototype for the IFC payload. Current
shape:

```json
{
  "source": "e8a0ba3c_Structural Package 25-05-2026.json",
  "node": "0:1:1:1:1",
  "scope": "within-part",
  "frame": "model coordinates as stored by the detector",
  "weld_count": 269,
  "total_length_mm": 18595.8,
  "welds": [
    {
      "weld_number": "W001",
      "joins": ["0:1:1:1:1:s4", "0:1:1:1:1:s26"],
      "method": "t-joint",
      "length_mm": 40.0,
      "path": [[-210.0, 0.0, 164.0]]
    }
  ]
}
```

Mapping to the IFC target:

| sidecar | IFC |
|---|---|
| `weld_number` | `IfcFastener.Name` — **once §2 is implemented** |
| `joins` | the `IfcRelConnects*` the fastener relates |
| `method` | `Pset_FastenerWeld.Type1` / `Type2` |
| `length_mm` | see §5.3 — **not** simply `l` |
| `path` | the fastener's representation |
| `frame` | must resolve to the IFC project coordinate system |

The `frame` field is load-bearing and easy to lose: the paths are in *model* coordinates as stored
by the detector, which is the same frame the AR pose maps from. Any IFC export has to place them in
the project coordinate system explicitly rather than assuming the two agree.

---

## 7. Open questions

- **Joint type coverage.** 61% of fragments have no detected type. Worth knowing whether that is a
  detection gap or genuinely ambiguous geometry before deciding how much effort it deserves.
- **Piece mark availability.** §2 assumes the piece mark is reachable at `extract` time. Confirm it
  is, or the scheme needs a different namespace.
- **Centroid sort tolerance.** Needs choosing against real data — coarse enough to be stable under
  solver noise, fine enough that no two joints tie.
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
