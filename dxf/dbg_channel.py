import sys, json, math
sys.path.insert(0, "/app")

from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Tool, TDF_Label
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.gp import gp_Ax1, gp_Pnt, gp_Dir, gp_Trsf
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

print("Loading...")
fname = "/app/uploads/9dd0ba2e_GT 132-33kV__G8702-C6-001-3D Rev02.STEP"
doc = TDocStd_Document(TCollection_ExtendedString("XDE"))
reader = STEPCAFControl_Reader()
reader.ReadFile(fname)
reader.Transfer(doc)

lab = TDF_Label()
TDF_Tool.Label_s(doc.Main().Data(), "0:1:1:1", lab)
shape = XCAFDoc_ShapeTool.GetShape_s(lab)

exp = TopExp_Explorer(shape, TopAbs_SOLID)
first = None
while exp.More():
    if not first:
        first = exp.Current()
    exp.Next()
s = first if first else shape

# Raw AABB
bb = Bnd_Box()
BRepBndLib.Add_s(s, bb)
xn, yn, zn, xx, yx, zx = bb.Get()
raw_dims = sorted([xx-xn, yx-yn, zx-zn], reverse=True)
print("AABB: %.1f x %.1f x %.1f" % (raw_dims[0], raw_dims[1], raw_dims[2]))
print("AABB actual: X=%.1f Y=%.1f Z=%.1f" % (xx-xn, yx-yn, zx-zn))

# Aligned
from app.pipeline.geom_alignment import align_by_longest_straight_edge
r = align_by_longest_straight_edge(s)
a = r[0]

bb2 = Bnd_Box()
BRepBndLib.Add_s(a, bb2)
xn2, yn2, zn2, xx2, yx2, zx2 = bb2.Get()
L = xx2 - xn2
H = yx2 - yn2
W = zx2 - zn2
print("Aligned: L=%.1f H=%.1f W=%.1f" % (L, H, W))

# Section dims
from app.pipeline.geometry_utils import compute_section_dims
sd = compute_section_dims(a)
sa = sd["area"]
Hc = sd["H"]
Wc = sd["W"]
print("SectionDims: area=%.1f H=%.1f W=%.1f" % (sa, Hc, Wc))

# Classify
from app.pipeline.classification import classify_profile
lp = "/app/app/pipeline/data/Aluminium_classifier_info.json"
cs = {"span_web": Hc, "span_flange": Wc, "area": sa, "length": L}
res = classify_profile(cs, lp)
if res:
    print("PASS1 MATCH: %s (%s)" % (res["Designation"], res["Category"]))
else:
    print("PASS1 NO MATCH: H=%.1f W=%.1f A=%.1f" % (Hc, Wc, sa))

# Check inflation
inflation = L / raw_dims[0] if raw_dims[0] > 0 else 1
print("L inflation: %.4f (%.1f%%)" % (inflation, (inflation-1)*100))
triggers = raw_dims[0] > 20.0 and L > raw_dims[0] * 1.03
print("Would trigger AABB retry: %s" % triggers)

# Try all 3 AABB axis rotations
def try_axis(shape_in, label):
    bb_t = Bnd_Box()
    BRepBndLib.Add_s(shape_in, bb_t)
    xnt, ynt, znt, xxt, yxt, zxt = bb_t.Get()
    Lt = xxt - xnt
    Ht = yxt - ynt
    Wt = zxt - znt
    print("  %s dims: L=%.1f H=%.1f W=%.1f" % (label, Lt, Ht, Wt))
    sdt = compute_section_dims(shape_in)
    sat = sdt["area"]
    Hct = sdt["H"]
    Wct = sdt["W"]
    print("  %s section: area=%.1f H=%.1f W=%.1f" % (label, sat, Hct, Wct))
    cst = {"span_web": Hct, "span_flange": Wct, "area": sat, "length": Lt}
    rt = classify_profile(cst, lp)
    if rt:
        print("  %s => MATCH %s (%s) score=%.2f" % (label, rt["Designation"], rt["Category"], rt["Match_score"]))
    else:
        print("  %s => NO MATCH" % label)
        # Show nearby
        with open(lp) as f:
            lib = json.load(f)
        for cat, ents in lib.items():
            for nm, info in ents.items():
                h, w, ar = info["height"], info["width"], info["csa"]
                dh = min(abs(h - Hct), abs(h - Wct))
                dw = min(abs(w - Hct), abs(w - Wct))
                if dh < 30 and dw < 30:
                    ea = abs(sat - ar) / max(ar, 1)
                    print("    %s/%s H=%.1f W=%.1f A=%.1f ea=%.2f%% dh=%.1f dw=%.1f" % (cat, nm, h, w, ar, ea*100, dh, dw))
    return rt

print("\n=== AABB Axis retries on ORIGINAL shape ===")

# Y-as-X: rotate +90 about Z
t1 = gp_Trsf()
t1.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(0,0,1)), math.radians(90))
rot1 = BRepBuilderAPI_Transform(s, t1, True).Shape()
try_axis(rot1, "Y-as-X")

# Z-as-X: rotate -90 about Y
t2 = gp_Trsf()
t2.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(0,1,0)), math.radians(-90))
rot2 = BRepBuilderAPI_Transform(s, t2, True).Shape()
try_axis(rot2, "Z-as-X")

# X-as-X (unrotated)
try_axis(s, "X-as-X(orig)")
