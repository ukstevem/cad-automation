import sys, json, glob, os, numpy as np
sys.path.insert(0,'/app'); sys.path.insert(0,'/app/tools')
import cv2
from app.services import charuco, multiview_fit as MVF, visibility as VIS
import pose_refine as PR, critical_edges as CE, line_check as LC

capdir, fitdir = sys.argv[1], sys.argv[2]
CANDS = ['plate_cal_00mm','plate_cal_02mm','plate_cal_05mm','plate_cal_10mm','plate_cal_20mm',
         'plate_plain','plate_slotted','plate_symmetric']
base = MVF.load_profile('outputs/calibration/RigCam_52FD1B1F.json')
board = charuco.build_board_from_config(base['board']); det = charuco.make_detector(board)
tv0 = np.asarray(json.load(open(os.path.join(fitdir,'fit.json')))['tvec'],float).ravel()
views=[]
for path in sorted(glob.glob(os.path.join(capdir,'*.png'))):
    p = base if '52FD' in path else MVF.load_profile('outputs/calibration/RigCam_B68DE55F.json')
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    v = MVF.build_view(img, p, board, det, label=path)
    v.update({'K':p['K'],'dist':p['dist'],'image':img,'tag':path}); views.append(v)

def recall(mesh, rv, tv, views):
    """
    What fraction of the edges actually IN the picture does this model account for?

    Confirmation alone is precision, and precision rewards poverty: a model with no holes cannot
    fail to explain them, so the plainest candidate wins by predicting least. Recall asks the
    opposite question - of the edge pixels visibly present on the part, how many lie on some
    projected model edge - and a model missing four holes is punished for the four rings of edge
    pixels it leaves unexplained.
    """
    got = []
    for v in views:
        pts, tan, z = CE.visible_feature_edges(mesh, np.asarray(rv).reshape(3,1),
                                               np.asarray(tv).reshape(3,1), v, step_px=2.0)
        if not len(pts):
            continue
        depth, _ = VIS.depth_buffer(mesh, np.asarray(rv).reshape(3,1),
                                    np.asarray(tv).reshape(3,1), v, downscale=1)
        solid = depth < VIS.FAR/2
        # observed edges, restricted to the part itself so the board and rig cannot contribute
        g = cv2.GaussianBlur(cv2.cvtColor(v['image'], cv2.COLOR_BGR2GRAY), (0,0), 1.2)
        lo, hi = np.percentile(g[solid], 25), np.percentile(g[solid], 90)
        ed = cv2.Canny(g, max(10, int(lo)), max(40, int(hi)))
        inner = cv2.erode(solid.astype(np.uint8), np.ones((7,7), np.uint8)) > 0
        ys, xs = np.nonzero((ed > 0) & inner)
        if len(xs) < 50:
            continue
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pts).query(np.stack([xs, ys], axis=1))
        got.append(100.0 * float((d < 4.0).mean()))
    return float(np.mean(got)) if got else 0.0


print("%-20s %11s %9s %9s" % ("candidate","confirmed","recall","score"), flush=True)
scores=[]
for name in CANDS:
    mesh = VIS.load_stl('outputs/ar_models/%s.stl' % name)
    model = json.load(open('outputs/ar_models/%s.json' % name))
    best=None
    for rf in model['resting_faces']:
        Rrest,_ = cv2.Rodrigues(np.asarray(rf['rvec'],float).reshape(3,1))
        for yaw in range(0,360,45):
            Ry,_ = cv2.Rodrigues(np.array([0.,0.,np.radians(yaw)]).reshape(3,1))
            rv = cv2.Rodrigues(Ry@Rrest)[0].ravel()
            r,t = PR.refine(mesh, rv, tv0.copy(), views, schedule=(30.,15.,8.,4.,2.),
                            iters=3, dof='seated', verbose=False)
            c,s = PR.score(mesh, r, t, views)
            if best is None or c>best[0]: best=(c,s,r,t)
    rc = recall(mesh, best[2], best[3], views)
    f = 0.0 if (best[0]+rc) == 0 else 2*best[0]*rc/(best[0]+rc)
    scores.append((name, best[0], best[1], rc, f))
    print("RES %-18s %10.0f%% %8.0f%% %8.0f" % (name.replace('plate_',''), best[0], rc, f), flush=True)
scores.sort(key=lambda x: -x[4])
print("RES BEST %s (score %.0f), runner-up %s at %.0f - margin %.0f"
      % (scores[0][0].replace('plate_',''), scores[0][4],
         scores[1][0].replace('plate_',''), scores[1][4], scores[0][4]-scores[1][4]), flush=True)
print("DONE", flush=True)
