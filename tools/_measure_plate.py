import sys, json, glob, os, numpy as np
sys.path.insert(0,'/app'); sys.path.insert(0,'/app/tools')
import cv2
from app.services import charuco, multiview_fit as MVF, visibility as VIS
import pose_refine as PR, critical_edges as CE, line_check as LC

capdir, fitdir, label = sys.argv[1], sys.argv[2], sys.argv[3]
base = MVF.load_profile('outputs/calibration/RigCam_52FD1B1F.json')
board = charuco.build_board_from_config(base['board']); det = charuco.make_detector(board)
# ALWAYS measure against the NOMINAL model - the whole point is detecting departure from drawing
mesh = VIS.load_stl('outputs/ar_models/plate_cal_00mm.stl')
model = json.load(open('outputs/ar_models/plate_cal_00mm.json'))
tv0 = np.asarray(json.load(open(os.path.join(fitdir,'fit.json')))['tvec'],float).ravel()
views=[]
for path in sorted(glob.glob(os.path.join(capdir,'*.png'))):
    p = base if '52FD' in path else MVF.load_profile('outputs/calibration/RigCam_B68DE55F.json')
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    v = MVF.build_view(img, p, board, det, label=path)
    v.update({'K':p['K'],'dist':p['dist'],'image':img,'tag':path}); views.append(v)
res=[]
for rf in model['resting_faces']:
    Rrest,_ = cv2.Rodrigues(np.asarray(rf['rvec'],float).reshape(3,1))
    best=None
    for yaw in range(0,360,30):
        Ry,_ = cv2.Rodrigues(np.array([0.,0.,np.radians(yaw)]).reshape(3,1))
        rv = cv2.Rodrigues(Ry@Rrest)[0].ravel()
        r,t = PR.refine(mesh, rv, tv0.copy(), views, schedule=(40.,20.,10.,5.,3.,2.),
                        iters=3, dof='seated', verbose=False)
        c,s = PR.score(mesh, r, t, views)
        if best is None or s>best[1]: best=(c,s,r,t)
    res.append((rf['index'],)+best)
res.sort(key=lambda x: -x[2])
i,c,s,r,t = res[0]
print("RES %s  face %d chosen (margin %.0f)  confirmed %.0f%%  silhouette %.0f%%"
      % (label, i, s-res[1][2], c, s), flush=True)
rv=np.asarray(r).ravel(); tv=np.asarray(t).ravel()
OUT = np.array([[0,220],[0,90],[90,0],[300,0],[300,220]], float); C = OUT.mean(axis=0)
HOLES = np.array([[55.,165.],[150.,60.],[245.,150.],[200.,195.]]) - C
out={}
for v in views:
    pts,tan,z,world = CE.visible_feature_edges(mesh, rv.reshape(3,1), tv.reshape(3,1), v,
                                               step_px=2.0, with_world=True)
    fx = float(np.asarray(v['K'],float).reshape(3,3)[0,0])
    off,found,_p,_b = LC.search_along_normals(LC.gradient_field(v['image'],None),pts,tan,z,fx,
                                              tol_mm=30.0, reach=1.0)
    R,_ = cv2.Rodrigues(rv.reshape(3,1)); obj=(R.T @ (world-tv).T).T
    dd = np.linalg.norm(obj[:,None,:2]-HOLES[None,:,:], axis=2)
    lab = np.where(dd.min(axis=1) < 16.0, dd.argmin(axis=1), 4)
    # TOP RIM ONLY. A 3 mm plate seen obliquely shows both rims of every hole, about 2 mm apart in
    # the image - the same size as the displacement being measured - so a search that may lock onto
    # either rim cannot resolve it. The model plane runs 0 to 3 mm; keep the upper edge.
    TOPZ = 3.0
    lab = np.where((obj[:,2] > TOPZ - 1.0) | (lab == 4), lab, 5)
    R2 = np.stack([evec2 := None], axis=0) if False else None
    for k in range(5):
        m = (lab==k) & found
        if m.sum() >= 6:
            # Keep the SIGNED offset and the normal, in the MODEL plane. A hole that has moved
            # sideways shows a signed offset of d.cos(theta) around its rim: averaging that gives
            # zero and averaging its magnitude gives roughly 2d/pi, which is why a 2mm hole read
            # 0.9mm. Fitting (dx,dy) to the pattern recovers the displacement itself.
            Rw,_ = cv2.Rodrigues(rv.reshape(3,1))
            nrm_img = np.stack([-tan[m][:,1], tan[m][:,0]], axis=1)
            # lift the image normal into the model plane via two nearby world points
            eps = 1.0
            base_uv = pts[m]
            far_uv = base_uv + nrm_img * eps
            K = np.asarray(v['K'],float).reshape(3,3); dist = np.asarray(v['dist'],float)
            rc = np.asarray(v['rvec_cam'],float).reshape(3,1); tc = np.asarray(v['tvec_cam'],float).reshape(3,1)
            Rc,_ = cv2.Rodrigues(rc)
            def unproj(uv, zc):
                xn = (uv[:,0]-K[0,2])/K[0,0]; yn = (uv[:,1]-K[1,2])/K[1,1]
                cam = np.stack([xn*zc, yn*zc, zc], axis=1)
                return (Rc.T @ (cam.T - tc)).T
            w0 = unproj(base_uv, z[m]); w1 = unproj(far_uv, z[m])
            dn_world = w1 - w0
            dn_model = (Rw.T @ dn_world.T).T[:, :2]
            nl = np.linalg.norm(dn_model, axis=1, keepdims=True)
            dn_model = dn_model / np.maximum(nl, 1e-9)
            mm_per_px = np.linalg.norm(w1 - w0, axis=1) / eps
            out.setdefault(k, []).append((off[m] * 1.0, dn_model, np.abs(off[m])))
for k in sorted(out):
    nm = 'outline' if k==4 else 'hole %d'%(k+1)
    sig = np.concatenate([b[0] for b in out[k]])
    nrm = np.vstack([b[1] for b in out[k]])
    mag = np.concatenate([b[2] for b in out[k]])
    extra = ''
    if k < 4 and len(sig) >= 12:
        # signed offset = dx*nx + dy*ny, solved robustly
        A = nrm; b = sig
        d = np.zeros(2)
        for _ in range(5):
            r_ = A @ d - b
            sc = 1.4826*np.median(np.abs(r_-np.median(r_))) + 1e-9
            w_ = 1.0/(1.0+(r_/(2.5*sc))**2)
            Aw = A*w_[:,None]
            d = np.linalg.solve(Aw.T@A + 1e-9*np.eye(2), Aw.T@b)
        extra = '   centre moved %5.2f mm' % float(np.linalg.norm(d))
    print("RES   %-9s median %5.2f mm  p90 %5.2f mm%s%s"
          % (nm, np.median(mag), np.percentile(mag,90), extra,
             '   <-- displaced' if k==1 else ''), flush=True)
os.makedirs(fitdir+'_final', exist_ok=True)
json.dump({'rvec':[float(x) for x in rv],'tvec':[float(x) for x in tv],
           'mesh':'plate_cal_00mm.stl','resting_index':int(i)},
          open(fitdir+'_final/fit.json','w'), indent=2)
print("DONE", flush=True)
