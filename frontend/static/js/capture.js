/**
 * Capture page — AR alignment spike (Phase 0).
 *
 * Workflow: pick a job + calibration profile → upload a photo → load the CAD model
 * (wireframe + numbered corner anchors) → build correspondences by clicking an anchor
 * in 3D then its location on the photo → Solve → the section's CAD edges are projected
 * and drawn over the photo. If the edges land on the real steel, alignment is proven.
 *
 * Manual-correspondence registration for the spike; marker-based + auto comes later.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MAX_PHOTO_W = 720;

export class CapturePage {
    constructor(api) {
        this.api = api;
        this.container = null;

        this._files = [];
        this._profiles = [];
        this._filename = null;
        this._profile = null;          // selected profile summary
        this._profileData = null;      // full profile (image_size etc.)

        this._img = null;              // HTMLImageElement (natural res)
        this._photoBlob = null;
        this._scale = 1;               // display/natural
        this._geometry = null;         // { edges, vertices, bbox }
        this._correspondences = [];    // [{ anchor, world:[x,y,z], image:[u,v] }]
        this._selectedAnchor = null;   // index awaiting a photo click
        this._overlay = null;          // [polyline2d] from solve
        this._lastSolve = null;

        // three.js
        this._three = null;
        this._anchorMeshes = [];
    }

    render(container) {
        this.container = container;
        this._cleanup();
        container.innerHTML = '<p aria-busy="true">Loading…</p>';
        this._init();
    }

    async _init() {
        try {
            const [filesResp, profResp] = await Promise.all([
                this.api.listFiles(),
                this.api.listCalibrationProfiles(),
            ]);
            this._files = filesResp.files || filesResp || [];
            this._profiles = profResp.profiles || [];
        } catch (err) {
            this.container.innerHTML = `<p class="error">Failed to load: ${this._esc(String(err?.message || err))}</p>`;
            return;
        }
        this.container.innerHTML = this._template();
        this._bind();
    }

    _cleanup() {
        this._disposeThree();
        this._correspondences = [];
        this._selectedAnchor = null;
        this._overlay = null;
        this._img = null;
        this._photoBlob = null;
    }

    // ---------------------------------------------------------------
    // Template
    // ---------------------------------------------------------------

    _template() {
        const fileOpts = ['<option value="">Select job…</option>']
            .concat(this._files.map(f => `<option value="${this._esc(f.filename)}">${this._esc(f.display_name || f.filename)}</option>`))
            .join('');
        const profOpts = ['<option value="">Select calibration…</option>']
            .concat(this._profiles.map(p => `<option value="${this._esc(p.name)}">${this._esc(p.name)} (${p.image_size ? p.image_size.join('×') : '?'})</option>`))
            .join('');

        return `
            <section class="capture-page">
                <h2>Capture &amp; Align <small>(Phase 0 spike)</small></h2>
                <p class="capture-intro">Overlay a part's CAD edges onto a photo to prove alignment.
                   Pick a job + the camera's calibration, upload a photo, match ≥4 corners, then Solve.</p>

                <div class="capture-markers">
                    <span>Printable AR markers:</span>
                    <label>size mm<input type="number" id="cap-mk-size" value="100" min="20" max="400"></label>
                    <label>count<input type="number" id="cap-mk-count" value="6" min="1" max="50"></label>
                    <label>start ID<input type="number" id="cap-mk-start" value="0" min="0"></label>
                    <button id="cap-mk-print" class="outline">Open marker PDF ↗</button>
                </div>

                <div class="capture-setup">
                    <label>Job<select id="cap-file">${fileOpts}</select></label>
                    <label>Calibration<select id="cap-profile">${profOpts}</select></label>
                    <label>Photo<input type="file" id="cap-photo-input" accept="image/*" disabled></label>
                    <button id="cap-load-model" class="outline" disabled>Load model</button>
                </div>
                <p id="cap-status" class="capture-status"></p>

                <div class="capture-stage">
                    <div class="capture-pane">
                        <header>Photo <small id="cap-photo-meta"></small></header>
                        <canvas id="cap-photo-canvas" class="capture-canvas"></canvas>
                    </div>
                    <div class="capture-pane">
                        <header>CAD model — click an anchor, then click it on the photo</header>
                        <div id="cap-3d" class="capture-3d"></div>
                    </div>
                </div>

                <div class="capture-actions">
                    <span id="cap-corr-count" class="capture-corr-count">0 correspondences</span>
                    <button id="cap-undo" class="outline secondary" disabled>Undo last</button>
                    <button id="cap-clear" class="outline secondary" disabled>Clear</button>
                    <button id="cap-solve" disabled>Solve &amp; Overlay</button>
                    <button id="cap-save" class="outline" disabled>Save capture</button>
                    <span id="cap-result" class="capture-result"></span>
                </div>
            </section>
        `;
    }

    _bind() {
        const $ = (s) => this.container.querySelector(s);
        $('#cap-file').addEventListener('change', (e) => { this._filename = e.target.value || null; this._refreshReady(); });
        $('#cap-profile').addEventListener('change', (e) => this._onProfile(e.target.value));
        $('#cap-photo-input').addEventListener('change', (e) => this._onPhoto(e));
        $('#cap-load-model').addEventListener('click', () => this._loadModel());
        $('#cap-undo').addEventListener('click', () => this._undo());
        $('#cap-clear').addEventListener('click', () => this._clearCorr());
        $('#cap-solve').addEventListener('click', () => this._solve());
        $('#cap-save').addEventListener('click', () => this._save());
        $('#cap-photo-canvas').addEventListener('click', (e) => this._onPhotoClick(e));
        $('#cap-mk-print').addEventListener('click', () => this._printMarkers());
    }

    _printMarkers() {
        const size = parseFloat(this.container.querySelector('#cap-mk-size').value) || 100;
        const count = parseInt(this.container.querySelector('#cap-mk-count').value) || 6;
        const start = parseInt(this.container.querySelector('#cap-mk-start').value) || 0;
        const p = new URLSearchParams({
            dictionary: 'DICT_APRILTAG_36h11', start, count, size_mm: size,
        });
        window.open(`/api/v1/ar/markers.pdf?${p.toString()}`, '_blank');
    }

    _onProfile(name) {
        this._profile = this._profiles.find(p => p.name === name) || null;
        this._profileData = null;
        if (name) {
            this.api.getCalibrationProfile(name).then(d => { this._profileData = d; this._validatePhotoRes(); });
        }
        this._refreshReady();
    }

    _refreshReady() {
        const ready = !!(this._filename && this._profile);
        this.container.querySelector('#cap-photo-input').disabled = !ready;
        this.container.querySelector('#cap-load-model').disabled = !(ready && this._img);
    }

    // ---------------------------------------------------------------
    // Photo
    // ---------------------------------------------------------------

    _onPhoto(e) {
        const file = e.target.files?.[0];
        if (!file) return;
        this._photoBlob = file;
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            this._img = img;
            this._scale = Math.min(1, MAX_PHOTO_W / img.naturalWidth);
            this._correspondences = [];
            this._overlay = null;
            this._drawPhoto();
            this._validatePhotoRes();
            this._refreshReady();
            this._refreshButtons();
            URL.revokeObjectURL(url);
        };
        img.src = url;
    }

    _validatePhotoRes() {
        const meta = this.container.querySelector('#cap-photo-meta');
        const status = this.container.querySelector('#cap-status');
        if (!this._img) return;
        const w = this._img.naturalWidth, h = this._img.naturalHeight;
        if (meta) meta.textContent = `${w}×${h}`;
        if (!this._profileData?.image_size) { if (status) status.textContent = ''; return; }
        const [pw, ph] = this._profileData.image_size;
        if (w !== pw || h !== ph) {
            this._resMismatch = true;
            if (status) {
                status.className = 'capture-status error';
                status.textContent = `⚠ Photo ${w}×${h} ≠ calibration ${pw}×${ph}. Intrinsics won't match — shoot at the calibrated resolution/lens. Solve is blocked.`;
            }
        } else {
            this._resMismatch = false;
            if (status) { status.className = 'capture-status ok'; status.textContent = `✓ Photo matches calibration (${pw}×${ph}).`; }
        }
        this._refreshButtons();
    }

    _drawPhoto() {
        const canvas = this.container.querySelector('#cap-photo-canvas');
        if (!canvas || !this._img) return;
        const w = Math.round(this._img.naturalWidth * this._scale);
        const h = Math.round(this._img.naturalHeight * this._scale);
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(this._img, 0, 0, w, h);

        // Correspondence markers
        ctx.lineWidth = 2;
        this._correspondences.forEach((c, i) => {
            const x = c.image[0] * this._scale, y = c.image[1] * this._scale;
            ctx.strokeStyle = '#16a34a'; ctx.fillStyle = '#16a34a';
            ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.stroke();
            ctx.font = '12px sans-serif';
            ctx.fillText(String(c.anchor), x + 7, y - 7);
        });

        // Overlay (projected CAD edges)
        if (this._overlay) {
            ctx.strokeStyle = 'rgba(220,38,38,0.9)';
            ctx.lineWidth = 1.5;
            for (const poly of this._overlay) {
                ctx.beginPath();
                poly.forEach((p, i) => {
                    const x = p[0] * this._scale, y = p[1] * this._scale;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }
        }
    }

    _onPhotoClick(e) {
        if (this._selectedAnchor == null || !this._img) {
            this._setStatus('Pick an anchor on the 3D model first, then click its spot on the photo.');
            return;
        }
        const canvas = e.currentTarget;
        const rect = canvas.getBoundingClientRect();
        const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
        const cy = (e.clientY - rect.top) * (canvas.height / rect.height);
        const u = cx / this._scale, v = cy / this._scale;   // natural-res pixels

        const world = this._geometry.vertices[this._selectedAnchor];
        // Replace any existing correspondence for this anchor.
        this._correspondences = this._correspondences.filter(c => c.anchor !== this._selectedAnchor);
        this._correspondences.push({ anchor: this._selectedAnchor, world, image: [u, v] });
        this._markAnchorUsed(this._selectedAnchor);
        this._selectedAnchor = null;
        this._overlay = null;
        this._drawPhoto();
        this._refreshButtons();
        this._setStatus(`Added anchor ${this._correspondences[this._correspondences.length - 1].anchor}. ${this._correspondences.length} correspondence(s).`);
    }

    // ---------------------------------------------------------------
    // 3D model
    // ---------------------------------------------------------------

    async _loadModel() {
        this._setStatus('Extracting CAD edges…');
        try {
            this._geometry = await this.api.getArGeometry(this._filename);
        } catch (err) {
            this._setStatus(`Failed to load model: ${err?.detail || err?.message || err}`, true);
            return;
        }
        this._build3D();
        this._setStatus(`Model loaded: ${this._geometry.summary.edges} edges, ${this._geometry.summary.vertices} anchors. Click an anchor to start.`);
    }

    _build3D() {
        this._disposeThree();
        const host = this.container.querySelector('#cap-3d');
        host.innerHTML = '';
        const W = host.clientWidth || 360, H = 360;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111418);
        const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 1e7);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(W, H);
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        host.appendChild(renderer.domElement);

        const { bbox, edges, vertices } = this._geometry;
        const ctr = new THREE.Vector3(
            (bbox.min[0] + bbox.max[0]) / 2,
            (bbox.min[1] + bbox.max[1]) / 2,
            (bbox.min[2] + bbox.max[2]) / 2,
        );
        const diag = Math.hypot(bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1], bbox.max[2] - bbox.min[2]) || 100;

        // Wireframe edges
        const pos = [];
        for (const poly of edges) {
            for (let i = 0; i < poly.length - 1; i++) {
                pos.push(poly[i][0], poly[i][1], poly[i][2], poly[i + 1][0], poly[i + 1][1], poly[i + 1][2]);
            }
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        scene.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x6aa9ff })));

        // Anchors as clickable spheres
        this._anchorMeshes = [];
        const r = diag / 120;
        vertices.forEach((v, i) => {
            const m = new THREE.Mesh(
                new THREE.SphereGeometry(r, 12, 12),
                new THREE.MeshBasicMaterial({ color: 0xcccccc }),
            );
            m.position.set(v[0], v[1], v[2]);
            m.userData.index = i;
            scene.add(m);
            this._anchorMeshes.push(m);
        });

        camera.position.set(ctr.x + diag, ctr.y - diag, ctr.z + diag * 0.6);
        camera.lookAt(ctr);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.copy(ctr);
        controls.update();
        const render = () => renderer.render(scene, camera);
        controls.addEventListener('change', render);

        // Click-to-pick (distinguish from orbit drag via movement threshold)
        const ray = new THREE.Raycaster();
        let down = null;
        renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
        renderer.domElement.addEventListener('pointerup', (e) => {
            if (!down) return;
            const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]);
            down = null;
            if (moved > 5) return;  // it was a drag
            const rect = renderer.domElement.getBoundingClientRect();
            const ndc = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1,
            );
            ray.setFromCamera(ndc, camera);
            const hit = ray.intersectObjects(this._anchorMeshes)[0];
            if (hit) { this._selectAnchor(hit.object.userData.index); render(); }
        });

        this._three = { scene, camera, renderer, controls, render };
        render();
    }

    _selectAnchor(index) {
        this._selectedAnchor = index;
        this._anchorMeshes.forEach((m, i) => {
            const used = this._correspondences.some(c => c.anchor === i);
            m.material.color.setHex(i === index ? 0xffd400 : used ? 0x16a34a : 0xcccccc);
        });
        this._setStatus(`Anchor ${index} selected — now click its location on the photo.`);
    }

    _markAnchorUsed(index) {
        const m = this._anchorMeshes[index];
        if (m) m.material.color.setHex(0x16a34a);
        this._three?.render();
    }

    // ---------------------------------------------------------------
    // Solve / save
    // ---------------------------------------------------------------

    _refreshButtons() {
        const n = this._correspondences.length;
        const $ = (s) => this.container.querySelector(s);
        $('#cap-corr-count').textContent = `${n} correspondence${n === 1 ? '' : 's'}`;
        $('#cap-undo').disabled = n === 0;
        $('#cap-clear').disabled = n === 0;
        $('#cap-solve').disabled = !(n >= 4 && this._img && !this._resMismatch);
        $('#cap-save').disabled = !this._lastSolve;
    }

    _undo() {
        const last = this._correspondences.pop();
        if (last) { const m = this._anchorMeshes[last.anchor]; if (m) m.material.color.setHex(0xcccccc); this._three?.render(); }
        this._overlay = null; this._drawPhoto(); this._refreshButtons();
    }

    _clearCorr() {
        this._correspondences = [];
        this._anchorMeshes.forEach(m => m.material.color.setHex(0xcccccc));
        this._three?.render();
        this._overlay = null; this._lastSolve = null; this._drawPhoto(); this._refreshButtons();
    }

    async _solve() {
        const corr = this._correspondences.map(c => ({ image: c.image, world: c.world }));
        this._setStatus('Solving pose…');
        try {
            const res = await this.api.solveAr(this._filename, this._profile.name, corr);
            this._overlay = res.overlay;
            this._lastSolve = res;
            this._drawPhoto();
            const rms = res.reproj_rms;
            const quality = rms < 2 ? 'excellent' : rms < 6 ? 'good' : rms < 15 ? 'fair' : 'poor';
            this.container.querySelector('#cap-result').innerHTML =
                `<span class="capture-rms-${quality}">RMS ${rms}px (${quality})</span> · cam ${res.camera_position.join(', ')} mm`;
            this._setStatus('Overlay drawn — do the red CAD edges land on the steel?');
            this._refreshButtons();
        } catch (err) {
            this._setStatus(`Solve failed: ${err?.detail || err?.message || err}`, true);
        }
    }

    async _save() {
        if (!this._lastSolve || !this._photoBlob) return;
        try {
            const resp = await this.api.saveCapture(this._filename, this._photoBlob, {
                profile: this._profile.name,
                correspondences: this._correspondences,
                pose: { rvec: this._lastSolve.rvec, tvec: this._lastSolve.tvec, camera_position: this._lastSolve.camera_position },
                reprojRms: this._lastSolve.reproj_rms,
            });
            this._setStatus(`Capture saved (${resp.capture_id}).`);
        } catch (err) {
            this._setStatus(`Save failed: ${err?.detail || err?.message || err}`, true);
        }
    }

    // ---------------------------------------------------------------
    // Utilities
    // ---------------------------------------------------------------

    _setStatus(msg, isError = false) {
        const el = this.container?.querySelector('#cap-status');
        if (el) { el.textContent = msg; el.className = `capture-status${isError ? ' error' : ''}`; }
    }

    _disposeThree() {
        if (!this._three) return;
        try {
            this._three.controls.dispose();
            this._three.renderer.dispose();
            this._three.renderer.domElement.remove();
        } catch { /* ignore */ }
        this._three = null;
        this._anchorMeshes = [];
    }

    _esc(str) {
        const el = document.createElement('span');
        el.textContent = str;
        return el.innerHTML;
    }
}
