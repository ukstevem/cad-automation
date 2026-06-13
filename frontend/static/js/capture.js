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

const MAX_PHOTO_W = 900;
const LOUPE_PX = 200;     // magnifier canvas size
const LOUPE_ZOOM = 8;     // magnification (shows full-res pixels)

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
        this._markers = null;          // detected markers [{id, corners, distance_mm}]
        this._conFiles = null;         // FileList for constellation calibration
        this._stream = null;           // live webcam MediaStream
        this._correspondences = [];    // [{ anchor, world:[x,y,z], image:[u,v] }]
        this._selectedAnchor = null;   // index awaiting a photo click
        this._overlayManual = null;    // red — manual-correspondence solve
        this._overlayAuto = null;      // cyan — marker auto-align (compare deviation)
        this._lastSolve = null;
        this._solving = false;         // guards against double-click on Solve
        this._saving = false;          // guards against double-click on Save
        this._saved = false;           // this solve already persisted

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
        this._stopStream();
        this._disposeThree();
        this._correspondences = [];
        this._selectedAnchor = null;
        this._overlayManual = null;
        this._overlayAuto = null;
        this._markers = null;
        this._img = null;
        this._photoBlob = null;
        this._lastSolve = null;
        this._saving = false;
        this._saved = false;
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
                    <span id="cap-mk-note" class="capture-mk-note"></span>
                </div>

                <div class="capture-setup">
                    <label>Job<select id="cap-file">${fileOpts}</select></label>
                    <label>Calibration<select id="cap-profile">${profOpts}</select></label>
                    <label>Photo (or use camera)<input type="file" id="cap-photo-input" accept="image/*" disabled></label>
                    <button id="cap-load-model" class="outline" disabled>Load model</button>
                </div>

                <div class="capture-livecam">
                    <span>Live camera:</span>
                    <select id="cap-cam-device"><option value="">default camera</option></select>
                    <button id="cap-cam-start" class="outline" disabled>Start camera</button>
                    <button id="cap-cam-capture" class="outline" disabled>Capture frame</button>
                    <button id="cap-cam-stop" class="outline secondary" disabled>Stop</button>
                    <video id="cap-cam-video" autoplay playsinline muted class="capture-cam-video"></video>
                </div>
                <p id="cap-status" class="capture-status"></p>

                <div class="capture-stage">
                    <div class="capture-pane">
                        <header>Photo <small id="cap-photo-meta"></small></header>
                        <div class="capture-photo-row">
                            <canvas id="cap-photo-canvas" class="capture-canvas"></canvas>
                            <div class="capture-loupe-box">
                                <canvas id="cap-loupe" width="${LOUPE_PX}" height="${LOUPE_PX}" class="capture-loupe"></canvas>
                                <div class="capture-loupe-cap">${LOUPE_ZOOM}× zoom — hover to place precisely</div>
                            </div>
                        </div>
                        <div class="capture-legend">
                            <span class="lg lg-red"></span> manual solve
                            <span class="lg lg-cyan"></span> auto-align
                            <span class="lg lg-blue"></span> detected marker
                        </div>
                    </div>
                    <div class="capture-pane">
                        <header>CAD model — click an anchor, then click it on the photo</header>
                        <div id="cap-3d" class="capture-3d"></div>
                    </div>
                </div>

                <div class="capture-actions">
                    <button id="cap-detect" class="outline" disabled>Detect markers</button>
                    <span id="cap-corr-count" class="capture-corr-count">0 correspondences</span>
                    <button id="cap-undo" class="outline secondary" disabled>Undo last</button>
                    <button id="cap-clear" class="outline secondary" disabled>Clear</button>
                    <button id="cap-solve" disabled>Solve &amp; Overlay</button>
                    <button id="cap-save" class="outline" disabled>Save capture</button>
                    <button id="cap-register" class="outline" disabled title="Store this marker's position from the current solve">Register marker</button>
                    <button id="cap-auto" class="outline" disabled title="Detect a registered marker and align automatically">Auto-align</button>
                    <span id="cap-result" class="capture-result"></span>
                </div>

                <article class="capture-constellation">
                    <header><strong>Constellation</strong> <small>— link many markers, anchor once, then any photo aligns</small></header>
                    <div class="capture-con-row">
                        <label>Calibration photos (≥2, close shots)
                            <input type="file" id="cap-con-files" accept="image/*" multiple>
                        </label>
                        <button id="cap-con-calibrate" class="outline" disabled>Calibrate constellation</button>
                        <button id="cap-con-anchor" class="outline" disabled title="Tie the constellation to the CAD using the current manual solve">Anchor to CAD (from current solve)</button>
                    </div>
                    <p id="cap-con-status" class="capture-con-status"></p>
                </article>
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
        $('#cap-photo-canvas').addEventListener('mousemove', (e) => this._updateLoupe(e));
        $('#cap-photo-canvas').addEventListener('mouseleave', () => this._clearLoupe());
        $('#cap-mk-print').addEventListener('click', () => this._printMarkers());
        $('#cap-detect').addEventListener('click', () => this._detectMarkers());
        $('#cap-register').addEventListener('click', () => this._registerMarker());
        $('#cap-auto').addEventListener('click', () => this._autoAlign());
        $('#cap-con-files').addEventListener('change', (e) => {
            this._conFiles = [...(e.target.files || [])];
            $('#cap-con-calibrate').disabled = this._conFiles.length < 2 || !this._profile;
        });
        $('#cap-con-calibrate').addEventListener('click', () => this._calibrateConstellation());
        $('#cap-con-anchor').addEventListener('click', () => this._anchorConstellation());
        $('#cap-cam-start').addEventListener('click', () => this._startCamera());
        $('#cap-cam-capture').addEventListener('click', () => this._captureLiveFrame());
        $('#cap-cam-stop').addEventListener('click', () => this._stopCamera());
        if (window.isSecureContext) $('#cap-cam-start').disabled = false;
    }

    _setConStatus(msg, isError = false) {
        const el = this.container?.querySelector('#cap-con-status');
        if (el) { el.textContent = msg; el.className = `capture-con-status${isError ? ' error' : ''}`; }
    }

    async _calibrateConstellation() {
        if (!this._conFiles || this._conFiles.length < 2 || !this._profile) {
            this._setConStatus('Select ≥2 photos and a calibration profile first.', true); return;
        }
        const size = parseFloat(this.container.querySelector('#cap-mk-size').value) || 100;
        const btn = this.container.querySelector('#cap-con-calibrate');
        btn.disabled = true; btn.setAttribute('aria-busy', 'true');
        this._setConStatus(`Detecting markers across ${this._conFiles.length} photos and linking…`);
        try {
            const res = await this.api.calibrateConstellation(this._filename, this._conFiles, {
                profile: this._profile.name, markerSizeMm: size,
            });
            const unl = res.unlinked.length ? ` · unlinked (need a co-visible shot): ${res.unlinked.join(', ')}` : '';
            this._setConStatus(`✓ Linked ${res.linked.length} markers: ${res.linked.join(', ')} (ref ${res.reference})${unl}. Now do a manual Solve on a wide shot, then Anchor to CAD.`);
        } catch (err) {
            this._setConStatus(`Calibrate failed: ${err?.detail || err?.message || err}`, true);
        } finally {
            btn.disabled = false; btn.removeAttribute('aria-busy');
        }
    }

    async _anchorConstellation() {
        if (!this._lastSolve || !this._photoBlob) {
            this._setConStatus('Do a manual Solve on this (wide) photo first, then Anchor.', true); return;
        }
        const size = parseFloat(this.container.querySelector('#cap-mk-size').value) || 100;
        const btn = this.container.querySelector('#cap-con-anchor');
        btn.disabled = true; btn.setAttribute('aria-busy', 'true');
        this._setConStatus('Anchoring constellation to the CAD frame…');
        try {
            const res = await this.api.anchorConstellation(this._filename, this._photoBlob, {
                profile: this._profile.name,
                modelRvec: this._lastSolve.rvec,
                modelTvec: this._lastSolve.tvec,
                markerSizeMm: size,
            });
            this._setConStatus(`✓ Anchored via marker ${res.anchored_via}. Registered ${res.count} markers: ${res.registered.join(', ')}. Auto-align now works on any photo showing these tags.`);
        } catch (err) {
            this._setConStatus(`Anchor failed: ${err?.detail || err?.message || err}`, true);
        } finally {
            btn.disabled = !(this._lastSolve && this._photoBlob);
            btn.removeAttribute('aria-busy');
        }
    }

    async _registerMarker() {
        if (!this._lastSolve || !this._photoBlob) return;
        const size = parseFloat(this.container.querySelector('#cap-mk-size').value) || 100;
        const btn = this.container.querySelector('#cap-register');
        if (btn) { btn.disabled = true; btn.setAttribute('aria-busy', 'true'); }
        this._setStatus('Registering marker (linking it to the CAD frame)…');
        try {
            const res = await this.api.registerMarker(this._filename, this._photoBlob, {
                profile: this._profile.name,
                modelRvec: this._lastSolve.rvec,
                modelTvec: this._lastSolve.tvec,
                markerSizeMm: size,
            });
            const ids = (res.registered || []).join(', ');
            this._setStatus(`✓ Registered ${res.registered.length} marker(s): ${ids}. Upload another photo showing any of them and hit Auto-align — no clicking.`);
            this.container.querySelector('#cap-auto').disabled = !(this._img && this._profile);
        } catch (err) {
            this._setStatus(`Register failed: ${err?.detail || err?.message || err}`, true);
        } finally {
            if (btn) { btn.removeAttribute('aria-busy'); btn.disabled = !this._lastSolve; }
        }
    }

    async _autoAlign() {
        if (!this._img || !this._profile || !this._photoBlob) return;
        const btn = this.container.querySelector('#cap-auto');
        if (btn) { btn.disabled = true; btn.setAttribute('aria-busy', 'true'); }
        this._setStatus('Detecting registered marker and aligning…');
        try {
            const res = await this.api.autoSolve(this._filename, this._photoBlob, { profile: this._profile.name });
            this._overlayAuto = res.overlay;
            this._markers = res.markers || [];
            this._lastSolve = null;        // auto pose isn't a manual solve
            this._drawPhoto();
            this._refreshButtons();
            const ids = (res.marker_ids_used || []).join(', ');
            this._setStatus(`✓ Auto-aligned (cyan) from ${this._markers.length} marker(s): ${ids} · RMS ${res.reproj_rms}px — no clicking.`);
        } catch (err) {
            this._setStatus(`Auto-align failed: ${err?.detail || err?.message || err}`, true);
        } finally {
            if (btn) { btn.removeAttribute('aria-busy'); btn.disabled = !(this._img && this._profile); }
        }
    }

    async _detectMarkers() {
        if (!this._img || !this._profile || !this._photoBlob) return;
        const size = parseFloat(this.container.querySelector('#cap-mk-size').value) || 100;
        const btn = this.container.querySelector('#cap-detect');
        if (btn) { btn.disabled = true; btn.setAttribute('aria-busy', 'true'); }
        this._setStatus('Detecting markers…');
        try {
            const res = await this.api.detectMarkers(this._filename, this._photoBlob, {
                profile: this._profile.name, markerSizeMm: size,
            });
            this._markers = res.markers || [];
            this._drawPhoto();
            if (this._markers.length) {
                const ids = this._markers.map(m => `ID ${m.id} @ ~${(m.distance_mm / 1000).toFixed(2)} m`).join(', ');
                this._setStatus(`✓ Detected ${this._markers.length} marker(s): ${ids}`);
            } else {
                this._setStatus('No markers detected — make sure the WHOLE tag is visible (not occluded), sharp, and well lit.', true);
            }
        } catch (err) {
            this._setStatus(`Detect failed: ${err?.detail || err?.message || err}`, true);
        } finally {
            if (btn) { btn.disabled = false; btn.removeAttribute('aria-busy'); }
        }
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
        const detectBtn = this.container.querySelector('#cap-detect');
        if (detectBtn) detectBtn.disabled = !(this._img && this._profile);
        const autoBtn = this.container.querySelector('#cap-auto');
        if (autoBtn) autoBtn.disabled = !(this._img && this._profile);
    }

    // ---------------------------------------------------------------
    // Photo
    // ---------------------------------------------------------------

    _onPhoto(e) {
        const file = e.target.files?.[0];
        if (file) this._setPhotoFromBlob(file);
    }

    _setPhotoFromBlob(blob) {
        this._photoBlob = blob;
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
            this._img = img;
            this._scale = Math.min(1, MAX_PHOTO_W / img.naturalWidth);
            this._correspondences = [];
            this._overlayManual = null;
            this._overlayAuto = null;
            this._markers = null;
            this._drawPhoto();
            this._validatePhotoRes();
            this._refreshReady();
            this._refreshButtons();
            URL.revokeObjectURL(url);
        };
        img.src = url;
    }

    // ── Live webcam (ditch the phone) ──────────────────────────────

    async _startCamera() {
        if (!window.isSecureContext) {
            this._setStatus('Live camera needs localhost or HTTPS (you\'re on a plain-IP origin).', true);
            return;
        }
        this._stopStream();
        const deviceId = this.container.querySelector('#cap-cam-device')?.value || undefined;
        try {
            this._stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: 1920 }, height: { ideal: 1080 },
                },
                audio: false,
            });
        } catch (err) {
            this._setStatus(`Camera error: ${err?.message || err}`, true);
            return;
        }
        const video = this.container.querySelector('#cap-cam-video');
        video.srcObject = this._stream;
        this.container.querySelector('#cap-cam-capture').disabled = false;
        this.container.querySelector('#cap-cam-stop').disabled = false;
        this.container.querySelector('#cap-cam-start').disabled = true;
        this._enumerateCameras();
        this._setStatus('Camera live — frame the part + markers, then Capture frame.');
    }

    _stopCamera() {
        this._stopStream();
        const v = this.container.querySelector('#cap-cam-video');
        if (v) v.srcObject = null;
        this.container.querySelector('#cap-cam-capture').disabled = true;
        this.container.querySelector('#cap-cam-stop').disabled = true;
        this.container.querySelector('#cap-cam-start').disabled = false;
    }

    async _captureLiveFrame() {
        const video = this.container.querySelector('#cap-cam-video');
        if (!video || !video.videoWidth) return;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.95));
        if (!blob) return;
        blob.name = `live_${video.videoWidth}x${video.videoHeight}.jpg`;
        this._setPhotoFromBlob(blob);
        this._setStatus(`Frame captured (${video.videoWidth}×${video.videoHeight}) — Load model, then detect / solve / auto-align.`);
    }

    _stopStream() {
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
            this._stream = null;
        }
    }

    async _enumerateCameras() {
        if (!navigator.mediaDevices?.enumerateDevices) return;
        try {
            const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
            const sel = this.container.querySelector('#cap-cam-device');
            if (sel && devs.length) {
                const cur = sel.value;
                sel.innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${this._esc(d.label || `Camera ${i + 1}`)}</option>`).join('');
                if (cur) sel.value = cur;
            }
        } catch { /* ignore */ }
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

        // Detected markers (blue outline + ID/distance)
        if (this._markers) {
            ctx.lineWidth = 2; ctx.strokeStyle = '#2563eb'; ctx.fillStyle = '#2563eb';
            for (const m of this._markers) {
                const c = m.corners.map(p => [p[0] * this._scale, p[1] * this._scale]);
                ctx.beginPath();
                c.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
                ctx.closePath(); ctx.stroke();
                // Readable label: white text on a blue pill, above the marker.
                const txt = `ID ${m.id}`;
                ctx.font = 'bold 15px sans-serif';
                const tw = ctx.measureText(txt).width;
                const lx = c[0][0], ly = c[0][1] - 6;
                ctx.fillStyle = '#2563eb';
                ctx.fillRect(lx - 3, ly - 16, tw + 8, 20);
                ctx.fillStyle = '#fff';
                ctx.fillText(txt, lx + 1, ly - 1);
                ctx.fillStyle = '#2563eb';
            }
        }

        // Projected CAD edges: red = manual solve (solid), cyan = auto-align (dashed
        // so it doesn't fully hide red when the two coincide closely).
        const drawOverlay = (polys, color, dashed) => {
            if (!polys) return;
            ctx.strokeStyle = color; ctx.lineWidth = 1.5;
            ctx.setLineDash(dashed ? [7, 5] : []);
            for (const poly of polys) {
                ctx.beginPath();
                poly.forEach((p, i) => {
                    const x = p[0] * this._scale, y = p[1] * this._scale;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }
            ctx.setLineDash([]);
        };
        drawOverlay(this._overlayManual, 'rgba(220,38,38,0.95)', false);   // solid red
        drawOverlay(this._overlayAuto, 'rgba(6,182,212,0.95)', true);      // dashed cyan
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

    _updateLoupe(e) {
        const canvas = this.container.querySelector('#cap-photo-canvas');
        const loupe = this.container.querySelector('#cap-loupe');
        if (!this._img || !loupe) return;
        const rect = canvas.getBoundingClientRect();
        const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
        const cy = (e.clientY - rect.top) * (canvas.height / rect.height);
        const nx = cx / this._scale, ny = cy / this._scale;     // natural-res coords
        const src = LOUPE_PX / LOUPE_ZOOM;                       // natural px shown
        const ctx = loupe.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, LOUPE_PX, LOUPE_PX);
        ctx.drawImage(this._img, nx - src / 2, ny - src / 2, src, src, 0, 0, LOUPE_PX, LOUPE_PX);
        ctx.strokeStyle = 'rgba(255,60,60,0.9)'; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(LOUPE_PX / 2, 0); ctx.lineTo(LOUPE_PX / 2, LOUPE_PX);
        ctx.moveTo(0, LOUPE_PX / 2); ctx.lineTo(LOUPE_PX, LOUPE_PX / 2);
        ctx.stroke();
    }

    _clearLoupe() {
        const loupe = this.container.querySelector('#cap-loupe');
        if (loupe) loupe.getContext('2d').clearRect(0, 0, LOUPE_PX, LOUPE_PX);
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
        this._updateMarkerSizing();
        this._setStatus(`Model loaded: ${this._geometry.summary.edges} edges, ${this._geometry.summary.vertices} anchors. Click an anchor to start.`);
    }

    _updateMarkerSizing() {
        const obb = this._geometry?.obb;
        const note = this.container.querySelector('#cap-mk-note');
        if (!obb || obb.length < 3) { if (note) note.textContent = ''; return; }
        const [minD, midD, maxD] = obb;
        // A marker (black square) plus its white quiet zone (~14%) must fit on the
        // face it sits on. Cap to the smallest cross-section so it fits any face,
        // with margin; round to 5 mm.
        const rec = Math.max(15, Math.floor((minD * 0.75) / 5) * 5);
        const sizeInput = this.container.querySelector('#cap-mk-size');
        if (sizeInput) sizeInput.value = rec;
        if (note) {
            note.textContent = `Element OBB ${minD}×${midD}×${maxD} mm → marker ≤ ~${rec} mm to fit the ${minD} mm face (larger faces allow bigger).`;
        }
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
        $('#cap-solve').disabled = this._solving || !(n >= 4 && this._img && !this._resMismatch);
        $('#cap-save').disabled = this._saving || this._saved || !this._lastSolve;
        $('#cap-register').disabled = !this._lastSolve;
        const anchorBtn = this.container.querySelector('#cap-con-anchor');
        if (anchorBtn) anchorBtn.disabled = !(this._lastSolve && this._photoBlob);
    }

    _undo() {
        const last = this._correspondences.pop();
        if (last) { const m = this._anchorMeshes[last.anchor]; if (m) m.material.color.setHex(0xcccccc); this._three?.render(); }
        this._overlayManual = null; this._drawPhoto(); this._refreshButtons();
    }

    _clearCorr() {
        this._correspondences = [];
        this._anchorMeshes.forEach(m => m.material.color.setHex(0xcccccc));
        this._three?.render();
        this._overlayManual = null; this._lastSolve = null; this._drawPhoto(); this._refreshButtons();
    }

    async _solve() {
        if (this._solving) return;
        this._solving = true;
        this._refreshButtons();
        const sbtn = this.container.querySelector('#cap-solve');
        if (sbtn) sbtn.setAttribute('aria-busy', 'true');
        const corr = this._correspondences.map(c => ({ image: c.image, world: c.world }));
        this._setStatus('Solving pose…');
        try {
            const res = await this.api.solveAr(this._filename, this._profile.name, corr);
            this._overlayManual = res.overlay;
            this._lastSolve = res;
            this._saved = false;        // a fresh solve can be saved again
            this._drawPhoto();
            const rms = res.reproj_rms;
            const quality = rms < 2 ? 'excellent' : rms < 6 ? 'good' : rms < 15 ? 'fair' : 'poor';
            this.container.querySelector('#cap-result').innerHTML =
                `<span class="capture-rms-${quality}">RMS ${rms}px (${quality})</span> · cam ${res.camera_position.join(', ')} mm`;
            this._setStatus('Overlay drawn — do the red CAD edges land on the steel?');
        } catch (err) {
            this._setStatus(`Solve failed: ${err?.detail || err?.message || err}`, true);
        } finally {
            this._solving = false;
            if (sbtn) sbtn.removeAttribute('aria-busy');
            this._refreshButtons();
        }
    }

    async _save() {
        if (this._saving || this._saved || !this._lastSolve || !this._photoBlob) return;
        this._saving = true;
        this._refreshButtons();
        const btn = this.container.querySelector('#cap-save');
        if (btn) { btn.setAttribute('aria-busy', 'true'); btn.textContent = 'Saving…'; }
        try {
            const resp = await this.api.saveCapture(this._filename, this._photoBlob, {
                profile: this._profile.name,
                correspondences: this._correspondences,
                pose: { rvec: this._lastSolve.rvec, tvec: this._lastSolve.tvec, camera_position: this._lastSolve.camera_position },
                reprojRms: this._lastSolve.reproj_rms,
            });
            this._saved = true;
            this._setStatus(`Capture saved (${resp.capture_id}).`);
        } catch (err) {
            this._setStatus(`Save failed: ${err?.detail || err?.message || err}`, true);
        } finally {
            this._saving = false;
            if (btn) { btn.removeAttribute('aria-busy'); btn.textContent = 'Save capture'; }
            this._refreshButtons();
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
