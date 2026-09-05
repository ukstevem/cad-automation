/**
 * Calibrate page — camera intrinsics via a ChArUco board.
 *
 * Capture board views two ways: live webcam (getUserMedia, laptop/USB cams) or
 * uploaded image set (any camera, incl. IP/CCTV the browser can't open). Each kept
 * frame is checked server-side for corner coverage; the full set is then sent to
 * /calibration/compute, which saves a named camera profile for the AR overlay tool.
 *
 * Note: getUserMedia only works in a secure context — http://localhost or HTTPS.
 * Over a plain-IP origin the live capture is disabled; use the upload path instead.
 */

export class CalibratePage {
    constructor(api) {
        this.api = api;
        this.container = null;

        this._board = null;            // current board params
        this._frames = [];             // [{ blob, url, corners, ok, name }]
        this._stream = null;           // active MediaStream
        this._videoDevices = [];
    }

    render(container) {
        this.container = container;
        this._cleanup();
        container.innerHTML = '<p aria-busy="true">Loading calibration…</p>';
        this._init();
    }

    async _init() {
        try {
            const defaults = await this.api.getBoardDefaults();
            // Restore the user's last board config so the dictionary/size don't
            // silently revert to defaults on navigation (a mismatch = zero detection).
            this._board = this._loadSavedBoard() || { ...defaults.default_board };
            this._dictionaries = defaults.dictionaries;
            this._minViews = defaults.min_views;
            this._recommendedViews = defaults.recommended_views;
            this._opencvAvailable = defaults.opencv_available;
        } catch (err) {
            this.container.innerHTML =
                `<p class="error">Could not load calibration config: ${this._esc(String(err?.message || err))}</p>`;
            return;
        }
        this.container.innerHTML = this._template();
        this._bindEvents();
        this._loadProfiles();
    }

    _cleanup() {
        this._stopStream();
        for (const f of this._frames) {
            if (f.url) URL.revokeObjectURL(f.url);
        }
        this._frames = [];
    }

    _stopStream() {
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
            this._stream = null;
        }
    }

    // ---------------------------------------------------------------
    // Template
    // ---------------------------------------------------------------

    _template() {
        const b = this._board;
        const secure = window.isSecureContext;
        const dictOptions = this._dictionaries
            .map(d => `<option value="${d}" ${d === b.dictionary ? 'selected' : ''}>${d}</option>`)
            .join('');

        return `
            <section class="calibrate-page">
                <h2>Camera Calibration</h2>
                <p class="calibrate-intro">
                    Compute a camera's intrinsics from ChArUco board views. Print the board,
                    capture it from ~${this._recommendedViews} angles (or upload an image set),
                    then calibrate and save a named camera profile.
                </p>

                ${this._opencvAvailable ? '' : `
                    <p class="calibrate-warn">⚠ OpenCV is not installed in the running container.
                    Rebuild the image (<code>docker compose up -d --build</code>) to enable calibration.</p>`}

                <article class="calibrate-board-config">
                    <header><strong>1. Board</strong></header>
                    <div class="calibrate-board-grid">
                        <label>Squares X<input type="number" id="cal-sx" min="3" step="1" value="${b.squares_x}"></label>
                        <label>Squares Y<input type="number" id="cal-sy" min="3" step="1" value="${b.squares_y}"></label>
                        <label>Square (mm)<input type="number" id="cal-square" min="1" step="0.1" value="${b.square_mm}"></label>
                        <label>Marker (mm)<input type="number" id="cal-marker" min="1" step="0.1" value="${b.marker_mm}"></label>
                        <label>Dictionary<select id="cal-dict">${dictOptions}</select></label>
                    </div>
                    <div class="calibrate-board-actions">
                        <button id="cal-board-download" class="outline">Open printable board ↗</button>
                        <small>Print at <strong>100% / actual size</strong>, then measure a printed square and put
                        that measured value in <em>Square (mm)</em> before calibrating.</small>
                    </div>
                    <p class="calibrate-ruler-links">
                        Printable ruler / tape (for the scale reference in your test photos):
                        <a href="https://www.printablerulers.net/" target="_blank" rel="noopener">printablerulers.net</a>
                        ·
                        <a href="https://www.101planners.com/printable-ruler-online-ruler/" target="_blank" rel="noopener">101planners</a>.
                        Print at 100%, then check scale against a credit card (exactly <strong>85.6 mm</strong> wide).
                    </p>
                </article>

                <article class="calibrate-capture">
                    <header><strong>2. Capture views</strong></header>

                    <div class="calibrate-live ${secure ? '' : 'calibrate-disabled'}">
                        <div class="calibrate-live-controls">
                            <select id="cal-device" ${secure ? '' : 'disabled'}>
                                <option value="">${secure ? 'Select camera…' : 'Live capture needs localhost/HTTPS'}</option>
                            </select>
                            <select id="cal-res" ${secure ? '' : 'disabled'} title="Resolution is requested by this page, not set in the camera's own utility">
                                <option value="1920x1080">1920x1080</option>
                                <option value="2560x1440">2560x1440</option>
                                <option value="3840x2160">3840x2160 (4K)</option>
                            </select>
                            <button id="cal-rescan" class="outline secondary" ${secure ? '' : 'disabled'}>Rescan</button>
                            <button id="cal-start-cam" class="outline" ${secure ? '' : 'disabled'}>Start camera</button>
                            <button id="cal-capture" class="outline" disabled>Capture frame</button>
                            <button id="cal-stop-cam" class="outline secondary" disabled>Stop</button>
                        </div>
                        <video id="cal-video" autoplay playsinline muted class="calibrate-video"></video>
                        <p id="cal-live-status" class="calibrate-live-status"></p>
                    </div>

                    <div class="calibrate-upload">
                        <label>…or add image files:
                            <input type="file" id="cal-upload" accept="image/*" multiple>
                        </label>
                    </div>
                </article>

                <article class="calibrate-frames">
                    <header>
                        <strong>3. Captured views</strong>
                        <span id="cal-frame-count" class="calibrate-frame-count"></span>
                        <button id="cal-clear-frames" class="outline secondary calibrate-clear">Clear all</button>
                    </header>
                    <div id="cal-frame-grid" class="calibrate-frame-grid">
                        <p class="calibrate-empty">No views yet. Capture or upload board images above.</p>
                    </div>
                </article>

                <article class="calibrate-run">
                    <header><strong>4. Calibrate &amp; save</strong></header>
                    <div class="calibrate-run-grid">
                        <label>Profile name<input type="text" id="cal-name" placeholder="e.g. logitech-c920-bay1" required></label>
                        <label>Camera label (optional)<input type="text" id="cal-label" placeholder="e.g. Logitech C920 @ 1920x1080"></label>
                    </div>
                    <button id="cal-compute" class="calibrate-compute">Calibrate</button>
                    <div id="cal-result"></div>
                </article>

                <article class="calibrate-profiles">
                    <header><strong>Saved profiles</strong></header>
                    <div id="cal-profile-list"><p aria-busy="true">Loading…</p></div>
                </article>
            </section>
        `;
    }

    _bindEvents() {
        const $ = (id) => this.container.querySelector(id);

        // Board param changes → keep this._board in sync
        ['cal-sx', 'cal-sy', 'cal-square', 'cal-marker', 'cal-dict'].forEach(id => {
            $('#' + id)?.addEventListener('change', () => this._syncBoard());
        });

        $('#cal-board-download')?.addEventListener('click', () => {
            this._syncBoard();
            window.open(this.api.getBoardImageUrl(this._board), '_blank');
        });

        $('#cal-start-cam')?.addEventListener('click', () => this._startCamera());
        // Rescan primes camera permission first: a USB camera plugged in after page load, or
        // before access was granted, is otherwise invisible to enumerateDevices.
        $('#cal-rescan')?.addEventListener('click', async () => {
            this._setLiveStatus('Scanning for cameras...');
            await this._enumerateDevices({ prime: true });
            const n = (this._videoDevices || []).length;
            this._setLiveStatus(n ? `Found ${n} camera${n === 1 ? '' : 's'}.`
                                  : 'No cameras found.', !n);
        });
        $('#cal-stop-cam')?.addEventListener('click', () => this._stopCamera());
        $('#cal-capture')?.addEventListener('click', () => this._captureFrame());
        $('#cal-upload')?.addEventListener('change', (e) => this._onUpload(e));
        $('#cal-clear-frames')?.addEventListener('click', () => this._clearFrames());
        $('#cal-compute')?.addEventListener('click', () => this._compute());

        if (window.isSecureContext) this._enumerateDevices({ prime: true });
        this._renderFrames();
    }

    _weakViewAdvice(frame) {
        // Markers and corners fail for different reasons, and saying "coverage/lighting" for both
        // is actively misleading: a frame with every marker found but no corners is a perfectly
        // good photograph of a board whose DIMENSIONS do not match what is configured here.
        const b = this._board;
        const size = frame.imageSize ? `${frame.imageSize[0]}x${frame.imageSize[1]}` : 'unknown size';
        const cfg = `${b.squares_x}x${b.squares_y}, ${b.square_mm}mm, ${b.dictionary}`;
        if (frame.markers > 0 && frame.corners === 0) {
            const fix = frame.suggestion
                ? ` ${frame.suggestion.message} Press "Use ${frame.suggestion.squares_x}x`
                  + `${frame.suggestion.squares_y}" to correct it.`
                : ' Set Squares X/Y to match the printed board.';
            return `Board MISMATCH — found ${frame.markers} markers but 0 corners, so the `
                 + `dictionary is right and the layout is not.${fix} Sent: ${cfg} · ${size}`;
        }
        if (frame.markers === 0) {
            return `No markers found at all — check focus, that the board fills more of the frame, `
                 + `and that the dictionary matches. Sent: ${cfg} · ${size}`;
        }
        return `Weak view — ${frame.markers} markers, only ${frame.corners} corners. More angle `
             + `variety and frame-edge coverage. Sent: ${cfg} · ${size}`;
    }

    _applySuggestion(sug) {
        if (!sug) return;
        this.container.querySelector('#cal-sx').value = sug.squares_x;
        this.container.querySelector('#cal-sy').value = sug.squares_y;
        this._syncBoard();
        this._setLiveStatus(`Board set to ${sug.squares_x}x${sug.squares_y}. Recapture your views `
                            + `- the ones already taken were checked against the wrong layout.`);
    }

    _syncBoard() {
        const $ = (id) => this.container.querySelector(id);
        this._board = {
            squares_x: parseInt($('#cal-sx').value) || 5,
            squares_y: parseInt($('#cal-sy').value) || 7,
            square_mm: parseFloat($('#cal-square').value) || 30,
            marker_mm: parseFloat($('#cal-marker').value) || 22,
            dictionary: $('#cal-dict').value,
        };
        try {
            localStorage.setItem('cal-board', JSON.stringify(this._board));
        } catch { /* storage unavailable — non-fatal */ }
    }

    _loadSavedBoard() {
        try {
            const raw = localStorage.getItem('cal-board');
            const b = raw ? JSON.parse(raw) : null;
            if (b && b.dictionary && b.squares_x && b.squares_y) return b;
        } catch { /* ignore */ }
        return null;
    }

    // ---------------------------------------------------------------
    // Live webcam
    // ---------------------------------------------------------------

    async _enumerateDevices({ prime = false } = {}) {
        if (!navigator.mediaDevices?.enumerateDevices) return;
        try {
            let devices = await navigator.mediaDevices.enumerateDevices();
            let vids = devices.filter(d => d.kind === 'videoinput');

            // Until camera permission is granted the browser hides device LABELS and, on
            // Chromium, can withhold additional cameras entirely - so a USB camera simply does
            // not appear and the list shows only the built-in one. The cure is to open a stream
            // briefly, which triggers the permission prompt, then enumerate again and drop it.
            // The old code carried a comment saying this was needed and then never did it.
            if (prime && (!vids.length || vids.some(d => !d.label))) {
                try {
                    const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
                    tmp.getTracks().forEach(t => t.stop());
                    devices = await navigator.mediaDevices.enumerateDevices();
                    vids = devices.filter(d => d.kind === 'videoinput');
                } catch (err) {
                    this._setLiveStatus(
                        `Camera permission refused (${err?.name || err}). Allow camera access for ` +
                        `this site, then press Rescan.`, true);
                }
            }

            this._videoDevices = vids;
            const sel = this.container.querySelector('#cal-device');
            if (!sel) return;
            if (vids.length) {
                const keep = sel.value;
                sel.innerHTML = vids
                    .map((d, i) => `<option value="${d.deviceId}">${this._esc(d.label || `Camera ${i + 1}`)}</option>`)
                    .join('');
                if (keep && vids.some(d => d.deviceId === keep)) sel.value = keep;
            }
            if (vids.length && vids.every(d => !d.label)) {
                this._setLiveStatus('Cameras found but unnamed - press Rescan and allow access ' +
                                    'to see which is which.', true);
            }
        } catch { /* ignore */ }
    }

    async _startCamera() {
        this._syncBoard();
        this._stopStream();
        const sel = this.container.querySelector('#cal-device');
        const deviceId = sel?.value || undefined;
        // Resolution is a property of the CAPTURE REQUEST, not of the camera - a camera utility
        // like Logi Tune sets field of view, focus and exposure, but the frame size comes from
        // whoever opens the stream. Requested as 'ideal' rather than 'exact' so an unsupported
        // size degrades instead of failing outright; the delivered size is reported below, and a
        // profile records whatever actually arrived.
        const resSel = this.container.querySelector('#cal-res');
        const [rw, rh] = (resSel?.value || '1920x1080').split('x').map(Number);
        const constraints = {
            video: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                width: { ideal: rw },
                height: { ideal: rh },
            },
            audio: false,
        };
        try {
            this._stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (err) {
            this._setLiveStatus(`Camera error: ${err?.message || err}`, true);
            return;
        }
        const video = this.container.querySelector('#cal-video');
        video.srcObject = this._stream;
        this.container.querySelector('#cal-capture').disabled = false;
        this.container.querySelector('#cal-stop-cam').disabled = false;
        this.container.querySelector('#cal-start-cam').disabled = true;
        // Show the resolution ACTUALLY delivered. width/height are requested as 'ideal', which
        // browsers treat as a hint - a camera that hands back 1280x720 would be calibrated at
        // 720p, and the resolution gate then rejects every 1080p capture at the rig.
        const track = this._stream.getVideoTracks()[0];
        const st = track?.getSettings?.() || {};
        const res = (st.width && st.height) ? `${st.width}x${st.height}` : 'unknown resolution';
        const warn = (st.width && st.width < rw);
        this._setLiveStatus(
            `Camera live at ${res}${warn ? ` - LOWER than the ${rw}x${rh} requested; the profile `
            + 'will record what arrived, and captures must later match it' : ''} - point it at the `
            + 'board and capture from several angles.', warn);
        // Labels may now be available; refresh the device list.
        this._enumerateDevices();
    }

    _stopCamera() {
        this._stopStream();
        const video = this.container.querySelector('#cal-video');
        if (video) video.srcObject = null;
        this.container.querySelector('#cal-capture').disabled = true;
        this.container.querySelector('#cal-stop-cam').disabled = true;
        this.container.querySelector('#cal-start-cam').disabled = false;
        this._setLiveStatus('');
    }

    async _captureFrame() {
        const video = this.container.querySelector('#cal-video');
        if (!video || !video.videoWidth) return;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.95));
        if (!blob) return;
        blob.name = `cap_${Date.now()}.jpg`;
        this._setLiveStatus('Checking frame…');
        await this._addFrame(blob);
    }

    _setLiveStatus(msg, isError = false) {
        const el = this.container.querySelector('#cal-live-status');
        if (!el) return;
        el.textContent = msg;
        el.classList.toggle('error', isError);
    }

    // ---------------------------------------------------------------
    // Upload
    // ---------------------------------------------------------------

    async _onUpload(e) {
        const files = [...(e.target.files || [])];
        e.target.value = '';
        for (const f of files) {
            await this._addFrame(f);
        }
    }

    // ---------------------------------------------------------------
    // Frame set
    // ---------------------------------------------------------------

    async _addFrame(blob) {
        this._syncBoard();
        let detection = null;
        try {
            detection = await this.api.detectBoard(blob, this._board);
        } catch (err) {
            detection = { detected: false, corners: 0, error: err?.detail || err?.message };
        }
        const frame = {
            blob,
            url: URL.createObjectURL(blob),
            corners: detection.corners || 0,
            markers: detection.markers || 0,
            imageSize: detection.image_size || null,
            suggestion: detection.suggestion || null,
            ok: !!detection.detected,
            name: blob.name || 'image',
        };
        this._frames.push(frame);
        this._renderFrames();
        if (this._stream) {
            this._setLiveStatus(
                frame.ok
                    ? `Kept — ${frame.corners} corners. ${this._frames.filter(f => f.ok).length} good view(s) so far.`
                    : this._weakViewAdvice(frame),
                !frame.ok
            );
        }
    }

    _clearFrames() {
        for (const f of this._frames) if (f.url) URL.revokeObjectURL(f.url);
        this._frames = [];
        this._renderFrames();
    }

    _renderFrames() {
        const grid = this.container.querySelector('#cal-frame-grid');
        const countEl = this.container.querySelector('#cal-frame-count');
        if (!grid) return;

        const good = this._frames.filter(f => f.ok).length;
        if (countEl) {
            countEl.textContent = this._frames.length
                ? `${good} good / ${this._frames.length} total (need ≥ ${this._minViews})`
                : '';
        }

        if (!this._frames.length) {
            grid.innerHTML = '<p class="calibrate-empty">No views yet. Capture or upload board images above.</p>';
            return;
        }

        grid.innerHTML = this._frames.map((f, i) => `
            <div class="calibrate-thumb ${f.ok ? 'good' : 'weak'}" data-i="${i}">
                <img src="${f.url}" alt="view ${i + 1}">
                <span class="calibrate-thumb-badge">${f.corners}</span>
                <button class="calibrate-thumb-remove" data-i="${i}" title="Remove">×</button>
            </div>
        `).join('');

        grid.querySelectorAll('.calibrate-thumb-remove').forEach(btn => {
            btn.addEventListener('click', () => this._removeFrame(parseInt(btn.dataset.i)));
        });
    }

    _removeFrame(i) {
        const f = this._frames[i];
        if (f?.url) URL.revokeObjectURL(f.url);
        this._frames.splice(i, 1);
        this._renderFrames();
    }

    // ---------------------------------------------------------------
    // Compute
    // ---------------------------------------------------------------

    async _compute() {
        this._syncBoard();
        const name = this.container.querySelector('#cal-name').value.trim();
        const label = this.container.querySelector('#cal-label').value.trim();
        const resultEl = this.container.querySelector('#cal-result');

        if (!name) {
            resultEl.innerHTML = '<p class="error">Enter a profile name first.</p>';
            return;
        }
        const usable = this._frames.filter(f => f.ok);
        if (usable.length < this._minViews) {
            resultEl.innerHTML = `<p class="error">Need at least ${this._minViews} good views (have ${usable.length}).</p>`;
            return;
        }

        const btn = this.container.querySelector('#cal-compute');
        btn.disabled = true;
        btn.setAttribute('aria-busy', 'true');
        btn.textContent = 'Calibrating…';
        resultEl.innerHTML = '';

        try {
            // Send all frames (server re-detects and skips weak ones).
            const blobs = this._frames.map(f => {
                const b = f.blob;
                if (!b.name) b.name = f.name;
                return b;
            });
            const profile = await this.api.computeCalibration(blobs, name, this._board, label);
            this._renderResult(profile);
            this._loadProfiles();
        } catch (err) {
            resultEl.innerHTML = `<p class="error">Calibration failed: ${this._esc(String(err?.detail || err?.message || err))}</p>`;
        } finally {
            btn.disabled = false;
            btn.removeAttribute('aria-busy');
            btn.textContent = 'Calibrate';
        }
    }

    _renderResult(p) {
        const resultEl = this.container.querySelector('#cal-result');
        const k = p.intrinsics || {};
        const rms = p.rms_reproj_error_px;
        // Absolute-pixel RMS isn't comparable across resolutions: 2 px on a 36 MP
        // phone image is proportionally far better than 2 px at 1080p. Normalise to
        // a 1920-px-wide reference before bucketing quality.
        const width = (p.image_size && p.image_size[0]) || 1920;
        const normRms = rms * (1920 / width);
        const quality = normRms < 0.5 ? 'excellent' : normRms < 1.0 ? 'good' : normRms < 2.0 ? 'fair' : 'poor';
        const scaleNote = width > 2200
            ? ` (≈${normRms.toFixed(2)} px normalised to 1080p)` : '';
        resultEl.innerHTML = `
            <div class="calibrate-result calibrate-result-${quality}">
                <p><strong>Saved “${this._esc(p.name)}”.</strong>
                   RMS reprojection error <strong>${rms} px</strong>${scaleNote} (${quality}) ·
                   ${p.views_used}/${p.views_total} views used ·
                   ${p.image_size[0]}×${p.image_size[1]}</p>
                <table class="calibrate-intrinsics">
                    <tr><td>fx</td><td>${k.fx}</td><td>fy</td><td>${k.fy}</td></tr>
                    <tr><td>cx</td><td>${k.cx}</td><td>cy</td><td>${k.cy}</td></tr>
                </table>
                <small>${quality === 'fair' || quality === 'poor' ? 'Tip: drop weak views and recapture with more angle variety and frame-edge coverage for a lower error.' : 'Good result — ready for the overlay tool.'}</small>
            </div>
        `;
    }

    // ---------------------------------------------------------------
    // Saved profiles
    // ---------------------------------------------------------------

    async _loadProfiles() {
        const el = this.container.querySelector('#cal-profile-list');
        if (!el) return;
        try {
            const resp = await this.api.listCalibrationProfiles();
            const profiles = resp.profiles || [];
            if (!profiles.length) {
                el.innerHTML = '<p class="calibrate-empty">No saved profiles yet.</p>';
                return;
            }
            el.innerHTML = profiles.map(p => `
                <div class="calibrate-profile-row" data-name="${this._esc(p.name)}">
                    <div class="calibrate-profile-main">
                        <strong>${this._esc(p.name)}</strong>
                        ${p.camera_label ? `<span class="calibrate-profile-label">${this._esc(p.camera_label)}</span>` : ''}
                    </div>
                    <div class="calibrate-profile-meta">
                        RMS ${p.rms_reproj_error_px ?? '?'} px ·
                        ${p.image_size ? p.image_size.join('×') : '?'} ·
                        ${p.views_used ?? '?'} views ·
                        ${this._formatDate(p.created_at)}
                    </div>
                    <button class="calibrate-profile-delete outline secondary" data-name="${this._esc(p.name)}" title="Delete">Delete</button>
                </div>
            `).join('');

            el.querySelectorAll('.calibrate-profile-delete').forEach(btn => {
                btn.addEventListener('click', () => this._deleteProfile(btn.dataset.name));
            });
        } catch (err) {
            el.innerHTML = `<p class="error">Failed to load profiles: ${this._esc(String(err?.message || err))}</p>`;
        }
    }

    async _deleteProfile(name) {
        if (!confirm(`Delete calibration profile “${name}”?`)) return;
        try {
            await this.api.deleteCalibrationProfile(name);
            this._loadProfiles();
        } catch (err) {
            alert('Failed to delete: ' + (err?.message || err));
        }
    }

    // ---------------------------------------------------------------
    // Utilities
    // ---------------------------------------------------------------

    _esc(str) {
        const el = document.createElement('span');
        el.textContent = str;
        return el.innerHTML;
    }

    _formatDate(isoStr) {
        if (!isoStr) return '—';
        try {
            return new Date(isoStr).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
        } catch {
            return isoStr;
        }
    }
}
