/**
 * AR Fit page - run the multi-view CAD pose fit and, above all, LOOK at the result.
 *
 * The overlays are the deliverable on this screen, not the RMS. A collapsed or mis-seated pose
 * scores a low residual quite happily - that has happened repeatedly on this rig - so the
 * pictures come first, the numbers second, and the degenerate/visibility flags are shown as
 * warnings rather than buried in a JSON blob.
 */
const POLL_MS = 2000;

export class ArFitPage {
    constructor(api) {
        this.api = api;
        this.sources = null;
        this.taskId = null;
        this.timer = null;
        this.container = null;
    }

    render(container) {
        this.container = container;
        container.innerHTML = `
            <hgroup>
                <h2>AR Fit</h2>
                <p>Fit the CAD model to a captured pair and check the overlay.</p>
            </hgroup>
            <div id="arfit-form"><p aria-busy="true">Loading sources...</p></div>
            <div id="arfit-status"></div>
            <div id="arfit-result"></div>
        `;
        this._loadSources();
    }

    destroy() { this._stopPolling(); }

    _stopPolling() {
        if (this.timer) { clearInterval(this.timer); this.timer = null; }
    }

    async _loadSources() {
        const el = document.getElementById('arfit-form');
        try {
            const r = await fetch('/api/v1/ar-fit/sources');
            if (!r.ok) throw new Error(`sources ${r.status}`);
            this.sources = await r.json();
        } catch (e) {
            el.innerHTML = `<article class="error">Could not load sources: ${e.message}</article>`;
            return;
        }
        this._renderForm();
    }

    _renderForm() {
        const s = this.sources;
        const el = document.getElementById('arfit-form');
        if (!s.captures.length) {
            el.innerHTML = `<article>No capture sets found in <code>outputs/ar_captures/</code>.
                Copy a pair of board-in-shot photos there first.</article>`;
            return;
        }
        const opts = (arr, val, label) => arr.map(x =>
            `<option value="${val(x)}">${label(x)}</option>`).join('');

        el.innerHTML = `
        <form id="arfit-run">
          <div class="grid">
            <label>Capture set
              <select name="captures" id="arfit-captures">
                ${opts(s.captures, c => c.name,
                       c => `${c.name} (${c.images} image${c.images === 1 ? '' : 's'})` +
                            (c.looks_like_calibration ? ' - calibration set, not for fitting' : ''))}
              </select>
              <small id="arfit-capture-note"></small>
            </label>
            <label>Model
              <select name="model">${opts(s.models, m => m.file, m => m.file)}</select>
            </label>
          </div>
          <div class="grid">
            <label>Camera A profile
              <select name="profile">
                ${opts(s.profiles, p => p.file, p => `${p.name} - RMS ${p.rms ?? '?'}px`)}
              </select>
            </label>
            <label>Camera B profile <small>(matched by the serial in the filename)</small>
              <select name="profileB">
                <option value="">- same as A -</option>
                ${opts(s.profiles, p => p.file, p => `${p.name} - RMS ${p.rms ?? '?'}px`)}
              </select>
            </label>
          </div>
          <details>
            <summary>Advanced</summary>
            <div class="grid">
              <label>Canny low <input type="number" name="canny_low" placeholder="auto"></label>
              <label>Canny high <input type="number" name="canny_high" placeholder="auto"></label>
              <label>Working margin (mm) <input type="number" name="working_margin" value="150"></label>
            </div>
            <div class="grid">
              <label>Coarse grid (mm) <input type="number" name="coarse_step" value="60"></label>
              <label>Coarse yaw (deg) <input type="number" name="coarse_yaw" value="10"></label>
            </div>
            <label><input type="checkbox" name="coarse" checked> Coarse (x, y, yaw) scan
              <small>- slower, but a wrong start converges confidently to nonsense</small></label>
            <label><input type="checkbox" name="full_6dof"> Solve all 6 DOF
              <small>- default is planar; the part rests on the board plane, so its height and
              tilt are known</small></label>
          </details>
          <button type="submit">Run fit</button>
        </form>`;

        // A fit wants a capture pair. Selecting a 36-image calibration set starts a scan that
        // takes many minutes and cannot produce a useful answer, so say so before it is run.
        const sel = document.getElementById('arfit-captures');
        const note = document.getElementById('arfit-capture-note');
        const updateNote = () => {
            const c = s.captures.find(x => x.name === sel.value);
            if (c && c.looks_like_calibration) {
                note.innerHTML = `<mark>${c.images} images - this looks like a calibration set.
                    A fit needs the 2-4 photos of one arrangement; scanning this many is slow
                    and will not give a better pose.</mark>`;
            } else {
                note.textContent = '';
            }
        };
        sel.addEventListener('change', updateNote);
        updateNote();

        document.getElementById('arfit-run').addEventListener('submit', e => {
            e.preventDefault();
            const c = s.captures.find(x => x.name === sel.value);
            if (c && c.looks_like_calibration &&
                !confirm(`"${c.name}" has ${c.images} images and looks like a calibration set. `
                       + `A fit normally uses 2-4. This will take several minutes. Run anyway?`)) {
                return;
            }
            this._run(new FormData(e.target));
        });
    }

    async _run(fd) {
        const num = k => {
            const v = fd.get(k);
            return v === null || v === '' ? null : Number(v);
        };
        const camB = fd.get('profileB');
        const body = {
            captures: fd.get('captures'),
            profile: fd.get('profile'),
            model: fd.get('model'),
            coarse: fd.get('coarse') === 'on',
            full_6dof: fd.get('full_6dof') === 'on',
            coarse_step: num('coarse_step') ?? 60,
            coarse_yaw: num('coarse_yaw') ?? 10,
            working_margin: num('working_margin') ?? 150,
            canny_low: num('canny_low'),
            canny_high: num('canny_high'),
        };
        // The serial before the timestamp in the filename is what picks the profile, so derive
        // the tag from the profile name rather than asking the user to retype it.
        if (camB) {
            const tag = camB.replace(/\.json$/, '').split('_').pop();
            body.cam_profiles = [`${tag}=${camB}`];
        }

        const status = document.getElementById('arfit-status');
        document.getElementById('arfit-result').innerHTML = '';
        status.innerHTML = `<article aria-busy="true">Starting fit...</article>`;
        try {
            const r = await fetch('/api/v1/ar-fit/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const j = await r.json();
            if (!r.ok) throw new Error(j.detail || `run ${r.status}`);
            this.taskId = j.task_id;
            this.outName = j.out;
            this.lastRequest = {
                captures: body.captures, profile: body.profile, model: body.model,
                cam_profiles: body.cam_profiles, canny_low: body.canny_low,
                canny_high: body.canny_high, working_margin: body.working_margin,
            };
            this._poll();
        } catch (e) {
            status.innerHTML = `<article class="error">Failed to start: ${e.message}</article>`;
        }
    }

    _poll() {
        this._stopPolling();
        const status = document.getElementById('arfit-status');
        let seconds = 0;
        const tick = async () => {
            seconds += POLL_MS / 1000;
            try {
                const r = await fetch(`/api/v1/ar-fit/status/${this.taskId}`);
                const t = await r.json();
                if (t.status === 'completed') {
                    this._stopPolling();
                    status.innerHTML = `<article><strong>Fit complete</strong> in ${seconds | 0}s</article>`;
                    this._showResult(t.results || t.result);
                } else if (t.status === 'failed') {
                    this._stopPolling();
                    status.innerHTML = `<article class="error"><strong>Fit failed</strong><br>
                        <small>${t.error || 'unknown error'}</small></article>`;
                } else {
                    status.innerHTML = `<article aria-busy="true">Fitting (${seconds | 0}s)
                        - a coarse scan takes a couple of minutes</article>`;
                }
            } catch (e) {
                this._stopPolling();
                status.innerHTML = `<article class="error">Lost the task: ${e.message}</article>`;
            }
        };
        this.timer = setInterval(tick, POLL_MS);
        tick();
    }

    async _adjust(spec) {
        const [axis, degrees] = spec.split(',');
        const status = document.getElementById('arfit-status');
        status.innerHTML = `<article aria-busy="true">Re-solving at ${degrees}&deg; about
            ${axis.toUpperCase()}...</article>`;
        try {
            const r = await fetch('/api/v1/ar-fit/adjust', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...this.lastRequest, name: this.outName,
                                       axis, degrees: Number(degrees) }),
            });
            const j = await r.json();
            if (!r.ok) throw new Error(j.detail || `adjust ${r.status}`);
            this.taskId = j.task_id;
            this._poll();
        } catch (e) {
            status.innerHTML = `<article class="error">Adjust failed: ${e.message}</article>`;
        }
    }

    _bindRotation() {
        document.querySelectorAll('[data-rot]').forEach(b =>
            b.addEventListener('click', () => this._adjust(b.dataset.rot)));
    }

    _showResult(res) {
        const el = document.getElementById('arfit-result');
        if (!res) { el.innerHTML = '<article>No result payload.</article>'; return; }
        const info = res.info || {};
        const visible = (info.visible_fraction ?? 1) * 100;
        const r3 = a => (a || []).map(x => Number(x).toFixed(3)).join(', ');
        const r1 = a => (a || []).map(x => Number(x).toFixed(1)).join(', ');

        const warnings = [];
        if (info.degenerate) {
            warnings.push(`<strong>Degenerate fit.</strong> ${info.degenerate}`);
        }
        if (visible < 90) {
            warnings.push(`Only <strong>${visible.toFixed(1)}%</strong> of CAD points project
                inside the images - part of the model is off-frame, so the residual is computed
                on less than the whole part.`);
        }
        // A repetitive structure - a truss, a ladder frame - is genuinely ambiguous to an edge
        // cost: flipped end-for-end or slid along by a bay it looks nearly the same. Measured on
        // this rig, two poses 180 deg apart differed by ~3px on a ~23px baseline. Say so, rather
        // than letting a plausible-looking RMS imply more certainty than exists.
        if (res.ambiguity_px !== undefined && res.ambiguity_px < 8) {
            warnings.push(`<strong>The two orientations are ${res.ambiguity_px} px apart.</strong> `
                + 'That is not enough to choose between them. Compare the overlays below against '
                + 'a feature that is only at one end - an end plate, a bracket - and take the one '
                + 'that matches.');
        }
        warnings.push('<strong>Pose ambiguity on repetitive parts.</strong> An edge cost cannot '
            + 'reliably localise a regular truss along its own axis, or tell it end-for-end: '
            + 'those poses score almost identically. Check the overlay against a known feature '
            + '(an end plate, a bracket) rather than trusting the RMS.');
        const warnHtml = warnings.length
            ? `<article class="error">${warnings.join('<br><br>')}</article>` : '';

        // Cache-bust. A re-solve writes the SAME overlay filenames, so without this the browser
        // serves the previous image from cache: the numbers update and the picture does not,
        // which looks exactly like the button having done nothing.
        const stamp = Date.now();
        const grid = list => (list || []).map(p => `
            <figure style="margin:0">
              <img src="/outputs/${p}?v=${stamp}" alt="overlay" style="width:100%">
              <figcaption><small>${p.split('/').pop()}</small></figcaption>
            </figure>`).join('');
        const overlays = grid(res.overlays);

        // On a repetitive part the two orientations score within a couple of pixels, so the
        // solver has no basis to choose. Show both at equal weight and let the eye settle it -
        // an end plate or bracket makes it obvious in a second.
        const alts = (res.alternatives || []).map(a => `
          <article>
            <header><strong>Alternative: ${a.label}</strong> &mdash; RMS
              <strong>${a.rms_after_px} px</strong> (per view ${a.per_view_rms_px.join(', ')})
              ${a.yaw_deg !== undefined ? `, yaw ${a.yaw_deg}&deg;` : ''}</header>
            ${grid(a.overlays)}
          </article>`).join('');

        el.innerHTML = `
          ${warnHtml}
          <article>
            <header><strong>${res.init_source && res.init_source.startsWith('rotated')
                ? res.init_source.replace(/^rotated/, 'Seating: rotated') + ' &mdash; '
                : ''}Check the overlay before the numbers.</strong>
              Amber is the starting guess, cyan the fit, magenta the edge pixels it matched
              against. A low RMS at a visibly wrong pose is a wrong pose.</header>
            ${overlays || '<p>No overlay images.</p>'}
          </article>
          <article>
            <header><strong>Seating</strong> &mdash; roll about a part's own axis is not
              reliably recoverable from two views (only ~3% of this part's geometry differs
              under it, and the hole pattern that would settle it is below the cameras'
              resolution). Spin it to match, and the pose is re-solved around that seating.
            </header>
            <div role="group">
              <button class="secondary" data-rot="x,90">Rotate X 90&deg;</button>
              <button class="secondary" data-rot="y,90">Rotate Y 90&deg;</button>
              <button class="secondary" data-rot="y,180">Flip Y 180&deg;</button>
              <button class="secondary" data-rot="z,90">Rotate Z 90&deg;</button>
            </div>
            <small>Y is the part's long axis. Re-solving takes a few seconds &mdash; no coarse
              search, just a refit from the rotated seating.</small>
          </article>
          ${alts}
          <article>
            <table>
              <tbody>
                <tr><th>RMS</th><td>${info.rms_before_px} &rarr;
                    <strong>${info.rms_after_px} px</strong></td></tr>
                <tr><th>Per view</th><td>${r1(info.per_view_rms_px)} px</td></tr>
                <tr><th>Visible</th><td>${visible.toFixed(1)}% of ${info.n_points} CAD points</td></tr>
                <tr><th>Mode</th><td>${info.mode || '6-DOF'}${
                    info.yaw_deg !== undefined ? ` (yaw ${info.yaw_deg}&deg;)` : ''}</td></tr>
                <tr><th>rvec</th><td><code>${r3(res.rvec)}</code></td></tr>
                <tr><th>tvec</th><td><code>${r1(res.tvec)} mm</code> (board frame)</td></tr>
                <tr><th>Init</th><td>${res.init_source || '?'}</td></tr>
                <tr><th>Hidden lines</th><td>${res.hidden_line_removal
                    ? `removed via <code>${res.mesh}</code>`
                    : '<mark>NOT removed - no .stl beside the model. The fit is matching '
                      + 'far-side edges no camera can see.</mark>'}</td></tr>
                ${(res.visibility_rounds || []).length ? `<tr><th>Visibility</th><td>` +
                    res.visibility_rounds.map((h, i) =>
                        `round ${i + 1}: ${h.visible_per_view.join(' / ')} pts visible ` +
                        `(${(h.visible_fraction * 100).toFixed(0)}%), RMS ${h.rms_after_px}`
                    ).join('<br>') + `</td></tr>` : ''}
              </tbody>
            </table>
            ${(res.failures || []).length
                ? `<small>Skipped: ${res.failures.map(f => `${f.photo} (${f.error})`).join('; ')}</small>`
                : ''}
          </article>`;
        this._bindRotation();
    }
}
