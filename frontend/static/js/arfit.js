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
              <select name="captures">
                ${opts(s.captures, c => c.name, c => `${c.name} (${c.images} images)`)}
              </select>
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

        document.getElementById('arfit-run').addEventListener('submit', e => {
            e.preventDefault();
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
        const warnHtml = warnings.length
            ? `<article class="error">${warnings.join('<br><br>')}</article>` : '';

        const overlays = (res.overlays || []).map(p => `
            <figure>
              <img src="/outputs/${p}" alt="overlay" style="width:100%">
              <figcaption><small>${p.split('/').pop()}</small></figcaption>
            </figure>`).join('');

        el.innerHTML = `
          ${warnHtml}
          <article>
            <header><strong>Check the overlay before the numbers.</strong>
              Amber is the starting guess, cyan the fit, magenta the edge pixels it matched
              against. A low RMS at a visibly wrong pose is a wrong pose.</header>
            ${overlays || '<p>No overlay images.</p>'}
          </article>
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
              </tbody>
            </table>
            ${(res.failures || []).length
                ? `<small>Skipped: ${res.failures.map(f => `${f.photo} (${f.error})`).join('; ')}</small>`
                : ''}
          </article>`;
    }
}
