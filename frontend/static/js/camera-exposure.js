/**
 * Exposure control for a live camera, with a meter that says what the setting is actually doing.
 *
 * WHY A METER AND NOT JUST A SLIDER. "Improve the picture" has a hard ceiling here that has
 * nothing to do with how the picture looks: blow out the ChArUco board's white squares and
 * detection fails, which loses the world frame and with it every pose derived from it. And the
 * reason to raise exposure at all is not brightness for its own sake, it is the DARK end - creases
 * inside shadowed webs are exactly the edges that go unmeasurable, and lifting the floor is what
 * makes them carry contrast. So the two numbers worth watching are how close the top is to
 * clipping and how far the bottom sits above the noise.
 *
 * WHAT THIS DOES NOT COVER. Only cameras attached to the machine running the browser. The weld
 * cell's cameras live on the rig host and are driven by v4l2 there - see
 * `tools/webcam_capture.py expose`, which sweeps exposure and saves a frame per setting.
 *
 * Raising exposure does NOT invalidate a calibration: intrinsics are geometry, not brightness.
 * That holds only while focus stays put, so focus is reported and pinned to manual where the
 * camera allows it - a focus shift changes the intrinsics silently and no later inspection of the
 * photographs can detect it.
 */

const KNOBS = ['exposureTime', 'exposureCompensation', 'brightness'];

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/**
 * Attach the control + meter to `host` for the given MediaStream and <video>.
 * Returns a stop() to be called when the stream is torn down.
 */
export function attachExposureControls(host, stream, video, onStatus) {
    const track = stream?.getVideoTracks?.()[0];
    if (!host || !track || !video) return () => {};
    const caps = (track.getCapabilities && track.getCapabilities()) || {};
    const cur = (track.getSettings && track.getSettings()) || {};
    const knob = KNOBS.find(k => caps[k] && typeof caps[k].min === 'number');

    let meterTimer = null;
    const stop = () => {
        if (meterTimer) { clearInterval(meterTimer); meterTimer = null; }
        host.hidden = true;
        host.innerHTML = '';
    };

    host.hidden = false;
    if (!knob) {
        // Driver support for these constraints is patchy on Windows. Saying so plainly beats a
        // dead slider, and the meter is still worth having - it lets the vendor tool's slider be
        // judged against real numbers instead of by eye.
        host.innerHTML = `<p class="capture-expo-warn">No exposure control is available for
            <b>${esc(track.label || 'this camera')}</b> through the browser. Set it in the vendor
            tool (Logi Tune / Logitech Camera Settings) and use the meter below to judge it.</p>
            <div class="capture-expo-meter"></div>`;
    } else {
        const c = caps[knob];
        const step = c.step || (c.max - c.min) / 100 || 1;
        const val = cur[knob] ?? ((c.min + c.max) / 2);
        const focus = caps.focusMode ? (cur.focusMode || 'unknown') : 'not reported';
        host.innerHTML = `
            <div class="capture-expo-row">
                <label>exposure
                    <input type="range" class="capture-expo-range" min="${c.min}" max="${c.max}"
                           step="${step}" value="${val}">
                </label>
                <span class="capture-expo-val">${val}</span>
                <button type="button" class="outline secondary capture-expo-auto">back to auto</button>
                <span class="capture-expo-focus">${esc(knob)} · focus: <b>${esc(String(focus))}</b></span>
            </div>
            <div class="capture-expo-meter"></div>`;

        const range = host.querySelector('.capture-expo-range');
        range.addEventListener('input', async () => {
            host.querySelector('.capture-expo-val').textContent = range.value;
            const adv = { [knob]: Number(range.value) };
            if (caps.exposureMode?.includes?.('manual')) adv.exposureMode = 'manual';
            try {
                await track.applyConstraints({ advanced: [adv] });
            } catch (err) {
                onStatus?.(`Camera refused that exposure: ${err?.message || err}`, true);
            }
        });
        host.querySelector('.capture-expo-auto').addEventListener('click', async () => {
            if (!caps.exposureMode?.includes?.('continuous')) return;
            try {
                await track.applyConstraints({ advanced: [{ exposureMode: 'continuous' }] });
                onStatus?.('Exposure back to auto — it will drift between shots.');
            } catch (err) { /* the camera simply declined */ }
        });
        if (caps.focusMode?.includes?.('manual') && cur.focusMode !== 'manual') {
            track.applyConstraints({ advanced: [{ focusMode: 'manual' }] }).catch(() => {});
        }
    }

    const out = host.querySelector('.capture-expo-meter');
    const cv = document.createElement('canvas');
    cv.width = 192; cv.height = 108;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    meterTimer = setInterval(() => {
        if (!video.videoWidth) return;
        ctx.drawImage(video, 0, 0, cv.width, cv.height);
        const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
        const hist = new Uint32Array(256);
        for (let i = 0; i < d.length; i += 4) {
            hist[(d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114) | 0]++;
        }
        const total = d.length / 4;
        const pct = (p) => {
            let want = total * p, run = 0;
            for (let v = 0; v < 256; v++) { run += hist[v]; if (run >= want) return v; }
            return 255;
        };
        const clip = 100 * (hist[254] + hist[255]) / total;
        const bright = pct(0.90), dark = pct(0.10);
        let cls = 'ok', msg;
        if (clip > 0.5) {
            cls = 'bad';
            msg = `too bright — ${clip.toFixed(1)}% of pixels clipped. The board's white squares go
                   first, and losing the board loses the world frame. Bring it down until this is
                   under 0.5%.`;
        } else if (bright < 205) {
            cls = 'warn';
            msg = `room to raise — bright end ${bright}, headroom to 255. Lifting it lifts the dark
                   end too, which is where creases become measurable.`;
        } else {
            msg = `good — bright end ${bright}, no clipping, dark end ${dark}. A higher dark end
                   means more creases carry usable contrast.`;
        }
        out.className = `capture-expo-meter is-${cls}`;
        out.innerHTML = `<span class="capture-expo-nums">bright p90 <b>${bright}</b> ·
            dark p10 <b>${dark}</b> · clipped <b>${clip.toFixed(2)}%</b></span>
            <span class="capture-expo-msg">${msg}</span>`;
    }, 400);

    return stop;
}
