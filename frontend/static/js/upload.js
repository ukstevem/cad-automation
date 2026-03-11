/**
 * Upload page - drag-drop, progress, results display
 */
import { createCheckRow, createResultCard, createVerdictBanner } from './components.js';
import { formatFileSize } from './utils.js';

const MAX_SIZE = 200 * 1024 * 1024;
const ALLOWED_EXTENSIONS = ['.step', '.stp'];

export class UploadPage {
    constructor(api) {
        this.api = api;
        this.container = null;
    }

    render(container) {
        this.container = container;
        container.innerHTML = this._template();
        this._bindEvents();
    }

    _template() {
        return `
            <section>
                <h2>Upload STEP File</h2>
                <p>Drag and drop a STEP file to validate and analyse its geometry.</p>

                <div class="upload-zone" id="drop-zone" tabindex="0" role="button" aria-label="Upload STEP file">
                    <div class="upload-zone-content" id="zone-content">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                        <p><strong>Drop STEP file here</strong></p>
                        <p><small>.step, .stp | Max ${formatFileSize(MAX_SIZE)}</small></p>
                    </div>
                    <input type="file" id="file-input" accept=".step,.stp" hidden>
                </div>

                <div id="progress-area" hidden>
                    <div class="progress-wrapper">
                        <progress id="progress-bar" value="0" max="100"></progress>
                        <span id="progress-text">Uploading... 0%</span>
                    </div>
                </div>

                <div id="error-area" hidden></div>
            </section>

            <section id="results-area" hidden></section>
        `;
    }

    _bindEvents() {
        const zone = this.container.querySelector('#drop-zone');
        const input = this.container.querySelector('#file-input');

        zone.addEventListener('click', () => input.click());
        zone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
        });

        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) this._handleUpload(file);
        });

        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) this._handleUpload(file);
            input.value = '';
        });
    }

    _validateClientSide(file) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            return `Invalid file type "${ext}". Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`;
        }
        if (file.size > MAX_SIZE) {
            return `File too large (${formatFileSize(file.size)}). Maximum: ${formatFileSize(MAX_SIZE)}`;
        }
        if (file.size === 0) {
            return 'File is empty.';
        }
        return null;
    }

    async _handleUpload(file) {
        const errorArea = this.container.querySelector('#error-area');
        const progressArea = this.container.querySelector('#progress-area');
        const resultsArea = this.container.querySelector('#results-area');

        // Reset
        errorArea.hidden = true;
        resultsArea.hidden = true;

        // Client-side validation
        const clientError = this._validateClientSide(file);
        if (clientError) {
            this._showError(clientError);
            return;
        }

        // Show progress
        progressArea.hidden = false;
        const progressBar = this.container.querySelector('#progress-bar');
        const progressText = this.container.querySelector('#progress-text');
        progressBar.value = 0;
        progressText.textContent = 'Uploading... 0%';

        // Disable drop zone during upload
        const zone = this.container.querySelector('#drop-zone');
        zone.classList.add('uploading');

        try {
            const data = await this.api.uploadFile(file, false, (pct) => {
                progressBar.value = pct;
                if (pct < 100) {
                    progressText.textContent = `Uploading... ${pct}%`;
                } else {
                    progressText.textContent = 'Validating file...';
                }
            });

            progressArea.hidden = true;
            this._renderResults(data, file);

        } catch (err) {
            progressArea.hidden = true;
            const msg = err?.data?.detail?.message || err?.message || 'Upload failed. Please try again.';
            this._showError(msg);
        } finally {
            zone.classList.remove('uploading');
        }
    }

    _showError(message) {
        const errorArea = this.container.querySelector('#error-area');
        errorArea.hidden = false;
        errorArea.innerHTML = `
            <article class="result-card fail">
                <header><strong>Error</strong></header>
                <div class="result-card-body"><p>${message}</p></div>
            </article>
        `;
    }

    _renderResults(data, file) {
        const resultsArea = this.container.querySelector('#results-area');
        resultsArea.hidden = false;

        const { overallPass, checks } = this._assessWorkability(data);

        // Verdict banner
        const passCount = checks.filter(c => c.pass).length;
        const subtitle = overallPass
            ? 'File is ready for processing'
            : `${passCount} of ${checks.length} checks passed`;
        let html = createVerdictBanner(overallPass, subtitle);

        // File info card
        const info = data.file_info || {};
        html += createResultCard('File Information', `
            <table>
                <tbody>
                    <tr><td>Filename</td><td>${info.original_filename || file.name}</td></tr>
                    <tr><td>Size</td><td>${formatFileSize(info.size_bytes || file.size)}</td></tr>
                    <tr><td>Extension</td><td>${info.extension || '-'}</td></tr>
                    <tr><td>ID</td><td><code>${info.unique_id || '-'}</code></td></tr>
                </tbody>
            </table>
        `, 'neutral');

        // Validation checks card
        let checksHtml = checks.map(c => createCheckRow(c.label, c.pass, c.detail)).join('');
        html += createResultCard('Validation Checks', checksHtml, overallPass ? 'pass' : 'fail');

        // Next steps card - direct user to Analysis tab
        if (data.success !== false) {
            const savedName = info.saved_filename || '';
            html += createResultCard('Next Steps', `
                <p>Your file has been uploaded and validated. Head to the
                <a href="#analysis" data-page="analysis" class="nav-link"><strong>Analysis</strong></a>
                tab to inspect the assembly tree and classify parts.</p>
            `, 'neutral');
        }

        // Upload another button
        html += '<button class="upload-another" id="upload-another">Upload another file</button>';

        resultsArea.innerHTML = html;

        this.container.querySelector('#upload-another').addEventListener('click', () => {
            resultsArea.hidden = true;
            resultsArea.innerHTML = '';
        });
    }

    _assessWorkability(data) {
        const checks = [];

        checks.push({
            label: 'File uploaded',
            pass: data.success !== false || data.file_info != null,
            detail: data.file_info ? formatFileSize(data.file_info.size_bytes) : null,
        });

        const ext = data.file_info?.extension;
        checks.push({
            label: 'Valid extension',
            pass: ext != null,
            detail: ext || null,
        });

        checks.push({
            label: 'STEP content validated',
            pass: data.success !== false,
            detail: data.success !== false ? 'Valid STEP header' : 'Invalid content',
        });

        const overallPass = checks.every(c => c.pass);
        return { overallPass, checks };
    }
}
