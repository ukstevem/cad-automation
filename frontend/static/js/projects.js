/**
 * Projects page — manage scope-of-supply projects that aggregate
 * multiple analysis runs and support cross-file nesting.
 */

export class ProjectsPage {
    constructor(api) {
        this.api = api;
        this.container = null;

        /** @type {string|null} currently selected project number */
        this._selectedProject = null;

        /** @type {Object|null} full project data from backend */
        this._projectData = null;

        /** @type {Array|null} available analysis files from /analysis/files */
        this._availableFiles = null;

        // ── Nesting state ──
        this._nestingTaskId = null;
        this._nestingPollTimer = null;
        this._nestingRunning = false;
        this._nestingCuttingList = null;
        this._nestingDefaultStockLength = 6000;
        this._nestingDefaultStockQty = 20;
        this._nestingKerf = 3;
    }

    render(container) {
        this.container = container;
        this._cleanup();
        container.innerHTML = this._template();
        this._bindEvents();
        this._loadProjects();
    }

    _cleanup() {
        if (this._nestingPollTimer) {
            clearInterval(this._nestingPollTimer);
            this._nestingPollTimer = null;
        }
        this._nestingTaskId = null;
        this._nestingRunning = false;
        this._nestingCuttingList = null;
        this._selectedProject = null;
        this._projectData = null;
        this._availableFiles = null;
    }

    // ---------------------------------------------------------------
    // Template
    // ---------------------------------------------------------------

    _template() {
        return `
            <section class="projects-page">
                <div class="projects-header">
                    <h2>Projects</h2>
                    <button id="new-project-btn" class="outline">+ New Project</button>
                </div>

                <div class="projects-layout">
                    <div class="projects-list-panel">
                        <div id="projects-list" class="projects-list">
                            <p aria-busy="true">Loading projects...</p>
                        </div>
                    </div>

                    <div class="projects-detail-panel" id="project-detail">
                        <p class="projects-empty-hint">Select a project from the list, or create a new one.</p>
                    </div>
                </div>
            </section>
        `;
    }

    _bindEvents() {
        this.container.querySelector('#new-project-btn')
            ?.addEventListener('click', () => this._showNewProjectDialog());
    }

    // ---------------------------------------------------------------
    // Project list
    // ---------------------------------------------------------------

    async _loadProjects() {
        try {
            const resp = await this.api.listProjects();
            this._renderProjectList(resp.projects || []);
        } catch (err) {
            this._renderProjectList([]);
            console.error('Failed to load projects:', err);
        }
    }

    _renderProjectList(projects) {
        const el = this.container.querySelector('#projects-list');
        if (!el) return;

        if (projects.length === 0) {
            el.innerHTML = '<p class="projects-empty">No projects yet. Click <strong>+ New Project</strong> to get started.</p>';
            return;
        }

        el.innerHTML = projects.map(p => `
            <div class="project-card ${this._selectedProject === p.project_number ? 'selected' : ''}"
                 data-project="${this._esc(p.project_number)}">
                <div class="project-card-title">${this._esc(p.project_number)}</div>
                <div class="project-card-meta">
                    ${p.analysis_count} analysis run${p.analysis_count !== 1 ? 's' : ''}
                    &middot; ${this._formatDate(p.created_at)}
                </div>
            </div>
        `).join('');

        el.querySelectorAll('.project-card').forEach(card => {
            card.addEventListener('click', () => {
                const pn = card.dataset.project;
                this._selectProject(pn);
            });
        });
    }

    // ---------------------------------------------------------------
    // New project dialog
    // ---------------------------------------------------------------

    _showNewProjectDialog() {
        const existing = document.getElementById('new-project-dialog');
        if (existing) existing.remove();

        const dialog = document.createElement('dialog');
        dialog.id = 'new-project-dialog';
        dialog.className = 'projects-dialog';
        dialog.innerHTML = `
            <form method="dialog" class="projects-dialog-form">
                <h3>New Project</h3>
                <label for="new-project-number">Project Number</label>
                <input type="text" id="new-project-number" placeholder="e.g. P25001" required autofocus>
                <div class="projects-dialog-actions">
                    <button type="button" id="new-project-cancel" class="outline">Cancel</button>
                    <button type="submit">Create</button>
                </div>
            </form>
        `;
        document.body.appendChild(dialog);
        dialog.showModal();

        dialog.querySelector('#new-project-cancel').addEventListener('click', () => {
            dialog.close();
            dialog.remove();
        });

        dialog.querySelector('form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const pn = dialog.querySelector('#new-project-number').value.trim();
            if (!pn) return;
            try {
                await this.api.createProject(pn);
                dialog.close();
                dialog.remove();
                await this._loadProjects();
                this._selectProject(pn);
            } catch (err) {
                const msg = err?.detail || err?.message || 'Failed to create project';
                alert(msg);
            }
        });

        dialog.addEventListener('cancel', () => dialog.remove());
    }

    // ---------------------------------------------------------------
    // Project detail view
    // ---------------------------------------------------------------

    async _selectProject(projectNumber) {
        // Stop any existing poll from a previous project
        if (this._nestingPollTimer) {
            clearInterval(this._nestingPollTimer);
            this._nestingPollTimer = null;
        }

        this._selectedProject = projectNumber;
        this._nestingCuttingList = null;
        this._nestingRunning = false;
        this._nestingTaskId = null;

        // Highlight in list
        this.container.querySelectorAll('.project-card').forEach(c => {
            c.classList.toggle('selected', c.dataset.project === projectNumber);
        });

        const detail = this.container.querySelector('#project-detail');
        if (!detail) return;
        detail.innerHTML = '<p aria-busy="true">Loading project...</p>';

        try {
            this._projectData = await this.api.getProject(projectNumber);
            if (!this._availableFiles) {
                const filesResp = await this.api.listFiles();
                this._availableFiles = filesResp.files || filesResp || [];
            }
            this._renderProjectDetail();

            // Reconnect to a saved nesting task if one exists
            const savedTaskId = this._projectData.nesting_task_id;
            if (savedTaskId) {
                await this._reconnectNesting(savedTaskId);
            }
        } catch (err) {
            detail.innerHTML = `<p class="error">Failed to load project: ${this._esc(String(err?.message || err))}</p>`;
        }
    }

    async _reconnectNesting(taskId) {
        const NESTING_BASE = await this.api.getNestingBase();
        try {
            const resp = await fetch(
                `${NESTING_BASE}/api/v1/nesting/status/${encodeURIComponent(taskId)}`
            ).then(r => r.json());

            if (resp.status === 'completed') {
                // Fetch and display the completed cutting list
                this._nestingTaskId = taskId;
                const cl = await fetch(
                    `${NESTING_BASE}/api/v1/nesting/cutting-list/${encodeURIComponent(taskId)}`
                ).then(r => r.json());
                this._nestingCuttingList = cl;
                this._renderCuttingList();

            } else if (resp.status === 'running' || resp.status === 'pending') {
                // Resume polling
                this._nestingTaskId = taskId;
                this._nestingRunning = true;
                this._updateNestingButton();
                this._pollNesting(taskId);

            } else if (resp.status === 'failed') {
                // Clear the saved task — it's dead
                this._clearSavedNestingTask();
            }
        } catch {
            // Nesting service unreachable — leave the saved task for later
            console.warn('Could not reconnect to nesting service for task', taskId);
        }
    }

    _renderProjectDetail() {
        const detail = this.container.querySelector('#project-detail');
        if (!detail || !this._projectData) return;

        const proj = this._projectData;
        const analyses = proj.analyses || [];

        // Check which files have CNC analysis (for nesting eligibility)
        const hasAnalyses = analyses.length > 0;

        let html = `
            <div class="project-detail-header">
                <h3>${this._esc(proj.project_number)}</h3>
                <div class="project-detail-actions">
                    <button id="project-delete-btn" class="outline contrast" title="Delete project">Delete</button>
                </div>
            </div>

            <div class="project-analyses-section">
                <div class="project-analyses-header">
                    <h4>Scope of Supply</h4>
                    <button id="add-analysis-btn" class="outline">+ Add Analysis Run</button>
                </div>

                ${analyses.length === 0
                    ? '<p class="projects-empty">No analysis runs added yet. Click <strong>+ Add Analysis Run</strong> to build the scope of supply.</p>'
                    : this._renderAnalysesTable(analyses)
                }
            </div>

            <div class="project-nesting-section" ${!hasAnalyses ? 'hidden' : ''}>
                <div class="project-nesting-header">
                    <h4>Nesting</h4>
                    <button id="project-nesting-btn" class="outline" ${this._nestingRunning ? 'disabled' : ''}>
                        ${this._nestingRunning ? 'Running...' : 'Run Nesting'}
                    </button>
                </div>
                <div id="project-nesting-results"></div>
            </div>
        `;

        detail.innerHTML = html;

        // Bind events
        detail.querySelector('#add-analysis-btn')
            ?.addEventListener('click', () => this._showAddAnalysisDialog());

        detail.querySelector('#project-delete-btn')
            ?.addEventListener('click', () => this._deleteProject());

        detail.querySelector('#project-nesting-btn')
            ?.addEventListener('click', () => this._showNestingDialog());

        // Bind quantity inputs
        detail.querySelectorAll('.project-qty-input').forEach(input => {
            input.addEventListener('change', () => this._onQtyChange());
        });

        // Bind remove buttons
        detail.querySelectorAll('.project-remove-analysis').forEach(btn => {
            btn.addEventListener('click', () => {
                const fn = btn.dataset.filename;
                this._removeAnalysis(fn);
            });
        });

        // Re-render cutting list if we have one
        if (this._nestingCuttingList) {
            this._renderCuttingList();
        }
    }

    _renderAnalysesTable(analyses) {
        const rows = analyses.map(a => {
            const displayName = a.display_name || a.filename;
            return `
                <tr>
                    <td class="project-analysis-name" title="${this._esc(a.filename)}">${this._esc(displayName)}</td>
                    <td class="project-analysis-qty">
                        <input type="number" class="project-qty-input"
                               data-filename="${this._esc(a.filename)}"
                               min="1" step="1" value="${a.quantity || 1}">
                    </td>
                    <td class="project-analysis-date">${this._formatDate(a.added_at)}</td>
                    <td class="project-analysis-actions">
                        <button class="project-remove-analysis outline contrast"
                                data-filename="${this._esc(a.filename)}" title="Remove">x</button>
                    </td>
                </tr>
            `;
        }).join('');

        return `
            <table class="project-analyses-table">
                <thead>
                    <tr><th>Analysis File</th><th>Qty</th><th>Added</th><th></th></tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    }

    // ---------------------------------------------------------------
    // Add analysis dialog
    // ---------------------------------------------------------------

    _showAddAnalysisDialog() {
        const existing = document.getElementById('add-analysis-dialog');
        if (existing) existing.remove();

        const files = this._availableFiles || [];
        const alreadyAdded = new Set((this._projectData?.analyses || []).map(a => a.filename));

        const available = files.filter(f => !alreadyAdded.has(f.filename));

        if (available.length === 0) {
            alert('No additional analysis files available. Upload and analyse a STEP file first.');
            return;
        }

        const dialog = document.createElement('dialog');
        dialog.id = 'add-analysis-dialog';
        dialog.className = 'projects-dialog projects-dialog-wide';
        dialog.innerHTML = `
            <form method="dialog" class="projects-dialog-form">
                <h3>Add Analysis Run</h3>
                <p>Select one or more analysed files to add to this project.</p>
                <div class="add-analysis-file-list">
                    ${available.map(f => `
                        <label class="add-analysis-file-item">
                            <input type="checkbox" value="${this._esc(f.filename)}"
                                   data-display="${this._esc(f.display_name || f.filename)}">
                            <span>${this._esc(f.display_name || f.filename)}</span>
                        </label>
                    `).join('')}
                </div>
                <div class="projects-dialog-actions">
                    <button type="button" id="add-analysis-cancel" class="outline">Cancel</button>
                    <button type="submit">Add Selected</button>
                </div>
            </form>
        `;

        document.body.appendChild(dialog);
        dialog.showModal();

        dialog.querySelector('#add-analysis-cancel').addEventListener('click', () => {
            dialog.close();
            dialog.remove();
        });

        dialog.querySelector('form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const checked = dialog.querySelectorAll('input[type="checkbox"]:checked');
            if (checked.length === 0) return;

            const newEntries = [...checked].map(cb => ({
                filename: cb.value,
                quantity: 1,
                display_name: cb.dataset.display || cb.value,
            }));

            dialog.close();
            dialog.remove();
            await this._addAnalyses(newEntries);
        });

        dialog.addEventListener('cancel', () => dialog.remove());
    }

    async _addAnalyses(newEntries) {
        const current = (this._projectData?.analyses || []).map(a => ({
            filename: a.filename,
            quantity: a.quantity,
            display_name: a.display_name,
            added_at: a.added_at,
        }));

        const merged = [...current, ...newEntries];

        try {
            this._projectData = await this.api.updateProjectAnalyses(
                this._selectedProject, merged
            );
            this._renderProjectDetail();
            this._loadProjects(); // refresh counts in sidebar
        } catch (err) {
            alert('Failed to add analyses: ' + (err?.message || err));
        }
    }

    async _removeAnalysis(filename) {
        const current = (this._projectData?.analyses || [])
            .filter(a => a.filename !== filename)
            .map(a => ({
                filename: a.filename,
                quantity: a.quantity,
                display_name: a.display_name,
                added_at: a.added_at,
            }));

        try {
            this._projectData = await this.api.updateProjectAnalyses(
                this._selectedProject, current
            );
            this._renderProjectDetail();
            this._loadProjects();
        } catch (err) {
            alert('Failed to remove analysis: ' + (err?.message || err));
        }
    }

    async _onQtyChange() {
        const inputs = this.container.querySelectorAll('.project-qty-input');
        const updated = (this._projectData?.analyses || []).map(a => {
            const input = [...inputs].find(i => i.dataset.filename === a.filename);
            return {
                filename: a.filename,
                quantity: input ? Math.max(1, parseInt(input.value) || 1) : a.quantity,
                display_name: a.display_name,
                added_at: a.added_at,
            };
        });

        try {
            this._projectData = await this.api.updateProjectAnalyses(
                this._selectedProject, updated
            );
            this._loadProjects(); // refresh sidebar counts
        } catch (err) {
            console.error('Failed to update quantities:', err);
        }
    }

    async _deleteProject() {
        if (!confirm(`Delete project "${this._selectedProject}"? This cannot be undone.`)) return;

        try {
            await this.api.deleteProject(this._selectedProject);
            this._selectedProject = null;
            this._projectData = null;
            const detail = this.container.querySelector('#project-detail');
            if (detail) detail.innerHTML = '<p class="projects-empty-hint">Select a project from the list, or create a new one.</p>';
            await this._loadProjects();
        } catch (err) {
            alert('Failed to delete project: ' + (err?.message || err));
        }
    }

    // ---------------------------------------------------------------
    // Nesting
    // ---------------------------------------------------------------

    async _showNestingDialog() {
        if (this._nestingRunning) return;

        const btn = this.container.querySelector('#project-nesting-btn');
        if (btn) { btn.disabled = true; btn.textContent = 'Loading items...'; }

        let nestingData;
        try {
            nestingData = await this.api.getProjectNestingItems(this._selectedProject);
        } catch (err) {
            alert('Failed to build nesting items: ' + (err?.detail || err?.message || err));
            if (btn) { btn.disabled = false; btn.textContent = 'Run Nesting'; }
            return;
        }

        if (btn) { btn.disabled = false; btn.textContent = 'Run Nesting'; }

        if (!nestingData.items || nestingData.items.length === 0) {
            alert('No section items found for nesting. Ensure CNC analysis has been run on the added files.');
            return;
        }

        // Show errors if any
        if (nestingData.errors && nestingData.errors.length > 0) {
            console.warn('Nesting item warnings:', nestingData.errors);
        }

        this._showNestingSettingsDialog(nestingData);
    }

    _showNestingSettingsDialog(nestingData) {
        const items = nestingData.items;
        const sections = nestingData.sections || [];

        // Count per section
        const sectionInfo = new Map();
        for (const it of items) {
            const existing = sectionInfo.get(it.section);
            if (!existing) {
                sectionInfo.set(it.section, { count: 1, maxLen: it.length });
            } else {
                existing.count++;
                existing.maxLen = Math.max(existing.maxLen, it.length);
            }
        }

        const existing = document.getElementById('project-nesting-dialog');
        if (existing) existing.remove();

        const sectionRows = [...sectionInfo.entries()]
            .sort((a, b) => a[0].localeCompare(b[0]))
            .map(([sec, info]) => `
                <tr class="nesting-section-row" data-section="${this._esc(sec)}">
                    <td><label class="nesting-include-label"><input type="checkbox" class="nesting-include-chk" checked> ${this._esc(sec)}</label></td>
                    <td class="nesting-section-count">${info.count} pcs</td>
                    <td class="nesting-section-maxlen">${info.maxLen} mm</td>
                    <td colspan="2" class="nesting-stock-cell">
                        <div class="nesting-stock-entries" data-section="${this._esc(sec)}">
                            <div class="nesting-stock-entry">
                                <input type="number" class="nesting-stock-len" min="100" step="100" placeholder="length">
                                <span class="nesting-stock-x">x</span>
                                <input type="number" class="nesting-stock-qty" min="1" step="1" placeholder="qty">
                                <button type="button" class="nesting-stock-remove outline" title="Remove">-</button>
                            </div>
                        </div>
                        <button type="button" class="nesting-stock-add outline" data-section="${this._esc(sec)}" title="Add stock length">+</button>
                    </td>
                </tr>
            `).join('');

        const dialog = document.createElement('dialog');
        dialog.id = 'project-nesting-dialog';
        dialog.className = 'nesting-settings-dialog';
        dialog.innerHTML = `
            <form method="dialog" class="nesting-settings-form">
                <h3>Nesting Settings — ${this._esc(this._selectedProject)}</h3>
                <p class="nesting-settings-summary">${items.length} section pieces across ${sectionInfo.size} profile(s)</p>
                ${nestingData.errors && nestingData.errors.length > 0
                    ? `<p class="nesting-warnings">${nestingData.errors.map(e => this._esc(e)).join('<br>')}</p>`
                    : ''}

                <div class="nesting-settings-defaults">
                    <div class="nesting-settings-field">
                        <label for="pn-stock-length">Default Stock Length (mm)</label>
                        <input type="number" id="pn-stock-length" min="100" step="100"
                               value="${this._nestingDefaultStockLength}">
                    </div>
                    <div class="nesting-settings-field">
                        <label for="pn-stock-qty">Default Stock Qty</label>
                        <input type="number" id="pn-stock-qty" min="1" step="1"
                               value="${this._nestingDefaultStockQty}">
                    </div>
                    <div class="nesting-settings-field">
                        <label for="pn-kerf">Kerf / Blade Width (mm)</label>
                        <input type="number" id="pn-kerf" min="0" step="1"
                               value="${this._nestingKerf}">
                    </div>
                </div>

                <table class="nesting-overrides-table">
                    <thead>
                        <tr><th>Section</th><th>Items</th><th>Max Len</th><th>Stock (leave blank for default)</th></tr>
                    </thead>
                    <tbody>${sectionRows}</tbody>
                </table>

                <div class="nesting-settings-actions">
                    <button type="button" id="pn-nesting-cancel" class="outline">Cancel</button>
                    <button type="submit">Run Nesting</button>
                </div>
            </form>
        `;

        document.body.appendChild(dialog);
        dialog.showModal();
        dialog.querySelector('#pn-stock-length')?.focus();

        // Add/remove stock entry handlers
        dialog.addEventListener('click', (e) => {
            if (e.target.classList.contains('nesting-stock-add')) {
                const sec = e.target.dataset.section;
                const container = dialog.querySelector(`.nesting-stock-entries[data-section="${sec}"]`);
                if (!container) return;
                const entry = document.createElement('div');
                entry.className = 'nesting-stock-entry';
                entry.innerHTML = `
                    <input type="number" class="nesting-stock-len" min="100" step="100" placeholder="length">
                    <span class="nesting-stock-x">x</span>
                    <input type="number" class="nesting-stock-qty" min="1" step="1" placeholder="qty">
                    <button type="button" class="nesting-stock-remove outline" title="Remove">-</button>
                `;
                container.appendChild(entry);
            }
            if (e.target.classList.contains('nesting-stock-remove')) {
                const entry = e.target.closest('.nesting-stock-entry');
                const container = entry?.parentElement;
                // Keep at least one entry row
                if (container && container.querySelectorAll('.nesting-stock-entry').length > 1) {
                    entry.remove();
                } else if (entry) {
                    // Clear the inputs instead of removing the last row
                    entry.querySelectorAll('input').forEach(i => { i.value = ''; });
                }
            }
        });

        dialog.querySelector('#pn-nesting-cancel').addEventListener('click', () => {
            dialog.close();
            dialog.remove();
        });

        dialog.querySelector('form').addEventListener('submit', (e) => {
            e.preventDefault();
            const stockLen = parseInt(dialog.querySelector('#pn-stock-length').value) || 6000;
            const stockQty = parseInt(dialog.querySelector('#pn-stock-qty').value) || 20;
            const kerf = parseInt(dialog.querySelector('#pn-kerf').value) || 3;

            this._nestingDefaultStockLength = stockLen;
            this._nestingDefaultStockQty = stockQty;
            this._nestingKerf = kerf;

            // Collect excluded sections and per-section stock overrides
            const excludedSections = new Set();
            const stockPerSection = [];
            for (const row of dialog.querySelectorAll('.nesting-section-row')) {
                const sec = row.dataset.section;
                const included = row.querySelector('.nesting-include-chk')?.checked;
                if (!included) {
                    excludedSections.add(sec);
                    continue;
                }
                // Collect all stock entries for this section
                const stockEntries = [];
                for (const entry of row.querySelectorAll('.nesting-stock-entry')) {
                    const len = parseInt(entry.querySelector('.nesting-stock-len')?.value);
                    const qty = parseInt(entry.querySelector('.nesting-stock-qty')?.value);
                    if (len > 0 && qty > 0) {
                        stockEntries.push({ length: len, qty });
                    }
                }
                if (stockEntries.length > 0) {
                    stockPerSection.push({ section: sec, stock: stockEntries });
                }
            }

            const filteredItems = excludedSections.size > 0
                ? items.filter(it => !excludedSections.has(it.section))
                : items;

            if (filteredItems.length === 0) {
                alert('All sections are excluded. Include at least one section to run nesting.');
                return;
            }

            dialog.close();
            dialog.remove();
            this._submitNesting(filteredItems, stockPerSection, stockLen, stockQty, kerf);
        });

        dialog.addEventListener('cancel', () => dialog.remove());
    }

    async _submitNesting(items, stockPerSection, defaultLen, defaultQty, kerf) {
        this._nestingRunning = true;
        this._nestingCuttingList = null;
        this._updateNestingButton();

        const body = {
            job_label: this._selectedProject || 'project-nesting',
            items,
            stock_per_section: stockPerSection,
            default_stock: [{ length: defaultLen, qty: defaultQty }],
            kerf,
            time_limit: 300.0,
        };

        const NESTING_BASE = await this.api.getNestingBase();

        fetch(`${NESTING_BASE}/api/v1/nesting/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        })
        .then(r => {
            if (!r.ok) return r.json().then(e => { throw new Error(e.detail || `HTTP ${r.status}`); });
            return r.json();
        })
        .then(resp => {
            if (resp.task_id) {
                this._nestingTaskId = resp.task_id;
                // Persist task ID so we can reconnect after navigation
                this._saveNestingTask(resp.task_id);
                this._pollNesting(resp.task_id);
            } else {
                this._nestingRunning = false;
                this._updateNestingButton();
            }
        })
        .catch(err => {
            console.error('Failed to start nesting:', err);
            this._nestingRunning = false;
            this._updateNestingButton();
            alert(`Nesting failed to start: ${err.message}`);
        });
    }

    async _pollNesting(taskId) {
        if (this._nestingPollTimer) clearInterval(this._nestingPollTimer);

        const NESTING_BASE = await this.api.getNestingBase();
        this._renderNestingProgress({ phase: 0, description: 'Starting nesting solver...' });

        this._nestingPollTimer = setInterval(() => {
            fetch(`${NESTING_BASE}/api/v1/nesting/status/${encodeURIComponent(taskId)}`)
                .then(r => r.json())
                .then(resp => {
                    if (resp.status === 'completed') {
                        clearInterval(this._nestingPollTimer);
                        this._nestingPollTimer = null;
                        this._nestingRunning = false;
                        this._updateNestingButton();

                        fetch(`${NESTING_BASE}/api/v1/nesting/cutting-list/${encodeURIComponent(taskId)}`)
                            .then(r => r.json())
                            .then(cl => {
                                this._nestingCuttingList = cl;
                                this._renderCuttingList();
                            })
                            .catch(err => console.error('Failed to fetch cutting list:', err));

                        // Keep task_id saved so results can be re-shown on navigation

                    } else if (resp.status === 'failed') {
                        clearInterval(this._nestingPollTimer);
                        this._nestingPollTimer = null;
                        this._nestingRunning = false;
                        this._updateNestingButton();
                        this._clearSavedNestingTask();
                        alert(`Nesting failed: ${resp.error || 'unknown error'}`);

                    } else {
                        this._renderNestingProgress(resp.progress || {});
                    }
                })
                .catch(() => { /* network hiccup — keep polling */ });
        }, 2000);
    }

    _updateNestingButton() {
        const btn = this.container?.querySelector('#project-nesting-btn');
        if (!btn) return;
        btn.disabled = this._nestingRunning;
        btn.textContent = this._nestingRunning ? 'Running...' : 'Run Nesting';
    }

    _renderNestingProgress(progress) {
        const panel = this.container?.querySelector('#project-nesting-results');
        if (!panel) return;

        const desc = progress.description || 'Working...';
        const pct = progress.percent || 0;
        const secInfo = (progress.section_index && progress.section_count)
            ? ` (section ${progress.section_index}/${progress.section_count}: ${progress.section || ''})`
            : '';

        panel.innerHTML = `
            <div class="nesting-progress">
                <p class="nesting-progress-text" aria-busy="true">
                    ${this._esc(desc)}${secInfo}
                </p>
                ${pct > 0 ? `<progress value="${pct}" max="100"></progress>` : '<progress></progress>'}
            </div>
        `;
    }

    _renderCuttingList() {
        const panel = this.container?.querySelector('#project-nesting-results');
        if (!panel || !this._nestingCuttingList) return;

        const cl = this._nestingCuttingList;
        const totals = cl.totals || {};
        const sections = cl.sections || [];

        // ── Compute synopsis stats ──
        let totalStockLen = 0;
        let totalUsedLen = 0;
        for (const sec of sections) {
            for (const bar of (sec.bars || [])) {
                totalStockLen += bar.stock_length_mm || 0;
                totalUsedLen += bar.used_length_mm || 0;
            }
        }
        const overallUtil = totalStockLen > 0
            ? Math.round((totalUsedLen / totalStockLen) * 100)
            : 0;

        // Per-section synopsis rows
        const synopsisSections = sections.map(sec => {
            const summ = sec.summary || {};
            let secStock = 0, secUsed = 0;
            for (const bar of (sec.bars || [])) {
                secStock += bar.stock_length_mm || 0;
                secUsed += bar.used_length_mm || 0;
            }
            const secUtil = secStock > 0 ? Math.round((secUsed / secStock) * 100) : 0;
            const statusCls = sec.phase1_status === 'optimal' ? 'nesting-status-optimal'
                : sec.phase1_status === 'feasible' ? 'nesting-status-feasible'
                : 'nesting-status-other';
            return `<tr>
                <td>${this._esc(sec.designation)}</td>
                <td>${summ.items_placed ?? '?'}</td>
                <td>${summ.stocks_used ?? '?'}</td>
                <td>
                    <div class="synopsis-util-bar">
                        <div class="synopsis-util-fill" style="width:${secUtil}%"></div>
                    </div>
                    <span class="synopsis-util-label">${secUtil}%</span>
                </td>
                <td>${(summ.total_waste_mm ?? secStock - secUsed) > 0 ? ((summ.total_waste_mm ?? (secStock - secUsed)) / 1000).toFixed(1) + ' m' : '—'}</td>
                <td><span class="nesting-status-badge ${statusCls}">${this._esc(sec.phase1_status || '?')}</span></td>
            </tr>`;
        }).join('');

        let html = `
            <div class="nesting-results-header">
                <span class="nesting-results-title">Cutting List</span>
                <div class="nesting-results-actions">
                    <button class="nesting-pdf-btn outline">PDF</button>
                    <button class="nesting-csv-btn outline">CSV</button>
                    <button class="nesting-close-btn outline">x</button>
                </div>
            </div>

            <div class="nesting-synopsis">
                <div class="nesting-synopsis-kpis">
                    <div class="synopsis-kpi">
                        <span class="synopsis-kpi-value">${totals.total_items_placed ?? '?'}</span>
                        <span class="synopsis-kpi-label">Placed</span>
                    </div>
                    <div class="synopsis-kpi">
                        <span class="synopsis-kpi-value">${totals.total_stocks_used ?? '?'}</span>
                        <span class="synopsis-kpi-label">Bars Used</span>
                    </div>
                    <div class="synopsis-kpi">
                        <span class="synopsis-kpi-value">${overallUtil}%</span>
                        <span class="synopsis-kpi-label">Utilisation</span>
                    </div>
                    <div class="synopsis-kpi">
                        <span class="synopsis-kpi-value">${totals.total_waste_mm != null ? (totals.total_waste_mm / 1000).toFixed(1) + ' m' : '?'}</span>
                        <span class="synopsis-kpi-label">Total Waste</span>
                    </div>
                    ${(totals.total_items_unassigned ?? 0) > 0 ? `
                    <div class="synopsis-kpi synopsis-kpi-warn">
                        <span class="synopsis-kpi-value">${totals.total_items_unassigned}</span>
                        <span class="synopsis-kpi-label">Unassigned</span>
                    </div>` : ''}
                </div>

                <table class="nesting-synopsis-table">
                    <thead><tr><th>Section</th><th>Placed</th><th>Bars</th><th>Utilisation</th><th>Waste</th><th>Status</th></tr></thead>
                    <tbody>${synopsisSections}</tbody>
                </table>
            </div>

            <details class="nesting-full-detail">
                <summary class="nesting-full-detail-summary">Full cutting list detail</summary>
                <div class="nesting-full-detail-content">
        `;

        for (const section of sections) {
            const summ = section.summary || {};
            const statusBadge = section.phase1_status === 'optimal'
                ? '<span class="nesting-status-badge nesting-status-optimal">optimal</span>'
                : section.phase1_status === 'feasible'
                    ? '<span class="nesting-status-badge nesting-status-feasible">feasible</span>'
                    : `<span class="nesting-status-badge nesting-status-other">${this._esc(section.phase1_status || '?')}</span>`;

            html += `
                <details class="nesting-section-details">
                    <summary class="nesting-section-summary">
                        <strong>${this._esc(section.designation)}</strong>
                        — ${summ.items_placed ?? '?'} placed, ${summ.stocks_used ?? '?'} bars
                        ${statusBadge}
                    </summary>
                    <div class="nesting-bars-container">
            `;

            for (const bar of (section.bars || [])) {
                const usePct = bar.stock_length_mm > 0
                    ? Math.round((bar.used_length_mm / bar.stock_length_mm) * 100)
                    : 0;

                html += `
                    <div class="nesting-bar">
                        <div class="nesting-bar-header">
                            <span class="nesting-bar-label">${this._esc(bar.bar_label)}</span>
                            <span class="nesting-bar-stock">${bar.stock_length_mm} mm</span>
                            <span class="nesting-bar-usage">${usePct}% used</span>
                            <span class="nesting-bar-waste">waste: ${bar.waste_mm} mm</span>
                        </div>
                        <div class="nesting-bar-visual" title="${bar.used_length_mm} / ${bar.stock_length_mm} mm">
                `;

                for (const cut of (bar.cuts || [])) {
                    const widthPct = bar.stock_length_mm > 0
                        ? (cut.length_mm / bar.stock_length_mm) * 100
                        : 0;
                    const label = cut.member || cut.ref_id || `Cut ${cut.cut_no}`;
                    html += `<div class="nesting-cut-block" style="width:${widthPct.toFixed(1)}%" title="${this._esc(label)}: ${cut.length_mm} mm${cut.parent ? ' (' + this._esc(cut.parent) + ')' : ''}">${cut.length_mm}</div>`;
                }

                if (bar.waste_mm > 0 && bar.stock_length_mm > 0) {
                    const wastePct = (bar.waste_mm / bar.stock_length_mm) * 100;
                    html += `<div class="nesting-waste-block" style="width:${wastePct.toFixed(1)}%">${bar.waste_mm}</div>`;
                }

                html += `
                        </div>
                        <table class="nesting-cuts-table">
                            <thead><tr><th>#</th><th>Member</th><th>Parent</th><th>Source</th><th>Length</th></tr></thead>
                            <tbody>
                `;

                for (const cut of (bar.cuts || [])) {
                    html += `<tr>
                        <td>${cut.cut_no}</td>
                        <td>${this._esc(cut.member || cut.ref_id || '—')}</td>
                        <td>${this._esc(cut.parent || '—')}</td>
                        <td>${this._esc(cut.source_file || '—')}</td>
                        <td>${cut.length_mm} mm</td>
                    </tr>`;
                }

                html += `</tbody></table></div>`;
            }

            if (section.unassigned && section.unassigned.length > 0) {
                html += `<div class="nesting-unassigned">
                    <strong>Unassigned (${section.unassigned.length})</strong>
                    <ul>`;
                for (const u of section.unassigned) {
                    html += `<li>${this._esc(u.member_name || u.ref_id || '?')} — ${u.length} mm</li>`;
                }
                html += `</ul></div>`;
            }

            html += `</div></details>`;
        }

        html += `</div></details>`; // close nesting-full-detail

        panel.innerHTML = html;

        panel.querySelector('.nesting-pdf-btn')?.addEventListener('click', () => {
            if (!this._selectedProject) return;
            const a = document.createElement('a');
            a.href = this.api.getProjectNestingPdfUrl(this._selectedProject);
            a.download = '';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        });

        panel.querySelector('.nesting-csv-btn')?.addEventListener('click', async () => {
            if (!this._nestingTaskId) return;
            const NESTING_BASE = await this.api.getNestingBase();
            const a = document.createElement('a');
            a.href = `${NESTING_BASE}/api/v1/nesting/cutting-list/${encodeURIComponent(this._nestingTaskId)}/csv`;
            a.download = '';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        });

        panel.querySelector('.nesting-close-btn')?.addEventListener('click', () => {
            panel.innerHTML = '';
        });
    }

    // ---------------------------------------------------------------
    // Nesting task persistence
    // ---------------------------------------------------------------

    _saveNestingTask(taskId) {
        if (!this._selectedProject) return;
        this.api.updateProjectNestingTask(
            this._selectedProject, taskId, new Date().toISOString()
        ).catch(err => console.warn('Failed to save nesting task:', err));
    }

    _clearSavedNestingTask() {
        if (!this._selectedProject) return;
        this.api.updateProjectNestingTask(this._selectedProject, null, null)
            .catch(err => console.warn('Failed to clear nesting task:', err));
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
            const d = new Date(isoStr);
            return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
        } catch {
            return isoStr;
        }
    }
}
