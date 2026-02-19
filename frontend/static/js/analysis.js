/**
 * Analysis page - side-by-side assembly tree + 3D STL viewer
 *
 * Progressive workflow:
 *   1. Analyse a STEP file -> tree appears, root-level STLs generated in background
 *   2. Assembly nodes become selectable (clickable to preview) once STLs are ready
 *   3. User clicks "Explode" on an assembly -> children STLs generated -> children become selectable
 *   4. Multi-solid parts can be exploded into individual solids
 *   5. Exploding an instance also explodes all sibling instances of the same reference
 *   6. Parts show Postprocess / Bought-out buttons once selectable
 */
import { STLViewer } from './stl-viewer.js';

export class AnalysisPage {
    constructor(api) {
        this.api = api;
        this.container = null;

        /** @type {Map<string, string>} node id -> classification (postprocess / bought-out) */
        this.classifications = new Map();

        /** @type {Map<string, string>} nodeId -> STL URL */
        this.stlMap = new Map();

        /** @type {Set<string>} node IDs that have been exploded */
        this.explodedNodes = new Set();

        /** @type {Map<string, Array<{name: string, nodeId: string}>>} refId -> solid children info */
        this._solidChildrenCache = new Map();

        /** @type {Map<string, {name: string, parentName: string|null}>} nodeId -> {name, parentName} */
        this._parentMap = new Map();

        /** @type {Map<string, number>} key -> setInterval timer id */
        this._explodePollTimers = new Map();

        /** @type {string|null} currently selected node shown in viewer */
        this._selectedNodeId = null;

        /** @type {STLViewer|null} single viewer instance */
        this._viewer = null;

        /** @type {number|null} root STL poll timer */
        this._stlPollTimer = null;

        /** @type {number|null} analysis background-task poll timer */
        this._analysisPollTimer = null;

        /** @type {string|null} current uploaded filename being analysed */
        this._currentFilename = null;

        /** @type {number|null} debounce timer for project state save */
        this._saveTimer = null;
    }

    render(container) {
        this.container = container;
        this._cleanup();
        container.innerHTML = this._template();
        this._bindEvents();
        this._loadFiles();
    }

    // ---------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------

    _cleanup() {
        if (this._analysisPollTimer) {
            clearInterval(this._analysisPollTimer);
            this._analysisPollTimer = null;
        }
        if (this._stlPollTimer) {
            clearInterval(this._stlPollTimer);
            this._stlPollTimer = null;
        }
        for (const timerId of this._explodePollTimers.values()) {
            clearInterval(timerId);
        }
        this._explodePollTimers.clear();
        if (this._viewer) {
            this._viewer.dispose();
            this._viewer = null;
        }
        if (this._saveTimer) {
            clearTimeout(this._saveTimer);
            this._saveTimer = null;
        }
        this.stlMap.clear();
        this.explodedNodes.clear();
        this.classifications.clear();
        this._solidChildrenCache.clear();
        this._parentMap.clear();
        this._selectedNodeId = null;
        this._currentFilename = null;
    }

    // ---------------------------------------------------------------
    // Template
    // ---------------------------------------------------------------

    _template() {
        return `
            <section>
                <h2>Assembly Analysis</h2>
                <p>Select an uploaded STEP file to inspect its assembly hierarchy.</p>

                <div class="analysis-controls">
                    <select id="file-select" aria-label="Select STEP file">
                        <option value="">Loading files...</option>
                    </select>
                    <button id="analyze-btn" disabled>Analyze</button>
                </div>

                <div id="analysis-error" hidden></div>
                <div id="analysis-loading" hidden>
                    <p aria-busy="true"><span id="analysis-status-msg">Analysing assembly structure...</span> <span id="elapsed-timer"></span></p>
                </div>
            </section>

            <section id="tree-results" hidden>
                <div id="analysis-summary"></div>
                <div id="stl-progress" class="stl-progress" hidden></div>

                <div class="analysis-workspace">
                    <div class="workspace-tree-panel">
                        <div id="assembly-tree-container" class="assembly-tree"></div>
                    </div>
                    <div class="workspace-viewer-panel">
                        <div id="stl-viewer-panel" class="stl-viewer-panel">
                            <div class="stl-viewer-placeholder">
                                Click a node in the tree to preview its 3D model
                            </div>
                        </div>
                    </div>
                </div>

                <div id="classification-tables" class="classification-tables-section" hidden></div>
            </section>
        `;
    }

    // ---------------------------------------------------------------
    // Event binding
    // ---------------------------------------------------------------

    _bindEvents() {
        const select = this.container.querySelector('#file-select');
        const btn = this.container.querySelector('#analyze-btn');

        select.addEventListener('change', () => {
            btn.disabled = !select.value;
        });

        btn.addEventListener('click', () => {
            const filename = select.value;
            if (filename) this._analyze(filename);
        });
    }

    // ---------------------------------------------------------------
    // File list
    // ---------------------------------------------------------------

    async _loadFiles() {
        const select = this.container.querySelector('#file-select');
        try {
            const data = await this.api.listFiles();
            const files = data.files || [];
            if (files.length === 0) {
                select.innerHTML = '<option value="">No STEP files uploaded yet</option>';
                return;
            }
            select.innerHTML = '<option value="">-- Choose a file --</option>' +
                files.map(f => {
                    const display = f.display_name || f.filename;
                    const date = f.uploaded_at ? new Date(f.uploaded_at).toLocaleString() : '';
                    const label = date ? `${date}  —  ${display}` : display;
                    return `<option value="${this._esc(f.filename)}">${this._esc(label)}</option>`;
                }).join('');
        } catch {
            select.innerHTML = '<option value="">Failed to load files</option>';
        }
    }

    // ---------------------------------------------------------------
    // Analyze
    // ---------------------------------------------------------------

    async _analyze(filename) {
        const errorEl = this.container.querySelector('#analysis-error');
        const loadingEl = this.container.querySelector('#analysis-loading');
        const resultsEl = this.container.querySelector('#tree-results');
        const btn = this.container.querySelector('#analyze-btn');

        errorEl.hidden = true;
        resultsEl.hidden = true;
        loadingEl.hidden = false;
        btn.disabled = true;
        this._cleanup();
        this._currentFilename = filename;

        // Elapsed timer — keep running until analysis finishes (including poll time)
        const timerEl = loadingEl.querySelector('#elapsed-timer');
        const start = Date.now();
        const elapsedTimer = setInterval(() => {
            const secs = Math.floor((Date.now() - start) / 1000);
            if (timerEl) timerEl.textContent = `${secs}s elapsed`;
        }, 1000);

        const done = (data) => {
            clearInterval(elapsedTimer);
            loadingEl.hidden = true;
            btn.disabled = false;
            this._handleAnalysisResult(data, filename);
        };

        const fail = (err) => {
            clearInterval(elapsedTimer);
            loadingEl.hidden = true;
            btn.disabled = false;
            const msg = err?.detail?.message || err?.error || err?.message || 'Analysis failed.';
            errorEl.hidden = false;
            errorEl.innerHTML = `<article class="result-card fail"><header><strong>Analysis Error</strong></header><div class="result-card-body"><p>${this._esc(msg)}</p></div></article>`;
            errorEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };

        try {
            const data = await this.api.getAssemblyTree(filename);

            if (data.analysis_task_id) {
                // Server is parsing in background — poll for completion
                this._pollAnalysisTask(data.analysis_task_id, filename, done, fail);
            } else {
                // Cache hit — instant result
                done(data);
            }
        } catch (err) {
            fail(err);
        }
    }

    /**
     * Poll GET /analysis/status/{taskId} until analysis completes or fails.
     * The timer is stored on this._analysisPollTimer so _cleanup() can cancel it.
     *
     * The try/catch wraps ONLY the API call so that errors thrown by done() or
     * fail() propagate normally instead of being silently swallowed.
     */
    _pollAnalysisTask(taskId, filename, done, fail) {
        const statusMsgEl = this.container.querySelector('#analysis-status-msg');
        const setMsg = (msg) => { if (statusMsgEl) statusMsgEl.textContent = msg; };

        this._analysisPollTimer = setInterval(async () => {
            let data;
            try {
                data = await this.api.getAnalysisStatus(taskId);
            } catch (err) {
                if (err?.status === 404) {
                    // Server lost the task — likely a container restart
                    clearInterval(this._analysisPollTimer);
                    this._analysisPollTimer = null;
                    fail({ error: 'Analysis task was lost (server may have restarted). Please try again.' });
                }
                // Other network hiccup — silently retry next tick
                return;
            }

            if (data.status === 'completed') {
                clearInterval(this._analysisPollTimer);
                this._analysisPollTimer = null;
                done(data);
            } else if (data.status === 'failed') {
                clearInterval(this._analysisPollTimer);
                this._analysisPollTimer = null;
                fail(data);
            } else {
                // pending or running — update status message so user can see progress
                const label = data.status === 'running'
                    ? 'Parsing STEP file...'
                    : 'Waiting for worker thread...';
                setMsg(label);
            }
        }, 2000);
    }

    /**
     * Render the tree and kick off STL polling from a completed analysis response.
     * Used by both the instant cache-hit path and the async poll path.
     */
    _handleAnalysisResult(data, filename) {
        this._renderTree(data);

        if (data.project_state) {
            this._restoreProjectState(data.project_state);
        }

        // Only trigger STL generation if we don't already have cached STL results.
        // _restoreProjectState populates stlMap from project_state.stl_map, so if
        // the user has prior results this check is false and we skip generation.
        if (this.stlMap.size === 0) {
            this._triggerSTL(filename);
        }
    }

    /**
     * Explicitly request STL generation from the backend and start polling.
     * The POST endpoint is idempotent — it returns the existing task if one
     * is already pending, running, or completed.
     */
    async _triggerSTL(filename) {
        try {
            const data = await this.api.generateSTL(filename);
            if (data.task_id) {
                this._pollSTLProgress(data.task_id, filename);
            }
        } catch (err) {
            console.error('Failed to trigger STL generation:', err);
        }
    }

    // ---------------------------------------------------------------
    // Tree rendering
    // ---------------------------------------------------------------

    _renderTree(data) {
        const resultsEl = this.container.querySelector('#tree-results');
        const summaryEl = this.container.querySelector('#analysis-summary');
        const treeEl = this.container.querySelector('#assembly-tree-container');

        resultsEl.hidden = false;

        const s = data.summary || {};
        summaryEl.innerHTML = `
            <div class="analysis-summary">
                <div class="summary-stat"><strong>${s.total_assemblies || 0}</strong> Assemblies</div>
                <div class="summary-stat"><strong>${s.total_parts || 0}</strong> Parts</div>
                <div class="summary-stat"><strong>${s.total_solids || 0}</strong> Solids</div>
            </div>
        `;

        const nodes = data.assembly_tree || [];
        treeEl.innerHTML = '<ul>' + nodes.map(n => this._renderNode(n, 0)).join('') + '</ul>';

        this._buildParentMap(nodes, null);
        this._bindTreeEvents(treeEl);
    }

    _renderNode(node, depth) {
        const hasChildren = node.children && node.children.length > 0;
        const toggleClass = hasChildren ? 'expanded' : 'leaf';
        const childrenHtml = hasChildren
            ? '<ul>' + node.children.map(c => this._renderNode(c, depth + 1)).join('') + '</ul>'
            : '';

        const badgeLabel = this._badgeLabel(node.node_type);
        const solidInfo = node.node_type !== 'assembly' && node.solid_count !== undefined
            ? `<span class="node-solid-count">${node.solid_count} solid${node.solid_count !== 1 ? 's' : ''}</span>`
            : '';

        const actions = this._actionsHtml(node);

        const instanceRef = node.instance_ref && node.instance_ref !== node.name
            ? `<span class="node-instance-ref" title="Instance reference">${this._esc(node.instance_ref)}</span>`
            : '';

        const spinner = '<span class="node-stl-spinner" hidden></span>';
        const refId = node.ref_id || node.id;
        const isMirrored = !!node.is_mirrored;
        // Chirality key: same ref + same mirror state share STLs; different states need separate STLs
        const chiralKey = `${refId}:${isMirrored ? 'M' : 'N'}`;
        const mirroredBadge = isMirrored
            ? '<span class="node-mirrored-badge" title="Mirrored instance">Mirror</span>'
            : '';

        return `
            <li class="tree-node" data-node-id="${this._esc(node.id)}" data-node-type="${this._esc(node.node_type)}" data-node-name="${this._esc(node.name)}" data-ref-id="${this._esc(refId)}" data-chiral-key="${this._esc(chiralKey)}" data-is-mirrored="${isMirrored}" data-solid-count="${node.solid_count || 0}" data-depth="${depth}">
                <div class="tree-node-row">
                    <button class="tree-toggle ${toggleClass}" aria-label="Toggle">\u25B6</button>
                    <span class="tree-node-name">${this._esc(node.name)}</span>
                    ${instanceRef}
                    <span class="node-type-badge ${this._esc(node.node_type)}">${badgeLabel}</span>
                    ${mirroredBadge}
                    ${solidInfo}
                    ${spinner}
                    ${actions}
                </div>
                ${childrenHtml}
            </li>
        `;
    }

    _badgeLabel(nodeType) {
        const labels = {
            assembly: 'Assembly',
            part_single_solid: 'Part',
            part_multi_solid: 'Multi-solid',
            part_no_solid: 'No solid',
            solid: 'Solid',
        };
        return labels[nodeType] || nodeType;
    }

    // ---------------------------------------------------------------
    // Classification summary tables
    // ---------------------------------------------------------------

    /**
     * Walk the assembly tree data and build a map of nodeId → {name, parentName}.
     * Called once after rendering so lookups are O(1) when updating the tables.
     */
    _buildParentMap(nodes, parentName) {
        for (const node of nodes) {
            this._parentMap.set(node.id, { name: node.name, parentName });
            if (node.children && node.children.length > 0) {
                this._buildParentMap(node.children, node.name);
            }
        }
    }

    /**
     * Re-render the two classification summary tables from the current
     * this.classifications map. Called after every classify or restore.
     *
     * Deduplicates by ref_id so each unique part type appears exactly once
     * with a Qty column showing the number of classified instances.
     */
    _updateClassificationTables() {
        const el = this.container.querySelector('#classification-tables');
        if (!el) return;

        const treeEl = this.container.querySelector('#assembly-tree-container');

        // Group by ref_id within each action bucket.
        // Map: refId -> { name, usedIn, qty }
        const postprocess = new Map();
        const bought = new Map();

        for (const [nodeId, action] of this.classifications) {
            const targetMap = action === 'postprocess' ? postprocess
                            : action === 'bought-out'  ? bought
                            : null;
            if (!targetMap) continue;

            // Resolve ref_id for deduplication
            const domEl = treeEl?.querySelector(`.tree-node[data-node-id="${CSS.escape(nodeId)}"]`);
            const refId = domEl?.dataset.refId || nodeId;

            if (targetMap.has(refId)) {
                targetMap.get(refId).qty++;
                continue;
            }

            // First occurrence — resolve name and parent
            let name, usedIn;
            const info = this._parentMap.get(nodeId);
            if (info) {
                name = info.name;
                usedIn = info.parentName || '—';
            } else if (domEl) {
                name = domEl.dataset.nodeName || nodeId;
                const parentAssembly = domEl.parentElement?.closest('.tree-node[data-node-type="assembly"]');
                usedIn = parentAssembly?.dataset.nodeName || '—';
            } else {
                continue;
            }

            targetMap.set(refId, { name, usedIn, qty: 1 });
        }

        const ppRows = Array.from(postprocess.values());
        const boRows = Array.from(bought.values());

        const hasData = ppRows.length > 0 || boRows.length > 0;
        el.hidden = !hasData;
        if (!hasData) return;

        const renderTable = (rows, label, cls) => `
            <div class="classification-table-card">
                <div class="classification-table-header ${cls}">
                    <span>${label}</span>
                    <span class="classification-table-count">${rows.length}</span>
                </div>
                ${rows.length === 0
                    ? '<p class="classification-table-empty">None classified yet</p>'
                    : `<table class="classification-table">
                        <thead><tr><th>Part</th><th>Used In</th><th class="classification-table-qty-col">Qty</th></tr></thead>
                        <tbody>${rows.map(r =>
                            `<tr><td>${this._esc(r.name)}</td><td class="classification-table-parent">${this._esc(r.usedIn)}</td><td class="classification-table-qty">${r.qty}</td></tr>`
                        ).join('')}</tbody>
                    </table>`
                }
            </div>
        `;

        el.innerHTML = `
            <div class="classification-tables-grid">
                ${renderTable(ppRows, 'Postprocess', 'postprocess')}
                ${renderTable(boRows, 'Bought Out', 'bought-out')}
            </div>
        `;
    }

    _actionsHtml(node) {
        const btns = [];
        switch (node.node_type) {
            case 'assembly':
                btns.push('<button class="btn-explode" data-action="explode" hidden>Explode</button>');
                break;
            case 'part_multi_solid':
                btns.push('<button class="btn-explode" data-action="explode" hidden>Explode</button>');
                btns.push('<button class="btn-bought-out" data-action="bought-out" hidden>Bought-out</button>');
                break;
            case 'part_single_solid':
                btns.push('<button class="btn-postprocess" data-action="postprocess" hidden>Postprocess</button>');
                btns.push('<button class="btn-bought-out" data-action="bought-out" hidden>Bought-out</button>');
                break;
            case 'part_no_solid':
                btns.push('<button class="btn-bought-out" data-action="bought-out" hidden>Bought-out</button>');
                break;
        }
        return btns.length ? `<span class="node-actions">${btns.join('')}</span>` : '';
    }

    // ---------------------------------------------------------------
    // Tree event handling
    // ---------------------------------------------------------------

    _bindTreeEvents(treeEl) {
        treeEl.addEventListener('click', (e) => {
            // 1. Toggle expand/collapse
            const toggle = e.target.closest('.tree-toggle:not(.leaf)');
            if (toggle) {
                const li = toggle.closest('.tree-node');
                const childUl = li.querySelector(':scope > ul');
                if (childUl) {
                    const collapsed = childUl.hidden;
                    childUl.hidden = !collapsed;
                    toggle.classList.toggle('expanded', collapsed);
                }
                return;
            }

            // 2. Explode button (assembly or multi-solid)
            const explodeBtn = e.target.closest('.btn-explode');
            if (explodeBtn) {
                const li = explodeBtn.closest('.tree-node');
                this._explodeNode(li);
                return;
            }

            // 3. Classification buttons (Postprocess, Bought-out)
            const actionBtn = e.target.closest('[data-action]:not(.btn-explode)');
            if (actionBtn) {
                const li = actionBtn.closest('.tree-node');
                const nodeId = li.dataset.nodeId;
                const action = actionBtn.dataset.action;
                this._classifyNode(li, nodeId, action);
                return;
            }

            // 4. Click on a selectable node row -> load STL in viewer
            const row = e.target.closest('.tree-node-row.node-selectable');
            if (row) {
                const li = row.closest('.tree-node');
                const nodeId = li.dataset.nodeId;
                this._selectNodeForPreview(nodeId);
            }
        });
    }

    // ---------------------------------------------------------------
    // Selectability state
    // ---------------------------------------------------------------

    _updateTreeSelectability() {
        const treeEl = this.container.querySelector('#assembly-tree-container');
        if (!treeEl) return;

        for (const li of treeEl.querySelectorAll('.tree-node')) {
            const nodeId = li.dataset.nodeId;
            const hasStl = this.stlMap.has(nodeId);
            const isExploded = this.explodedNodes.has(nodeId);
            const isClassified = this.classifications.has(nodeId);
            const row = li.querySelector(':scope > .tree-node-row');

            // Selectable = has an STL, not classified, not exploded
            const selectable = hasStl && !isClassified && !isExploded;
            row.classList.toggle('node-selectable', selectable);

            // Explode button: visible if has STL and not already exploded
            const explodeBtn = row.querySelector('.btn-explode');
            if (explodeBtn) {
                explodeBtn.hidden = !hasStl || isExploded;
            }

            // Action buttons (postprocess, bought-out): visible when has STL, not classified, not exploded
            for (const btn of row.querySelectorAll('.btn-postprocess, .btn-bought-out')) {
                btn.hidden = !hasStl || isClassified || isExploded;
            }

            row.classList.toggle('node-selected', this._selectedNodeId === nodeId);
        }
    }

    /**
     * Populate stlMap from task results (which include node_id + stl_file).
     */
    _populateStlMap(results, filename) {
        if (!results) return;
        const runId = filename.substring(0, 8);
        for (const r of results) {
            if (r.stl_file && r.node_id) {
                this.stlMap.set(r.node_id, `/outputs/stl/${runId}/${r.stl_file}`);
            }
        }
        this._debouncedSave();
    }

    // ---------------------------------------------------------------
    // Click-to-preview
    // ---------------------------------------------------------------

    _selectNodeForPreview(nodeId) {
        if (this._selectedNodeId === nodeId) return;

        const url = this.stlMap.get(nodeId);
        if (!url) return;

        this._selectedNodeId = nodeId;

        // Update tree highlighting
        const treeEl = this.container.querySelector('#assembly-tree-container');
        for (const row of treeEl.querySelectorAll('.tree-node-row')) {
            const li = row.closest('.tree-node');
            row.classList.toggle('node-selected', li.dataset.nodeId === nodeId);
        }

        this._loadInViewer(url);
    }

    _loadInViewer(url) {
        const panel = this.container.querySelector('#stl-viewer-panel');
        if (!panel) return;

        const doLoad = () => {
            if (!this._viewer) {
                panel.innerHTML = '';
                this._viewer = new STLViewer(panel);
            }

            panel.classList.add('loading');
            this._viewer.loadSTL(url)
                .then(() => { panel.classList.remove('loading'); })
                .catch(() => { panel.classList.remove('loading'); });
        };

        if (!this._viewer) {
            requestAnimationFrame(doLoad);
        } else {
            doLoad();
        }
    }

    // ---------------------------------------------------------------
    // Root-level STL progress polling
    // ---------------------------------------------------------------

    _pollSTLProgress(taskId, filename) {
        const progressEl = this.container.querySelector('#stl-progress');
        if (!progressEl) return;

        progressEl.hidden = false;
        progressEl.innerHTML = '<p class="stl-progress-text">Generating 3D previews...</p><progress></progress>';

        this._stlPollTimer = setInterval(async () => {
            try {
                const status = await this.api.getSTLStatus(taskId);

                if (status.status === 'running') {
                    const pct = status.progress || 0;
                    const current = status.current_item || '';
                    const completed = status.completed_items || 0;
                    const total = status.total_items || 0;
                    const label = total > 0
                        ? `Generating STL ${completed}/${total}${current ? ': ' + current : ''}...`
                        : 'Generating 3D previews...';
                    progressEl.innerHTML = `<p class="stl-progress-text">${this._esc(label)}</p><progress value="${pct}" max="100"></progress>`;
                } else if (status.status === 'completed') {
                    clearInterval(this._stlPollTimer);
                    this._stlPollTimer = null;
                    progressEl.innerHTML = '<p class="stl-progress-text stl-progress-done">3D previews ready</p>';
                    setTimeout(() => { progressEl.hidden = true; }, 3000);

                    this._populateStlMap(status.results, filename);

                    // Auto-mark root assembly nodes as exploded (their children are the ones with STLs)
                    const treeEl = this.container.querySelector('#assembly-tree-container');
                    if (treeEl) {
                        for (const rootLi of treeEl.querySelectorAll(':scope > ul > .tree-node')) {
                            if (rootLi.dataset.nodeType === 'assembly') {
                                this.explodedNodes.add(rootLi.dataset.nodeId);
                            }
                        }
                    }

                    this._updateTreeSelectability();

                    // Auto-select first available node
                    const firstId = Array.from(this.stlMap.keys())[0];
                    if (firstId) {
                        this._selectNodeForPreview(firstId);
                    }
                } else if (status.status === 'failed') {
                    clearInterval(this._stlPollTimer);
                    this._stlPollTimer = null;
                    progressEl.innerHTML = `<p class="stl-progress-text stl-progress-error">STL generation failed: ${this._esc(status.error || 'unknown error')}</p>`;
                }
            } catch {
                // Silently retry on transient network issues
            }
        }, 2000);
    }

    // ---------------------------------------------------------------
    // Explode workflow
    // ---------------------------------------------------------------

    /**
     * Find sibling instances with the same chiral-key (ref_id + mirror state) and
     * node_type that are not already exploded.
     *
     * Chirality matters for STL generation: a mirrored instance produces a
     * geometrically different STL and must not be pooled with its non-mirrored
     * counterparts. Classification sync (in _classifyNode) ignores chirality
     * because the logical part is the same regardless of orientation.
     */
    _findSiblings(chiralKey, excludeNodeId, nodeType) {
        if (!chiralKey) return [];
        const treeEl = this.container.querySelector('#assembly-tree-container');
        if (!treeEl) return [];
        const sel = `.tree-node[data-chiral-key="${CSS.escape(chiralKey)}"][data-node-type="${CSS.escape(nodeType)}"]`;
        return Array.from(treeEl.querySelectorAll(sel))
            .filter(li => li.dataset.nodeId !== excludeNodeId && !this.explodedNodes.has(li.dataset.nodeId));
    }

    /**
     * Explode an assembly or multi-solid node, including chiral siblings.
     */
    async _explodeNode(li) {
        const nodeId = li.dataset.nodeId;
        const nodeType = li.dataset.nodeType;
        const chiralKey = li.dataset.chiralKey;

        if (this.explodedNodes.has(nodeId)) return;

        // Only group siblings that share the same chirality (mirror state)
        const siblings = this._findSiblings(chiralKey, nodeId, nodeType);

        if (nodeType === 'assembly') {
            await this._explodeAssembly(li, siblings);
        } else if (nodeType === 'part_multi_solid') {
            await this._explodeMultiSolid(li, siblings);
        }
    }

    /**
     * Mark a single node as visually exploded.
     */
    _markExploded(li) {
        const nodeId = li.dataset.nodeId;
        this.explodedNodes.add(nodeId);

        const row = li.querySelector(':scope > .tree-node-row');
        row.classList.add('node-exploded');

        const explodeBtn = row.querySelector('.btn-explode');
        if (explodeBtn) explodeBtn.hidden = true;

        // Auto-expand children
        const childUl = li.querySelector(':scope > ul');
        const toggleBtn = row.querySelector('.tree-toggle');
        if (childUl && childUl.hidden) {
            childUl.hidden = false;
            if (toggleBtn) toggleBtn.classList.add('expanded');
        }
    }

    _showChildSpinners(li) {
        const childUl = li.querySelector(':scope > ul');
        if (!childUl) return;
        for (const childLi of childUl.querySelectorAll(':scope > .tree-node')) {
            if (!this.stlMap.has(childLi.dataset.nodeId)) {
                const spinner = childLi.querySelector(':scope > .tree-node-row > .node-stl-spinner');
                if (spinner) spinner.hidden = false;
            }
        }
    }

    _hideChildSpinners(li) {
        const childUl = li.querySelector(':scope > ul');
        if (!childUl) return;
        for (const spinner of childUl.querySelectorAll(':scope > .tree-node > .tree-node-row > .node-stl-spinner')) {
            spinner.hidden = true;
        }
    }

    // -- Assembly explode --

    async _explodeAssembly(li, siblings) {
        const nodeId = li.dataset.nodeId;
        const filename = this._currentFilename;

        // Mark this node + all siblings as exploded
        this._markExploded(li);
        for (const sib of siblings) this._markExploded(sib);

        this._debouncedSave();

        // Check if children already have STLs (from sibling explosion or cache)
        const childUl = li.querySelector(':scope > ul');
        if (childUl) {
            const children = Array.from(childUl.querySelectorAll(':scope > .tree-node'));
            if (children.length > 0 && children.every(c => this.stlMap.has(c.dataset.nodeId))) {
                this._updateTreeSelectability();
                return;
            }
        }

        // Show spinners on children that don't have STLs yet
        const allLis = [li, ...siblings];
        for (const el of allLis) this._showChildSpinners(el);

        try {
            const data = await this.api.generateSTLChildren(filename, nodeId);
            if (data.task_id) {
                this._pollAssemblyExplodeTask(data.task_id, filename, allLis);
            }
        } catch (err) {
            console.error('Explode STL generation failed:', err);
            for (const el of allLis) this._hideChildSpinners(el);
        }
    }

    _pollAssemblyExplodeTask(taskId, filename, allLis) {
        const key = `assembly:${taskId}`;
        const timer = setInterval(async () => {
            try {
                const status = await this.api.getSTLStatus(taskId);

                if (status.status === 'completed') {
                    clearInterval(timer);
                    this._explodePollTimers.delete(key);

                    this._populateStlMap(status.results, filename);
                    for (const el of allLis) this._hideChildSpinners(el);
                    this._updateTreeSelectability();

                } else if (status.status === 'failed') {
                    clearInterval(timer);
                    this._explodePollTimers.delete(key);
                    for (const el of allLis) this._hideChildSpinners(el);
                    console.error('Explode task failed:', status.error);
                }
            } catch {
                // Silently retry
            }
        }, 2000);

        this._explodePollTimers.set(key, timer);
    }

    // -- Multi-solid explode --

    async _explodeMultiSolid(li, siblings) {
        const nodeId = li.dataset.nodeId;
        const chiralKey = li.dataset.chiralKey;
        const filename = this._currentFilename;

        // Mark this node + all siblings as exploded
        this._markExploded(li);
        for (const sib of siblings) this._markExploded(sib);

        // Check if solid children are already cached (from sibling explosion, same chirality)
        const cached = this._solidChildrenCache.get(chiralKey);
        if (cached) {
            const allLis = [li, ...siblings];
            for (const el of allLis) this._createSolidChildrenDOM(el, cached);
            this._updateTreeSelectability();
            this._debouncedSave();
            return;
        }

        // Show spinner on the multi-solid node itself while generating
        const allLis = [li, ...siblings];
        for (const el of allLis) {
            const spinner = el.querySelector(':scope > .tree-node-row > .node-stl-spinner');
            if (spinner) spinner.hidden = false;
        }

        this._debouncedSave();

        try {
            const data = await this.api.generateSTLSolids(filename, nodeId);
            if (data.task_id) {
                this._pollSolidsTask(data.task_id, filename, chiralKey, allLis);
            }
        } catch (err) {
            console.error('Multi-solid explode failed:', err);
            for (const el of allLis) {
                const spinner = el.querySelector(':scope > .tree-node-row > .node-stl-spinner');
                if (spinner) spinner.hidden = true;
            }
        }
    }

    _pollSolidsTask(taskId, filename, chiralKey, allLis) {
        const key = `solids:${chiralKey}`;
        const timer = setInterval(async () => {
            try {
                const status = await this.api.getSTLStatus(taskId);

                if (status.status === 'completed') {
                    clearInterval(timer);
                    this._explodePollTimers.delete(key);

                    this._populateStlMap(status.results, filename);

                    // Cache solid children info keyed by chiralKey
                    const childrenInfo = (status.results || [])
                        .filter(r => r.stl_file && r.node_id)
                        .map(r => ({ name: r.name, nodeId: r.node_id }));
                    this._solidChildrenCache.set(chiralKey, childrenInfo);

                    // Create DOM children and hide spinners
                    for (const el of allLis) {
                        this._createSolidChildrenDOM(el, childrenInfo);
                        const spinner = el.querySelector(':scope > .tree-node-row > .node-stl-spinner');
                        if (spinner) spinner.hidden = true;
                    }

                    this._updateTreeSelectability();
                    this._debouncedSave();

                } else if (status.status === 'failed') {
                    clearInterval(timer);
                    this._explodePollTimers.delete(key);
                    for (const el of allLis) {
                        const spinner = el.querySelector(':scope > .tree-node-row > .node-stl-spinner');
                        if (spinner) spinner.hidden = true;
                    }
                    console.error('Solids task failed:', status.error);
                }
            } catch {
                // Silently retry
            }
        }, 2000);

        this._explodePollTimers.set(key, timer);
    }

    /**
     * Dynamically add synthetic child <li> elements for individual solids.
     */
    _createSolidChildrenDOM(li, childrenInfo) {
        // Remove any existing synthetic children UL (idempotent)
        const existingUl = li.querySelector(':scope > ul');
        if (existingUl) existingUl.remove();

        const childUl = document.createElement('ul');
        const parentDepth = parseInt(li.dataset.depth) || 0;

        for (const child of childrenInfo) {
            const childLi = document.createElement('li');
            childLi.className = 'tree-node';
            childLi.dataset.nodeId = child.nodeId;
            childLi.dataset.nodeType = 'solid';
            childLi.dataset.nodeName = child.name;
            childLi.dataset.refId = child.nodeId;
            childLi.dataset.solidCount = '0';
            childLi.dataset.depth = String(parentDepth + 1);

            childLi.innerHTML = `
                <div class="tree-node-row">
                    <button class="tree-toggle leaf" aria-label="Toggle">\u25B6</button>
                    <span class="tree-node-name">${this._esc(child.name)}</span>
                    <span class="node-type-badge solid">Solid</span>
                    <span class="node-stl-spinner" hidden></span>
                    <span class="node-actions">
                        <button class="btn-postprocess" data-action="postprocess" hidden>Postprocess</button>
                        <button class="btn-bought-out" data-action="bought-out" hidden>Bought-out</button>
                    </span>
                </div>
            `;
            childUl.appendChild(childLi);
        }

        li.appendChild(childUl);

        // Update toggle button: was leaf, now has children
        const toggle = li.querySelector(':scope > .tree-node-row > .tree-toggle');
        if (toggle) {
            toggle.classList.remove('leaf');
            toggle.classList.add('expanded');
        }
    }

    // ---------------------------------------------------------------
    // Classification (Postprocess / Bought-out)
    // ---------------------------------------------------------------

    _classifyNode(li, nodeId, action) {
        const refId = li.dataset.refId;
        const nodeType = li.dataset.nodeType;
        const treeEl = this.container.querySelector('#assembly-tree-container');

        // Collect all nodes to classify:
        //   • all DOM elements with the same node_id (duplicated due to shared prototype subtrees)
        //   • all instances sharing the same ref_id and node_type (sibling instances of the same prototype)
        const toClassify = new Set([li]);

        if (treeEl) {
            for (const el of treeEl.querySelectorAll(
                `.tree-node[data-node-id="${CSS.escape(nodeId)}"]`
            )) {
                toClassify.add(el);
            }
            if (refId) {
                for (const el of treeEl.querySelectorAll(
                    `.tree-node[data-ref-id="${CSS.escape(refId)}"][data-node-type="${CSS.escape(nodeType)}"]`
                )) {
                    toClassify.add(el);
                }
            }
        }

        for (const el of toClassify) {
            const elNodeId = el.dataset.nodeId;
            this.classifications.set(elNodeId, action);
            el.classList.add('node-classified');
            el.dataset.classification = action;

            const actionsEl = el.querySelector('.node-actions');
            if (actionsEl) actionsEl.hidden = true;

            const row = el.querySelector(':scope > .tree-node-row');
            if (row) row.classList.remove('node-selectable');
        }

        this._updateClassificationTables();
        this._debouncedSave();
    }

    // ---------------------------------------------------------------
    // Project state persistence
    // ---------------------------------------------------------------

    _restoreProjectState(state) {
        const treeEl = this.container.querySelector('#assembly-tree-container');
        if (!treeEl) return;

        // Restore stlMap
        if (state.stl_map) {
            for (const [nodeId, url] of Object.entries(state.stl_map)) {
                this.stlMap.set(nodeId, url);
            }
        }

        // Restore solid children cache (keyed by chiralKey: "<ref_id>:M" or "<ref_id>:N")
        if (state.solid_children) {
            for (const [chiralKey, children] of Object.entries(state.solid_children)) {
                this._solidChildrenCache.set(chiralKey, children);
            }
        }

        // Restore exploded nodes
        if (state.exploded_nodes) {
            for (const nodeId of state.exploded_nodes) {
                this.explodedNodes.add(nodeId);
                // Apply to ALL matching DOM elements (shared refs = duplicate node_ids)
                const allLis = treeEl.querySelectorAll(`.tree-node[data-node-id="${CSS.escape(nodeId)}"]`);
                for (const li of allLis) {
                    const row = li.querySelector(':scope > .tree-node-row');
                    if (row) row.classList.add('node-exploded');
                    // Auto-expand
                    const childUl = li.querySelector(':scope > ul');
                    const toggleBtn = row?.querySelector('.tree-toggle');
                    if (childUl && childUl.hidden) {
                        childUl.hidden = false;
                        if (toggleBtn) toggleBtn.classList.add('expanded');
                    }
                    // Recreate synthetic children for multi-solid nodes
                    if (li.dataset.nodeType === 'part_multi_solid') {
                        const chiralKey = li.dataset.chiralKey;
                        const cached = this._solidChildrenCache.get(chiralKey);
                        if (cached) {
                            this._createSolidChildrenDOM(li, cached);
                        }
                    }
                }
            }
        }

        // Restore classifications
        if (state.classifications) {
            for (const [nodeId, action] of Object.entries(state.classifications)) {
                this.classifications.set(nodeId, action);
                const allLis = treeEl.querySelectorAll(`.tree-node[data-node-id="${CSS.escape(nodeId)}"]`);
                for (const li of allLis) {
                    li.classList.add('node-classified');
                    li.dataset.classification = action;
                    const actions = li.querySelector('.node-actions');
                    if (actions) actions.hidden = true;
                }
            }
        }

        // Apply selectability based on restored state
        this._updateTreeSelectability();
        this._updateClassificationTables();

        // Auto-select first available node
        if (this.stlMap.size > 0) {
            const firstId = Array.from(this.stlMap.keys())[0];
            if (firstId) {
                this._selectNodeForPreview(firstId);
            }
        }
    }

    _debouncedSave() {
        if (this._saveTimer) clearTimeout(this._saveTimer);
        this._saveTimer = setTimeout(() => {
            this._saveTimer = null;
            this._saveProjectState();
        }, 1000);
    }

    async _saveProjectState() {
        const filename = this._currentFilename;
        if (!filename) return;

        const state = {
            classifications: Object.fromEntries(this.classifications),
            exploded_nodes: Array.from(this.explodedNodes),
            stl_map: Object.fromEntries(this.stlMap),
            solid_children: Object.fromEntries(this._solidChildrenCache),
        };

        try {
            await this.api.saveProjectState(filename, state);
        } catch (err) {
            console.error('Failed to save project state:', err);
        }
    }

    // ---------------------------------------------------------------
    // Utilities
    // ---------------------------------------------------------------

    _formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    _esc(str) {
        const el = document.createElement('span');
        el.textContent = str;
        return el.innerHTML;
    }
}
