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

        /** @type {Map<string, string>} nodeId -> refId (populated from tree data, not DOM) */
        this._nodeRefMap = new Map();

        /** @type {Map<string, number>} key -> setInterval timer id */
        this._explodePollTimers = new Map();

        /** @type {string|null} currently selected node shown in viewer */
        this._selectedNodeId = null;

        /** @type {Map<string, number>|null} nodeId -> mesh index when multi-solid scene is loaded */
        this._multiSolidMeshMap = null;
        /** @type {string|null} nodeId of the parent multi-solid currently loaded as scene */
        this._multiSolidParentId = null;
        /** @type {Array<number>|null} default colors for each mesh in the current scene */
        this._multiSolidDefaultColors = null;

        /** @type {Map<string, number[]>} nodeId -> 4x4 column-major placement matrix */
        this._placements = new Map();

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

        /** @type {Array|null} cached part-level consolidation groups (null = not yet loaded) */
        this._consolidationGroups = null;

        /** @type {Array|null} cached solid-level consolidation groups (cross-ref) */
        this._solidConsolidationGroups = null;

        /** @type {Array|null} cached intra-part solid consolidation groups */
        this._intraSolidGroups = null;

        /** @type {number|null} consolidation poll timer */
        this._consolidatePollTimer = null;

        /** @type {boolean} true while a consolidation task is running */
        this._consolidating = false;

        /** @type {Array|null} raw assembly_tree nodes from API response */
        this._treeData = null;

        /** @type {Map<string, number>} refId -> total instance count in tree (for qty display) */
        this._refIdInstanceCount = new Map();

        /** @type {Object|null} CNC analysis results keyed by ref_id (null = not yet loaded) */
        this._cncAnalysisResults = null;

        /** @type {number|null} CNC analysis poll timer */
        this._cncPollTimer = null;

        /** @type {boolean} true while a CNC analysis task is running */
        this._cncAnalysing = false;

        /** @type {string} last project number entered by user */
        this._lastProjectNumber = '';

        /** @type {string} last steel grade entered by user */
        this._lastSteelGrade = 'S275';

        // ── Nesting state ──
        /** @type {string|null} nesting task id from the nesting service */
        this._nestingTaskId = null;
        /** @type {number|null} nesting poll timer */
        this._nestingPollTimer = null;
        /** @type {boolean} true while a nesting job is running */
        this._nestingRunning = false;
        /** @type {Object|null} nesting result from the service */
        this._nestingResult = null;
        /** @type {Object|null} cutting list from the service */
        this._nestingCuttingList = null;
        /** @type {number} default stock length in mm */
        this._nestingDefaultStockLength = 6000;
        /** @type {number} default stock qty */
        this._nestingDefaultStockQty = 20;
        /** @type {number} kerf in mm */
        this._nestingKerf = 3;
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
        this._multiSolidMeshMap = null;
        this._multiSolidParentId = null;
        this._multiSolidDefaultColors = null;
        if (this._saveTimer) {
            clearTimeout(this._saveTimer);
            this._saveTimer = null;
        }
        if (this._consolidatePollTimer) {
            clearInterval(this._consolidatePollTimer);
            this._consolidatePollTimer = null;
        }
        if (this._cncPollTimer) {
            clearInterval(this._cncPollTimer);
            this._cncPollTimer = null;
        }
        if (this._nestingPollTimer) {
            clearInterval(this._nestingPollTimer);
            this._nestingPollTimer = null;
        }
        this._nestingTaskId = null;
        this._nestingRunning = false;
        this._nestingResult = null;
        this._nestingCuttingList = null;
        this.stlMap.clear();
        this.explodedNodes.clear();
        this.classifications.clear();
        this._solidChildrenCache.clear();
        this._parentMap.clear();
        this._nodeRefMap.clear();
        this._refIdInstanceCount.clear();
        this._selectedNodeId = null;
        this._currentFilename = null;
        this._consolidationGroups = null;
        this._solidConsolidationGroups = null;
        this._intraSolidGroups = null;
        this._consolidating = false;
        this._treeData = null;
        this._cncAnalysisResults = null;
        this._cncAnalysing = false;
        this._projectStateRestored = false;
        // Note: _lastProjectNumber and _lastSteelGrade are intentionally NOT reset
        // on cleanup so values persist across file selections within the same session.
    }

    // ---------------------------------------------------------------
    // Template
    // ---------------------------------------------------------------

    _template() {
        return `
            <section>
                <h2>Assembly Analysis</h2>
                <p>Select an uploaded STEP or IFC file to inspect its assembly hierarchy.</p>

                <div class="analysis-controls">
                    <select id="file-select" aria-label="Select CAD file">
                        <option value="">Loading files...</option>
                    </select>
                    <button id="analyze-btn" disabled>Analyze</button>
                </div>

                <div id="file-preview-panel" class="file-preview-panel" hidden>
                    <div id="file-preview-viewer" class="file-preview-viewer"></div>
                    <p class="file-preview-label" id="file-preview-label"></p>
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
                        <div class="tree-toolbar">
                            <input id="tree-search" type="search" placeholder="Filter parts…" class="tree-search-input">
                            <span id="classification-progress" class="classification-progress"></span>
                        </div>
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

                <div id="parts-list-bar" class="parts-list-bar" hidden>
                    <button id="show-parts-list-btn" class="outline">BOM</button>
                </div>
                <div id="parts-list-panel" class="parts-list-panel" hidden></div>
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
            this._loadFilePreview(select.value);
        });

        btn.addEventListener('click', () => {
            const filename = select.value;
            if (filename) this._analyze(filename);
        });

        this.container.addEventListener('click', (e) => {
            if (e.target.id === 'show-parts-list-btn') {
                this._togglePartsList();
            }
        });

        // Search input is inside #tree-results (added when tree renders), so bind via delegation
        this.container.addEventListener('input', (e) => {
            if (e.target.id === 'tree-search') {
                this._filterTree(e.target.value);
            }
        });
        this.container.addEventListener('search', (e) => {
            if (e.target.id === 'tree-search') {
                this._filterTree('');
            }
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
    // File preview (shown on file-select change, before Analyze is clicked)
    // ---------------------------------------------------------------

    _loadFilePreview(filename) {
        const panel = this.container.querySelector('#file-preview-panel');
        const viewerEl = this.container.querySelector('#file-preview-viewer');
        const label = this.container.querySelector('#file-preview-label');

        if (!filename) {
            panel.hidden = true;
            viewerEl.innerHTML = '';
            return;
        }

        // Show panel with a loading placeholder immediately
        label.textContent = filename;
        panel.hidden = false;
        viewerEl.innerHTML = '<p class="file-preview-status" aria-busy="true">Generating preview…</p>';

        // The server generates the PNG thumbnail on first request (may take several
        // seconds for a large assembly) then caches it for subsequent selections.
        // A plain <img> handles the async wait naturally via onload/onerror.
        const enc = encodeURIComponent(filename);
        const img = new Image();
        img.className = 'file-preview-img';
        img.alt = filename;

        img.onload = () => {
            viewerEl.innerHTML = '';
            viewerEl.appendChild(img);
        };

        img.onerror = () => {
            viewerEl.innerHTML = '<p class="file-preview-status">Preview unavailable — click Analyze to generate one.</p>';
        };

        // Setting src kicks off the request.  If the user picks a different file
        // before this completes the stale img is simply dropped.
        img.src = `/api/v1/stl/thumbnail/${enc}`;
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
        this._projectStateRestored = true;

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
        this._treeData = nodes;
        treeEl.innerHTML = '<ul>' + nodes.map(n => this._renderNode(n, 0)).join('') + '</ul>';

        this._buildParentMap(nodes, null);
        this._extractPlacements(nodes);
        this._bindTreeEvents(treeEl);

        // Show the "All Parts" button once the tree is available
        const partsBar = this.container.querySelector('#parts-list-bar');
        if (partsBar) partsBar.hidden = false;
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
            // Also record nodeId → refId so BOM walk never
            // needs a DOM querySelector to resolve refId (fails for hidden nodes).
            if (node.ref_id) {
                this._nodeRefMap.set(node.id, node.ref_id);
                // Count instances per refId — used to display correct qty for solid bodies
                this._refIdInstanceCount.set(
                    node.ref_id,
                    (this._refIdInstanceCount.get(node.ref_id) || 0) + 1
                );
            }
            if (node.children && node.children.length > 0) {
                this._buildParentMap(node.children, node.name);
            }
        }
    }

    /** Walk tree data and store any non-identity placement matrices by nodeId. */
    _extractPlacements(nodes) {
        for (const node of nodes) {
            if (node.placement) {
                this._placements.set(node.id, node.placement);
            }
            if (node.children && node.children.length > 0) {
                this._extractPlacements(node.children);
            }
        }
    }

    _actionsHtml(node) {
        const btns = [];
        switch (node.node_type) {
            case 'assembly':
                btns.push('<button class="btn-explode"     data-action="explode"     hidden>▶ Explode</button>');
                btns.push('<button class="btn-bought-out"  data-action="bought-out"  hidden>BO</button>');
                btns.push('<button class="btn-exclude"     data-action="exclude"     hidden>EXC</button>');
                btns.push('<button class="btn-unclassify"  data-action="unclassify"  hidden>✕</button>');
                break;
            case 'part_multi_solid':
                btns.push('<button class="btn-explode"     data-action="explode"     hidden>▶ Solids</button>');
                btns.push('<button class="btn-postprocess" data-action="postprocess" hidden>CNC</button>');
                btns.push('<button class="btn-bought-out"  data-action="bought-out"  hidden>BO</button>');
                btns.push('<button class="btn-exclude"     data-action="exclude"     hidden>EXC</button>');
                btns.push('<button class="btn-unclassify"  data-action="unclassify"  hidden>✕</button>');
                break;
            case 'part_single_solid':
                btns.push('<button class="btn-postprocess" data-action="postprocess" hidden>CNC</button>');
                btns.push('<button class="btn-bought-out"  data-action="bought-out"  hidden>BO</button>');
                btns.push('<button class="btn-exclude"     data-action="exclude"     hidden>EXC</button>');
                btns.push('<button class="btn-unclassify"  data-action="unclassify"  hidden>✕</button>');
                break;
            case 'part_no_solid':
                btns.push('<button class="btn-bought-out"  data-action="bought-out"  hidden>BO</button>');
                btns.push('<button class="btn-exclude"     data-action="exclude"     hidden>EXC</button>');
                btns.push('<button class="btn-unclassify"  data-action="unclassify"  hidden>✕</button>');
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

            // 3. Classification buttons (Postprocess, Bought-out) and Unclassify
            const actionBtn = e.target.closest('[data-action]:not(.btn-explode)');
            if (actionBtn) {
                const li = actionBtn.closest('.tree-node');
                const nodeId = li.dataset.nodeId;
                const action = actionBtn.dataset.action;
                if (action === 'unclassify') {
                    this._unclassifyNode(li, nodeId);
                } else {
                    this._classifyNode(li, nodeId, action);
                }
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

        // Hover highlighting for children in multi-solid / assembly scene
        treeEl.addEventListener('mouseover', (e) => {
            const row = e.target.closest('.tree-node-row');
            if (!row || row._hoverActive) return;
            if (!this._multiSolidMeshMap) return;
            const li = row.closest('.tree-node');
            if (!li) return;
            const nodeId = li.dataset.nodeId;
            if (!this._multiSolidMeshMap.has(nodeId)) return;

            row._hoverActive = true;
            row.classList.add('node-hover-highlight');
            this._highlightMesh(nodeId);
        });

        treeEl.addEventListener('mouseout', (e) => {
            const row = e.target.closest('.tree-node-row');
            if (!row || !row._hoverActive) return;
            // Only unhighlight if actually leaving the row, not entering a child element
            if (row.contains(e.relatedTarget)) return;

            row._hoverActive = false;
            row.classList.remove('node-hover-highlight');
            this._unhighlightAllMeshes();
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
            const currentAction = li.dataset.classification || null;
            const row = li.querySelector(':scope > .tree-node-row');

            // Selectable = has an STL (exploded parents remain clickable so the user
            // can still load the assembly/compound STL to see where it sits in the model).
            row.classList.toggle('node-selectable', hasStl);

            // Explode button: visible when has STL and not already exploded
            const explodeBtn = row.querySelector('.btn-explode');
            if (explodeBtn) {
                explodeBtn.hidden = !hasStl || isExploded;
            }

            // BO button: visible when has STL; stays visible even when exploded
            // so the user can mark exploded items as bought-out
            const boughtOutBtn = row.querySelector('.btn-bought-out');
            if (boughtOutBtn) {
                boughtOutBtn.hidden = !hasStl;
                boughtOutBtn.classList.toggle('btn-active', isClassified && currentAction === 'bought-out');
            }

            // EXC button: visible when has STL; stays visible even when exploded
            const excludeBtn = row.querySelector('.btn-exclude');
            if (excludeBtn) {
                excludeBtn.hidden = !hasStl;
                excludeBtn.classList.toggle('btn-active', isClassified && currentAction === 'exclude');
            }

            // CNC button: visible when has STL and not exploded; active when currently CNC
            const ppBtn = row.querySelector('.btn-postprocess');
            if (ppBtn) {
                ppBtn.hidden = !hasStl || isExploded;
                ppBtn.classList.toggle('btn-active', isClassified && currentAction === 'postprocess');
            }

            // Unclassify button: visible when classified (even if exploded, to undo BO)
            const unclassifyBtn = row.querySelector('.btn-unclassify');
            if (unclassifyBtn) {
                unclassifyBtn.hidden = !isClassified;
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

        const treeEl = this.container.querySelector('#assembly-tree-container');
        const targetLi = treeEl?.querySelector(`.tree-node[data-node-id="${CSS.escape(nodeId)}"]`);

        // Case 1: exploded parent (multi-solid or assembly) -> load all children as a scene
        if (targetLi
            && this.explodedNodes.has(nodeId)
            && (targetLi.dataset.nodeType === 'part_multi_solid'
                || targetLi.dataset.nodeType === 'assembly')) {
            this._selectedNodeId = nodeId;
            for (const row of treeEl.querySelectorAll('.tree-node-row')) {
                const li = row.closest('.tree-node');
                row.classList.toggle('node-selected', li.dataset.nodeId === nodeId);
            }
            this._loadMultiSolidScene(nodeId);
            return;
        }

        // Case 2: child whose parent scene is already loaded -> highlight in-place
        if (targetLi
            && this._multiSolidMeshMap
            && this._multiSolidMeshMap.has(nodeId)) {
            this._selectedNodeId = nodeId;
            this._highlightMesh(nodeId);
            for (const row of treeEl.querySelectorAll('.tree-node-row')) {
                const li = row.closest('.tree-node');
                row.classList.toggle('node-selected', li.dataset.nodeId === nodeId);
            }
            return;
        }

        // Case 3: normal single-STL path
        const url = this.stlMap.get(nodeId);
        if (!url) return;

        this._selectedNodeId = nodeId;
        this._multiSolidMeshMap = null;
        this._multiSolidParentId = null;
        this._multiSolidDefaultColors = null;

        for (const row of treeEl.querySelectorAll('.tree-node-row')) {
            const li = row.closest('.tree-node');
            row.classList.toggle('node-selected', li.dataset.nodeId === nodeId);
        }

        this._loadInViewer(url);
    }

    /**
     * Select a tree node and scroll it into view (used by BOM row clicks).
     * Expands collapsed ancestor nodes so the target is visible.
     */
    _selectAndScrollToNode(nodeId) {
        const treeEl = this.container.querySelector('#assembly-tree-container');
        if (!treeEl) return;

        // Find the tree node <li> for this nodeId
        const targetLi = treeEl.querySelector(`.tree-node[data-node-id="${nodeId}"]`);
        if (!targetLi) return;

        // Expand any collapsed ancestor <ul> elements so the node is visible
        let el = targetLi.parentElement;
        while (el && el !== treeEl) {
            if (el.tagName === 'UL' && el.hidden) {
                el.hidden = false;
                // Update the toggle button on the parent <li>
                const parentLi = el.closest('.tree-node');
                if (parentLi) {
                    const toggle = parentLi.querySelector(':scope > .tree-node-row .tree-toggle');
                    if (toggle) toggle.classList.add('expanded');
                }
            }
            el = el.parentElement;
        }

        // Select the node (loads preview + highlights)
        this._selectNodeForPreview(nodeId);

        // Scroll the tree node row into view
        const row = targetLi.querySelector('.tree-node-row');
        if (row) {
            row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // ---------------------------------------------------------------
    // Multi-solid scene (hover-to-highlight)
    // ---------------------------------------------------------------

    /** Scene colours for multi-solid / assembly hover-to-highlight */
    static _DEFAULT_COLOR = 0x4a90d9;   // blue — everything when no hover
    static _HIGHLIGHT_COLOR = 0x4a90d9; // blue — hovered item stays blue
    static _DIM_COLOR = 0x888888;       // grey — non-hovered items
    static _DEFAULT_OPACITY = 1.0;
    static _DIM_OPACITY = 0.18;         // smoked glass

    async _loadMultiSolidScene(parentNodeId) {
        const treeEl = this.container.querySelector('#assembly-tree-container');
        const parentLi = treeEl?.querySelector(`.tree-node[data-node-id="${CSS.escape(parentNodeId)}"]`);
        if (!parentLi) return;

        // Gather children — from cache (multi-solid) or from DOM (assembly)
        let children;
        if (parentLi.dataset.nodeType === 'part_multi_solid') {
            const chiralKey = parentLi.dataset.chiralKey;
            children = this._solidChildrenCache.get(chiralKey);
        } else {
            // Assembly: direct child <li> nodes from the DOM
            const childUl = parentLi.querySelector(':scope > ul');
            if (childUl) {
                children = Array.from(childUl.querySelectorAll(':scope > .tree-node'))
                    .map(li => ({ nodeId: li.dataset.nodeId, name: li.dataset.nodeName || li.dataset.nodeId }));
            }
        }
        if (!children || children.length === 0) return;

        const items = [];
        const meshMap = new Map();
        const defaultColors = [];

        const DC = AnalysisPage._DEFAULT_COLOR;
        for (let i = 0; i < children.length; i++) {
            const url = this.stlMap.get(children[i].nodeId);
            if (!url) continue;
            const placement = this._placements.get(children[i].nodeId) || null;
            items.push({ url, color: DC, opacity: AnalysisPage._DEFAULT_OPACITY, label: children[i].name, placement });
            meshMap.set(children[i].nodeId, items.length - 1);
            defaultColors.push(DC);
        }

        if (items.length === 0) return;

        this._multiSolidMeshMap = meshMap;
        this._multiSolidParentId = parentNodeId;
        this._multiSolidDefaultColors = defaultColors;

        const panel = this.container.querySelector('#stl-viewer-panel');
        if (!panel) return;

        if (!this._viewer) {
            panel.innerHTML = '';
            this._viewer = new STLViewer(panel);
        }

        panel.classList.add('loading');
        try {
            await this._viewer.loadScene(items);
        } finally {
            panel.classList.remove('loading');
        }
    }

    _highlightMesh(nodeId) {
        if (!this._multiSolidMeshMap || !this._viewer) return;
        const HL = AnalysisPage._HIGHLIGHT_COLOR;
        const DIM_C = AnalysisPage._DIM_COLOR;
        const DIM_O = AnalysisPage._DIM_OPACITY;
        for (const [nid, idx] of this._multiSolidMeshMap) {
            if (nid === nodeId) {
                this._viewer.setMeshColor(idx, HL, 1.0);
            } else {
                this._viewer.setMeshColor(idx, DIM_C, DIM_O);
            }
        }
    }

    _unhighlightAllMeshes() {
        if (!this._multiSolidMeshMap || !this._viewer) return;
        const DC = AnalysisPage._DEFAULT_COLOR;
        const OP = AnalysisPage._DEFAULT_OPACITY;
        for (const [, idx] of this._multiSolidMeshMap) {
            this._viewer.setMeshColor(idx, DC, OP);
        }
    }

    _loadInViewer(url) {
        // Clear multi-solid scene state when loading a single STL
        this._multiSolidMeshMap = null;
        this._multiSolidParentId = null;
        this._multiSolidDefaultColors = null;

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

                    // Cache solid children info keyed by chiralKey.
                    // Include all solids with a node_id, even those whose STL
                    // generation failed (stl_file may be null) — they still need
                    // to appear in the BOM and tree for classification purposes.
                    const childrenInfo = (status.results || [])
                        .filter(r => r.node_id)
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

        const parentRefId = li.dataset.refId;
        const parentNodeId = li.dataset.nodeId;
        const parentName = li.dataset.nodeName || null;

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
                        <button class="btn-postprocess" data-action="postprocess" hidden>CNC</button>
                        <button class="btn-bought-out"  data-action="bought-out"  hidden>BO</button>
                        <button class="btn-exclude"     data-action="exclude"     hidden>EXC</button>
                        <button class="btn-unclassify"  data-action="unclassify"  hidden>✕</button>
                    </span>
                </div>
            `;
            childUl.appendChild(childLi);

            // Register in parent/node maps so BOM walk can display solid body rows.
            // _solidParentRefId lets _buildBOMItems look up the correct instance qty.
            // _solidParentNodeId lets BOM inherit the parent's classification.
            this._parentMap.set(child.nodeId, {
                name: child.name,
                parentName,
                _solidParentRefId: parentRefId,
                _solidParentNodeId: parentNodeId,
            });
            this._nodeRefMap.set(child.nodeId, child.nodeId);
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
        }

        this._updateTreeSelectability();
        this._updateProgress();
        this._debouncedSave();

        // Auto-advance: select next unclassified sibling with an STL
        this._selectNextUnclassified(li);
    }

    /**
     * Find the next visible, unclassified tree node after `currentLi` and select it.
     * Walks forward through all tree-node elements in DOM order.
     */
    _selectNextUnclassified(currentLi) {
        const treeEl = this.container.querySelector('#assembly-tree-container');
        if (!treeEl) return;

        const allNodes = [...treeEl.querySelectorAll('.tree-node')];
        const currentIdx = allNodes.indexOf(currentLi);
        if (currentIdx < 0) return;

        // Search forward from current position, then wrap around
        for (let offset = 1; offset < allNodes.length; offset++) {
            const candidate = allNodes[(currentIdx + offset) % allNodes.length];
            const candidateId = candidate.dataset.nodeId;

            // Skip already-classified nodes
            if (this.classifications.has(candidateId)) continue;

            // Skip nodes without STL (not selectable)
            if (!this.stlMap.has(candidateId)) continue;

            // Skip exploded nodes (their children are the actionable items)
            if (this.explodedNodes.has(candidateId)) continue;

            // Found the next unclassified item — select and scroll to it
            this._selectNodeForPreview(candidateId);
            const row = candidate.querySelector(':scope > .tree-node-row');
            if (row) {
                row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }
            return;
        }
    }

    _unclassifyNode(li, nodeId) {
        const refId = li.dataset.refId;
        const nodeType = li.dataset.nodeType;
        const treeEl = this.container.querySelector('#assembly-tree-container');

        // Mirror _classifyNode's collection logic — same ref_id + node_type peers
        const toUnclassify = new Set([li]);
        if (treeEl) {
            for (const el of treeEl.querySelectorAll(
                `.tree-node[data-node-id="${CSS.escape(nodeId)}"]`
            )) {
                toUnclassify.add(el);
            }
            if (refId) {
                for (const el of treeEl.querySelectorAll(
                    `.tree-node[data-ref-id="${CSS.escape(refId)}"][data-node-type="${CSS.escape(nodeType)}"]`
                )) {
                    toUnclassify.add(el);
                }
            }
        }

        for (const el of toUnclassify) {
            this.classifications.delete(el.dataset.nodeId);
            el.classList.remove('node-classified');
            delete el.dataset.classification;
        }

        this._updateTreeSelectability();
        this._updateProgress();
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
                }
            }
        }

        // Apply selectability based on restored state
        this._updateTreeSelectability();
        this._updateProgress();

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

        // Don't save if project state hasn't been restored yet — avoids
        // overwriting existing classifications with empty data during the
        // window between page load and project_state restore.
        if (!this._projectStateRestored) return;

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
    // Progress counter + tree filter
    // ---------------------------------------------------------------

    /**
     * Update the classification progress counter in the tree toolbar.
     * Shows unique-part counts (by ref_id) for CNC and Bought-out.
     */
    _updateProgress() {
        const progressEl = this.container.querySelector('#classification-progress');
        if (!progressEl) return;

        // Build per-ref_id classification map (last classification wins per ref_id)
        const refIdCl = new Map();
        for (const [nodeId, action] of this.classifications) {
            const refId = this._nodeRefMap.get(nodeId) || nodeId;
            refIdCl.set(refId, action);
        }

        let cnc = 0, bo = 0, exc = 0;
        for (const action of refIdCl.values()) {
            if (action === 'postprocess') cnc++;
            else if (action === 'bought-out') bo++;
            else if (action === 'exclude') exc++;
        }

        const parts = [];
        if (cnc > 0) parts.push(`<span class="prog-cnc">${cnc} CNC</span>`);
        if (bo > 0)  parts.push(`<span class="prog-bo">${bo} BO</span>`);
        if (exc > 0) parts.push(`<span class="prog-exc">${exc} EXC</span>`);
        progressEl.innerHTML = parts.length
            ? parts.join(' <span class="prog-sep">·</span> ')
            : '';
    }

    /**
     * Filter tree nodes by name. Matching nodes are shown; non-matching hidden.
     * Parents of matching nodes are always expanded and shown.
     */
    _filterTree(query) {
        const treeEl = this.container.querySelector('#assembly-tree-container');
        if (!treeEl) return;

        const q = query.trim().toLowerCase();

        if (!q) {
            // Clear filter — restore all nodes to visible
            for (const li of treeEl.querySelectorAll('.tree-node')) {
                li.style.display = '';
            }
            return;
        }

        // First pass: mark each node as matching or not
        const matching = new Set();
        for (const li of treeEl.querySelectorAll('.tree-node')) {
            const name = (li.dataset.nodeName || '').toLowerCase();
            if (name.includes(q)) matching.add(li);
        }

        // Second pass: also include ancestors of matching nodes
        const visible = new Set(matching);
        for (const li of matching) {
            let ancestor = li.parentElement?.closest('.tree-node');
            while (ancestor) {
                visible.add(ancestor);
                // Expand ancestor so child is reachable
                const childUl = ancestor.querySelector(':scope > ul');
                const toggle = ancestor.querySelector(':scope > .tree-node-row .tree-toggle');
                if (childUl) {
                    childUl.hidden = false;
                    if (toggle) toggle.classList.add('expanded');
                }
                ancestor = ancestor.parentElement?.closest('.tree-node');
            }
        }

        // Apply visibility
        for (const li of treeEl.querySelectorAll('.tree-node')) {
            li.style.display = visible.has(li) ? '' : 'none';
        }
    }

    // ---------------------------------------------------------------
    // BOM view — explode-aware classified items
    // ---------------------------------------------------------------

    _togglePartsList() {
        const panel = this.container.querySelector('#parts-list-panel');
        const btn = this.container.querySelector('#show-parts-list-btn');
        if (!panel) return;

        if (!panel.hidden) {
            panel.hidden = true;
            btn.textContent = 'BOM';
            return;
        }

        const filename = this._currentFilename;
        if (!filename) return;

        // Load cached consolidation and CNC analysis data (best-effort, non-blocking)
        Promise.allSettled([
            this.api.getConsolidation(filename)
                .then(resp => {
                    if (resp?.groups) {
                        this._consolidationGroups = resp.groups;
                        this._solidConsolidationGroups = resp.solid_groups || [];
                        this._intraSolidGroups = resp.intra_solid_groups || [];
                        this._consolidating = false;
                    }
                }),
            this.api.getCncResult(filename)
                .then(resp => {
                    if (resp?.results) {
                        this._cncAnalysisResults = resp.results;
                        this._cncAnalysing = false;
                    }
                }),
        ]).finally(() => {
            this._renderPartsList(this._consolidationGroups);
        });
    }

    /**
     * Walk the assembly tree respecting the user's explode decisions and
     * return a Map of BOM items keyed by ref_id (or solidNodeId for solid bodies).
     *
     * Only the current "effective" level is walked:
     *   - Unexploded assemblies → one item (the assembly itself)
     *   - Exploded assemblies   → recurse into children
     *   - Multi-solid, solid-exploded → individual solid body items
     *   - Multi-solid, not exploded   → one item
     *   - Single-solid / no-solid    → one item
     */
    _buildBOMItems() {
        const itemMap = new Map();
        this._walkForBOM(this._treeData || [], null, itemMap);

        // Merge solid body items that belong to the same solid consolidation group
        // (e.g. solid 2 of part 1091 is geometrically identical to solid 0 of part 1085)
        this._mergeSolidGroups(itemMap);

        // Merge intra-part duplicate solids
        // (e.g. solids 0,1,3,4 of a weldment are four identical plates)
        this._mergeIntraSolidGroups(itemMap);

        const cncItems      = [];
        const boItems       = [];
        const unknownItems  = [];
        const excludedItems = [];

        for (const item of itemMap.values()) {
            const classifiedId = item.nodeIds.find(nid => this.classifications.has(nid));
            let action = classifiedId ? this.classifications.get(classifiedId) : null;

            // Inherit classification from parent multi-solid for exploded solid children
            if (!action) {
                const parentInfo = this._parentMap.get(item.key);
                if (parentInfo?._solidParentNodeId) {
                    action = this.classifications.get(parentInfo._solidParentNodeId) || null;
                }
            }

            if (action === 'postprocess') {
                // Check CNC analysis result — separate unknowns from resolved items
                const info = this._cncResultForItem(item);
                const rType = info?.result?.type;
                if (rType === 'unknown') {
                    unknownItems.push(item);
                } else {
                    cncItems.push(item);
                }
            }
            else if (action === 'bought-out') boItems.push(item);
            else if (action === 'exclude') excludedItems.push(item);
            // unclassified → skip
        }

        // Sort each section: by name
        const byName = (a, b) => a.name.localeCompare(b.name);
        unknownItems.sort(byName);
        cncItems.sort(byName);
        boItems.sort(byName);
        excludedItems.sort(byName);

        return { cncItems, boItems, unknownItems, excludedItems };
    }

    /**
     * Post-walk pass: merge BOM items whose solid nodeIds belong to the same
     * solid consolidation group.
     *
     * Solid nodeIds in the BOM have the form "<ref_id>:s<N>" (matching the
     * synthetic ids used by the STL generator).  The solid_groups from the
     * consolidation worker describe which (ref_id, solid_index) pairs share
     * identical geometry across different parent multi-solid parts.
     *
     * When merging, the first encountered item is kept as canonical; subsequent
     * same-group items have their qty, mirroredCount, nodeIds, and parentNames
     * folded in, then removed from the map.
     */
    _mergeSolidGroups(itemMap) {
        const groups = this._solidConsolidationGroups;
        if (!groups || groups.length === 0) return;

        // Build lookup: solid nodeId → group object
        const solidIdToGroup = new Map();
        for (const sg of groups) {
            for (const m of sg.members) {
                solidIdToGroup.set(`${m.ref_id}:s${m.solid_index}`, sg);
            }
        }

        // Track which key is canonical for each group (first one seen wins)
        const groupCanonicalKey = new Map();

        for (const key of [...itemMap.keys()]) {
            const sg = solidIdToGroup.get(key);
            if (!sg) continue;

            if (groupCanonicalKey.has(sg)) {
                // Fold this item into the canonical item
                const canonicalKey = groupCanonicalKey.get(sg);
                const canonical = itemMap.get(canonicalKey);
                const item = itemMap.get(key);
                if (canonical && item) {
                    canonical.qty += item.qty;
                    canonical.mirroredCount += item.mirroredCount;
                    canonical.nodeIds.push(...item.nodeIds);
                    if (!canonical._memberQtys) canonical._memberQtys = [canonical.qty - item.qty];
                    canonical._memberQtys.push(item.qty);
                    for (const p of item.parentNames) {
                        if (!canonical.parentNames.includes(p)) canonical.parentNames.push(p);
                    }
                    for (const [p, c] of Object.entries(item.parentCounts || {})) {
                        canonical.parentCounts[p] = (canonical.parentCounts[p] || 0) + c;
                    }
                    itemMap.delete(key);
                }
            } else {
                groupCanonicalKey.set(sg, key);
                // canonical item keeps its name — it already has a human-readable name
            }
        }
    }

    /**
     * Post-walk pass: merge BOM rows for identical solid bodies within the same
     * multi-solid prototype (intra-part duplicate solids).
     *
     * For example, a weldment whose STEP compound contains four geometrically
     * identical gusset plates will appear as four separate solid rows when
     * solid-exploded.  The backend reports these in intra_solid_groups; this
     * method folds them into a single row so the BOM shows qty×(plates per
     * instance × number of instances) rather than separate rows per solid body.
     */
    _mergeIntraSolidGroups(itemMap) {
        const groups = this._intraSolidGroups;
        if (!groups || groups.length === 0) return;

        for (const ig of groups) {
            const keys = ig.solid_indices.map(idx => `${ig.ref_id}:s${idx}`);
            const presentKeys = keys.filter(k => itemMap.has(k));
            if (presentKeys.length < 2) continue;

            const [canonicalKey, ...restKeys] = presentKeys;
            const canonical = itemMap.get(canonicalKey);
            if (!canonical._memberQtys) canonical._memberQtys = [canonical.qty];
            for (const key of restKeys) {
                const item = itemMap.get(key);
                if (item) {
                    canonical.qty += item.qty;
                    canonical.mirroredCount += item.mirroredCount;
                    canonical._memberQtys.push(item.qty);
                    canonical.nodeIds.push(...item.nodeIds);
                    for (const p of item.parentNames) {
                        if (!canonical.parentNames.includes(p)) canonical.parentNames.push(p);
                    }
                    for (const [p, c] of Object.entries(item.parentCounts || {})) {
                        canonical.parentCounts[p] = (canonical.parentCounts[p] || 0) + c;
                    }
                    itemMap.delete(key);
                }
            }
        }
    }

    _walkForBOM(nodes, parentName, itemMap) {
        for (const node of nodes) {
            const refId     = node.ref_id || node.id;
            const isMirr    = !!node.is_mirrored;

            if (node.node_type === 'assembly') {
                if (this.explodedNodes.has(node.id)) {
                    this._walkForBOM(node.children || [], node.name, itemMap);
                } else {
                    this._bomUpsert(itemMap, refId, node.name, node.id, isMirr, parentName);
                }

            } else if (node.node_type === 'part_multi_solid') {
                if (this.explodedNodes.has(node.id)) {
                    const chiralKey = `${refId}:${isMirr ? 'M' : 'N'}`;
                    const altKey    = `${refId}:${isMirr ? 'N' : 'M'}`;
                    const solids    = this._solidChildrenCache.get(chiralKey)
                                   ?? this._solidChildrenCache.get(altKey);
                    if (solids && solids.length > 0) {
                        for (const solid of solids) {
                            this._bomUpsert(itemMap, solid.nodeId, solid.name,
                                            solid.nodeId, isMirr, node.name);
                        }
                    } else {
                        // Solid children not yet loaded — show the multi-solid itself
                        this._bomUpsert(itemMap, refId, node.name, node.id, isMirr, parentName);
                    }
                } else {
                    this._bomUpsert(itemMap, refId, node.name, node.id, isMirr, parentName);
                }

            } else {
                // part_single_solid / part_no_solid
                this._bomUpsert(itemMap, refId, node.name, node.id, isMirr, parentName);
            }
        }
    }

    _bomUpsert(itemMap, key, name, nodeId, isMirrored, parentName) {
        if (itemMap.has(key)) {
            const item = itemMap.get(key);
            item.qty++;
            if (isMirrored) item.mirroredCount++;
            item.nodeIds.push(nodeId);
            if (parentName) {
                if (!item.parentNames.includes(parentName)) {
                    item.parentNames.push(parentName);
                }
                item.parentCounts[parentName] = (item.parentCounts[parentName] || 0) + 1;
            }
        } else {
            const parentCounts = {};
            if (parentName) parentCounts[parentName] = 1;
            itemMap.set(key, {
                key,
                name,
                nodeIds: [nodeId],
                qty: 1,
                mirroredCount: isMirrored ? 1 : 0,
                parentNames: parentName ? [parentName] : [],
                parentCounts,
            });
        }
    }

    _renderPartsList(consolidationGroups = null) {
        const panel = this.container.querySelector('#parts-list-panel');
        if (!panel) return;

        const { cncItems, boItems, unknownItems, excludedItems } = this._buildBOMItems();
        const totalClassified = cncItems.length + boItems.length + unknownItems.length + excludedItems.length;

        // Build a ref_id → consolidation group lookup for merged display
        const refToGroup = new Map();
        if (consolidationGroups) {
            for (const g of consolidationGroups) {
                for (const rid of g.ref_ids) {
                    refToGroup.set(rid, g);
                }
            }
        }

        const renderRows = (items, isCnc = false) => {
            // When consolidation is active, merge items in the same group into one row
            if (consolidationGroups && refToGroup.size > 0) {
                const seen = new Set();
                const merged = [];

                for (const item of items) {
                    const g = refToGroup.get(item.key);
                    if (g) {
                        if (seen.has(g)) continue;
                        seen.add(g);
                        // Combine qty/mirrored from all group members that are classified
                        const members = g.ref_ids
                            .map(rid => items.find(it => it.key === rid))
                            .filter(Boolean);
                        const qty = members.reduce((s, m) => s + m.qty, 0);
                        const mirr = members.reduce((s, m) => s + m.mirroredCount, 0);
                        const parents = [...new Set(members.flatMap(m => m.parentNames))];
                        const mergeTag = members.length > 1
                            ? ` <span class="parts-list-group-tag" title="${members.length} identical CAD definitions merged — qty is combined total">&times;${members.length}</span>`
                            : '';
                        const mirrorNote = mirr > 0
                            ? ` <span class="parts-list-mirror">(+${mirr}M)</span>` : '';
                        // Per-occurrence qty breakdown when items are consolidated
                        let qtyBreakdown = '';
                        if (members.length > 1) {
                            // Collect per-member qtys; if a member was already
                            // merged at the solid level it carries _memberQtys
                            const memberQtys = members.map(m => m.qty);
                            const allSame = memberQtys.every(q => q === memberQtys[0]);
                            if (allSame) {
                                qtyBreakdown = ` <span class="parts-list-qty-detail" title="${memberQtys[0]} per occurrence × ${members.length} occurrences">(${memberQtys[0]}×${members.length})</span>`;
                            } else {
                                qtyBreakdown = ` <span class="parts-list-qty-detail" title="Per occurrence: ${memberQtys.join(', ')}">(${memberQtys.join('+')})</span>`;
                            }
                        } else if (members.length === 1 && members[0]._memberQtys && members[0]._memberQtys.length > 1) {
                            // Single group member but already merged at solid level
                            qtyBreakdown = this._qtyBreakdownHtml(members[0]);
                        }
                        // Merge parentCounts across all members
                        const mergedCounts = {};
                        for (const m of members) {
                            for (const [p, c] of Object.entries(m.parentCounts || {})) {
                                mergedCounts[p] = (mergedCounts[p] || 0) + c;
                            }
                        }
                        const parentsCells = parents.length > 0
                            ? parents.map(p => {
                                const c = mergedCounts[p] || 1;
                                return `<span class="used-in-item"><span class="parent-qty">${c}</span>${this._esc(p)}</span>`;
                            }).join('')
                            : '<span class="used-in-item">—</span>';
                        // Use first group member's result for CNC badge
                        const cncHtml = isCnc && members[0]
                            ? this._cncResultHtml(members[0], filename) : '';
                        const grpNid = members[0]?.nodeIds?.[0] || '';
                        merged.push(`<tr data-bom-node-id="${grpNid}">
                            <td>${this._esc(g.canonical_name)}${mergeTag}${cncHtml ? ' ' + cncHtml : ''}</td>
                            <td class="parts-list-qty">${qty}${mirrorNote}</td>
                            <td class="parts-list-parents">${parentsCells}</td>
                        </tr>`);
                    } else {
                        merged.push(isCnc ? this._cncBomRow(item, filename) : this._bomRow(item));
                    }
                }
                return merged.join('');
            }
            return items.map(item =>
                isCnc ? this._cncBomRow(item, filename) : this._bomRow(item)
            ).join('');
        };

        const CL_BADGE = {
            postprocess: '<span class="parts-list-cl-badge postprocess">CNC</span>',
            'bought-out': '<span class="parts-list-cl-badge bought-out">BO</span>',
            exclude: '<span class="parts-list-cl-badge exclude">EXC</span>',
        };

        const filename = this._currentFilename;

        // Determine which bulk downloads are available
        let hasDxf = false, hasNc1 = false;
        if (this._cncAnalysisResults) {
            for (const r of Object.values(this._cncAnalysisResults)) {
                const checks = r.type === 'multi_solid' ? (r.solids || []) : [r];
                for (const s of checks) {
                    if (s.dxf_path) hasDxf = true;
                    if (s.nc1_path) hasNc1 = true;
                }
            }
        }
        const enc = encodeURIComponent(filename || '');
        const dxfZipLink = hasDxf
            ? `<a href="/api/v1/cnc-analysis/download-all/${enc}/dxf" class="parts-cnc-dl-btn" download>\u2193\u00a0DXF</a>`
            : '';
        const nc1ZipLink = hasNc1
            ? `<a href="/api/v1/cnc-analysis/download-all/${enc}/nc1" class="parts-cnc-dl-btn" download>\u2193\u00a0NC1</a>`
            : '';

        const hasAnyResults = Object.keys(this._cncAnalysisResults || {}).length > 0;
        const analyseBtn = this._cncAnalysing
            ? `<button class="parts-cnc-analyse-btn outline" disabled>Analysing\u2026</button>`
            : `<button class="parts-cnc-analyse-btn outline">Analyse</button>`
              + (hasAnyResults ? `<button class="parts-cnc-reanalyse-btn outline" title="Clear cache and re-run analysis">\u21ba\u00a0Re-analyse</button>` : '');

        let tbody = '';
        if (unknownItems.length > 0) {
            tbody += `<tr class="parts-list-section-header parts-list-section-unknown">
                <td colspan="3">\u26a0 Unmatched Sections &middot; ${unknownItems.length} — profile not in library</td>
            </tr>`;
            tbody += renderRows(unknownItems, true);
        }
        if (cncItems.length > 0) {
            tbody += `<tr class="parts-list-section-header parts-list-section-pp">
                <td colspan="2">Post Process &middot; ${cncItems.length}</td>
                <td class="parts-list-section-action">${analyseBtn}${dxfZipLink}${nc1ZipLink}</td>
            </tr>`;
            tbody += renderRows(cncItems, true);
        }
        if (boItems.length > 0) {
            tbody += `<tr class="parts-list-section-header parts-list-section-bo">
                <td colspan="3">Bought Out &middot; ${boItems.length}</td>
            </tr>`;
            tbody += renderRows(boItems, false);
        }
        if (excludedItems.length > 0) {
            tbody += `<tr class="parts-list-section-header parts-list-section-exc">
                <td colspan="3">Excluded &middot; ${excludedItems.length}</td>
            </tr>`;
            tbody += renderRows(excludedItems, false);
        }
        if (totalClassified === 0) {
            tbody = `<tr><td colspan="3" class="parts-list-empty-msg">No items classified yet — use CNC / BO / EXC buttons in the tree.</td></tr>`;
        }

        const consolidateBtn = this._consolidating
            ? `<button class="parts-consolidate-btn outline" disabled>Consolidating…</button>`
            : (consolidationGroups
                ? `<button class="parts-consolidate-btn outline" title="Re-run consolidation">Consolidate</button>`
                : `<button class="parts-consolidate-btn">Consolidate</button>`);

        const bomDlBtn = totalClassified > 0
            ? `<button class="parts-bom-xlsx-btn outline" title="Download BOM as Excel with thumbnails">\u2193\u00a0BOM (.xlsx)</button>`
            + `<button class="parts-bom-dl-btn outline" title="Download BOM as JSON">\u2193\u00a0JSON</button>`
            : '';

        // Nesting button — only show when we have CNC section results
        const hasNestableSections = this._hasNestableSections(cncItems, unknownItems);
        const nestingBtn = hasNestableSections
            ? (this._nestingRunning
                ? `<button class="parts-nesting-btn outline" disabled>Nesting\u2026</button>`
                : `<button class="parts-nesting-btn outline">\u2702 Nesting</button>`)
            : '';

        panel.innerHTML = `
            <div class="parts-list-card">
                <div class="parts-list-header">
                    <span>BOM${totalClassified > 0 ? ' &middot; ' + totalClassified : ''}</span>
                    <div class="parts-list-header-actions">
                        ${consolidateBtn}
                        ${bomDlBtn}
                        ${nestingBtn}
                        <button id="parts-list-close-btn" class="outline parts-list-close">&#x2715;</button>
                    </div>
                </div>
                <div class="parts-list-scroll">
                    <table class="parts-list-table">
                        <thead>
                            <tr>
                                <th>Part</th>
                                <th class="parts-list-qty">Qty</th>
                                <th class="parts-list-used-in-header">Used In</th>
                            </tr>
                        </thead>
                        <tbody>${tbody}</tbody>
                    </table>
                </div>
                <div id="nesting-results-panel" class="nesting-results-panel" ${this._nestingCuttingList ? '' : 'hidden'}></div>
            </div>
        `;
        panel.hidden = false;

        panel.querySelector('#parts-list-close-btn')?.addEventListener('click', () => {
            panel.hidden = true;
            const btn = this.container.querySelector('#show-parts-list-btn');
            if (btn) btn.textContent = 'BOM';
        });

        panel.querySelector('.parts-consolidate-btn')?.addEventListener('click', () => {
            this._startConsolidation();
        });

        const allCncItems = [...unknownItems, ...cncItems];
        panel.querySelector('.parts-cnc-analyse-btn')?.addEventListener('click', () => {
            this._startCncAnalysis(allCncItems, false);
        });

        panel.querySelector('.parts-cnc-reanalyse-btn')?.addEventListener('click', () => {
            this._startCncAnalysis(allCncItems, true);
        });

        panel.querySelector('.parts-bom-dl-btn')?.addEventListener('click', () => {
            this._downloadBOM(allCncItems, boItems, unknownItems, excludedItems);
        });

        panel.querySelector('.parts-bom-xlsx-btn')?.addEventListener('click', () => {
            this._downloadBOMXlsx();
        });

        const allNestableItems = [...cncItems, ...unknownItems];
        panel.querySelector('.parts-nesting-btn')?.addEventListener('click', () => {
            this._showNestingSettingsDialog(allNestableItems);
        });

        // Render existing cutting list if we have one
        if (this._nestingCuttingList) {
            this._renderCuttingList();
        }

        // Click a BOM row → select + scroll to the first occurrence in the tree
        panel.querySelector('.parts-list-table tbody')?.addEventListener('click', (e) => {
            const tr = e.target.closest('tr[data-bom-node-id]');
            if (!tr) return;
            const nodeId = tr.dataset.bomNodeId;
            if (!nodeId) return;
            // Highlight the active BOM row
            for (const r of panel.querySelectorAll('.bom-row-active')) {
                r.classList.remove('bom-row-active');
            }
            tr.classList.add('bom-row-active');
            this._selectAndScrollToNode(nodeId);
        });
    }

    _parentCellsHtml(item) {
        const names = item.parentNames || [];
        if (names.length === 0) return '<span class="used-in-item">—</span>';
        const counts = item.parentCounts || {};
        return names.map(p => {
            const c = counts[p] || 1;
            return `<span class="used-in-item"><span class="parent-qty">${c}</span>${this._esc(p)}</span>`;
        }).join('');
    }

    _qtyBreakdownHtml(item) {
        const mq = item._memberQtys;
        if (!mq || mq.length < 2) return '';
        const allSame = mq.every(q => q === mq[0]);
        if (allSame) {
            return ` <span class="parts-list-qty-detail" title="${mq[0]} per occurrence × ${mq.length} occurrences">(${mq[0]}×${mq.length})</span>`;
        }
        return ` <span class="parts-list-qty-detail" title="Per occurrence: ${mq.join(', ')}">(${mq.join('+')})</span>`;
    }

    _bomRow(item) {
        const mirrorNote = item.mirroredCount > 0
            ? ` <span class="parts-list-mirror">(+${item.mirroredCount}M)</span>` : '';
        const nid = item.nodeIds?.[0] || '';
        return `<tr data-bom-node-id="${nid}">
            <td>${this._esc(item.name)}</td>
            <td class="parts-list-qty">${item.qty}${mirrorNote}</td>
            <td class="parts-list-parents">${this._parentCellsHtml(item)}</td>
        </tr>`;
    }

    _cncBomRow(item, filename) {
        const mirrorNote = item.mirroredCount > 0
            ? ` <span class="parts-list-mirror">(+${item.mirroredCount}M)</span>` : '';
        const cncHtml = this._cncResultHtml(item, filename);
        const nid = item.nodeIds?.[0] || '';
        return `<tr data-bom-node-id="${nid}">
            <td>${this._esc(item.name)}${cncHtml ? ' ' + cncHtml : ''}</td>
            <td class="parts-list-qty">${item.qty}${mirrorNote}</td>
            <td class="parts-list-parents">${this._parentCellsHtml(item)}</td>
        </tr>`;
    }

    /**
     * Resolve the CNC analysis result for a BOM item.
     * For single-solid parts: looks up by item.key (= XCAF ref_id).
     * For solid bodies from exploded multi-solids: looks up via _solidParentRefId,
     * then picks the correct per-solid result using the solid index encoded in nodeId.
     */
    _cncResultForItem(item) {
        if (!this._cncAnalysisResults) return null;

        // Direct lookup (single-solid part)
        const direct = this._cncAnalysisResults[item.key];
        if (direct) return { result: direct, xcafRefId: item.key, solidIdx: null };

        // Solid body from exploded multi-solid
        const parentInfo = this._parentMap.get(item.key);
        if (parentInfo?._solidParentRefId) {
            const xcafRefId = parentInfo._solidParentRefId;
            const parentResult = this._cncAnalysisResults[xcafRefId];
            if (!parentResult) return null;

            if (parentResult.type === 'multi_solid') {
                // Try to extract solid index from nodeId (expected format "<ref_id>:s<N>")
                const match = item.key.match(/:s(\d+)$/);
                const solidIdx = match ? parseInt(match[1]) : 0;
                const solidResult = parentResult.solids?.[solidIdx];
                if (solidResult) return { result: solidResult, xcafRefId, solidIdx };
                // Fall back to parent result if index extraction fails
                return { result: parentResult, xcafRefId, solidIdx: null };
            }
            // Non-multi_solid result stored under parent ref_id
            return { result: parentResult, xcafRefId, solidIdx: null };
        }

        return null;
    }

    /**
     * Return HTML string with result badge and optional download link for a CNC BOM item.
     */
    _cncResultHtml(item, filename) {
        const info = this._cncResultForItem(item);
        if (!info) return '';

        const { result, xcafRefId, solidIdx } = info;
        let badge = '';
        let downloadLink = '';

        // Construct safe_ref: ref_id with colons→hyphens, plus "-s{N}" for solid N>0
        const baseSafeRef = xcafRefId.replace(/:/g, '-');
        const safeRef = (solidIdx != null && solidIdx > 0)
            ? `${baseSafeRef}-s${solidIdx}`
            : baseSafeRef;

        // Confidence pill helper
        const confLevel = result.confidence || '';
        const confBadge = confLevel
            ? ` <span class="cnc-confidence cnc-confidence-${confLevel.toLowerCase()}">${confLevel}</span>`
            : '';

        switch (result.type) {
            case 'plate': {
                const { L, W, T } = result.dims || {};
                const dimStr = (L && W && T)
                    ? ` ${Math.round(L)}\u00d7${Math.round(W)}\u00d7${Math.round(T)}`
                    : '';
                badge = `<span class="cnc-badge cnc-badge-plate">PLATE${this._esc(dimStr)}</span>${confBadge}`;
                if (filename) {
                    const url = `/api/v1/cnc-analysis/download/${encodeURIComponent(filename)}/${encodeURIComponent(safeRef)}/dxf`;
                    downloadLink = `<a href="${url}" class="cnc-download" download>\u2193DXF</a>`;
                }
                break;
            }
            case 'section': {
                const label = result.designation
                    ? `${result.category || ''} ${result.designation}`.trim()
                    : result.category || 'SECTION';
                badge = `<span class="cnc-badge cnc-badge-section">${this._esc(label)}</span>${confBadge}`;
                if (filename && result.nc1_path) {
                    const url = `/api/v1/cnc-analysis/download/${encodeURIComponent(filename)}/${encodeURIComponent(safeRef)}/nc1`;
                    downloadLink = `<a href="${url}" class="cnc-download" download>\u2193NC1</a>`;
                }
                break;
            }
            case 'multi_solid':
                badge = `<span class="cnc-badge cnc-badge-multi">MULTI (${result.n_solids})</span>`;
                break;
            case 'unknown': {
                const msg = result.message ? this._esc(result.message) : '';
                const dims = result.dims;
                const csa = result.section_area;
                let detail = '';
                if (dims) {
                    detail += `L=${dims.L} H=${dims.H} W=${dims.W}`;
                    if (csa) detail += ` CSA=${csa}`;
                }
                badge = `<span class="cnc-badge cnc-badge-unknown">UNMATCHED SECTION</span>`
                      + (detail ? `<span class="cnc-unknown-dims">${detail} mm</span>` : '')
                      + (msg ? `<span class="cnc-unknown-msg" title="${msg}">${msg}</span>` : '');
                break;
            }
            default:
                return '';
        }

        return badge + (downloadLink ? ' ' + downloadLink : '');
    }

    /**
     * Download BOM as Excel (.xlsx) with embedded STL thumbnails from the server.
     */
    async _downloadBOMXlsx() {
        const filename = this._currentFilename;
        if (!filename) return;

        const btn = document.querySelector('.parts-bom-xlsx-btn');
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Generating\u2026';
        }

        try {
            const resp = await fetch(`/api/v1/cnc-analysis/bom-xlsx/${encodeURIComponent(filename)}`);
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail?.error || `HTTP ${resp.status}`);
            }
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            // Extract filename from Content-Disposition or use default
            const cd = resp.headers.get('Content-Disposition') || '';
            const match = cd.match(/filename="?([^"]+)"?/);
            a.download = match ? match[1] : `${this._lastProjectNumber || 'project'}-BOM.xlsx`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 60_000);
        } catch (e) {
            console.error('BOM XLSX download failed:', e);
            alert(`BOM Excel download failed: ${e.message}`);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = '\u2193\u00a0BOM (.xlsx)';
            }
        }
    }

    /**
     * Generate and download a BOM JSON file with full ordering information.
     * Includes CNC items with dims, weight, qty, filenames; and bought-out items.
     */
    _downloadBOM(cncItems, boItems, unknownItems = [], excludedItems = []) {
        const DENSITY = 7.85e-6; // kg/mm³ (steel)
        const r3 = v => (v != null && isFinite(v)) ? Math.round(v * 1000) / 1000 : null;
        const r1 = v => (v != null && isFinite(v)) ? Math.round(v * 10)   / 10   : null;

        // Apply part-level consolidation groups — same merge logic as the visual
        // BOM table uses in renderRows().  cncItems/boItems are pre-solid-consolidation
        // but not yet part-group consolidated; without this step the exported BOM
        // would list separate rows for every XCAF ref_id even when they represent
        // geometrically identical parts already merged on screen.
        const _consolidate = (items) => {
            const groups = this._consolidationGroups;
            if (!groups || groups.length === 0) return items;
            const refToGroup = new Map();
            for (const g of groups) {
                for (const rid of g.ref_ids) refToGroup.set(rid, g);
            }
            const seen = new Map();   // group object → merged item
            const out  = [];
            for (const item of items) {
                const g = refToGroup.get(item.key);
                if (g) {
                    if (seen.has(g)) {
                        // Fold into the canonical entry
                        const c = seen.get(g);
                        c.qty          += item.qty;
                        c.mirroredCount += item.mirroredCount;
                        c._memberQtys.push(item.qty);
                        for (const p of item.parentNames) {
                            if (!c.parentNames.includes(p)) c.parentNames.push(p);
                        }
                    } else {
                        // First member: shallow-clone and apply canonical name
                        const merged = Object.assign({}, item, {
                            name: g.canonical_name,
                            _memberQtys: [item.qty],
                        });
                        seen.set(g, merged);
                        out.push(merged);
                    }
                } else {
                    out.push(item);
                }
            }
            return out;
        };

        const mergedCnc      = _consolidate(cncItems);
        const mergedBo       = _consolidate(boItems);
        const mergedUnknown  = _consolidate(unknownItems);
        const mergedExcluded = _consolidate(excludedItems);

        const _cncEntry = (item) => {
            const info = this._cncResultForItem(item);
            const res  = info?.result ?? null;

            // Weight — prefer stored mass_kg, fall back to volume × density, then dims × density
            let massEach = res?.mass_kg ?? null;
            if (massEach == null && res?.volume_mm3 != null) {
                massEach = r3(res.volume_mm3 * DENSITY);
            }
            if (massEach == null && res?.dims) {
                const { L, W, T } = res.dims;
                if (L && W && T) massEach = r3(L * W * T * DENSITY);
            }

            const baseName = fn => fn
                ? fn.replace(/\\/g, '/').split('/').pop()
                : null;

            return {
                name:             item.name,
                ref_id:           item.key,
                qty:              item.qty,
                qty_per_occurrence: item._memberQtys ?? null,
                parent_assemblies: (item.parentNames || []).map(p => ({
                    name: p,
                    qty: (item.parentCounts || {})[p] || 1,
                })),
                type:             res?.type        ?? null,
                category:         res?.category    ?? null,
                designation:      res?.designation ?? null,
                profile_type:     res?.profile_type ?? null,
                dims_mm:          res?.dims        ?? null,
                volume_mm3:       res?.volume_mm3  ?? null,
                mass_kg_each:     massEach,
                total_weight_kg:  massEach != null ? r3(massEach * item.qty) : null,
                dxf_file:         baseName(res?.dxf_path),
                nc1_file:         baseName(res?.nc1_path),
                holes:            res?.holes      ?? null,
                end_cuts:         res?.end_cuts   ?? null,
                match_score:      res?.match_score ?? null,
                analysed_at:      res?.analysed_at ?? null,
            };
        };

        const cnc_entries     = mergedCnc.map(_cncEntry);
        const unknown_entries = mergedUnknown.map(_cncEntry);

        const _simpleEntry = (item) => ({
            name:             item.name,
            qty:              item.qty,
            qty_per_occurrence: item._memberQtys ?? null,
            parent_assemblies: (item.parentNames || []).map(p => ({
                name: p,
                qty: (item.parentCounts || {})[p] || 1,
            })),
        });

        const bo_entries       = mergedBo.map(_simpleEntry);
        const excluded_entries = mergedExcluded.map(_simpleEntry);

        const totalCncQty      = cnc_entries.reduce((s, e) => s + e.qty, 0);
        const totalWeight      = cnc_entries.reduce((s, e) => s + (e.total_weight_kg ?? 0), 0);
        const totalBoQty       = bo_entries.reduce((s, e) => s + e.qty, 0);
        const totalUnknownQty  = unknown_entries.reduce((s, e) => s + e.qty, 0);
        const totalExcludedQty = excluded_entries.reduce((s, e) => s + e.qty, 0);

        const bom = {
            project_number:  this._lastProjectNumber  || null,
            steel_grade:     this._lastSteelGrade     || null,
            step_file:       this._currentFilename    || null,
            generated_at:    new Date().toISOString(),
            unknown_items:   unknown_entries,
            cnc_items:       cnc_entries,
            bought_out_items: bo_entries,
            excluded_items:  excluded_entries,
            summary: {
                total_unknown_types:       unknown_entries.length,
                total_unknown_qty:         totalUnknownQty,
                total_cnc_types:           cnc_entries.length,
                total_cnc_qty:             totalCncQty,
                total_estimated_weight_kg: r1(totalWeight),
                total_bought_out_types:    bo_entries.length,
                total_bought_out_qty:      totalBoQty,
                total_excluded_types:      excluded_entries.length,
                total_excluded_qty:        totalExcludedQty,
            },
        };

        const blob = new Blob([JSON.stringify(bom, null, 2)], { type: 'application/json' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `${this._lastProjectNumber || 'project'}-bom.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
    }

    /**
     * Show the settings dialog, then collect XCAF ref_ids and start the analysis.
     */
    _startCncAnalysis(cncItems, force = false) {
        if (this._cncAnalysing) return;
        const filename = this._currentFilename;
        if (!filename || cncItems.length === 0) return;

        this._showCncSettingsDialog((projectNumber, steelGrade) => {
            const refIdSet = new Set();
            const memberIds = {};
            const parentNames = {};

            for (const item of cncItems) {
                let xcafRefId = item.key;

                // For solid body items, use the parent multi-solid's XCAF ref_id
                const parentInfo = this._parentMap.get(item.key);
                if (parentInfo?._solidParentRefId) {
                    xcafRefId = parentInfo._solidParentRefId;
                }

                refIdSet.add(xcafRefId);
                if (!memberIds[xcafRefId]) {
                    memberIds[xcafRefId] = parentInfo?._solidParentRefId
                        ? (parentInfo.parentName || item.name)
                        : item.name;
                }
                if (!parentNames[xcafRefId]) {
                    // Use the first parentNames entry as the external reference (e.g. C25001).
                    // Fall back to the item's own name if no parent is available.
                    const firstParent = item.parentNames?.[0];
                    parentNames[xcafRefId] = firstParent || item.name || '';
                }
            }

            if (refIdSet.size === 0) return;

            this._cncAnalysing = true;
            this._renderPartsList(this._consolidationGroups);

            // When force=true append ?force=1 to the URL so the router bypasses
            // the cache even if api.js is an older cached version without the flag.
            const _cncUrl = `/api/v1/cnc-analysis/analyse/${encodeURIComponent(filename)}${force ? '?force=1' : ''}`;
            const _cncBody = { ref_ids: [...refIdSet], member_ids: memberIds, parent_names: parentNames, project_number: projectNumber, steel_grade: steelGrade, force: force };
            const _cncPromise = fetch(_cncUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(_cncBody) }).then(r => r.json());
            _cncPromise
                .then(resp => {
                    if (resp.cnc_task_id) {
                        this._pollCncAnalysis(resp.cnc_task_id);
                    } else if (resp.status === 'completed') {
                        // All ref_ids were already analysed — backend returned inline.
                        // Merge into existing results so prior partial runs are preserved.
                        this._cncAnalysing = false;
                        if (resp.results) {
                            this._cncAnalysisResults = Object.assign(
                                {}, this._cncAnalysisResults || {}, resp.results
                            );
                        }
                        this._renderPartsList(this._consolidationGroups);
                    } else {
                        this._cncAnalysing = false;
                        this._renderPartsList(this._consolidationGroups);
                    }
                })
                .catch(err => {
                    console.error('Failed to start CNC analysis:', err);
                    this._cncAnalysing = false;
                    this._renderPartsList(this._consolidationGroups);
                });
        });
    }

    /**
     * Show a <dialog> modal to collect project number and steel grade before
     * starting CNC analysis.  Calls onConfirm(projectNumber, steelGrade) when
     * the user clicks Analyse; does nothing if the user cancels.
     */
    _showCncSettingsDialog(onConfirm) {
        // Remove any stale dialog left from a previous call
        const existing = document.getElementById('cnc-settings-dialog');
        if (existing) existing.remove();

        const dialog = document.createElement('dialog');
        dialog.id = 'cnc-settings-dialog';
        dialog.className = 'cnc-settings-dialog';
        dialog.innerHTML = `
            <form method="dialog" class="cnc-settings-form">
                <h3 class="cnc-settings-title">CNC Analysis Settings</h3>
                <div class="cnc-settings-field">
                    <label for="cnc-project-number">Project Number</label>
                    <input type="text" id="cnc-project-number"
                           placeholder="e.g. C25001"
                           value="${this._esc(this._lastProjectNumber)}"
                           autocomplete="off">
                </div>
                <div class="cnc-settings-field">
                    <label for="cnc-steel-grade">Material Grade</label>
                    <input type="text" id="cnc-steel-grade"
                           placeholder="e.g. S275, 6061-T6"
                           value="${this._esc(this._lastSteelGrade)}"
                           autocomplete="off">
                    <small class="cnc-settings-hint">Aluminium grades (6061, 6082 etc.) use the AL section library</small>
                </div>
                <div class="cnc-settings-actions">
                    <button type="button" id="cnc-settings-cancel" class="outline">Cancel</button>
                    <button type="submit">Analyse</button>
                </div>
            </form>
        `;

        document.body.appendChild(dialog);
        dialog.showModal();

        // Focus the first input
        dialog.querySelector('#cnc-project-number')?.focus();

        dialog.querySelector('#cnc-settings-cancel').addEventListener('click', () => {
            dialog.close();
            dialog.remove();
        });

        dialog.querySelector('form').addEventListener('submit', (e) => {
            e.preventDefault();
            const projNum = dialog.querySelector('#cnc-project-number').value.trim();
            const grade   = dialog.querySelector('#cnc-steel-grade').value.trim() || 'S275';
            this._lastProjectNumber = projNum;
            this._lastSteelGrade   = grade;
            dialog.close();
            dialog.remove();
            onConfirm(projNum, grade);
        });

        // ESC key — browser fires 'cancel' on <dialog>
        dialog.addEventListener('cancel', () => {
            dialog.remove();
        });
    }

    _pollCncAnalysis(taskId) {
        if (this._cncPollTimer) clearInterval(this._cncPollTimer);

        this._cncPollTimer = setInterval(() => {
            this.api.getCncStatus(taskId)
                .then(resp => {
                    if (resp.status === 'completed') {
                        clearInterval(this._cncPollTimer);
                        this._cncPollTimer = null;
                        this._cncAnalysing = false;
                        // Merge new results into any pre-existing ones so that
                        // results from earlier partial runs are not discarded.
                        this._cncAnalysisResults = Object.assign(
                            {}, this._cncAnalysisResults || {}, resp.results || {}
                        );

                        const panel = this.container?.querySelector('#parts-list-panel');
                        if (panel && !panel.hidden) {
                            this._renderPartsList(this._consolidationGroups);
                        }
                    } else if (resp.status === 'failed') {
                        clearInterval(this._cncPollTimer);
                        this._cncPollTimer = null;
                        this._cncAnalysing = false;
                        console.error('CNC analysis failed:', resp.error);

                        // Reload cached results from the server — the worker may have
                        // saved partial results progressively before timing out.
                        const fn = this._currentFilename;
                        if (fn) {
                            this.api.getCncResult(fn)
                                .then(r => {
                                    if (r?.results) {
                                        this._cncAnalysisResults = Object.assign(
                                            {}, this._cncAnalysisResults || {}, r.results
                                        );
                                    }
                                })
                                .catch(() => {})
                                .finally(() => {
                                    const panel = this.container?.querySelector('#parts-list-panel');
                                    if (panel && !panel.hidden) {
                                        this._renderPartsList(this._consolidationGroups);
                                    }
                                });
                        } else {
                            const panel = this.container?.querySelector('#parts-list-panel');
                            if (panel && !panel.hidden) {
                                this._renderPartsList(this._consolidationGroups);
                            }
                        }
                    }
                    // pending/running → keep polling
                })
                .catch(() => { /* network hiccup — keep polling */ });
        }, 2000);
    }

    _startConsolidation() {
        if (this._consolidating) return;
        const filename = this._currentFilename;
        if (!filename) return;

        this._consolidating = true;
        this._renderPartsList(null);  // Re-render to show "Consolidating…" button state

        this.api.startConsolidation(filename)
            .then(resp => {
                if (resp.groups) {
                    this._consolidationGroups = resp.groups;
                    this._solidConsolidationGroups = resp.solid_groups || [];
                    this._intraSolidGroups = resp.intra_solid_groups || [];
                    this._consolidating = false;
                    this._renderPartsList(this._consolidationGroups);
                } else if (resp.consolidation_task_id) {
                    this._pollConsolidation(resp.consolidation_task_id);
                } else {
                    this._consolidating = false;
                    this._renderPartsList(null);
                }
            })
            .catch(err => {
                console.error('Failed to start consolidation:', err);
                this._consolidating = false;
                this._renderPartsList(null);
            });
    }

    _pollConsolidation(taskId) {
        if (this._consolidatePollTimer) clearInterval(this._consolidatePollTimer);

        this._consolidatePollTimer = setInterval(() => {
            this.api.getConsolidationStatus(taskId)
                .then(resp => {
                    if (resp.status === 'completed') {
                        clearInterval(this._consolidatePollTimer);
                        this._consolidatePollTimer = null;
                        this._consolidating = false;
                        this._consolidationGroups = resp.groups;
                        this._solidConsolidationGroups = resp.solid_groups || [];
                        this._intraSolidGroups = resp.intra_solid_groups || [];

                        const panel = this.container?.querySelector('#parts-list-panel');
                        if (panel && !panel.hidden) {
                            this._renderPartsList(this._consolidationGroups);
                        }
                    } else if (resp.status === 'failed') {
                        clearInterval(this._consolidatePollTimer);
                        this._consolidatePollTimer = null;
                        this._consolidating = false;
                        console.error('Consolidation failed:', resp.error);
                        const panel = this.container?.querySelector('#parts-list-panel');
                        if (panel && !panel.hidden) {
                            this._renderPartsList(null);
                        }
                    }
                    // pending/running → keep polling
                })
                .catch(() => {/* network hiccup — keep polling */});
        }, 2000);
    }

    /**
     * Classify all instances of a part by nodeId list.
     * Used by the parts list panel to classify parts that may not yet be
     * visible in the DOM (unexploded assemblies).
     */
    _classifyAllInstances(nodeIds, action) {
        const treeEl = this.container.querySelector('#assembly-tree-container');

        for (const nodeId of nodeIds) {
            this.classifications.set(nodeId, action);

            if (treeEl) {
                for (const li of treeEl.querySelectorAll(
                    `.tree-node[data-node-id="${CSS.escape(nodeId)}"]`
                )) {
                    li.classList.add('node-classified');
                    li.dataset.classification = action;
                }
            }
        }

        this._updateTreeSelectability();
        this._updateProgress();
        this._debouncedSave();
    }

    // ---------------------------------------------------------------
    // Nesting integration
    // ---------------------------------------------------------------

    /**
     * Check whether there are any CNC section items suitable for nesting.
     * Only type=section results have a designation and length for nesting.
     */
    _hasNestableSections(cncItems, unknownItems) {
        const allItems = [...cncItems, ...unknownItems];
        for (const item of allItems) {
            const info = this._cncResultForItem(item);
            if (info?.result?.type === 'section' && info.result.designation && info.result.dims?.L) {
                return true;
            }
        }
        return false;
    }

    /**
     * Build the nesting request items array from CNC-analysed BOM items.
     * Expands each BOM row by its instance count so nesting gets one item per piece.
     */
    _buildNestingItems(bomItems) {
        const items = [];
        let idx = 0;

        for (const item of bomItems) {
            const info = this._cncResultForItem(item);
            if (!info?.result) continue;
            const r = info.result;
            if (r.type !== 'section' || !r.designation || !r.dims?.L) continue;

            const section = r.designation;
            const length = Math.round(r.dims.L);
            const parentName = item.parentNames?.[0] || '';

            // Expand by qty — one nesting item per physical piece
            for (let i = 0; i < item.qty; i++) {
                items.push({
                    item_index: idx++,
                    ref_id: item.key,
                    section,
                    length,
                    parent: parentName,
                    member_name: item.name,
                });
            }
        }

        return items;
    }

    /**
     * Show the nesting settings dialog — stock length, qty, kerf, per-section overrides.
     */
    _showNestingSettingsDialog(bomItems) {
        const nestingItems = this._buildNestingItems(bomItems);
        if (nestingItems.length === 0) {
            alert('No section items available for nesting. Run CNC analysis first.');
            return;
        }

        // Collect unique sections and their max item length
        const sectionInfo = new Map();
        for (const it of nestingItems) {
            const existing = sectionInfo.get(it.section);
            if (!existing) {
                sectionInfo.set(it.section, { count: 1, maxLen: it.length });
            } else {
                existing.count++;
                existing.maxLen = Math.max(existing.maxLen, it.length);
            }
        }

        const existing = document.getElementById('nesting-settings-dialog');
        if (existing) existing.remove();

        const sectionRows = [...sectionInfo.entries()]
            .sort((a, b) => a[0].localeCompare(b[0]))
            .map(([sec, info]) => `
                <tr class="nesting-section-row" data-section="${this._esc(sec)}">
                    <td class="nesting-section-name">${this._esc(sec)}</td>
                    <td class="nesting-section-count">${info.count} pcs</td>
                    <td class="nesting-section-maxlen">${info.maxLen} mm</td>
                    <td><input type="number" class="nesting-stock-len" min="100" step="100" placeholder="default" title="Stock length override for this section"></td>
                    <td><input type="number" class="nesting-stock-qty" min="1" step="1" placeholder="default" title="Stock qty override for this section"></td>
                </tr>
            `).join('');

        const dialog = document.createElement('dialog');
        dialog.id = 'nesting-settings-dialog';
        dialog.className = 'nesting-settings-dialog';
        dialog.innerHTML = `
            <form method="dialog" class="nesting-settings-form">
                <h3 class="nesting-settings-title">\u2702 Nesting Settings</h3>
                <p class="nesting-settings-summary">${nestingItems.length} section pieces across ${sectionInfo.size} profile(s)</p>

                <div class="nesting-settings-defaults">
                    <div class="nesting-settings-field">
                        <label for="nesting-stock-length">Default Stock Length (mm)</label>
                        <input type="number" id="nesting-stock-length" min="100" step="100"
                               value="${this._nestingDefaultStockLength}">
                    </div>
                    <div class="nesting-settings-field">
                        <label for="nesting-stock-qty">Default Stock Qty</label>
                        <input type="number" id="nesting-stock-qty" min="1" step="1"
                               value="${this._nestingDefaultStockQty}">
                    </div>
                    <div class="nesting-settings-field">
                        <label for="nesting-kerf">Kerf / Blade Width (mm)</label>
                        <input type="number" id="nesting-kerf" min="0" step="1"
                               value="${this._nestingKerf}">
                    </div>
                </div>

                ${sectionInfo.size > 1 ? `
                <details class="nesting-overrides-details">
                    <summary>Per-section stock overrides</summary>
                    <table class="nesting-overrides-table">
                        <thead>
                            <tr><th>Section</th><th>Items</th><th>Max Len</th><th>Stock Len</th><th>Stock Qty</th></tr>
                        </thead>
                        <tbody>${sectionRows}</tbody>
                    </table>
                </details>` : ''}

                <div class="nesting-settings-actions">
                    <button type="button" id="nesting-cancel" class="outline">Cancel</button>
                    <button type="submit">Run Nesting</button>
                </div>
            </form>
        `;

        document.body.appendChild(dialog);
        dialog.showModal();
        dialog.querySelector('#nesting-stock-length')?.focus();

        dialog.querySelector('#nesting-cancel').addEventListener('click', () => {
            dialog.close();
            dialog.remove();
        });

        dialog.querySelector('form').addEventListener('submit', (e) => {
            e.preventDefault();
            const stockLen = parseInt(dialog.querySelector('#nesting-stock-length').value) || 6000;
            const stockQty = parseInt(dialog.querySelector('#nesting-stock-qty').value) || 20;
            const kerf = parseInt(dialog.querySelector('#nesting-kerf').value) || 3;

            this._nestingDefaultStockLength = stockLen;
            this._nestingDefaultStockQty = stockQty;
            this._nestingKerf = kerf;

            // Collect per-section overrides
            const stockPerSection = [];
            for (const row of dialog.querySelectorAll('.nesting-section-row')) {
                const sec = row.dataset.section;
                const lenInput = row.querySelector('.nesting-stock-len');
                const qtyInput = row.querySelector('.nesting-stock-qty');
                const len = parseInt(lenInput?.value);
                const qty = parseInt(qtyInput?.value);
                if (len > 0 && qty > 0) {
                    stockPerSection.push({ section: sec, stock: [{ length: len, qty }] });
                }
            }

            dialog.close();
            dialog.remove();
            this._submitNesting(nestingItems, stockPerSection, stockLen, stockQty, kerf);
        });

        dialog.addEventListener('cancel', () => { dialog.remove(); });
    }

    /**
     * Submit the nesting job to the nesting service.
     */
    async _submitNesting(items, stockPerSection, defaultLen, defaultQty, kerf) {
        this._nestingRunning = true;
        this._nestingResult = null;
        this._nestingCuttingList = null;
        this._renderPartsList(this._consolidationGroups);

        const body = {
            job_label: this._lastProjectNumber || this._currentFilename || 'nesting',
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
                this._pollNesting(resp.task_id);
            } else {
                this._nestingRunning = false;
                this._renderPartsList(this._consolidationGroups);
            }
        })
        .catch(err => {
            console.error('Failed to start nesting:', err);
            this._nestingRunning = false;
            alert(`Nesting failed to start: ${err.message}`);
            this._renderPartsList(this._consolidationGroups);
        });
    }

    /**
     * Poll the nesting service for task completion.
     */
    async _pollNesting(taskId) {
        if (this._nestingPollTimer) clearInterval(this._nestingPollTimer);

        const NESTING_BASE = await this.api.getNestingBase();

        // Show initial progress
        this._renderNestingProgress({ phase: 0, description: 'Starting nesting solver\u2026' });

        this._nestingPollTimer = setInterval(() => {
            fetch(`${NESTING_BASE}/api/v1/nesting/status/${encodeURIComponent(taskId)}`)
                .then(r => r.json())
                .then(resp => {
                    if (resp.status === 'completed') {
                        clearInterval(this._nestingPollTimer);
                        this._nestingPollTimer = null;
                        this._nestingRunning = false;

                        // Fetch the cutting list
                        fetch(`${NESTING_BASE}/api/v1/nesting/cutting-list/${encodeURIComponent(taskId)}`)
                            .then(r => r.json())
                            .then(cl => {
                                this._nestingCuttingList = cl;
                                this._nestingResult = resp.result || null;
                                this._renderPartsList(this._consolidationGroups);
                            })
                            .catch(err => {
                                console.error('Failed to fetch cutting list:', err);
                                this._renderPartsList(this._consolidationGroups);
                            });

                    } else if (resp.status === 'failed') {
                        clearInterval(this._nestingPollTimer);
                        this._nestingPollTimer = null;
                        this._nestingRunning = false;
                        console.error('Nesting failed:', resp.error);
                        alert(`Nesting failed: ${resp.error || 'unknown error'}`);
                        this._renderPartsList(this._consolidationGroups);

                    } else {
                        // running/pending — update progress
                        this._renderNestingProgress(resp.progress || {});
                    }
                })
                .catch(() => { /* network hiccup — keep polling */ });
        }, 2000);
    }

    /**
     * Render inline progress indicator while nesting is running.
     */
    _renderNestingProgress(progress) {
        const panel = this.container?.querySelector('#nesting-results-panel');
        if (!panel) return;
        panel.hidden = false;

        const phase = progress.phase || 0;
        const desc = progress.description || 'Working\u2026';
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

    /**
     * Render the cutting list results inside the nesting panel.
     */
    _renderCuttingList() {
        const panel = this.container?.querySelector('#nesting-results-panel');
        if (!panel || !this._nestingCuttingList) return;
        panel.hidden = false;

        const cl = this._nestingCuttingList;
        const totals = cl.totals || {};
        const sections = cl.sections || [];

        let html = `
            <div class="nesting-results-header">
                <span class="nesting-results-title">Cutting List</span>
                <div class="nesting-results-actions">
                    <button class="nesting-csv-btn outline">\u2193 CSV</button>
                    <button class="nesting-close-btn outline">\u2715</button>
                </div>
            </div>
            <div class="nesting-totals">
                <span>Placed: <strong>${totals.total_items_placed ?? '?'}</strong></span>
                <span>Unassigned: <strong>${totals.total_items_unassigned ?? 0}</strong></span>
                <span>Stocks used: <strong>${totals.total_stocks_used ?? '?'}</strong></span>
                <span>Total waste: <strong>${totals.total_waste_mm != null ? (totals.total_waste_mm / 1000).toFixed(1) + ' m' : '?'}</strong></span>
            </div>
        `;

        for (const section of sections) {
            const summ = section.summary || {};
            const statusBadge = section.phase1_status === 'optimal'
                ? '<span class="nesting-status-badge nesting-status-optimal">optimal</span>'
                : section.phase1_status === 'feasible'
                    ? '<span class="nesting-status-badge nesting-status-feasible">feasible</span>'
                    : `<span class="nesting-status-badge nesting-status-other">${this._esc(section.phase1_status || '?')}</span>`;

            html += `
                <details class="nesting-section-details" open>
                    <summary class="nesting-section-summary">
                        <strong>${this._esc(section.designation)}</strong>
                        \u2014 ${summ.items_placed ?? '?'} placed, ${summ.stocks_used ?? '?'} bars
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

                // Visual representation of cuts on the bar
                for (const cut of (bar.cuts || [])) {
                    const widthPct = bar.stock_length_mm > 0
                        ? (cut.length_mm / bar.stock_length_mm) * 100
                        : 0;
                    const label = cut.member || cut.ref_id || `Cut ${cut.cut_no}`;
                    html += `<div class="nesting-cut-block" style="width:${widthPct.toFixed(1)}%" title="${this._esc(label)}: ${cut.length_mm} mm${cut.parent ? ' (' + this._esc(cut.parent) + ')' : ''}">${cut.length_mm}</div>`;
                }

                // Waste block
                if (bar.waste_mm > 0 && bar.stock_length_mm > 0) {
                    const wastePct = (bar.waste_mm / bar.stock_length_mm) * 100;
                    html += `<div class="nesting-waste-block" style="width:${wastePct.toFixed(1)}%">${bar.waste_mm}</div>`;
                }

                html += `
                        </div>
                        <table class="nesting-cuts-table">
                            <thead><tr><th>#</th><th>Member</th><th>Parent</th><th>Length</th></tr></thead>
                            <tbody>
                `;

                for (const cut of (bar.cuts || [])) {
                    html += `<tr>
                        <td>${cut.cut_no}</td>
                        <td>${this._esc(cut.member || cut.ref_id || '—')}</td>
                        <td>${this._esc(cut.parent || '—')}</td>
                        <td>${cut.length_mm} mm</td>
                    </tr>`;
                }

                html += `</tbody></table></div>`;
            }

            // Unassigned items
            if (section.unassigned && section.unassigned.length > 0) {
                html += `<div class="nesting-unassigned">
                    <strong>\u26a0 Unassigned (${section.unassigned.length})</strong>
                    <ul>`;
                for (const u of section.unassigned) {
                    html += `<li>${this._esc(u.member_name || u.ref_id || '?')} \u2014 ${u.length} mm</li>`;
                }
                html += `</ul></div>`;
            }

            html += `</div></details>`;
        }

        panel.innerHTML = html;

        // CSV download
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

        // Close button
        panel.querySelector('.nesting-close-btn')?.addEventListener('click', () => {
            panel.hidden = true;
        });
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
