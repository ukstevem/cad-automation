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

        /** @type {Array|null} unified native BOM rows from analysis.native_bom */
        this._nativeBom = null;

        /** @type {'instance'|'consolidated'} current Full BOM view mode */
        this._nativeBomView = 'instance';

        /** @type {'step'|'ifc'} source of the current analysis (determines world-placement handling) */
        this._source = 'step';

        /** @type {boolean} whether the viewer is showing the full assembly with highlight */
        this._showFullAssembly = false;

        /** @type {Map<string, number[]>} tree nodeId → column-major 4x4 world transform (STEP only) */
        this._worldPlacements = new Map();

        /** @type {Map<string, number>} tree nodeId → mesh index in the loaded assembly scene */
        this._meshIndexByNodeId = new Map();

        /** @type {boolean} whether clicks on meshes build a group selection instead of selecting a tree node */
        this._groupMode = false;

        /** @type {Set<string>} currently selected node_ids when group mode is on (yet to be grouped) */
        this._groupSelection = new Set();

        /**
         * @type {Map<string, {id: string, name: string, node_ids: string[], created_at: string}>}
         * Custom groups keyed by group id, persisted under project_state.groups.
         */
        this.groups = new Map();

        /** @type {string|null} currently isolated group id (everything else dimmed) */
        this._isolatedGroupId = null;

        /** @type {Set<string>} group ids currently hidden from the viewer */
        this._hiddenGroupIds = new Set();

        /** @type {Set<string>} group ids whose member list is expanded in the panel */
        this._expandedGroups = new Set();

        /** @type {Map<string, number>} refId -> total instance count in tree (for qty display) */
        this._refIdInstanceCount = new Map();

        /** @type {Object|null} CNC analysis results keyed by ref_id (null = not yet loaded) */
        this._cncAnalysisResults = null;

        /** @type {{state:string, consolidated_at:string|null, analyzed_at:string|null,
         *          cnc_ref_count:number, stale_cnc_refs:string[]}|null}
         *  Consolidation/CNC freshness for gating the Analyse/Download buttons. */
        this._cncState = null;

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
        this._nativeBom = null;
        this._source = 'step';
        this._showFullAssembly = false;
        this._worldPlacements?.clear();
        this._meshIndexByNodeId?.clear();
        this._groupMode = false;
        this._groupSelection?.clear();
        this.groups?.clear();
        this._isolatedGroupId = null;
        this._hiddenGroupIds?.clear();
        this._expandedGroups?.clear();
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
                        <div class="viewer-toolbar">
                            <button id="show-full-assembly-btn" class="outline" title="Show the whole assembly in place; selected parts highlight">Show Full Assembly</button>
                            <button id="group-mode-btn" class="outline" title="Toggle group-selection mode: click parts to build a selection, then create a named group" hidden>Group Mode</button>
                            <span id="group-selection-counter" class="group-selection-counter" hidden></span>
                            <button id="group-create-btn" class="outline" hidden>Create Group</button>
                            <button id="group-add-btn" class="outline" hidden>Add to Group…</button>
                            <button id="group-clear-btn" class="outline" hidden>Clear Selection</button>
                            <button id="viewer-groups-btn" class="outline" title="Open the groups panel" hidden>Groups</button>
                            <button id="viewer-maximize-btn" class="outline viewer-maximize-btn" title="Maximise viewer (Esc to restore)" aria-label="Maximise viewer">&#x26F6;</button>
                        </div>
                        <div id="stl-viewer-panel" class="stl-viewer-panel">
                            <div class="stl-viewer-placeholder">
                                Click a node in the tree to preview its 3D model
                            </div>
                        </div>
                        <div id="viewer-load-status" class="viewer-load-status" hidden></div>
                    </div>
                </div>

                <div id="parts-list-bar" class="parts-list-bar" hidden>
                    <button id="show-parts-list-btn" class="outline">BOM</button>
                    <button id="show-native-bom-btn" class="outline" title="Full BOM including unclassified and bought-out parts">Full BOM</button>
                    <button id="show-groups-btn" class="outline" title="Manage custom groupings of parts">Groups</button>
                </div>
                <div id="parts-list-panel" class="parts-list-panel" hidden></div>
                <div id="native-bom-panel" class="parts-list-panel" hidden></div>
                <div id="groups-panel" class="parts-list-panel" hidden></div>
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
            } else if (e.target.id === 'show-native-bom-btn') {
                this._toggleNativeBom();
            } else if (e.target.id === 'show-groups-btn') {
                this._toggleGroupsPanel();
            } else if (e.target.id === 'show-full-assembly-btn') {
                this._toggleFullAssembly();
            } else if (e.target.id === 'group-mode-btn') {
                this._toggleGroupMode();
            } else if (e.target.id === 'group-create-btn') {
                this._promptCreateGroup();
            } else if (e.target.id === 'group-add-btn') {
                this._showAddToGroupPicker(e.target);
            } else if (e.target.id === 'group-clear-btn') {
                this._clearGroupSelection();
            } else if (e.target.id === 'viewer-groups-btn') {
                this._toggleGroupsPanel();
            } else if (e.target.closest('#viewer-maximize-btn')) {
                this._toggleViewerMaximized();
            }
        });

        // Esc restores the viewer from maximised state.
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const panel = this.container?.querySelector('.workspace-viewer-panel');
                if (panel?.classList.contains('viewer-maximized')) {
                    this._toggleViewerMaximized();
                }
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
        // Classify the (restored) frontier with the lightweight pass, then
        // auto-apply — runs ahead of any heavy CNC analysis.
        this._classifyFrontier();

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
        this._nativeBom = Array.isArray(data.native_bom) ? data.native_bom : [];
        this._source = (data.summary && data.summary.source) || 'step';
        treeEl.innerHTML = '<ul>' + nodes.map(n => this._renderNode(n, 0)).join('') + '</ul>';

        this._buildParentMap(nodes, null);
        this._extractPlacements(nodes);
        this._computeWorldPlacements(nodes);
        this._bindTreeEvents(treeEl);

        // Show the "All Parts" button once the tree is available
        const partsBar = this.container.querySelector('#parts-list-bar');
        if (partsBar) partsBar.hidden = false;

        // Seed the classification breakdown (restored project state will refresh
        // it again once classifications are applied).
        this._updateProgress();
    }

    /**
     * Human-recognisable name for a tree node.  STEP single-part assemblies
     * often wrap their geometry in a generic "SOLID"/"COMPOUND" leaf while the
     * real name sits on the parent assembly.  For solid-bearing leaves with a
     * generic name, fall back to the parent name; no-solid artifacts keep the
     * generic name so they stay visually distinct from their solid sibling.
     */
    _displayNodeName(node, parentName) {
        const GENERIC = this._genericLeafNames ||
            (this._genericLeafNames = new Set(['solid', 'compound', 'shape', 'unnamed', '']));
        const nm = (node.name || '').trim();
        if (node.node_type !== 'part_no_solid'
            && GENERIC.has(nm.toLowerCase())
            && parentName && parentName.trim()) {
            return parentName.trim();
        }
        return node.name || '';
    }

    _renderNode(node, depth, parentName = null) {
        const hasChildren = node.children && node.children.length > 0;
        const toggleClass = hasChildren ? 'expanded' : 'leaf';
        const childrenHtml = hasChildren
            ? '<ul>' + node.children.map(c => this._renderNode(c, depth + 1, node.name)).join('') + '</ul>'
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
                    <span class="tree-node-name">${this._esc(this._displayNodeName(node, parentName))}</span>
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
            this._parentMap.set(node.id, { name: node.name, parentName, nodeType: node.node_type });
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
        if (!url) {
            // No STL for this node, but if full assembly is on we can still highlight.
            if (this._showFullAssembly && this._meshIndexByNodeId.has(nodeId)) {
                this._selectedNodeId = nodeId;
                for (const row of treeEl.querySelectorAll('.tree-node-row')) {
                    const li = row.closest('.tree-node');
                    row.classList.toggle('node-selected', li.dataset.nodeId === nodeId);
                }
                this._highlightSelectedInAssembly(nodeId);
            }
            return;
        }

        this._selectedNodeId = nodeId;
        this._multiSolidMeshMap = null;
        this._multiSolidParentId = null;
        this._multiSolidDefaultColors = null;

        for (const row of treeEl.querySelectorAll('.tree-node-row')) {
            const li = row.closest('.tree-node');
            row.classList.toggle('node-selected', li.dataset.nodeId === nodeId);
        }

        if (this._showFullAssembly) {
            this._highlightSelectedInAssembly(nodeId);
        } else {
            this._loadInViewer(url);
        }
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
    // Full assembly view — load every generated STL, highlight selection
    // ---------------------------------------------------------------

    /**
     * Column-major 4x4 matrix multiply (mirrors _mat4_mul in analysis.py).
     * Used to walk tree nodes and accumulate local placements into world-space.
     */
    _matMul4(a, b) {
        const r = new Array(16).fill(0);
        for (let col = 0; col < 4; col++) {
            for (let row = 0; row < 4; row++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    sum += a[k * 4 + row] * b[col * 4 + k];
                }
                r[col * 4 + row] = sum;
            }
        }
        return r;
    }

    /**
     * Walk the assembly tree and record each node's world transform.
     *
     * STEP: tree nodes carry local placements; world = parent_world × local.
     * IFC:  tree nodes carry world-space placements AND STLs are generated in
     *       world coordinates (use-world-coords), so we skip accumulation and
     *       leave _worldPlacements empty — loadScene will then place meshes
     *       at origin, which is correct for pre-transformed IFC geometry.
     */
    _computeWorldPlacements(nodes) {
        this._worldPlacements.clear();
        if (this._source === 'ifc') return;

        const IDENT = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
        const walk = (list, parentMat) => {
            for (const node of list || []) {
                const local = Array.isArray(node.placement) && node.placement.length === 16
                    ? node.placement
                    : IDENT;
                const world = this._matMul4(parentMat, local);
                this._worldPlacements.set(node.id, world);
                if (node.children && node.children.length > 0) {
                    walk(node.children, world);
                }
            }
        };
        walk(nodes, IDENT);
    }

    async _toggleFullAssembly() {
        this._showFullAssembly = !this._showFullAssembly;
        const btn = this.container.querySelector('#show-full-assembly-btn');
        if (btn) {
            btn.textContent = this._showFullAssembly ? 'Hide Full Assembly' : 'Show Full Assembly';
            btn.classList.toggle('viewer-active-toggle', this._showFullAssembly);
        }

        if (this._showFullAssembly) {
            await this._loadFullAssemblyView();
            if (this._selectedNodeId && this._meshIndexByNodeId.has(this._selectedNodeId)) {
                this._highlightSelectedInAssembly(this._selectedNodeId);
            }
        } else {
            // Exiting full-assembly view — drop group-mode state too.
            this._groupMode = false;
            this._groupSelection.clear();
            this._isolatedGroupId = null;
            this._renderViewerLoadStatus(null, []);
            // Revert: if a node is selected with an STL, show it alone; else placeholder.
            if (this._selectedNodeId && this.stlMap.has(this._selectedNodeId)) {
                this._loadInViewer(this.stlMap.get(this._selectedNodeId));
            } else {
                this._showViewerPlaceholder();
            }
        }
        this._updateGroupToolbar();
    }

    async _loadFullAssemblyView() {
        const panel = this.container.querySelector('#stl-viewer-panel');
        if (!panel) return;

        if (!this.stlMap || this.stlMap.size === 0) {
            panel.innerHTML = '<div class="stl-viewer-placeholder">STL files not generated yet. Wait for preview generation to complete, then try again.</div>';
            return;
        }

        const items = [];
        for (const [nodeId, url] of this.stlMap) {
            const placement = this._worldPlacements.get(nodeId);
            items.push({
                url,
                nodeId,
                placement: (placement && placement.length === 16) ? placement : null,
                color: 0xaaaaaa,
                opacity: 1.0,
            });
        }

        if (!this._viewer) {
            panel.innerHTML = '';
            this._viewer = new STLViewer(panel);
        }

        panel.classList.add('loading');
        try {
            const summary = await this._viewer.loadScene(items);
            this._meshIndexByNodeId.clear();
            items.forEach((item, i) => this._meshIndexByNodeId.set(item.nodeId, i));
            this._renderViewerLoadStatus(summary, items);

            // Paint each mesh in its classification colour so overall progress
            // is visible at a glance.
            this._applyClassificationColors();

            // Picking: left-click a mesh → either toggle group selection (if in
            // Group Mode) or select the matching tree node.
            this._viewer.setOnMeshClick((nodeId) => {
                if (!nodeId) return;
                if (this._groupMode) {
                    this._toggleMeshInGroupSelection(nodeId);
                } else {
                    this._selectAndScrollToNode(nodeId);
                }
            });
            // Right-click a mesh → context menu.
            //   * Group Mode  → group-membership actions (remove from group).
            //   * Otherwise   → classification (CNC / BO / EXC).
            this._viewer.setOnMeshContextMenu((nodeId, ev) => {
                if (!nodeId) return;
                if (this._groupMode) {
                    this._showGroupContextMenu(nodeId, ev.clientX, ev.clientY);
                } else {
                    this._selectAndScrollToNode(nodeId);
                    this._showMeshContextMenu(nodeId, ev.clientX, ev.clientY);
                }
            });
        } catch (err) {
            console.warn('Full assembly load failed', err);
        } finally {
            panel.classList.remove('loading');
        }
    }

    /**
     * Render BOM-vs-scene reconciliation banner above the viewer panel.
     *
     * Shows: total loaded vs total requested, per-entity tally pulled from
     * native_bom (so the user can sanity-check 'I expected 1542 IfcBeam, the
     * scene rendered 1530 — 12 are missing'), and an expandable list of the
     * actual failed URLs. Independent of the loader-failure root cause; its
     * job is to make completeness observable.
     */
    _renderViewerLoadStatus(summary, items) {
        const statusEl = this.container.querySelector('#viewer-load-status');
        if (!statusEl) return;
        if (!summary) { statusEl.hidden = true; statusEl.innerHTML = ''; return; }

        const failedUrls = new Set(summary.failures.map(f => f.url));
        const bomByNodeId = new Map();
        if (Array.isArray(this._nativeBom)) {
            for (const r of this._nativeBom) bomByNodeId.set(r.node_id, r);
        }

        // Tally loaded/failed per entity across the items we actually asked for.
        const tally = new Map(); // entity -> {loaded, failed}
        for (const item of items) {
            const row = bomByNodeId.get(item.nodeId);
            const ent = row?.entity || 'Unknown';
            if (!tally.has(ent)) tally.set(ent, { loaded: 0, failed: 0 });
            if (failedUrls.has(item.url)) tally.get(ent).failed++;
            else tally.get(ent).loaded++;
        }

        const totalAsked = summary.total;
        const totalLoaded = summary.loaded;
        const totalFailed = summary.failures.length;

        const tallyHtml = Array.from(tally.entries())
            .sort((a, b) => (b[1].loaded + b[1].failed) - (a[1].loaded + a[1].failed))
            .map(([ent, c]) => {
                const cls = c.failed > 0 ? 'vls-entity vls-entity-fail' : 'vls-entity';
                const failPart = c.failed > 0 ? ` <span class="vls-fail-count">(${c.failed} failed)</span>` : '';
                return `<span class="${cls}"><strong>${this._esc(ent)}</strong> ${c.loaded}${failPart}</span>`;
            }).join('');

        const headerCls = totalFailed > 0 ? 'vls-header vls-header-warn' : 'vls-header vls-header-ok';
        const headerIcon = totalFailed > 0 ? '⚠' : '✓';
        const headerText = totalFailed > 0
            ? `${totalLoaded} of ${totalAsked} parts rendered (${totalFailed} failed)`
            : `${totalLoaded} of ${totalAsked} parts rendered`;

        const failuresList = totalFailed > 0
            ? `<details class="vls-failures">
                 <summary>Show failed URLs (${totalFailed})</summary>
                 <ul>${summary.failures.slice(0, 200).map(f =>
                     `<li><code>${this._esc(f.url.split('/').pop())}</code> — <span class="vls-fail-reason">${this._esc(f.reason)}</span></li>`
                 ).join('')}${totalFailed > 200 ? `<li>… ${totalFailed - 200} more</li>` : ''}</ul>
               </details>`
            : '';

        statusEl.innerHTML = `
            <div class="${headerCls}"><span class="vls-icon">${headerIcon}</span> ${headerText}</div>
            <div class="vls-tally">${tallyHtml}</div>
            ${failuresList}
        `;
        statusEl.hidden = false;
    }

    _highlightSelectedInAssembly(nodeId) {
        if (!this._viewer || this._meshIndexByNodeId.size === 0) return;
        // In group mode the single-node highlight is suppressed; group selection
        // is what we're visualising. Fall through to the full paint.
        if (this._groupMode) {
            this._paintAllMeshes();
            return;
        }
        const HIGHLIGHT = 0xff6600;
        const isolated = this._isolatedNodeIds();
        const hidden = this._hiddenNodeIds();
        const selectedIdx = this._meshIndexByNodeId.get(nodeId);
        this._meshIndexByNodeId.forEach((idx, nid) => {
            // A mesh is visible only when (a) it's within the isolate filter,
            // if any; AND (b) it isn't marked hidden by any hide-group flag.
            // Hide stacks on isolate — so you can isolate a parent group and
            // still hide one of its children.
            const inIsolate = !isolated || isolated.has(nid);
            const inHide    = hidden && hidden.has(nid);
            if (!inIsolate || inHide) {
                this._viewer.setMeshVisible(idx, false);
                return;
            }
            this._viewer.setMeshVisible(idx, true);
            if (nid === nodeId && selectedIdx != null) {
                this._viewer.setMeshColor(idx, HIGHLIGHT, 1.0);
            } else {
                const color = this._classificationColorFor(nid);
                this._viewer.setMeshColor(idx, color, 0.18);
            }
        });
    }

    /**
     * Paint every mesh considering (in order of precedence):
     *   1. Isolated group — members fully visible, non-members completely hidden
     *   2. Group-mode selection — selected parts yellow, others at class colour
     *   3. Classification colour only
     */
    _paintAllMeshes() {
        if (!this._viewer || this._meshIndexByNodeId.size === 0) return;
        const GROUP_SELECT = 0xfacc15; // yellow — pending group members
        const isolated = this._isolatedNodeIds();
        const hidden = this._hiddenNodeIds();
        this._meshIndexByNodeId.forEach((idx, nid) => {
            const inIsolate = !isolated || isolated.has(nid);
            const inHide    = hidden && hidden.has(nid);
            if (!inIsolate || inHide) {
                this._viewer.setMeshVisible(idx, false);
                return;
            }
            this._viewer.setMeshVisible(idx, true);
            let color, opacity;
            if (this._groupMode && this._groupSelection.has(nid)) {
                color = GROUP_SELECT;
                opacity = 1.0;
            } else {
                color = this._classificationColorFor(nid);
                opacity = 1.0;
            }
            this._viewer.setMeshColor(idx, color, opacity);
        });
    }

    _isolatedNodeIds() {
        if (!this._isolatedGroupId) return null;
        const g = this._groupById(this._isolatedGroupId);
        if (!g) return null;
        const out = this._groupNodeIdsRecursive(this._isolatedGroupId);
        return out.size > 0 ? out : null;
    }

    /** Union of node_ids belonging to any group currently flagged Hidden (including descendants). */
    _hiddenNodeIds() {
        if (this._hiddenGroupIds.size === 0) return null;
        const out = new Set();
        for (const gid of this._hiddenGroupIds) {
            for (const nid of this._groupNodeIdsRecursive(gid)) out.add(nid);
        }
        return out.size > 0 ? out : null;
    }

    /** Direct child groups of ``parentId`` (or all top-level when parentId is null). */
    _groupChildrenOf(parentId) {
        const out = [];
        for (const g of this.groups.values()) {
            if ((g.parent_id || null) === parentId) out.push(g);
        }
        return out;
    }

    /** Return the set of node_ids in ``groupId`` and all descendant groups. */
    _groupNodeIdsRecursive(groupId) {
        const out = new Set();
        const visit = (gid) => {
            const g = this._groupById(gid);
            if (!g) return;
            for (const nid of g.node_ids) out.add(nid);
            for (const child of this._groupChildrenOf(gid)) visit(child.id);
        };
        visit(groupId);
        return out;
    }

    /**
     * Resolve a group id to its record — handles both user groups (stored in
     * ``this.groups``) and the special virtual Unclassified group (computed
     * on the fly).  Callers receive the same shape either way.
     */
    _groupById(groupId) {
        if (groupId === '__unclassified__') return this._virtualUnclassifiedGroup();
        return this.groups.get(groupId) || null;
    }

    /**
     * Build a virtual group listing every native-BOM part with no resolved
     * classification.  The group is never persisted and cannot be edited
     * (rename/delete/parent/remove are blocked in the panel).  Returns null
     * when there's no BOM data yet — in that case the UI silently omits it.
     */
    _virtualUnclassifiedGroup() {
        if (!Array.isArray(this._nativeBom) || this._nativeBom.length === 0) return null;
        const nodeIds = [];
        for (const row of this._nativeBom) {
            const nid = row.node_id;
            if (!nid) continue;
            if (this._resolveClassification(nid) == null) nodeIds.push(nid);
        }
        return {
            id: '__unclassified__',
            name: 'Unclassified',
            parent_id: null,
            node_ids: nodeIds,
            _virtual: true,
        };
    }

    /** Collect ids of ``groupId`` and all its descendants — used for cycle prevention. */
    _groupAndDescendantIds(groupId) {
        const out = new Set();
        const visit = (gid) => {
            if (out.has(gid)) return;
            out.add(gid);
            for (const child of this._groupChildrenOf(gid)) visit(child.id);
        };
        visit(groupId);
        return out;
    }

    // Backwards-compatible alias — some earlier call sites still use this name.
    _applyClassificationColors() { this._paintAllMeshes(); }

    _classificationColorFor(nodeId) {
        const resolved = this._resolveClassification(nodeId);
        if (!resolved) return 0xaaaaaa;               // Unclassified
        if (resolved.mixed)                return 0xd97706;  // Mixed — amber
        if (resolved.action === 'postprocess') return 0x2563eb;  // CNC — blue
        if (resolved.action === 'bought-out')  return 0x16a34a;  // BO — green
        if (resolved.action === 'exclude')     return 0x7c3aed;  // EXC — purple
        return 0xaaaaaa;
    }

    /**
     * Repaint the assembly view to reflect the current classification state.
     * No-op unless full assembly is active. Keeps any existing selection
     * highlight by routing through _highlightSelectedInAssembly.
     */
    _refreshAssemblyColors() {
        if (!this._showFullAssembly) return;
        if (this._selectedNodeId && this._meshIndexByNodeId.has(this._selectedNodeId)) {
            this._highlightSelectedInAssembly(this._selectedNodeId);
        } else {
            this._applyClassificationColors();
        }
    }

    _showViewerPlaceholder() {
        const panel = this.container.querySelector('#stl-viewer-panel');
        if (!panel) return;
        if (this._viewer) {
            this._viewer.dispose();
            this._viewer = null;
        }
        panel.innerHTML = '<div class="stl-viewer-placeholder">Click a node in the tree to preview its 3D model</div>';
        this._meshIndexByNodeId.clear();
    }

    /**
     * Show a floating classification menu at the given viewport coordinates.
     * Reuses the tree's _classifyNode / _unclassifyNode so peer-propagation,
     * progress counter, and project-state save all behave identically to
     * clicking the buttons on the tree row.
     */
    _showMeshContextMenu(nodeId, x, y) {
        this._dismissMeshContextMenu();

        const treeEl = this.container.querySelector('#assembly-tree-container');
        const li = treeEl?.querySelector(`.tree-node[data-node-id="${CSS.escape(nodeId)}"]`);
        if (!li) return;

        const current = this.classifications.get(nodeId) || null;
        const label = li.dataset.nodeName || nodeId;

        const menu = document.createElement('div');
        menu.className = 'viewer-context-menu';
        menu.innerHTML = `
            <div class="vcm-header" title="${this._esc(label)}">${this._esc(label)}</div>
            <button type="button" data-action="postprocess" class="${current === 'postprocess' ? 'vcm-active' : ''}">CNC</button>
            <button type="button" data-action="bought-out" class="${current === 'bought-out' ? 'vcm-active' : ''}">Bought Out</button>
            <button type="button" data-action="exclude"    class="${current === 'exclude'    ? 'vcm-active' : ''}">Exclude</button>
            ${current ? '<button type="button" data-action="unclassify" class="vcm-clear">Clear classification</button>' : ''}
        `;

        menu.style.position = 'fixed';
        menu.style.left = `${x}px`;
        menu.style.top  = `${y}px`;
        document.body.appendChild(menu);

        // Nudge onto the page if the cursor was near the right/bottom edge
        const rect = menu.getBoundingClientRect();
        if (rect.right  > window.innerWidth)  menu.style.left = `${Math.max(0, window.innerWidth  - rect.width  - 4)}px`;
        if (rect.bottom > window.innerHeight) menu.style.top  = `${Math.max(0, window.innerHeight - rect.height - 4)}px`;

        menu.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-action]');
            if (!btn) return;
            const action = btn.dataset.action;
            if (action === 'unclassify') {
                this._unclassifyNode(li, nodeId);
            } else {
                this._classifyNode(li, nodeId, action);
            }
            this._dismissMeshContextMenu();
        });

        // Dismiss on any outside pointerdown / Escape. Use capture so OrbitControls
        // can't swallow the event first.
        const outside = (e) => {
            if (!menu.contains(e.target)) this._dismissMeshContextMenu();
        };
        const esc = (e) => { if (e.key === 'Escape') this._dismissMeshContextMenu(); };
        // Defer attachment a tick so the originating contextmenu click doesn't
        // immediately dismiss the freshly opened menu.
        setTimeout(() => {
            document.addEventListener('pointerdown', outside, { capture: true });
            document.addEventListener('keydown', esc);
        }, 0);

        this._meshContextMenu = { menu, outside, esc };
    }

    _dismissMeshContextMenu() {
        const ctx = this._meshContextMenu;
        if (!ctx) return;
        ctx.menu.remove();
        document.removeEventListener('pointerdown', ctx.outside, { capture: true });
        document.removeEventListener('keydown', ctx.esc);
        this._meshContextMenu = null;
    }

    /**
     * Right-click context menu shown *only in Group Mode*.  Tells the user
     * which group the clicked part belongs to (and its breadcrumb path), and
     * offers Remove-from-group plus a "select every visible instance sharing
     * this prototype" action.  Classification stays in the non-group-mode
     * menu.
     */
    _showGroupContextMenu(nodeId, x, y) {
        this._dismissMeshContextMenu();

        const label = this._nodeLabelFor(nodeId) || nodeId;
        const groupEntry = this._groupContainingNodeId(nodeId);
        const siblingCount = this._countVisibleSiblings(nodeId);

        const menu = document.createElement('div');
        menu.className = 'viewer-context-menu';
        menu.innerHTML = `
            <div class="vcm-header" title="${this._esc(label)}">${this._esc(label)}</div>
            ${siblingCount > 1 ? `
                <button type="button" data-action="select-siblings">Select all visible instances (${siblingCount})</button>
            ` : `
                <div class="vcm-subheader" style="font-style:italic;color:#64748b;">No other visible instances</div>
            `}
            ${groupEntry ? `
                <div class="vcm-subheader" title="${this._esc(groupEntry.groupPath)}">In group: ${this._esc(groupEntry.groupPath)}</div>
                <button type="button" data-action="remove-from-group"
                        data-group-id="${this._esc(groupEntry.group.id)}"
                        class="vcm-clear">Remove from group</button>
            ` : `
                <div class="vcm-subheader" style="font-style:italic;color:#64748b;">Not in any group</div>
            `}
            <button type="button" data-action="cancel">Cancel</button>
        `;

        menu.style.position = 'fixed';
        menu.style.left = `${x}px`;
        menu.style.top  = `${y}px`;
        document.body.appendChild(menu);

        const rect = menu.getBoundingClientRect();
        if (rect.right  > window.innerWidth)  menu.style.left = `${Math.max(0, window.innerWidth  - rect.width  - 4)}px`;
        if (rect.bottom > window.innerHeight) menu.style.top  = `${Math.max(0, window.innerHeight - rect.height - 4)}px`;

        menu.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-action]');
            if (!btn) return;
            const action = btn.dataset.action;
            if (action === 'remove-from-group') {
                this._removeNodeFromGroup(btn.dataset.groupId, nodeId);
            } else if (action === 'select-siblings') {
                this._selectAllVisibleSiblings(nodeId);
            }
            this._dismissMeshContextMenu();
        });

        const outside = (e) => { if (!menu.contains(e.target)) this._dismissMeshContextMenu(); };
        const esc = (e) => { if (e.key === 'Escape') this._dismissMeshContextMenu(); };
        setTimeout(() => {
            document.addEventListener('pointerdown', outside, { capture: true });
            document.addEventListener('keydown', esc);
        }, 0);
        this._meshContextMenu = { menu, outside, esc };
    }

    /**
     * Return a key that uniquely identifies ``nodeId``'s prototype, so any two
     * meshes sharing it represent the same physical part template.
     *   STEP: the XCAF ref_id (shared across instances); per-solid rows also
     *         key on the solid index so only the same solid-of-prototype matches.
     *   IFC:  IfcTypeProduct GlobalId (from ifc_metadata.type_guid) in the BOM.
     * Falls back to the node_id itself when no prototype info is resolvable.
     */
    _prototypeKeyFor(nodeId) {
        if (!nodeId) return null;
        if (this._source === 'step') {
            // Direct leaf: its own ref_id
            if (this._nodeRefMap?.has(nodeId)) {
                return `step:${this._nodeRefMap.get(nodeId)}`;
            }
            // Per-solid split row: "<instance>:s<N>" — prototype key is the
            // instance's ref_id plus the solid index.
            const m = nodeId.match(/^(.*):s(\d+)$/);
            if (m && this._nodeRefMap?.has(m[1])) {
                return `step:${this._nodeRefMap.get(m[1])}:s${m[2]}`;
            }
        }
        if (this._source === 'ifc' && Array.isArray(this._nativeBom)) {
            const row = this._nativeBom.find(r => r.node_id === nodeId);
            if (row?.type_guid) return `ifc:${row.type_guid}`;
        }
        return `solo:${nodeId}`;
    }

    /** True if the mesh for ``nodeId`` is currently rendered in the viewer. */
    _isMeshCurrentlyVisible(nodeId) {
        const isolated = this._isolatedNodeIds();
        const hidden   = this._hiddenNodeIds();
        const inIsolate = !isolated || isolated.has(nodeId);
        const inHide    = hidden && hidden.has(nodeId);
        return inIsolate && !inHide;
    }

    /** Count visible meshes that share the given node's prototype key. */
    _countVisibleSiblings(nodeId) {
        const key = this._prototypeKeyFor(nodeId);
        let count = 0;
        for (const nid of this._meshIndexByNodeId.keys()) {
            if (this._prototypeKeyFor(nid) !== key) continue;
            if (!this._isMeshCurrentlyVisible(nid)) continue;
            count += 1;
        }
        return count;
    }

    /** Add every visible mesh sharing the prototype to the group selection. */
    _selectAllVisibleSiblings(nodeId) {
        const key = this._prototypeKeyFor(nodeId);
        if (!key) return;
        for (const nid of this._meshIndexByNodeId.keys()) {
            if (this._prototypeKeyFor(nid) !== key) continue;
            if (!this._isMeshCurrentlyVisible(nid)) continue;
            this._groupSelection.add(nid);
        }
        this._updateGroupToolbar();
        this._refreshAssemblyColors();
    }

    // ---------------------------------------------------------------
    // Custom groups — multi-select in the viewer, persisted in project_state
    // ---------------------------------------------------------------

    _toggleGroupMode() {
        this._groupMode = !this._groupMode;
        if (!this._groupMode) {
            this._groupSelection.clear();
        }
        this._updateGroupToolbar();
        this._refreshAssemblyColors();
    }

    _toggleMeshInGroupSelection(nodeId) {
        if (!nodeId) return;
        if (this._groupSelection.has(nodeId)) {
            this._groupSelection.delete(nodeId);
        } else {
            this._groupSelection.add(nodeId);
        }
        this._updateGroupToolbar();
        this._refreshAssemblyColors();
    }

    _clearGroupSelection() {
        this._groupSelection.clear();
        this._updateGroupToolbar();
        this._refreshAssemblyColors();
    }

    _updateGroupToolbar() {
        const modeBtn   = this.container.querySelector('#group-mode-btn');
        const createBtn = this.container.querySelector('#group-create-btn');
        const clearBtn  = this.container.querySelector('#group-clear-btn');
        const counter   = this.container.querySelector('#group-selection-counter');
        const groupsBtn = this.container.querySelector('#viewer-groups-btn');
        const groupsPanel = this.container.querySelector('#groups-panel');

        const showModeBtn = !!this._showFullAssembly;
        const count = this._groupSelection.size;

        if (modeBtn) {
            modeBtn.hidden = !showModeBtn;
            modeBtn.textContent = this._groupMode ? 'Exit Group Mode' : 'Group Mode';
            modeBtn.classList.toggle('viewer-active-toggle', this._groupMode);
        }
        if (createBtn) {
            createBtn.hidden = !(this._groupMode && count > 0);
        }
        if (clearBtn) {
            clearBtn.hidden = !(this._groupMode && count > 0);
        }
        const addBtn = this.container.querySelector('#group-add-btn');
        if (addBtn) {
            addBtn.hidden = !(this._groupMode && count > 0 && this.groups.size > 0);
        }
        if (counter) {
            counter.hidden = !(this._groupMode && count > 0);
            counter.textContent = count > 0 ? `${count} selected` : '';
        }
        if (groupsBtn) {
            groupsBtn.hidden = !showModeBtn;
            const open = groupsPanel && !groupsPanel.hidden;
            groupsBtn.textContent = open ? 'Hide Groups' : 'Groups';
            groupsBtn.classList.toggle('viewer-active-toggle', open);
        }
    }

    _promptCreateGroup() {
        const count = this._groupSelection.size;
        if (count === 0) return;
        const defaultName = `Group ${this.groups.size + 1}`;
        const name = (prompt(`Name this group of ${count} part${count === 1 ? '' : 's'}:`, defaultName) || '').trim();
        if (!name) return;

        // Remove these node_ids from any existing group (uniqueness: one group per node_id).
        const selected = new Set(this._groupSelection);
        for (const g of this.groups.values()) {
            g.node_ids = g.node_ids.filter(nid => !selected.has(nid));
        }
        // Drop groups that both became empty AND have no child groups.  Parent
        // containers with zero direct members must survive; otherwise their
        // children get orphaned.
        for (const [gid, g] of [...this.groups]) {
            if (g.node_ids.length === 0 && this._groupChildrenOf(gid).length === 0) {
                this.groups.delete(gid);
            }
        }

        const id = `grp_${Math.random().toString(16).slice(2, 10)}`;
        const group = {
            id,
            name,
            parent_id: null,
            node_ids: [...selected],
            created_at: new Date().toISOString(),
        };
        this.groups.set(id, group);

        this._groupSelection.clear();
        this._updateGroupToolbar();
        this._refreshAssemblyColors();
        this._debouncedSave();
        this._renderGroupsPanelIfOpen();
    }

    /**
     * Open a floating picker listing existing groups — clicking one adds the
     * current group selection to that group (removing the same node_ids from
     * any other group to keep each part in a single group).
     */
    _showAddToGroupPicker(anchorEl) {
        this._dismissAddToGroupPicker();
        if (this._groupSelection.size === 0 || this.groups.size === 0) return;

        const rect = anchorEl.getBoundingClientRect();
        const menu = document.createElement('div');
        menu.className = 'viewer-context-menu';
        menu.innerHTML = `
            <div class="vcm-header">Add ${this._groupSelection.size} to group…</div>
            ${[...this.groups.values()].filter(g => !g._virtual).map(g => `
                <button type="button" data-group-id="${this._esc(g.id)}">
                    ${this._esc(g.name)} <small>(${g.node_ids.length})</small>
                </button>
            `).join('')}
        `;

        menu.style.position = 'fixed';
        menu.style.left = `${rect.left}px`;
        menu.style.top  = `${rect.bottom + 4}px`;
        document.body.appendChild(menu);

        const bounds = menu.getBoundingClientRect();
        if (bounds.right  > window.innerWidth)  menu.style.left = `${Math.max(0, window.innerWidth  - bounds.width  - 4)}px`;
        if (bounds.bottom > window.innerHeight) menu.style.top  = `${Math.max(0, rect.top - bounds.height - 4)}px`;

        menu.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-group-id]');
            if (!btn) return;
            this._addSelectionToGroup(btn.dataset.groupId);
            this._dismissAddToGroupPicker();
        });

        const outside = (e) => { if (!menu.contains(e.target) && e.target !== anchorEl) this._dismissAddToGroupPicker(); };
        const esc = (e) => { if (e.key === 'Escape') this._dismissAddToGroupPicker(); };
        setTimeout(() => {
            document.addEventListener('pointerdown', outside, { capture: true });
            document.addEventListener('keydown', esc);
        }, 0);
        this._addToGroupPicker = { menu, outside, esc };
    }

    _dismissAddToGroupPicker() {
        const ctx = this._addToGroupPicker;
        if (!ctx) return;
        ctx.menu.remove();
        document.removeEventListener('pointerdown', ctx.outside, { capture: true });
        document.removeEventListener('keydown', ctx.esc);
        this._addToGroupPicker = null;
    }

    _addSelectionToGroup(groupId) {
        const target = this.groups.get(groupId);
        if (!target || this._groupSelection.size === 0) return;
        const incoming = new Set(this._groupSelection);

        // Uniqueness: strip these node_ids from any other group first.
        for (const g of this.groups.values()) {
            if (g.id === groupId) continue;
            g.node_ids = g.node_ids.filter(nid => !incoming.has(nid));
        }
        // Drop emptied groups — but only if they have no child groups, so
        // parent containers survive.
        for (const [gid, g] of [...this.groups]) {
            if (gid === groupId) continue;
            if (g.node_ids.length === 0 && this._groupChildrenOf(gid).length === 0) {
                this.groups.delete(gid);
            }
        }

        // Append only new members (preserve order, skip duplicates).
        const existing = new Set(target.node_ids);
        for (const nid of incoming) {
            if (!existing.has(nid)) target.node_ids.push(nid);
        }

        this._groupSelection.clear();
        this._updateGroupToolbar();
        this._refreshAssemblyColors();
        this._debouncedSave();
        this._renderGroupsPanelIfOpen();
    }

    _deleteGroup(groupId) {
        const g = this.groups.get(groupId);
        if (!g) return;
        if (!confirm(`Delete group "${g.name}"? Any child groups will be re-parented to its parent; parts remain assigned to their own groups.`)) return;
        // Re-parent children to this group's parent so the tree survives.
        const newParent = g.parent_id || null;
        for (const child of this._groupChildrenOf(groupId)) {
            child.parent_id = newParent;
        }
        this.groups.delete(groupId);
        if (this._isolatedGroupId === groupId) this._isolatedGroupId = null;
        this._hiddenGroupIds.delete(groupId);
        this._debouncedSave();
        this._renderGroupsPanelIfOpen();
        this._refreshAssemblyColors();
    }

    /**
     * Floating menu listing CNC / BO / EXC / Clear that applies to every part
     * inside ``groupId`` (and its descendants).
     */
    _showGroupClassifyPicker(groupId, anchorEl) {
        this._dismissGroupClassifyPicker();
        const g = this.groups.get(groupId);
        if (!g) return;
        const count = this._groupNodeIdsRecursive(groupId).size;

        const menu = document.createElement('div');
        menu.className = 'viewer-context-menu';
        menu.innerHTML = `
            <div class="vcm-header">Classify "${this._esc(g.name)}" (${count} part${count === 1 ? '' : 's'})</div>
            <button type="button" data-action="postprocess">CNC</button>
            <button type="button" data-action="bought-out">Bought Out</button>
            <button type="button" data-action="exclude">Exclude</button>
            <button type="button" data-action="unclassify" class="vcm-clear">Clear classification</button>
        `;

        const rect = anchorEl.getBoundingClientRect();
        menu.style.position = 'fixed';
        menu.style.left = `${rect.left}px`;
        menu.style.top  = `${rect.bottom + 4}px`;
        document.body.appendChild(menu);

        const bounds = menu.getBoundingClientRect();
        if (bounds.right  > window.innerWidth)  menu.style.left = `${Math.max(0, window.innerWidth  - bounds.width  - 4)}px`;
        if (bounds.bottom > window.innerHeight) menu.style.top  = `${Math.max(0, rect.top - bounds.height - 4)}px`;

        menu.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-action]');
            if (!btn) return;
            this._dismissGroupClassifyPicker();
            this._classifyGroup(groupId, btn.dataset.action);
        });

        const outside = (e) => { if (!menu.contains(e.target) && e.target !== anchorEl) this._dismissGroupClassifyPicker(); };
        const esc = (e) => { if (e.key === 'Escape') this._dismissGroupClassifyPicker(); };
        setTimeout(() => {
            document.addEventListener('pointerdown', outside, { capture: true });
            document.addEventListener('keydown', esc);
        }, 0);
        this._groupClassifyPicker = { menu, outside, esc };
    }

    _dismissGroupClassifyPicker() {
        const ctx = this._groupClassifyPicker;
        if (!ctx) return;
        ctx.menu.remove();
        document.removeEventListener('pointerdown', ctx.outside, { capture: true });
        document.removeEventListener('keydown', ctx.esc);
        this._groupClassifyPicker = null;
    }

    /**
     * Open a floating picker listing valid parent candidates for ``groupId``.
     * Excludes the group itself and all its descendants to prevent cycles.
     */
    _showSetParentPicker(groupId, anchorEl) {
        this._dismissSetParentPicker();
        const g = this.groups.get(groupId);
        if (!g) return;
        const forbidden = this._groupAndDescendantIds(groupId);
        const candidates = [...this.groups.values()].filter(c => !forbidden.has(c.id));

        const menu = document.createElement('div');
        menu.className = 'viewer-context-menu';
        menu.innerHTML = `
            <div class="vcm-header">Set parent of "${this._esc(g.name)}"</div>
            <button type="button" data-parent-id="__new__" class="vcm-new">
                &#x2795;&nbsp;New parent group…
            </button>
            <button type="button" data-parent-id="__none__" class="${g.parent_id ? '' : 'vcm-active'}">
                (No parent — top level)
            </button>
            ${candidates.map(c => `
                <button type="button" data-parent-id="${this._esc(c.id)}" class="${g.parent_id === c.id ? 'vcm-active' : ''}">
                    ${this._esc(c.name)}
                </button>
            `).join('')}
        `;

        const rect = anchorEl.getBoundingClientRect();
        menu.style.position = 'fixed';
        menu.style.left = `${rect.left}px`;
        menu.style.top  = `${rect.bottom + 4}px`;
        document.body.appendChild(menu);

        const bounds = menu.getBoundingClientRect();
        if (bounds.right  > window.innerWidth)  menu.style.left = `${Math.max(0, window.innerWidth  - bounds.width  - 4)}px`;
        if (bounds.bottom > window.innerHeight) menu.style.top  = `${Math.max(0, rect.top - bounds.height - 4)}px`;

        menu.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-parent-id]');
            if (!btn) return;
            const choice = btn.dataset.parentId;
            if (choice === '__new__') {
                this._dismissSetParentPicker();
                this._createEmptyParentFor(groupId);
                return;
            }
            const pid = choice === '__none__' ? null : choice;
            this._setGroupParent(groupId, pid);
            this._dismissSetParentPicker();
        });

        const outside = (e) => { if (!menu.contains(e.target) && e.target !== anchorEl) this._dismissSetParentPicker(); };
        const esc = (e) => { if (e.key === 'Escape') this._dismissSetParentPicker(); };
        setTimeout(() => {
            document.addEventListener('pointerdown', outside, { capture: true });
            document.addEventListener('keydown', esc);
        }, 0);
        this._setParentPicker = { menu, outside, esc };
    }

    _dismissSetParentPicker() {
        const ctx = this._setParentPicker;
        if (!ctx) return;
        ctx.menu.remove();
        document.removeEventListener('pointerdown', ctx.outside, { capture: true });
        document.removeEventListener('keydown', ctx.esc);
        this._setParentPicker = null;
    }

    /**
     * Create a new (empty) parent group, inheriting the child's current parent,
     * and reassign the child under it.  Lets the user wrap one group in a new
     * parent in a single step.
     */
    _createEmptyParentFor(childId) {
        const child = this.groups.get(childId);
        if (!child) return;
        const defaultName = `Parent of ${child.name}`;
        const name = (prompt('Name the new parent group:', defaultName) || '').trim();
        if (!name) return;

        const id = `grp_${Math.random().toString(16).slice(2, 10)}`;
        const parent = {
            id,
            name,
            parent_id: child.parent_id || null,  // take the child's current parent, if any
            node_ids: [],
            created_at: new Date().toISOString(),
        };
        this.groups.set(id, parent);
        child.parent_id = id;

        this._debouncedSave();
        this._renderGroupsPanelIfOpen();
        this._refreshAssemblyColors();
    }

    _setGroupParent(groupId, parentId) {
        const g = this.groups.get(groupId);
        if (!g) return;
        if (parentId === groupId) return;  // defensive — UI already prevents this
        if (parentId && this._groupAndDescendantIds(groupId).has(parentId)) return;  // cycle guard
        g.parent_id = parentId || null;
        this._debouncedSave();
        this._renderGroupsPanelIfOpen();
        this._refreshAssemblyColors();
    }

    _toggleIsolateGroup(groupId) {
        this._isolatedGroupId = this._isolatedGroupId === groupId ? null : groupId;
        this._renderGroupsPanelIfOpen();
        this._refreshAssemblyColors();
    }

    _toggleHideGroup(groupId) {
        if (this._hiddenGroupIds.has(groupId)) {
            this._hiddenGroupIds.delete(groupId);
        } else {
            this._hiddenGroupIds.add(groupId);
        }
        this._renderGroupsPanelIfOpen();
        this._refreshAssemblyColors();
    }

    _renameGroup(groupId) {
        const g = this.groups.get(groupId);
        if (!g) return;
        const name = (prompt('Rename group:', g.name) || '').trim();
        if (!name || name === g.name) return;
        g.name = name;
        this._debouncedSave();
        this._renderGroupsPanelIfOpen();
    }

    _toggleViewerMaximized() {
        const panel = this.container?.querySelector('.workspace-viewer-panel');
        const btn   = this.container?.querySelector('#viewer-maximize-btn');
        if (!panel) return;
        const max = panel.classList.toggle('viewer-maximized');
        document.body.classList.toggle('viewer-maximized-active', max);
        if (btn) {
            btn.innerHTML = max ? '&#x26F7;' : '&#x26F6;';
            btn.title = max ? 'Restore viewer (Esc)' : 'Maximise viewer (Esc to restore)';
        }
        // Nudge the Three.js ResizeObserver — some browsers take a frame to
        // report the new bounding rect after a class toggle.
        requestAnimationFrame(() => {
            if (this._viewer && typeof this._viewer._onResize === 'function') {
                this._viewer._onResize();
            }
        });
    }

    _toggleGroupsPanel() {
        const panel = this.container.querySelector('#groups-panel');
        const btn = this.container.querySelector('#show-groups-btn');
        if (!panel) return;
        if (!panel.hidden) {
            panel.hidden = true;
            if (btn) btn.textContent = 'Groups';
            this._updateGroupToolbar();
            return;
        }
        this._renderGroupsPanel();
        if (btn) btn.textContent = 'Hide Groups';
        this._updateGroupToolbar();
    }

    _renderGroupsPanelIfOpen() {
        const panel = this.container?.querySelector('#groups-panel');
        if (panel && !panel.hidden) this._renderGroupsPanel();
    }

    _renderGroupsPanel() {
        const panel = this.container.querySelector('#groups-panel');
        if (!panel) return;
        const entries = [...this.groups.values()];

        const emptyMsg = `
            <div class="parts-list-empty-msg" style="padding:1rem;">
                No groups yet. Turn on <strong>Show Full Assembly</strong>, enable <strong>Group Mode</strong>,
                click parts in the viewer, then <strong>Create Group</strong>.
            </div>`;

        // Flatten the group tree depth-first so rows appear under their parents.
        const ordered = [];
        // Virtual Unclassified group pinned at the top when BOM data exists.
        const virt = this._virtualUnclassifiedGroup();
        if (virt && virt.node_ids.length > 0) {
            ordered.push({ g: virt, depth: 0 });
        }
        const walk = (parentId, depth) => {
            const kids = this._groupChildrenOf(parentId)
                .sort((a, b) => a.name.localeCompare(b.name));
            for (const g of kids) {
                ordered.push({ g, depth });
                walk(g.id, depth + 1);
            }
        };
        walk(null, 0);

        const rows = ordered.flatMap(({ g, depth }) => {
            const isolated = this._isolatedGroupId === g.id;
            const hidden   = this._hiddenGroupIds.has(g.id);
            const expanded = this._expandedGroups.has(g.id);
            const virtual  = !!g._virtual;
            const rowClass = [
                isolated ? 'group-isolated' : '',
                hidden   ? 'group-hidden'   : '',
                virtual  ? 'group-virtual'  : '',
            ].filter(Boolean).join(' ');
            const indent = depth > 0 ? `<span class="group-indent" style="--depth:${depth}"></span>` : '';
            const descCount = this._groupNodeIdsRecursive(g.id).size;
            const countCell = descCount > g.node_ids.length
                ? `${g.node_ids.length} <small class="group-desc-count">(+${descCount - g.node_ids.length})</small>`
                : `${g.node_ids.length}`;
            const expandBtn = g.node_ids.length > 0
                ? `<button class="outline group-expand-btn" title="Show / hide members">${expanded ? '▾' : '▸'}</button>`
                : '<span class="group-expand-placeholder"></span>';

            const out = [`<tr data-group-id="${this._esc(g.id)}" class="${rowClass}">
                <td>${indent}${expandBtn} ${this._esc(g.name)}${virtual ? ' <small class="group-virtual-tag">auto</small>' : ''}</td>
                <td class="parts-list-qty">${countCell}</td>
                <td class="group-actions">
                    <button class="outline group-classify-btn" title="Classify all parts in this group">Classify…</button>
                    <button class="outline group-isolate-btn" title="Show only this group (and its children)">${isolated ? 'Clear Isolate' : 'Isolate'}</button>
                    <button class="outline group-hide-btn" title="Hide this group (and its children) from the viewer">${hidden ? 'Show' : 'Hide'}</button>
                    ${virtual ? '' : `
                        <button class="outline group-parent-btn" title="Set parent group">Parent…</button>
                        <button class="outline group-rename-btn">Rename</button>
                        <button class="outline group-delete-btn">Delete</button>
                    `}
                </td>
            </tr>`];

            if (expanded && g.node_ids.length > 0) {
                for (const nid of g.node_ids) {
                    const label = this._nodeLabelFor(nid);
                    const current = this._resolveClassification(nid);
                    const badge = current
                        ? this._classificationLabel(current.action, current.origin, current.mixed)
                        : '';
                    out.push(`<tr class="group-member-row" data-group-id="${this._esc(g.id)}" data-node-id="${this._esc(nid)}">
                        <td><span class="group-indent" style="--depth:${depth + 1}"></span>${this._esc(label)} ${badge}</td>
                        <td class="parts-list-qty"></td>
                        <td class="group-actions">
                            <button class="outline group-member-cnc-btn ${current?.action === 'postprocess' ? 'cls-active-cnc' : ''}" title="Mark as CNC">CNC</button>
                            <button class="outline group-member-bo-btn  ${current?.action === 'bought-out'  ? 'cls-active-bo'  : ''}" title="Mark as Bought Out">BO</button>
                            <button class="outline group-member-exc-btn ${current?.action === 'exclude'     ? 'cls-active-exc' : ''}" title="Mark as Excluded">EXC</button>
                            ${current ? '<button class="outline group-member-clear-btn" title="Clear classification">&#x2715;</button>' : ''}
                            <button class="outline group-member-locate-btn" title="Select this part in the tree">Locate</button>
                            ${virtual ? '' : '<button class="outline group-member-remove-btn" title="Remove this part from the group">&#x2013;</button>'}
                        </td>
                    </tr>`);
                }
            }
            return out;
        }).join('');

        panel.innerHTML = `
            <div class="parts-list-card">
                <div class="parts-list-header">
                    <span>Groups${entries.length > 0 ? ' &middot; ' + entries.length : ''}</span>
                    <div class="parts-list-header-actions">
                        <button class="outline parts-list-close groups-panel-close">&#x2715;</button>
                    </div>
                </div>
                <div class="parts-list-scroll">
                    ${entries.length === 0 ? emptyMsg : `
                        <table class="parts-list-table">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th class="parts-list-qty">Parts</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>${rows}</tbody>
                        </table>`}
                </div>
            </div>`;
        panel.hidden = false;

        panel.querySelector('.groups-panel-close')?.addEventListener('click', () => {
            panel.hidden = true;
            const btn = this.container.querySelector('#show-groups-btn');
            if (btn) btn.textContent = 'Groups';
        });

        panel.querySelector('tbody')?.addEventListener('click', (e) => {
            const tr = e.target.closest('tr[data-group-id]');
            if (!tr) return;
            const gid = tr.dataset.groupId;

            // Member-row actions
            if (tr.classList.contains('group-member-row')) {
                const nid = tr.dataset.nodeId;
                if (e.target.closest('.group-member-remove-btn')) this._removeNodeFromGroup(gid, nid);
                else if (e.target.closest('.group-member-locate-btn'))
                    this._selectAndScrollToNode(this._bomNodeIdToTreeNodeId(nid));
                else if (e.target.closest('.group-member-cnc-btn'))   this._applyClassification(nid, 'postprocess');
                else if (e.target.closest('.group-member-bo-btn'))    this._applyClassification(nid, 'bought-out');
                else if (e.target.closest('.group-member-exc-btn'))   this._applyClassification(nid, 'exclude');
                else if (e.target.closest('.group-member-clear-btn')) this._applyClassification(nid, 'unclassify');
                return;
            }

            // Group-row actions
            if (e.target.closest('.group-expand-btn')) this._toggleGroupExpanded(gid);
            else if (e.target.closest('.group-classify-btn'))
                this._showGroupClassifyPicker(gid, e.target.closest('.group-classify-btn'));
            else if (e.target.closest('.group-isolate-btn')) this._toggleIsolateGroup(gid);
            else if (e.target.closest('.group-hide-btn')) this._toggleHideGroup(gid);
            else if (e.target.closest('.group-parent-btn')) this._showSetParentPicker(gid, e.target.closest('.group-parent-btn'));
            else if (e.target.closest('.group-rename-btn')) this._renameGroup(gid);
            else if (e.target.closest('.group-delete-btn')) this._deleteGroup(gid);
        });
    }

    _toggleGroupExpanded(groupId) {
        if (this._expandedGroups.has(groupId)) this._expandedGroups.delete(groupId);
        else this._expandedGroups.add(groupId);
        this._renderGroupsPanelIfOpen();
    }

    /**
     * Apply a classification action to a single node_id.  Prefers the existing
     * tree-driven path (`_classifyNode` / `_unclassifyNode`) so peer
     * propagation and DOM highlighting behave identically to clicking a
     * tree-row button.  For synthetic ids (e.g. "...:s0") with no tree <li>
     * we fall back to the minimal state update.
     */
    _applyClassification(nodeId, action) {
        const treeEl = this.container.querySelector('#assembly-tree-container');
        const li = treeEl?.querySelector(`.tree-node[data-node-id="${CSS.escape(nodeId)}"]`);
        if (li) {
            if (action === 'unclassify') this._unclassifyNode(li, nodeId);
            else this._classifyNode(li, nodeId, action);
            return;
        }
        // No tree element — minimal path, still keeps BOM/viewer/progress in sync.
        if (action === 'unclassify') this.classifications.delete(nodeId);
        else this.classifications.set(nodeId, action);
        this._updateProgress();
        this._debouncedSave();
        this._refreshNativeBomIfOpen();
        this._refreshAssemblyColors();
    }

    /**
     * Apply a classification to every part in a group and its descendants.
     * Always asks for confirmation since one click can affect many parts.
     */
    _classifyGroup(groupId, action) {
        const g = this.groups.get(groupId);
        if (!g) return;
        const nodes = [...this._groupNodeIdsRecursive(groupId)];
        if (nodes.length === 0) return;
        const label = action === 'postprocess' ? 'CNC'
                    : action === 'bought-out' ? 'Bought Out'
                    : action === 'exclude'    ? 'Exclude'
                    : 'Clear classification on';
        if (!confirm(`${label} ${nodes.length} part${nodes.length === 1 ? '' : 's'} in "${g.name}"?`)) return;
        for (const nid of nodes) {
            this._applyClassification(nid, action);
        }
    }

    _removeNodeFromGroup(groupId, nodeId) {
        const g = this.groups.get(groupId);
        if (!g) return;
        const before = g.node_ids.length;
        g.node_ids = g.node_ids.filter(nid => nid !== nodeId);
        if (g.node_ids.length === before) return;  // wasn't in group

        // Also drop it from any pending group-mode selection so the viewer's
        // yellow "selected" highlight disappears when the action completes.
        if (this._groupSelection.has(nodeId)) {
            this._groupSelection.delete(nodeId);
            this._updateGroupToolbar();
        }

        this._debouncedSave();
        this._renderGroupsPanelIfOpen();
        this._refreshAssemblyColors();
    }

    /**
     * Human label for a node_id — prefers the tree's node name, falls back to
     * the native_bom row's mark, and finally the raw id.  Handles both direct
     * tree nodes and synthetic ":s<N>" solid-child ids.
     */
    /**
     * Find the group that directly contains ``nodeId`` (uniqueness rule means
     * at most one), plus a breadcrumb path that includes any ancestor groups.
     * Returns null if the part isn't grouped.
     */
    _groupContainingNodeId(nodeId) {
        for (const g of this.groups.values()) {
            if (!g.node_ids.includes(nodeId)) continue;
            // Walk up parent_id to build "Grandparent → Parent → Group"
            const names = [];
            let cur = g;
            const guard = new Set();
            while (cur && !guard.has(cur.id)) {
                names.unshift(cur.name);
                guard.add(cur.id);
                cur = cur.parent_id ? this.groups.get(cur.parent_id) : null;
            }
            return { group: g, groupPath: names.join(' › ') };
        }
        return null;
    }

    _nodeLabelFor(nodeId) {
        if (!nodeId) return '';
        // Direct tree node
        const direct = this._parentMap?.get(nodeId);
        if (direct?.name) return direct.name;
        // Solid-split child — match against the BOM (which carries per-solid marks)
        if (Array.isArray(this._nativeBom)) {
            const row = this._nativeBom.find(r => r.node_id === nodeId);
            if (row?.mark) return row.mark;
        }
        // Fallback: translate instance-suffix to ref-suffix and try the tree DOM
        const treeId = this._bomNodeIdToTreeNodeId(nodeId);
        if (treeId !== nodeId) {
            const treeHit = this._parentMap?.get(treeId);
            if (treeHit?.name) return treeHit.name;
        }
        return nodeId;
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

        // Newly-revealed frontier: lightweight-classify the parts this explosion
        // exposed, then auto-apply (respects the frontier — won't descend into
        // nested unexploded assemblies).
        this._classifyFrontier();
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
        this._refreshNativeBomIfOpen();
        this._refreshAssemblyColors();

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
        this._refreshNativeBomIfOpen();
        this._refreshAssemblyColors();
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

        // Restore custom groups
        if (state.groups && typeof state.groups === 'object') {
            for (const [gid, g] of Object.entries(state.groups)) {
                if (!g || !Array.isArray(g.node_ids)) continue;
                this.groups.set(gid, {
                    id: g.id || gid,
                    name: g.name || gid,
                    parent_id: g.parent_id || null,
                    node_ids: g.node_ids.slice(),
                    created_at: g.created_at || null,
                });
            }
            // Rescue orphans: if parent_id points at a group that isn't
            // loaded (deleted at some point without re-parenting), promote
            // the child to top-level so it's visible in the panel again.
            let rescued = 0;
            for (const g of this.groups.values()) {
                if (g.parent_id && !this.groups.has(g.parent_id)) {
                    g.parent_id = null;
                    rescued += 1;
                }
            }
            if (rescued > 0) {
                // Persist the fix so we don't need to rescue every load.
                this._debouncedSave();
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
            groups: Object.fromEntries(this.groups),
        };

        try {
            await this.api.saveProjectState(filename, state);
        } catch (err) {
            console.error('Failed to save project state:', err);
        }
    }

    /**
     * Auto-apply classifications at the CURRENT explosion frontier (rw7.1).
     *
     * This is a FRONTIER assistant, not a tree-flattener: it respects the
     * user's top-down explode/categorise workflow and NEVER auto-descends.
     * Specifically the walk stops at:
     *   • assemblies the user has not exploded — they stay a single frontier
     *     item (this is what prevents classifying e.g. floor grating piece by
     *     piece; the user marks the grating assembly BO without exploding), and
     *   • any node already classified bought-out / exclude (out of scope).
     * For each EXPOSED, not-yet-classified part with a confident refined_class
     * (>=0.5) it applies the mapped action; low-confidence / MIXED parts are
     * left unset (flagged via the ⚠ badge) for review. Never overwrites a
     * manual decision; idempotent. Re-run whenever the frontier changes
     * (explode) or new CNC results arrive.
     */
    _autoClassifyFromRefinedResults() {
        if (!this._cncAnalysisResults || !this._projectStateRestored || !this._treeData) return;
        const MAP = {
            section: 'postprocess', plate: 'postprocess',
            formed_plate: 'postprocess', bent_section: 'postprocess',
            bought_out: 'bought-out', exclude: 'exclude',
        };
        let changed = 0;
        const applyTo = (node) => {
            if (this.classifications.has(node.id)) return;
            const res = this._refinedForNode(node);
            if (!res) return;
            const action = MAP[(res.refined_class || '').toLowerCase()];  // MIXED -> undefined
            const conf = res.refined_confidence;
            if (action && conf != null && conf >= 0.5) {
                this.classifications.set(node.id, action);
                changed++;
            }
        };
        const walk = (nodes) => {
            for (const node of nodes) {
                const cls = this.classifications.get(node.id);
                if (node.node_type === 'assembly') {
                    // Respect the human triage: don't descend into bought-out /
                    // excluded assemblies, or ones not yet exploded.
                    if (cls === 'bought-out' || cls === 'exclude') continue;
                    if (this.explodedNodes.has(node.id) && node.children) walk(node.children);
                    continue;
                }
                if (node.node_type === 'part_multi_solid') {
                    // Exploded -> classify each constituent solid (per-solid
                    // class from the parent's solids[]); unexploded -> classify
                    // the whole part by its aggregate (skips MIXED).
                    if (this.explodedNodes.has(node.id) && node.children) walk(node.children);
                    else applyTo(node);
                    continue;
                }
                // Single-solid part or a revealed SOLID node.
                applyTo(node);
            }
        };
        walk(this._treeData);
        if (changed > 0) {
            console.info(`Auto-classified ${changed} frontier parts from refined_class`);
            this._updateProgress();
            this._debouncedSave();
            this._refreshAssemblyColors();
            this._renderPartsList(this._consolidationGroups);
        }
    }

    /**
     * Collect the EXPOSED frontier part refs that still need a refined_class:
     * exposed = not inside an unexploded or bought-out/excluded assembly; needs
     * = not user-classified and no refined_class yet. Returns Map(refId -> name).
     */
    /**
     * Resolve a tree node to its refined-class result. Multi-solid SOLID nodes
     * (synthetic id "<parentRef>:s<N>") resolve to the parent's solids[N] —
     * they are never classified by their synthetic ref (it is not an XCAF label).
     */
    _refinedForNode(node) {
        if (!this._cncAnalysisResults) return null;
        if (node.node_type === 'solid') {
            const m = String(node.id).match(/^(.*):s(\d+)$/);
            const parentRef = m ? m[1] : node.id;
            const idx = m ? parseInt(m[2], 10) : 0;
            const res = this._cncAnalysisResults[parentRef];
            if (res && Array.isArray(res.solids) && res.solids[idx]) return res.solids[idx];
            return null;
        }
        return this._cncAnalysisResults[node.ref_id || node.id] || null;
    }

    _collectFrontierRefs() {
        const refs = new Map();
        const walk = (nodes) => {
            for (const node of nodes) {
                const cls = this.classifications.get(node.id);
                if (node.node_type === 'assembly') {
                    if (cls === 'bought-out' || cls === 'exclude') continue;
                    if (this.explodedNodes.has(node.id) && node.children) walk(node.children);
                    continue;
                }
                // SOLID nodes are covered by their parent multi-solid classify
                // (which returns solids[]); never send their synthetic ref.
                if (node.node_type === 'solid') continue;
                const refId = node.ref_id || node.id;
                const res = this._cncAnalysisResults && this._cncAnalysisResults[refId];
                if (!cls && !(res && res.refined_class != null) && !refs.has(refId)) {
                    refs.set(refId, node.name || '');
                }
                if (node.children && this.explodedNodes.has(node.id)) walk(node.children);
            }
        };
        if (this._treeData) walk(this._treeData);
        return refs;
    }

    /**
     * Lightweight-classify the current frontier (no NC1/DXF), merge the
     * refined_class results, then auto-apply. This is the cheap pass that runs
     * AS YOU EXPLORE — ahead of any heavy CNC analysis.
     */
    async _classifyFrontier() {
        if (!this._currentFilename || !this._treeData || !this._projectStateRestored) return;
        const refs = this._collectFrontierRefs();
        this._classifyInFlight = this._classifyInFlight || new Set();
        const refIds = [...refs.keys()].filter(r => !this._classifyInFlight.has(r));
        if (refIds.length) {
            refIds.forEach(r => this._classifyInFlight.add(r));
            const memberIds = {};
            for (const r of refIds) memberIds[r] = refs.get(r);
            try {
                const resp = await this.api.classifyParts(
                    this._currentFilename, refIds, memberIds, this._lastSteelGrade || '');
                if (resp?.results) {
                    this._cncAnalysisResults = this._cncAnalysisResults || {};
                    for (const [r, res] of Object.entries(resp.results)) {
                        if (res && !this._cncAnalysisResults[r]) this._cncAnalysisResults[r] = res;
                    }
                }
            } catch (e) {
                console.warn('classifyFrontier failed:', e);
            } finally {
                refIds.forEach(r => this._classifyInFlight.delete(r));
            }
        }
        // Always apply — an explode may reveal solids whose classes come from a
        // parent that is already cached (no new POST needed).
        this._autoClassifyFromRefinedResults();
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

        let cnc = 0, bo = 0, exc = 0, unclassified = 0;
        const rows = Array.isArray(this._nativeBom) ? this._nativeBom : null;

        if (rows && rows.length) {
            // Count one entry per unique prototype ref_id (the BOM unit), using
            // the same resolver as the BOM so solid-keyed and inherited
            // classifications are honoured. A ref is "classified" if any of its
            // instances resolves; the unclassified tally is the completeness
            // signal (0 ⇒ every fabricable part is triaged).
            const refAction = new Map();  // ref → action | null (real action wins)
            for (const row of rows) {
                const nid = row.node_id;
                if (!nid) continue;
                const m = nid.match(/^(.*):s\d+$/);
                const base = m ? m[1] : nid;
                const ref = this._nodeRefMap?.get(base) || base;
                const resolved = this._resolveClassification(nid);
                const action = resolved ? (resolved.action || 'postprocess') : null;
                if (!refAction.has(ref)) refAction.set(ref, action);
                else if (action != null && refAction.get(ref) == null) refAction.set(ref, action);
            }
            for (const action of refAction.values()) {
                if (action === 'postprocess') cnc++;
                else if (action === 'bought-out') bo++;
                else if (action === 'exclude') exc++;
                else unclassified++;
            }
        } else {
            // No BOM yet — legacy per-ref count from classifications (the
            // unclassified denominator isn't known until the BOM exists).
            const refIdCl = new Map();
            for (const [nodeId, action] of this.classifications) {
                refIdCl.set(this._nodeRefMap.get(nodeId) || nodeId, action);
            }
            for (const action of refIdCl.values()) {
                if (action === 'postprocess') cnc++;
                else if (action === 'bought-out') bo++;
                else if (action === 'exclude') exc++;
            }
        }

        const parts = [];
        if (cnc > 0) parts.push(`<span class="prog-cnc">${cnc} CNC</span>`);
        if (bo > 0)  parts.push(`<span class="prog-bo">${bo} BO</span>`);
        if (exc > 0) parts.push(`<span class="prog-exc">${exc} EXC</span>`);
        if (unclassified > 0) {
            parts.push(`<span class="prog-unc">${unclassified} unclassified</span>`);
        } else if (rows && rows.length && (cnc + bo + exc) > 0) {
            parts.push(`<span class="prog-done">✓ complete</span>`);
        }
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
            this._refreshCncState(filename),
        ]).finally(() => {
            this._autoClassifyFromRefinedResults();
            this._renderPartsList(this._consolidationGroups);
        });
    }

    /**
     * Refresh the CNC freshness state from the backend.  Sets this._cncState
     * to ``{state, consolidated_at, analyzed_at, cnc_ref_count, stale_cnc_refs}``
     * or null on failure.  Safe to call anytime — non-blocking, never throws.
     */
    async _refreshCncState(filename) {
        if (!filename) return;
        try {
            const resp = await this.api.getCncState(filename);
            this._cncState = resp || null;
        } catch (e) {
            this._cncState = null;
        }
    }

    // ---------------------------------------------------------------
    // Unified native BOM view (data-first, from analysis.native_bom)
    // ---------------------------------------------------------------

    _toggleNativeBom() {
        const panel = this.container.querySelector('#native-bom-panel');
        const btn = this.container.querySelector('#show-native-bom-btn');
        if (!panel) return;
        if (!panel.hidden) {
            panel.hidden = true;
            if (btn) btn.textContent = 'Full BOM';
            return;
        }

        // Best-effort: load any cached CNC results so DXF/NC1 download icons
        // can appear on matching rows.  Non-blocking — if the fetch is slow
        // or the file has no CNC analysis yet, the BOM still renders cleanly.
        const filename = this._currentFilename;
        if (filename && this._cncAnalysisResults == null && !this._cncLoading) {
            this._cncLoading = true;
            this.api.getCncResult(filename)
                .then(resp => {
                    if (resp?.results) this._cncAnalysisResults = resp.results;
                    if (panel && !panel.hidden) this._renderNativeBom();
                })
                .catch(() => {})
                .finally(() => { this._cncLoading = false; });
        }

        this._renderNativeBom();
        if (btn) btn.textContent = 'Hide Full BOM';
    }

    _renderNativeBom() {
        const panel = this.container.querySelector('#native-bom-panel');
        if (!panel) return;
        const rows = this._nativeBom || [];

        if (rows.length === 0) {
            panel.innerHTML = `
                <div class="parts-list-card">
                    <div class="parts-list-header">
                        <span>Full BOM</span>
                        <button class="outline parts-list-close native-bom-close">&#x2715;</button>
                    </div>
                    <div class="parts-list-empty-msg" style="padding:1rem;">
                        No native BOM data available for this file. Re-parse to regenerate.
                    </div>
                </div>`;
            panel.hidden = false;
            panel.querySelector('.native-bom-close')?.addEventListener('click', () => {
                panel.hidden = true;
                const btn = this.container.querySelector('#show-native-bom-btn');
                if (btn) btn.textContent = 'Full BOM';
            });
            return;
        }

        // Detect which optional (IFC/Tekla-specific) columns have any data
        const hasPhase     = rows.some(r => r.phase != null && r.phase !== '');
        const hasPour      = rows.some(r => r.pour != null && r.pour !== '');
        const hasFinish    = rows.some(r => r.finish != null && r.finish !== '');
        const hasPartClass = rows.some(r => r.part_class != null && r.part_class !== '');

        const sourceLabel = rows[0]?.source === 'ifc' ? 'IFC' : 'STEP';
        const classifiedCount = rows.filter(r => this.classifications.has(r.node_id)).length;

        const hasGroups = this.groups.size > 0;
        const view = this._nativeBomView === 'consolidated' ? 'consolidated' : 'instance';

        let header, body, summaryExtras;
        if (view === 'consolidated') {
            const cons = this._consolidateNativeBom(rows);
            summaryExtras = ` &middot; ${cons.length} unique`;
            header = `
                <tr>
                    <th>Entity</th>
                    <th>Profile</th>
                    <th>Grade</th>
                    <th class="nb-num">Length&nbsp;(mm)</th>
                    <th class="nb-num">Qty</th>
                    <th class="nb-num">Total&nbsp;Length&nbsp;(mm)</th>
                    <th class="nb-num">Total&nbsp;Weight&nbsp;(kg)</th>
                    <th>Used&nbsp;In</th>
                    <th>Classification</th>
                    ${hasGroups ? '<th>Groups</th>' : ''}
                </tr>`;
            body = cons.map(c => this._nativeBomConsolidatedRow(c, { hasGroups })).join('');
        } else {
            summaryExtras = '';
            header = `
                <tr>
                    <th>Mark</th>
                    <th>Entity</th>
                    <th>Profile</th>
                    <th>Grade</th>
                    <th class="nb-num">Length&nbsp;(mm)</th>
                    <th class="nb-num">Weight&nbsp;(kg)</th>
                    <th>Assembly</th>
                    ${hasGroups    ? '<th>Group</th>'  : ''}
                    ${hasPhase     ? '<th>Phase</th>'  : ''}
                    ${hasPour      ? '<th>Pour</th>'   : ''}
                    ${hasFinish    ? '<th>Finish</th>' : ''}
                    ${hasPartClass ? '<th>Class</th>'  : ''}
                    <th>Classification</th>
                </tr>`;
            body = rows.map(r => this._nativeBomRow(r, {
                hasPhase, hasPour, hasFinish, hasPartClass, hasGroups,
            })).join('');
        }

        const nestableCount = this._nestableCountFromNativeBom();
        const nestBtn = nestableCount > 0
            ? (this._nestingRunning
                ? `<button class="outline native-bom-nest-btn" disabled>Nesting…</button>`
                : `<button class="outline native-bom-nest-btn" title="Nest all CNC-classified sections from this BOM">✂ Nest Sections (${nestableCount})</button>`)
            : '';
        // DXF / NC1 zip buttons only when there's a CNC analysis available.
        const hasCncResults = this._cncAnalysisResults && Object.keys(this._cncAnalysisResults).length > 0;
        const dxfZipBtn = hasCncResults
            ? `<button class="outline native-bom-dxf-zip-btn" title="Download every generated DXF as a zip">↓ DXF&nbsp;zip</button>` : '';
        const nc1ZipBtn = hasCncResults
            ? `<button class="outline native-bom-nc1-zip-btn" title="Download every generated NC1 as a zip">↓ NC1&nbsp;zip</button>` : '';
        const csvBtn = `<button class="outline native-bom-csv-btn" title="Download this BOM as CSV (matches the current view)">↓ CSV</button>`;
        const jsonBtn = `<button class="outline native-bom-json-btn" title="Download this BOM as JSON (matches the current view)">↓ JSON</button>`;
        const xlsxBtn = hasCncResults
            ? `<button class="outline native-bom-xlsx-btn" title="Download BOM workbook (.xlsx) with embedded thumbnails">↓ Excel</button>` : '';
        const consolidateBtn = this._consolidating
            ? `<button class="outline native-bom-consolidate-btn" disabled>Consolidating…</button>`
            : (this._consolidationGroups
                ? `<button class="outline native-bom-consolidate-btn" title="Re-run geometric consolidation">Re-consolidate</button>`
                : `<button class="outline native-bom-consolidate-btn" title="Group identical-geometry parts across the whole tree">Consolidate</button>`);

        // "Analyse CNC" is only relevant when there's at least one row marked
        // as CNC (postprocess).  Re-analyse appears once some analysis has
        // already been done; otherwise it says "Analyse".
        const cncCount = (rows || []).reduce((n, r) => {
            const c = this._resolveClassification(r.node_id);
            return n + (c?.action === 'postprocess' ? 1 : 0);
        }, 0);
        const hasAnyCncResult = hasCncResults;
        const analyseCncBtn = cncCount > 0
            ? (this._cncAnalysing
                ? `<button class="outline native-bom-cnc-analyse-btn" disabled>Analysing…</button>`
                : `<button class="outline native-bom-cnc-analyse-btn" title="Run CNC geometric analysis on every CNC-classified part">${hasAnyCncResult ? 'Re-analyse CNC' : 'Analyse CNC'} (${cncCount})</button>`)
            : '';

        const viewToggle = `
            <div class="nb-view-toggle" role="tablist" aria-label="BOM view">
                <button class="outline nb-view-btn ${view === 'instance' ? 'nb-view-active' : ''}" data-view="instance">Per-instance</button>
                <button class="outline nb-view-btn ${view === 'consolidated' ? 'nb-view-active' : ''}" data-view="consolidated">Consolidated</button>
            </div>`;

        panel.innerHTML = `
            <div class="parts-list-card">
                <div class="parts-list-header">
                    <span>Full BOM &middot; ${rows.length} parts &middot; ${sourceLabel} &middot; ${classifiedCount} classified${summaryExtras}</span>
                    <div class="parts-list-header-actions">
                        ${viewToggle}
                        ${analyseCncBtn}
                        ${consolidateBtn}
                        ${csvBtn}
                        ${jsonBtn}
                        ${xlsxBtn}
                        ${dxfZipBtn}
                        ${nc1ZipBtn}
                        ${nestBtn}
                        <button class="outline parts-list-close native-bom-close">&#x2715;</button>
                    </div>
                </div>
                <div class="parts-list-scroll">
                    <table class="parts-list-table native-bom-table">
                        <thead>${header}</thead>
                        <tbody>${body}</tbody>
                    </table>
                </div>
                <div id="native-bom-nesting-results" class="nesting-results-panel" ${this._nestingCuttingList ? '' : 'hidden'}></div>
            </div>`;
        panel.hidden = false;

        panel.querySelector('.native-bom-close')?.addEventListener('click', () => {
            panel.hidden = true;
            const btn = this.container.querySelector('#show-native-bom-btn');
            if (btn) btn.textContent = 'Full BOM';
        });

        panel.querySelector('.native-bom-nest-btn')?.addEventListener('click', () => {
            this._showNestingSettingsDialog(this._buildNestingItemsFromNativeBom());
        });
        panel.querySelector('.native-bom-csv-btn')?.addEventListener('click', () => {
            this._downloadNativeBomCsv();
        });
        panel.querySelector('.native-bom-dxf-zip-btn')?.addEventListener('click', () => {
            const f = this._currentFilename;
            if (f) window.location.href = `/api/v1/cnc-analysis/download-all/${encodeURIComponent(f)}/dxf`;
        });
        panel.querySelector('.native-bom-nc1-zip-btn')?.addEventListener('click', () => {
            const f = this._currentFilename;
            if (f) window.location.href = `/api/v1/cnc-analysis/download-all/${encodeURIComponent(f)}/nc1`;
        });
        panel.querySelector('.native-bom-xlsx-btn')?.addEventListener('click', () => {
            this._downloadBOMXlsx();  // reuses Standard BOM's existing implementation
        });
        panel.querySelector('.native-bom-json-btn')?.addEventListener('click', () => {
            this._downloadNativeBomJson();
        });
        panel.querySelector('.native-bom-consolidate-btn')?.addEventListener('click', () => {
            this._startConsolidation();
        });
        panel.querySelector('.native-bom-cnc-analyse-btn')?.addEventListener('click', () => {
            // On stale-cnc, force=true so the backend wipes the orphaned stale
            // results + NC files and re-runs; otherwise it 409s and the stale
            // banner never clears.
            const force = this._cncState?.state === 'stale-cnc';
            this._startCncAnalysisFromNativeBom(force);
        });

        panel.querySelectorAll('.nb-view-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const v = btn.dataset.view === 'consolidated' ? 'consolidated' : 'instance';
                if (this._nativeBomView === v) return;
                this._nativeBomView = v;
                this._renderNativeBom();
            });
        });

        panel.querySelector('tbody')?.addEventListener('click', (e) => {
            const tr = e.target.closest('tr[data-node-id]');
            if (!tr) return;
            this._selectAndScrollToNode(this._bomNodeIdToTreeNodeId(tr.dataset.nodeId));
        });
    }

    /**
     * Translate a native-BOM row node_id (instance-based, e.g. "0:1:1:1:1:s0")
     * into the tree DOM's actual node_id for that element. Exploded solid children
     * in the tree are keyed by the prototype ref_id ("<refId>:s0"), not the
     * instance path, so selection must hop through _nodeRefMap.
     */
    _bomNodeIdToTreeNodeId(nodeId) {
        if (!nodeId) return nodeId;
        const m = nodeId.match(/^(.*):s(\d+)$/);
        if (m) {
            const refId = this._nodeRefMap?.get(m[1]);
            if (refId) return `${refId}:s${m[2]}`;
        }
        return nodeId;
    }

    /**
     * Recover a recognisable mark for a Full-BOM row whose stored mark is a
     * generic SOLID/COMPOUND.  Looks up the node's real type + parent name from
     * the tree so the parent-name fallback matches the tree view and the Excel
     * export (and so empty no-solid artifacts keep their generic name).  Falls
     * back to the stored mark for solid-split rows not present in _parentMap.
     */
    _bomRowDisplayMark(row) {
        const info = this._parentMap?.get(row.node_id);
        if (info) {
            return this._displayNodeName(
                { name: row.mark, node_type: info.nodeType }, info.parentName);
        }
        return row.mark || '';
    }

    _nativeBomRow(row, cols) {
        const nodeId = row.node_id || '';
        const resolved = this._resolveClassification(nodeId);
        const displayedCls = resolved
            ? this._classificationLabel(resolved.action, resolved.origin, resolved.mixed)
            : '<span class="nb-pending">Pending</span>';

        const length = row.length != null ? this._fmtNumber(row.length, 1) : '';
        const weight = row.weight != null ? this._fmtNumber(row.weight, 2) : '';
        const grade = row.grade || row.material || '';

        // CNC badge + DXF/NC1 download link when a CNC result exists for the
        // row's ref_id.  Rendered inline next to the part mark to match the
        // Standard BOM layout.
        const cncHtml = this._nativeBomCncHtml(row, this._currentFilename);

        let groupCell = '';
        if (cols.hasGroups) {
            const entry = this._groupContainingNodeId(nodeId);
            groupCell = `<td>${entry ? `<span class="nb-group" title="${this._esc(entry.groupPath)}">${this._esc(entry.groupPath)}</span>` : ''}</td>`;
        }

        return `<tr data-node-id="${this._esc(nodeId)}">
            <td>${this._esc(this._bomRowDisplayMark(row))}${cncHtml ? ' ' + cncHtml : ''}</td>
            <td><span class="nb-entity">${this._esc(row.entity || '')}</span></td>
            <td>${this._esc(row.profile || '')}</td>
            <td>${this._esc(grade)}</td>
            <td class="nb-num">${length}</td>
            <td class="nb-num">${weight}</td>
            <td>${this._esc(row.assembly_mark || '')}</td>
            ${groupCell}
            ${cols.hasPhase     ? `<td>${this._esc(row.phase ?? '')}</td>` : ''}
            ${cols.hasPour      ? `<td>${this._esc(row.pour ?? '')}</td>` : ''}
            ${cols.hasFinish    ? `<td>${this._esc(row.finish ?? '')}</td>` : ''}
            ${cols.hasPartClass ? `<td>${this._esc(row.part_class ?? '')}</td>` : ''}
            <td>${displayedCls}</td>
        </tr>`;
    }

    _classificationLabel(action, origin, mixed) {
        if (mixed) return '<span class="nb-cls nb-cls-mixed">Mixed</span>';
        let badge;
        if (action === 'postprocess')       badge = '<span class="nb-cls nb-cls-cnc">CNC</span>';
        else if (action === 'bought-out')   badge = '<span class="nb-cls nb-cls-bo">BO</span>';
        else if (action === 'exclude')      badge = '<span class="nb-cls nb-cls-exc">EXC</span>';
        else                                badge = this._esc(action);
        if (origin === 'parent')   return badge + ' <small>(inherited)</small>';
        if (origin === 'children') return badge + ' <small>(from components)</small>';
        return badge;
    }

    /**
     * Resolve a BOM row's classification by looking at:
     *   1. Direct hit on the row's node_id
     *   2. Parent (strip ":s<N>" suffix — for solid-split rows inheriting from
     *      the multi-solid parent)
     *   3. Descendants (any classification keyed under "{nodeId}:..." — for
     *      placeholder rows where the user classified individual solid children)
     * Returns {action, origin, mixed} or null if no classification found.
     */
    _resolveClassification(nodeId) {
        if (!nodeId) return null;
        if (this.classifications.has(nodeId)) {
            return { action: this.classifications.get(nodeId), origin: 'direct', mixed: false };
        }

        // Per-solid BOM rows carry instance-based node_ids ("0:1:1:1:1:s0").
        // The tree's exploded solid children, however, are keyed by the multi-solid
        // prototype's ref_id ("<refId>:s0") — that's where the user's classification
        // actually lives. Translate before falling back to structural inheritance.
        const solidMatch = nodeId.match(/^(.*):s(\d+)$/);
        if (solidMatch) {
            const instanceParent = solidMatch[1];
            const solidIdx = solidMatch[2];
            const refId = this._nodeRefMap?.get(instanceParent);
            if (refId) {
                const refSolidId = `${refId}:s${solidIdx}`;
                if (this.classifications.has(refSolidId)) {
                    return { action: this.classifications.get(refSolidId), origin: 'direct', mixed: false };
                }
            }
            // Fallback: parent multi-solid classified at its instance node_id
            if (this.classifications.has(instanceParent)) {
                return { action: this.classifications.get(instanceParent), origin: 'parent', mixed: false };
            }
        }

        // Placeholder rows: any classification on descendants counts toward this row
        const childPrefix = nodeId + ':';
        const childActions = new Set();
        for (const [nid, action] of this.classifications) {
            if (nid.startsWith(childPrefix)) childActions.add(action);
        }
        // Also check ref_id-keyed solid children (when the row is a multi-solid placeholder)
        const rowRefId = this._nodeRefMap?.get(nodeId);
        if (rowRefId && rowRefId !== nodeId) {
            const refPrefix = rowRefId + ':';
            for (const [nid, action] of this.classifications) {
                if (nid.startsWith(refPrefix)) childActions.add(action);
            }
        }
        if (childActions.size === 1) {
            return { action: [...childActions][0], origin: 'children', mixed: false };
        }
        if (childActions.size > 1) {
            return { action: null, origin: 'children', mixed: true };
        }
        return null;
    }

    _fmtNumber(v, dp) {
        if (typeof v !== 'number') v = parseFloat(v);
        if (!Number.isFinite(v)) return '';
        return v.toFixed(dp).replace(/\.?0+$/, '');
    }

    /**
     * Group native_bom rows by (entity, profile, grade, rounded length,
     * classification) so identical parts roll up into one summary row.
     * Rounding length to the nearest mm absorbs tiny float noise.
     */
    _consolidateNativeBom(rows) {
        const buckets = new Map();
        for (const row of rows) {
            const resolved = this._resolveClassification(row.node_id);
            const cls = resolved?.mixed ? 'mixed'
                     : (resolved?.action || 'pending');
            const entity = row.entity || '';
            const profile = row.profile || '';
            const grade = row.grade || row.material || '';
            const length = row.length != null ? Math.round(row.length) : null;
            const key = [entity, profile, grade, length == null ? '' : length, cls].join('|');

            let b = buckets.get(key);
            if (!b) {
                b = {
                    key, entity, profile, grade, length,
                    classification: resolved,
                    qty: 0,
                    totalWeight: 0,
                    totalLength: 0,
                    hasLength: length != null,
                    hasWeight: false,
                    groups: new Set(),
                    nodeIds: [],
                    parentNames: [],       // ordered list of distinct parents
                    parentCounts: {},      // parent_name → occurrence count
                };
                buckets.set(key, b);
            }
            b.qty += 1;
            if (typeof row.weight === 'number') { b.totalWeight += row.weight; b.hasWeight = true; }
            if (typeof row.length === 'number') b.totalLength += row.length;
            const entry = this._groupContainingNodeId(row.node_id);
            if (entry) b.groups.add(entry.group.name);
            b.nodeIds.push(row.node_id);

            // Used-In aggregation: row.assembly_mark is the immediate parent
            // assembly. Track distinct parents + per-parent occurrence counts
            // so the consolidated row can render "×4 in Weldment A, ×2 in …".
            const parent = row.assembly_mark;
            if (parent) {
                if (!(parent in b.parentCounts)) b.parentNames.push(parent);
                b.parentCounts[parent] = (b.parentCounts[parent] || 0) + 1;
            }
        }

        // Sort: classification precedence, then by profile, then length
        const clsOrder = { postprocess: 0, 'bought-out': 1, exclude: 2, mixed: 3, pending: 4 };
        const resolvedKey = b => clsOrder[b.classification?.action] ?? (b.classification?.mixed ? 3 : 4);
        return [...buckets.values()].sort((a, b) => {
            const d = resolvedKey(a) - resolvedKey(b);
            if (d !== 0) return d;
            const p = (a.profile || '').localeCompare(b.profile || '');
            if (p !== 0) return p;
            return (a.length || 0) - (b.length || 0);
        });
    }

    _nativeBomConsolidatedRow(c, cols) {
        const length = c.length != null ? c.length : '';
        const totalLength = c.hasLength ? this._fmtNumber(c.totalLength, 0) : '';
        const totalWeight = c.hasWeight ? this._fmtNumber(c.totalWeight, 1) : '';
        const clsLabel = c.classification
            ? this._classificationLabel(c.classification.action, c.classification.origin, c.classification.mixed)
            : '<span class="nb-pending">Pending</span>';

        // All rows in a consolidated bucket share (entity, profile, grade,
        // length, classification) so they share the same CNC result.  Ask for
        // the first member's CNC info to get the download link.
        const firstNid = c.nodeIds?.[0];
        const cncHtml = firstNid
            ? this._nativeBomCncHtml({ node_id: firstNid }, this._currentFilename)
            : '';

        let groupCell = '';
        if (cols.hasGroups) {
            const names = [...c.groups];
            if (names.length === 0) {
                groupCell = '<td></td>';
            } else if (names.length === 1) {
                groupCell = `<td><span class="nb-group" title="${this._esc(names[0])}">${this._esc(names[0])}</span></td>`;
            } else {
                groupCell = `<td><span class="nb-group" title="${this._esc(names.join(', '))}">${this._esc(names[0])} (+${names.length - 1})</span></td>`;
            }
        }

        return `<tr>
            <td><span class="nb-entity">${this._esc(c.entity)}</span>${cncHtml ? ' ' + cncHtml : ''}</td>
            <td>${this._esc(c.profile)}</td>
            <td>${this._esc(c.grade)}</td>
            <td class="nb-num">${length}</td>
            <td class="nb-num">${c.qty}</td>
            <td class="nb-num">${totalLength}</td>
            <td class="nb-num">${totalWeight}</td>
            <td class="parts-list-parents">${this._parentCellsHtml(c)}</td>
            <td>${clsLabel}</td>
            ${groupCell}
        </tr>`;
    }

    /** Escape a single CSV field per RFC 4180. */
    _csvField(v) {
        if (v == null) return '';
        const s = String(v);
        if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
        return s;
    }

    _classificationLabelPlain(resolved) {
        if (!resolved) return '';
        if (resolved.mixed) return 'Mixed';
        if (resolved.action === 'postprocess') return 'CNC';
        if (resolved.action === 'bought-out') return 'BO';
        if (resolved.action === 'exclude') return 'EXC';
        return resolved.action || '';
    }

    /**
     * Download a CSV of the Full BOM reflecting the active view.  Per-instance
     * keeps one row per physical part with every column (including GUIDs and
     * Tekla pset values when present); Consolidated rolls up identical rows
     * with Qty + Total Length + Total Weight.
     */
    _downloadNativeBomCsv() {
        const rows = Array.isArray(this._nativeBom) ? this._nativeBom : [];
        if (rows.length === 0) {
            alert('No BOM data to export.');
            return;
        }

        let lines;
        let tagSuffix;
        if (this._nativeBomView === 'consolidated') {
            const cons = this._consolidateNativeBom(rows);
            const headers = [
                'entity', 'profile', 'grade', 'length_mm',
                'qty', 'total_length_mm', 'total_weight_kg',
                'classification', 'groups',
            ];
            lines = [headers.join(',')];
            for (const c of cons) {
                const groups = [...c.groups].join('; ');
                lines.push([
                    this._csvField(c.entity),
                    this._csvField(c.profile),
                    this._csvField(c.grade),
                    this._csvField(c.length ?? ''),
                    this._csvField(c.qty),
                    this._csvField(c.hasLength ? Math.round(c.totalLength) : ''),
                    this._csvField(c.hasWeight ? c.totalWeight.toFixed(2) : ''),
                    this._csvField(this._classificationLabelPlain(c.classification)),
                    this._csvField(groups),
                ].join(','));
            }
            tagSuffix = 'consolidated';
        } else {
            const headers = [
                'node_id', 'guid', 'type_guid',
                'entity', 'mark', 'profile',
                'material', 'grade',
                'length_mm', 'weight_kg', 'volume_m3', 'area_m2',
                'assembly_mark', 'assembly_position',
                'pour', 'phase', 'finish', 'part_class',
                'source', 'classification', 'group',
            ];
            lines = [headers.join(',')];
            for (const r of rows) {
                const resolved = this._resolveClassification(r.node_id);
                const grp = this._groupContainingNodeId(r.node_id);
                lines.push([
                    this._csvField(r.node_id),
                    this._csvField(r.guid ?? ''),
                    this._csvField(r.type_guid ?? ''),
                    this._csvField(r.entity ?? ''),
                    this._csvField(this._bomRowDisplayMark(r)),
                    this._csvField(r.profile ?? ''),
                    this._csvField(r.material ?? ''),
                    this._csvField(r.grade ?? ''),
                    this._csvField(r.length ?? ''),
                    this._csvField(r.weight ?? ''),
                    this._csvField(r.volume ?? ''),
                    this._csvField(r.area ?? ''),
                    this._csvField(r.assembly_mark ?? ''),
                    this._csvField(r.assembly_position ?? ''),
                    this._csvField(r.pour ?? ''),
                    this._csvField(r.phase ?? ''),
                    this._csvField(r.finish ?? ''),
                    this._csvField(r.part_class ?? ''),
                    this._csvField(r.source ?? ''),
                    this._csvField(this._classificationLabelPlain(resolved)),
                    this._csvField(grp ? grp.groupPath : ''),
                ].join(','));
            }
            tagSuffix = 'per-instance';
        }

        // Prepend UTF-8 BOM so Excel opens accented / special chars cleanly.
        const csv = '﻿' + lines.join('\r\n') + '\r\n';
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const stem = this._lastProjectNumber
            || (this._currentFilename ? this._currentFilename.replace(/\.[^.]+$/, '') : 'project');
        a.download = `${stem}-full-bom-${tagSuffix}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Download the Full BOM as JSON.  Per-instance view dumps the raw rows
     * (plus resolved classification + group path); Consolidated dumps the
     * aggregated buckets.
     */
    _downloadNativeBomJson() {
        const rows = Array.isArray(this._nativeBom) ? this._nativeBom : [];
        if (rows.length === 0) {
            alert('No BOM data to export.');
            return;
        }

        const stem = this._lastProjectNumber
            || (this._currentFilename ? this._currentFilename.replace(/\.[^.]+$/, '') : 'project');
        let payload;
        let tagSuffix;

        if (this._nativeBomView === 'consolidated') {
            const cons = this._consolidateNativeBom(rows);
            tagSuffix = 'consolidated';
            payload = {
                view: 'consolidated',
                generated_at: new Date().toISOString(),
                project: stem,
                rows: cons.map(c => ({
                    entity: c.entity,
                    profile: c.profile,
                    grade: c.grade,
                    length_mm: c.length,
                    qty: c.qty,
                    total_length_mm: c.hasLength ? Math.round(c.totalLength) : null,
                    total_weight_kg: c.hasWeight ? +c.totalWeight.toFixed(2) : null,
                    classification: this._classificationLabelPlain(c.classification),
                    used_in: c.parentNames.map(p => ({ parent: p, qty: c.parentCounts[p] })),
                    groups: [...c.groups],
                })),
            };
        } else {
            tagSuffix = 'per-instance';
            payload = {
                view: 'per-instance',
                generated_at: new Date().toISOString(),
                project: stem,
                rows: rows.map(r => ({
                    ...r,
                    classification: this._classificationLabelPlain(this._resolveClassification(r.node_id)),
                    group: this._groupContainingNodeId(r.node_id)?.groupPath || null,
                })),
            };
        }

        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${stem}-full-bom-${tagSuffix}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    _refreshNativeBomIfOpen() {
        const panel = this.container?.querySelector('#native-bom-panel');
        if (panel && !panel.hidden) this._renderNativeBom();
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
                // part_single_solid / part_no_solid — recover the parent name
                // for generic SOLID leaves (no-solid artifacts keep theirs).
                this._bomUpsert(itemMap, refId, this._displayNodeName(node, parentName),
                                node.id, isMirr, parentName);
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
                const checks = Array.isArray(r.solids) ? r.solids : [r];
                for (const s of checks) {
                    if (s.dxf_path) hasDxf = true;
                    if (s.nc1_path) hasNc1 = true;
                }
            }
        }
        const enc = encodeURIComponent(filename || '');
        // Downloads share the consolidation freshness gate.  Render as
        // disabled <span> chips when locked so the row layout stays stable.
        const downloadsLocked = this._cncState && this._cncState.state !== 'fresh';
        const lockTitle = downloadsLocked
            ? (this._cncState?.state === 'missing'
                ? 'Run consolidation before downloading'
                : 'Outputs stale \u2014 re-run analysis first')
            : '';
        const dxfZipLink = hasDxf
            ? (downloadsLocked
                ? `<span class="parts-cnc-dl-btn disabled" title="${lockTitle}">\u2193\u00a0DXF</span>`
                : `<a href="/api/v1/cnc-analysis/download-all/${enc}/dxf" class="parts-cnc-dl-btn" download>\u2193\u00a0DXF</a>`)
            : '';
        const nc1ZipLink = hasNc1
            ? (downloadsLocked
                ? `<span class="parts-cnc-dl-btn disabled" title="${lockTitle}">\u2193\u00a0NC1</span>`
                : `<a href="/api/v1/cnc-analysis/download-all/${enc}/nc1" class="parts-cnc-dl-btn" download>\u2193\u00a0NC1</a>`)
            : '';

        const hasAnyResults = Object.keys(this._cncAnalysisResults || {}).length > 0;
        // Gate Analyse + Downloads on consolidation freshness (see _refreshCncState).
        // - missing/stale-tree: hard-disable Analyse, prompt to consolidate
        // - stale-cnc: Analyse stays enabled but auto-passes force=true so the
        //   backend clears the stale outputs/cnc dir and re-runs
        // - fresh: normal behaviour
        // Downloads (.dxf/.nc1 ZIP) and the Excel BOM gate on the same state.
        const cncState = this._cncState?.state || 'fresh';
        const analyseDisabledReason = cncState === 'missing'
            ? 'Run consolidation before CNC analysis'
            : cncState === 'stale-tree'
                ? 'Assembly was re-analysed \u2014 re-run consolidation first'
                : '';
        const analyseLabel = cncState === 'stale-cnc' ? 'Re-analyse (stale)' : 'Analyse';
        const analyseBtn = this._cncAnalysing
            ? `<button class="parts-cnc-analyse-btn outline" disabled>Analysing\u2026</button>`
            : (analyseDisabledReason
                ? `<button class="parts-cnc-analyse-btn outline" disabled title="${analyseDisabledReason}">${analyseLabel}</button>`
                : `<button class="parts-cnc-analyse-btn outline"${cncState === 'stale-cnc' ? ' data-force="1"' : ''}>${analyseLabel}</button>`)
              + (hasAnyResults && !analyseDisabledReason
                  ? `<button class="parts-cnc-reanalyse-btn outline" title="Clear cache and re-run analysis">\u21ba Re-analyse</button>`
                  : '');

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

        // BOM Excel download follows the same freshness gate as the NC/DXF ZIPs.
        const bomDlBtn = totalClassified > 0
            ? (downloadsLocked
                ? `<button class="parts-bom-xlsx-btn outline" disabled title="${lockTitle}">\u2193\u00a0BOM (.xlsx)</button>`
                : `<button class="parts-bom-xlsx-btn outline" title="Download BOM as Excel with thumbnails">\u2193\u00a0BOM (.xlsx)</button>`)
            + `<button class="parts-bom-dl-btn outline" title="Download BOM as JSON">\u2193\u00a0JSON</button>`
            : '';

        // Nesting button — only show when we have CNC section results
        const hasNestableSections = this._hasNestableSections(cncItems, unknownItems);
        const nestingBtn = hasNestableSections
            ? (this._nestingRunning
                ? `<button class="parts-nesting-btn outline" disabled>Nesting\u2026</button>`
                : `<button class="parts-nesting-btn outline">\u2702 Nesting</button>`)
            : '';

        // Banner reflecting consolidation/CNC freshness state.  Shown above the
        // table so the operator sees it before they reach the Analyse button
        // (which is also gated, but the banner explains *why*).
        let stateBanner = '';
        if (cncState === 'missing') {
            stateBanner = `<div class="parts-list-state-banner missing">
                <span>Consolidation has not been run for this assembly. CNC analysis and downloads are locked until consolidation completes.</span>
                <button class="parts-state-consolidate-btn">Consolidate now</button>
            </div>`;
        } else if (cncState === 'stale-tree') {
            stateBanner = `<div class="parts-list-state-banner stale">
                <span>The assembly was re-analysed after the last consolidation. Re-run consolidation before producing NC1 files.</span>
                <button class="parts-state-consolidate-btn">Re-consolidate</button>
            </div>`;
        } else if (cncState === 'stale-cnc') {
            const n = this._cncState?.stale_cnc_refs?.length || 0;
            stateBanner = `<div class="parts-list-state-banner stale">
                <span>${n} CNC result${n === 1 ? '' : 's'} predate the current consolidation. Re-analyse to clear stale NC files and refresh the BOM.</span>
            </div>`;
        }

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
                ${stateBanner}
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

        panel.querySelector('.parts-state-consolidate-btn')?.addEventListener('click', () => {
            this._startConsolidation();
        });

        const allCncItems = [...unknownItems, ...cncItems];
        panel.querySelector('.parts-cnc-analyse-btn')?.addEventListener('click', (ev) => {
            // When stale-cnc the button is rendered with data-force="1" so the
            // click here picks that up and bypasses the cache automatically.
            const forceFromState = ev.currentTarget?.dataset?.force === '1';
            this._startCncAnalysis(allCncItems, forceFromState);
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
            this._showNestingSettingsDialog(this._buildNestingItems(allNestableItems));
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

        // A multi-solid SOLID node ("<parentRef>:s<N>") — resolve to the
        // parent's solids[N] FIRST. The synthetic solid ref is not an XCAF
        // label, so any direct entry under it is stale; the authoritative
        // per-solid class lives in the parent result's solids[].
        const sm = String(item.key).match(/^(.*):s(\d+)$/);
        if (sm) {
            const parentRes = this._cncAnalysisResults[sm[1]];
            const idx = parseInt(sm[2], 10);
            if (parentRes && Array.isArray(parentRes.solids) && parentRes.solids[idx]) {
                return { result: parentRes.solids[idx], xcafRefId: sm[1], solidIdx: idx };
            }
        }

        // Direct lookup (single-solid part)
        const direct = this._cncAnalysisResults[item.key];
        if (direct) return { result: direct, xcafRefId: item.key, solidIdx: null };

        // Solid body from exploded multi-solid
        const parentInfo = this._parentMap.get(item.key);
        if (parentInfo?._solidParentRefId) {
            const xcafRefId = parentInfo._solidParentRefId;
            const parentResult = this._cncAnalysisResults[xcafRefId];
            if (!parentResult) return null;

            // Index solids[] whenever present — heavy results carry
            // type==='multi_solid', lightweight classify results just have solids[].
            if (Array.isArray(parentResult.solids)) {
                // Try to extract solid index from nodeId (expected format "<ref_id>:s<N>")
                const match = item.key.match(/:s(\d+)$/);
                const solidIdx = match ? parseInt(match[1]) : 0;
                const solidResult = parentResult.solids[solidIdx];
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
     * Resolve the CNC analysis result for a native_bom row.  Handles both
     * direct rows (node_id === ref_id, which happens for IFC or for STEP
     * placeholder rows) and per-solid rows ("<instance>:s<N>") by translating
     * the instance-path node_id back to the XCAF ref_id via _nodeRefMap.
     */
    _cncResultForNativeBomRow(row) {
        if (!this._cncAnalysisResults || !row?.node_id) return null;
        const m = row.node_id.match(/^(.*):s(\d+)$/);
        const instanceNodeId = m ? m[1] : row.node_id;
        const solidIdx = m ? parseInt(m[2], 10) : null;
        const xcafRefId = this._nodeRefMap?.get(instanceNodeId) || instanceNodeId;
        const refResult = this._cncAnalysisResults[xcafRefId];
        if (!refResult) return null;
        if (Array.isArray(refResult.solids) && solidIdx != null) {
            const solidResult = refResult.solids[solidIdx];
            if (solidResult) return { result: solidResult, xcafRefId, solidIdx };
            return { result: refResult, xcafRefId, solidIdx: null };
        }
        return { result: refResult, xcafRefId, solidIdx };
    }

    /** Native-BOM counterpart of _cncResultHtml: badge + optional DXF/NC1 link. */
    _nativeBomCncHtml(row, filename) {
        const info = this._cncResultForNativeBomRow(row);
        if (!info) return '';
        return this._renderCncInfo(info, filename);
    }

    /**
     * Return HTML string with result badge and optional download link for a CNC BOM item.
     */
    _cncResultHtml(item, filename) {
        const info = this._cncResultForItem(item);
        if (!info) return '';
        return this._renderCncInfo(info, filename);
    }

    /** Shared render path used by both Standard and Full BOM for CNC badges. */
    _renderCncInfo(info, filename) {
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

        // Refined decision-tree class (rw7.1): surfaces formed/bent/exclude/BO
        // that the base type can't express, with a review flag for low confidence.
        const refined = result.refined_class || '';
        const rConf = result.refined_confidence;
        let refinedBadge = '';
        if (refined) {
            const colors = {
                section: '#e6a13c', plate: '#4caf50', formed_plate: '#e3496b',
                bent_section: '#46c2c2', bought_out: '#8a7bd8', exclude: '#777',
            };
            const c = colors[refined.toLowerCase()] || '#999';
            const flag = (rConf != null && rConf < 0.5) ? ' ⚠' : '';
            const tip = `confidence ${rConf}${result.refined_reason ? ' — ' + result.refined_reason : ''}`;
            refinedBadge = ` <span class="cnc-refined" style="background:${c};color:#fff;border-radius:4px;padding:1px 6px;font-size:11px" title="${this._esc(tip)}">${this._esc(refined)}${flag}</span>`;
        }

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
                // Lightweight classification entry (no base type yet) — show the
                // refined-class badge alone so triage suggestions are visible.
                return refinedBadge;
        }

        return badge + refinedBadge + (downloadLink ? ' ' + downloadLink : '');
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
    /**
     * Start CNC geometric analysis for every native_bom row whose resolved
     * classification is 'postprocess'.  Reuses the settings dialog + backend
     * POST flow but builds the `ref_ids / member_ids / parent_names` payload
     * directly from native_bom rows (translating ":s<N>" suffixes back to
     * the XCAF ref_id via _nodeRefMap).
     */
    _startCncAnalysisFromNativeBom(force = false) {
        if (this._cncAnalysing) return;
        const filename = this._currentFilename;
        if (!filename) return;

        const rows = (this._nativeBom || []).filter(r => {
            const c = this._resolveClassification(r.node_id);
            return c?.action === 'postprocess';
        });
        if (rows.length === 0) {
            alert('No parts classified as CNC yet.');
            return;
        }

        const refIdSet = new Set();
        const memberIds = {};
        const parentNames = {};
        for (const row of rows) {
            const m = row.node_id.match(/^(.*):s\d+$/);
            const instanceNodeId = m ? m[1] : row.node_id;
            const refId = this._nodeRefMap?.get(instanceNodeId) || instanceNodeId;
            refIdSet.add(refId);
            if (!memberIds[refId]) memberIds[refId] = row.mark || '';
            if (!parentNames[refId]) parentNames[refId] = row.assembly_mark || row.mark || '';
        }
        if (refIdSet.size === 0) return;

        this._showCncSettingsDialog((projectNumber, steelGrade) => {
            this._cncAnalysing = true;
            this._rerenderOpenBomPanels();

            const url = `/api/v1/cnc-analysis/analyse/${encodeURIComponent(filename)}${force ? '?force=1' : ''}`;
            const body = {
                ref_ids: [...refIdSet],
                member_ids: memberIds,
                parent_names: parentNames,
                project_number: projectNumber,
                steel_grade: steelGrade,
                force,
            };
            fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            })
                .then(r => r.json())
                .then(resp => {
                    if (resp.cnc_task_id) {
                        this._pollCncAnalysis(resp.cnc_task_id);
                    } else if (resp.status === 'completed') {
                        this._cncAnalysing = false;
                        if (resp.results) {
                            this._cncAnalysisResults = Object.assign(
                                {}, this._cncAnalysisResults || {}, resp.results
                            );
                        }
                        this._rerenderOpenBomPanels();
                    } else {
                        this._cncAnalysing = false;
                        this._rerenderOpenBomPanels();
                    }
                })
                .catch(err => {
                    console.error('Failed to start CNC analysis:', err);
                    this._cncAnalysing = false;
                    this._rerenderOpenBomPanels();
                });
        });
    }

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
                        // Auto-classify newly analysed parts from refined_class.
                        this._autoClassifyFromRefinedResults();

                        // Refresh freshness — completion should move us to 'fresh'.
                        this._refreshCncState(this._currentFilename)
                            .finally(() => this._rerenderOpenBomPanels());
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
        this._rerenderOpenBomPanels();  // show "Consolidating…" state in whichever panel is open

        this.api.startConsolidation(filename)
            .then(resp => {
                if (resp.groups) {
                    this._consolidationGroups = resp.groups;
                    this._solidConsolidationGroups = resp.solid_groups || [];
                    this._intraSolidGroups = resp.intra_solid_groups || [];
                    this._consolidating = false;
                    this._rerenderOpenBomPanels();
                } else if (resp.consolidation_task_id) {
                    this._pollConsolidation(resp.consolidation_task_id);
                } else {
                    this._consolidating = false;
                    this._rerenderOpenBomPanels();
                }
            })
            .catch(err => {
                console.error('Failed to start consolidation:', err);
                this._consolidating = false;
                this._rerenderOpenBomPanels();
            });
    }

    /** Rerender whichever BOM panel (Standard or Full) is currently open. */
    _rerenderOpenBomPanels() {
        const standard = this.container?.querySelector('#parts-list-panel');
        if (standard && !standard.hidden) this._renderPartsList(this._consolidationGroups);
        const full = this.container?.querySelector('#native-bom-panel');
        if (full && !full.hidden) this._renderNativeBom();
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
                        // Refresh freshness state so the banner reflects the
                        // new consolidation (likely transitions missing → fresh
                        // or fresh → stale-cnc if CNC results predate this run).
                        this._refreshCncState(this._currentFilename)
                            .finally(() => this._rerenderOpenBomPanels());
                    } else if (resp.status === 'failed') {
                        clearInterval(this._consolidatePollTimer);
                        this._consolidatePollTimer = null;
                        this._consolidating = false;
                        console.error('Consolidation failed:', resp.error);
                        this._rerenderOpenBomPanels();
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
     * Build nesting items straight from ``analysis.native_bom``.
     *
     * Criteria: the row's resolved classification is ``postprocess`` AND its
     * entity is a length-cut profile (Section / HollowSection) AND it has
     * both a profile designation and a length.  Works uniformly for STEP
     * (populated post-CNC analysis) and IFC (populated at parse time from
     * Tekla psets).
     */
    /**
     * Entities that are physically not nestable (no bar-stock concept).  Any
     * BOM row whose entity is outside this set — STEP "Section"/"HollowSection"
     * as well as IFC "IfcBeam"/"IfcColumn"/"IfcMember" etc. — is eligible if
     * it carries a profile and a length.
     */
    static _NON_NESTABLE_ENTITIES = new Set([
        'Plate', 'IfcPlate',
        'MultiSolidPart',
        'Part',
        'IfcBuildingElementProxy',
        'IfcFooting', 'IfcPile',
    ]);

    _buildNestingItemsFromNativeBom() {
        const rows = Array.isArray(this._nativeBom) ? this._nativeBom : [];
        const items = [];
        let idx = 0;
        for (const row of rows) {
            const resolved = this._resolveClassification(row.node_id);
            if (resolved?.action !== 'postprocess') continue;
            if (AnalysisPage._NON_NESTABLE_ENTITIES.has(row.entity)) continue;
            if (!row.profile || !row.length) continue;
            items.push({
                item_index: idx++,
                ref_id: row.node_id,
                section: row.profile,
                length: Math.round(row.length),
                parent: row.assembly_mark || '',
                member_name: row.mark || '',
            });
        }
        return items;
    }

    /** Count how many Full-BOM rows currently qualify for nesting. */
    _nestableCountFromNativeBom() {
        return this._buildNestingItemsFromNativeBom().length;
    }

    /**
     * Show the nesting settings dialog — stock length, qty, kerf, per-section overrides.
     * Accepts pre-built nesting items so both the Standard-BOM path (which
     * synthesises from CNC analysis results) and the Full-BOM path (which
     * reads native_bom directly) can share the dialog.
     */
    _showNestingSettingsDialog(nestingItems) {
        if (!Array.isArray(nestingItems) || nestingItems.length === 0) {
            alert('No CNC-classified section parts available for nesting.');
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
