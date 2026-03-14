/**
 * Connections page — detect weld/bolt connections between solids.
 *
 * Workflow:
 *   1. Select an uploaded STEP file (must have been analysed first)
 *   2. Click a node in the assembly tree to scope the detection
 *   3. Click "Detect Connections" to launch the background worker
 *   4. Poll for results, display summary + table
 *   5. Click a connection row to highlight in the 3D viewer
 */
import { STLViewer } from './stl-viewer.js';

export class ConnectionsPage {
    constructor(api) {
        this.api = api;
        this.container = null;

        /** @type {string|null} */
        this._currentFilename = null;

        /** @type {string|null} selected node_id from the tree */
        this._selectedNodeId = null;

        /** @type {string|null} selected node name for display */
        this._selectedNodeName = null;

        /** @type {number|null} poll timer */
        this._pollTimer = null;

        /** @type {string|null} active task id */
        this._taskId = null;

        /** @type {STLViewer|null} */
        this._viewer = null;

        /** @type {Array} STL file list from preview generation */
        this._stlFiles = [];

        /** @type {Map<string, number>} node_id → mesh index in the viewer */
        this._solidIndexMap = new Map();

        /** @type {Array} connections from detection results */
        this._connections = [];

        /** @type {number} currently highlighted connection index (-1 = none) */
        this._activeConnIdx = -1;

        /** @type {number|null} preview poll timer */
        this._previewPollTimer = null;

        /** @type {string} detection scope */
        this._scope = 'all';

        /** @type {number|null} batch detection poll timer */
        this._batchPollTimer = null;

        /** @type {string|null} batch task id */
        this._batchTaskId = null;

        /** @type {Object|null} tree status data */
        this._treeStatus = null;

        /** @type {Array} full analysis tree nodes */
        this._treeNodes = [];

        /** @type {Map<string, number>} child "nodeId:sN" → compound flat index */
        this._childToCompoundIdx = new Map();

        /** @type {Function|null} delegated click handler on results container */
        this._onResultsClick = null;

        /** @type {Function|null} delegated change handler on results container */
        this._onResultsChange = null;
    }

    render(container) {
        this.container = container;
        this._cleanup();
        container.innerHTML = this._template();
        this._bindEvents();
        this._loadFiles();
    }

    _cleanup() {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
        if (this._previewPollTimer) {
            clearInterval(this._previewPollTimer);
            this._previewPollTimer = null;
        }
        if (this._batchPollTimer) {
            clearInterval(this._batchPollTimer);
            this._batchPollTimer = null;
        }
        if (this._viewer) {
            this._viewer.dispose();
            this._viewer = null;
        }
        // Remove delegated result handlers
        const resultsEl = this.container?.querySelector('#conn-results');
        if (resultsEl) {
            if (this._onResultsClick) {
                resultsEl.removeEventListener('click', this._onResultsClick);
                this._onResultsClick = null;
            }
            if (this._onResultsChange) {
                resultsEl.removeEventListener('change', this._onResultsChange);
                this._onResultsChange = null;
            }
        }
    }

    // ---------------------------------------------------------------
    // Template
    // ---------------------------------------------------------------

    _template() {
        return `
            <section style="font-size:0.8rem;">
                <h2 style="font-size:1.1rem; margin:0.3rem 0;">Connection Detection</h2>

                <div class="conn-controls" style="display:flex; gap:0.3rem; align-items:center; flex-wrap:wrap; margin-bottom:0.4rem;">
                    <select id="conn-file-select" aria-label="Select STEP file" style="max-width:280px; padding:2px 4px; font-size:0.78rem;">
                        <option value="">Loading files...</option>
                    </select>
                    <select id="conn-scope-select" aria-label="Detection scope" style="max-width:150px; padding:2px 4px; font-size:0.78rem;">
                        <option value="all">All Connections</option>
                        <option value="within-part">Within Part</option>
                        <option value="siblings">Between Parts</option>
                        <option value="cross-assembly">Cross-Assembly</option>
                    </select>
                    <button id="conn-detect-btn" disabled style="padding:2px 8px; font-size:0.78rem;" title="Shift+click to force re-detection">Detect</button>
                    <button id="conn-detect-all-btn" disabled style="padding:2px 8px; font-size:0.78rem; background:var(--pico-primary-background, #3b82f6);color:#fff;" title="Shift+click to force re-detection">Detect All</button>
                    <button id="conn-export-btn" disabled style="padding:2px 8px; font-size:0.78rem; background:var(--pico-secondary-background, #e2e8f0);color:var(--pico-secondary-color, #374151);">Export XLS</button>
                    <span id="conn-scope-label" style="font-size:0.75rem; color:#6b7280;"></span>
                </div>

                <div style="display:flex; gap:0.5rem; align-items:flex-start;">
                    <div id="conn-tree-panel" style="
                        min-width:220px; max-width:280px; max-height:calc(100vh - 140px); overflow-y:auto;
                        border:1px solid var(--pico-muted-border-color, #e2e8f0);
                        border-radius:4px; padding:0.25rem; font-size:0.75rem;
                        display:none;
                    "></div>

                    <div style="flex:1; min-width:0;">
                        <div id="conn-viewer-panel" style="
                            width:100%; height:340px;
                            border:1px solid var(--pico-muted-border-color, #e2e8f0);
                            border-radius:4px; background:#f8fafc;
                            display:none; position:relative; margin-bottom:0.4rem;
                        ">
                            <div id="conn-viewer-container" style="width:100%; height:100%;"></div>
                            <div id="conn-viewer-overlay" style="
                                position:absolute; top:0; left:0; right:0; bottom:0;
                                display:flex; align-items:center; justify-content:center;
                                background:rgba(248,250,252,0.85); font-size:0.8rem; color:#6b7280;
                                pointer-events:none;
                            ">
                                Select a node and run detection to preview
                            </div>
                        </div>

                        <div id="conn-status" style="display:none; margin-bottom:0.4rem;">
                            <small class="conn-status-text" style="font-size:0.75rem;">Detecting connections...</small>
                            <progress id="conn-progress" style="width:160px; height:8px;"></progress>
                        </div>
                        <div id="conn-summary" style="display:none; margin-bottom:0.4rem;"></div>
                        <div id="conn-results" style="display:none;"></div>
                    </div>
                </div>
            </section>
        `;
    }

    // ---------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------

    _bindEvents() {
        const select = this.container.querySelector('#conn-file-select');
        const scopeSelect = this.container.querySelector('#conn-scope-select');
        const btn = this.container.querySelector('#conn-detect-btn');

        select.addEventListener('change', async () => {
            this._currentFilename = select.value || null;
            this._selectedNodeId = null;
            this._selectedNodeName = null;
            this._updateScopeLabel();
            btn.disabled = true;
            this._hideResults();
            this._clearViewer();

            if (this._currentFilename) {
                detectAllBtn.disabled = false;
                await this._loadTree(this._currentFilename);
                // Don't auto-load cached results — wait for node selection
                // so results match the viewed STLs
            } else {
                detectAllBtn.disabled = true;
                this.container.querySelector('#conn-tree-panel').style.display = 'none';
            }
        });

        scopeSelect.addEventListener('change', () => {
            this._scope = scopeSelect.value;
            // Reload cached results for the new scope (null = no node selected; "" = all parts)
            if (this._currentFilename && this._selectedNodeId !== null) {
                this._loadCached(this._currentFilename, this._selectedNodeId, this._scope);
            }
        });

        btn.addEventListener('click', (e) => this._startDetection(e.shiftKey));

        const detectAllBtn = this.container.querySelector('#conn-detect-all-btn');
        detectAllBtn.addEventListener('click', (e) => this._startBatchDetection(e.shiftKey));

        const exportBtn = this.container.querySelector('#conn-export-btn');
        exportBtn.addEventListener('click', () => {
            if (this._currentFilename) {
                const url = this.api.getConnectionExportXlsxUrl(
                    this._currentFilename,
                    this._selectedNodeId || null,
                    this._scope,
                );
                window.location.href = url;
            }
        });
    }

    // ---------------------------------------------------------------
    // File list
    // ---------------------------------------------------------------

    async _loadFiles() {
        const select = this.container.querySelector('#conn-file-select');
        try {
            const data = await this.api.listFiles();
            const files = data.files || [];
            select.innerHTML = '<option value="">-- select a file --</option>';
            files.forEach(f => {
                const name = f.filename || f;
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                select.appendChild(opt);
            });
        } catch {
            select.innerHTML = '<option value="">Error loading files</option>';
        }
    }

    // ---------------------------------------------------------------
    // Assembly tree
    // ---------------------------------------------------------------

    async _loadTree(filename) {
        const panel = this.container.querySelector('#conn-tree-panel');
        panel.style.display = 'block';
        panel.innerHTML = '<em>Loading tree...</em>';

        try {
            const data = await this.api.getAssemblyTree(filename);
            const nodes = data.assembly_tree || [];
            this._treeNodes = nodes;
            if (nodes.length === 0) {
                panel.innerHTML = '<em>No assembly tree found — analyse the file first.</em>';
                return;
            }
            // If there's no single root assembly (multiple top-level nodes or
            // a single leaf), prepend a synthetic root so the user can scope
            // detection to the whole file.
            const singleRoot = nodes.length === 1 && nodes[0].children?.length > 0;
            const totalSolids = nodes.reduce((s, n) => s + (n.solid_count || 1), 0);
            const rootHtml = singleRoot ? '' : `
                <li style="margin:0 0 2px 0;">
                    <div class="conn-tree-row" data-node-id="" data-node-name="(All parts)"
                         data-solid-count="${totalSolids}" data-has-children="false"
                         style="padding:1px 3px; border-radius:3px; cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; line-height:1.3; font-style:italic; border-bottom:1px solid var(--pico-muted-border-color,#e2e8f0);"
                         title="Run detection on all parts in this file">
                        <span style="display:inline-block;width:0.8em;"></span>
                        (All parts) <span style="color:#6b7280;font-size:0.65rem;">(${totalSolids} solids)</span>
                    </div>
                </li>`;
            panel.innerHTML = '<ul style="list-style:none;padding:0;margin:0;">'
                + rootHtml
                + nodes.map(n => this._renderTreeNode(n)).join('')
                + '</ul>';
            this._bindTreeEvents(panel);
            // Fetch and apply tree status badges
            this._refreshTreeStatus();
        } catch {
            panel.innerHTML = '<em>Failed to load assembly tree.</em>';
        }
    }

    _renderTreeNode(node) {
        const hasChildren = node.children && node.children.length > 0;
        const solidCount = node.solid_count || 0;
        const badge = solidCount > 1
            ? `<span style="color:#6b7280;font-size:0.65rem;">(${solidCount})</span>`
            : '';
        const icon = hasChildren ? '\u25B6 ' : '';
        const childrenHtml = hasChildren
            ? `<ul style="list-style:none;padding-left:0.7rem;margin:0;display:none;">`
              + node.children.map(c => this._renderTreeNode(c)).join('')
              + '</ul>'
            : '';

        return `
            <li style="margin:0;">
                <div class="conn-tree-row" data-node-id="${this._esc(node.id)}"
                     data-node-name="${this._esc(node.name)}"
                     data-solid-count="${solidCount}"
                     data-has-children="${hasChildren}"
                     style="padding:1px 3px; border-radius:3px; cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; line-height:1.3;"
                     title="${this._esc(node.id)} — ${this._esc(node.name)} (${solidCount} solids)">
                    <span class="conn-tree-toggle" style="display:inline-block;width:0.8em;text-align:center;font-size:0.6rem;transition:transform 0.15s;${hasChildren ? '' : 'visibility:hidden;'}">${icon}</span>
                    ${this._esc(node.name)} ${badge}
                </div>
                ${childrenHtml}
            </li>
        `;
    }

    _bindTreeEvents(panel) {
        panel.addEventListener('click', (e) => {
            const row = e.target.closest('.conn-tree-row');
            if (!row) return;

            const hasChildren = row.dataset.hasChildren === 'true';
            const toggle = row.querySelector('.conn-tree-toggle');

            // Toggle expand/collapse
            if (hasChildren && e.target.closest('.conn-tree-toggle')) {
                const ul = row.parentElement.querySelector(':scope > ul');
                if (ul) {
                    const isOpen = ul.style.display !== 'none';
                    ul.style.display = isOpen ? 'none' : 'block';
                    toggle.style.transform = isOpen ? '' : 'rotate(90deg)';
                }
                return;
            }

            // Select this node as scope
            panel.querySelectorAll('.conn-tree-row').forEach(r => {
                r.style.background = '';
                r.style.fontWeight = '';
            });
            row.style.background = 'var(--pico-primary-background, #dbeafe)';
            row.style.fontWeight = '600';

            this._selectedNodeId = row.dataset.nodeId;
            this._selectedNodeName = row.dataset.nodeName;
            this._updateScopeLabel();
            this.container.querySelector('#conn-detect-btn').disabled = false;

            // Clear previous results immediately so the detect guard doesn't
            // block a new run while _loadCached is still pending
            this._hideResults();

            // Clean up any in-progress detection from a previous node
            if (this._pollTimer) {
                clearInterval(this._pollTimer);
                this._pollTimer = null;
            }
            this._taskId = null;
            this._hideStatus();

            // Start compound preview (shows all solids for this assembly)
            this._startPreview(this._selectedNodeId);

            // Load cached connection results (table only — no STL loading)
            this._loadCached(this._currentFilename, this._selectedNodeId, this._scope);
        });
    }

    _updateScopeLabel() {
        const label = this.container.querySelector('#conn-scope-label');
        if (this._selectedNodeName) {
            const idPart = this._selectedNodeId ? ` (${this._selectedNodeId})` : '';
            label.textContent = `Scope: ${this._selectedNodeName}${idPart}`;
        } else {
            label.textContent = '';
        }
    }

    _esc(s) {
        if (!s) return '';
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    // ---------------------------------------------------------------
    // 3D Viewer
    // ---------------------------------------------------------------

    _ensureViewer() {
        const viewerPanel = this.container.querySelector('#conn-viewer-panel');
        viewerPanel.style.display = 'block';

        if (!this._viewer) {
            const viewerContainer = this.container.querySelector('#conn-viewer-container');
            this._viewer = new STLViewer(viewerContainer);
        }
        return this._viewer;
    }

    _clearViewer() {
        this._stlFiles = [];
        this._solidIndexMap.clear();
        this._activeConnIdx = -1;
        if (this._viewer) {
            this._viewer.dispose();
            this._viewer = null;
        }
        const panel = this.container.querySelector('#conn-viewer-panel');
        if (panel) panel.style.display = 'none';
    }

    _setViewerOverlay(msg) {
        const overlay = this.container.querySelector('#conn-viewer-overlay');
        if (overlay) {
            overlay.textContent = msg || '';
            overlay.style.display = msg ? 'flex' : 'none';
        }
    }

    // ---------------------------------------------------------------
    // Preview STL generation
    // ---------------------------------------------------------------

    async _startPreview(nodeId) {
        if (this._previewPollTimer) {
            clearInterval(this._previewPollTimer);
            this._previewPollTimer = null;
        }

        this._ensureViewer();
        this._setViewerOverlay('Loading solid meshes...');

        try {
            // Empty nodeId = whole-file scope: batch preview all top-level nodes
            let resp;
            if (!nodeId) {
                const allIds = this._treeNodes.map(n => n.id).filter(Boolean);
                resp = await this.api.startConnectionPreviewBatch(
                    this._currentFilename, allIds,
                );
            } else {
                resp = await this.api.startConnectionPreview(
                    this._currentFilename, nodeId,
                );
            }

            // Fast path: files returned directly (cached on disk)
            if (resp.status === 'completed' && resp.files) {
                this._onPreviewReady(resp.files);
                return;
            }

            if (resp.status === 'completed' && resp.existing) {
                // Already done — load status to get files
                const statusResp = await this.api.getConnectionPreviewStatus(resp.task_id);
                if (statusResp.status === 'completed') {
                    this._onPreviewReady(statusResp.files);
                    return;
                }
            }

            // Slow path: poll for worker completion
            this._setViewerOverlay('Generating solid meshes...');
            this._previewPollTimer = setInterval(async () => {
                try {
                    const statusResp = await this.api.getConnectionPreviewStatus(resp.task_id);
                    if (statusResp.status === 'completed') {
                        clearInterval(this._previewPollTimer);
                        this._previewPollTimer = null;
                        this._onPreviewReady(statusResp.files);
                    } else if (statusResp.status === 'failed') {
                        clearInterval(this._previewPollTimer);
                        this._previewPollTimer = null;
                        this._setViewerOverlay(`Preview failed: ${statusResp.error}`);
                    }
                } catch {
                    // Network blip — keep polling
                }
            }, 1500);
        } catch (err) {
            this._setViewerOverlay(`Preview error: ${err.message || err.error || 'Unknown'}`);
        }
    }

    async _onPreviewReady(files) {
        this._stlFiles = files || [];
        this._solidIndexMap.clear();

        const items = [];
        for (let i = 0; i < this._stlFiles.length; i++) {
            const f = this._stlFiles[i];
            if (!f.url) continue;
            this._solidIndexMap.set(f.node_id, items.length);
            items.push({
                url: f.url,
                color: 0xaaaaaa,
                opacity: 0.6,
                label: f.name,
            });
        }

        // Build child-to-compound index mapping from the analysis tree.
        // The compound enumerates solids depth-first in label order, matching
        // the tree's leaf order — so flat index N maps to the Nth leaf solid.
        this._buildChildToCompoundMap();

        if (items.length === 0) {
            this._setViewerOverlay('No solids to display');
            return;
        }

        this._setViewerOverlay('Loading meshes...');
        try {
            const viewer = this._ensureViewer();
            await viewer.loadScene(items);
            this._setViewerOverlay(null);
        } catch (err) {
            this._setViewerOverlay(`Load failed: ${err.message}`);
        }
    }

    _buildChildToCompoundMap() {
        this._childToCompoundIdx.clear();
        if (!this._selectedNodeId || !this._treeNodes.length) return;

        // Find the selected node in the tree
        const node = this._findTreeNode(this._treeNodes, this._selectedNodeId);
        if (!node) return;

        // Walk depth-first: leaf solid_count determines how many compound
        // indices belong to each leaf. Compound entries use "parentId:sN".
        let flatIdx = 0;
        const walk = (n) => {
            if (n.children && n.children.length > 0) {
                for (const child of n.children) walk(child);
            } else {
                const count = n.solid_count || 0;
                for (let s = 0; s < count; s++) {
                    const childKey = `${n.id}:s${s}`;
                    this._childToCompoundIdx.set(childKey, flatIdx);
                    flatIdx++;
                }
            }
        };
        walk(node);
    }

    _findTreeNode(nodes, targetId) {
        for (const n of nodes) {
            if (n.id === targetId) return n;
            if (n.children) {
                const found = this._findTreeNode(n.children, targetId);
                if (found) return found;
            }
        }
        return null;
    }

    // ---------------------------------------------------------------
    // Highlight a connection in the viewer
    // ---------------------------------------------------------------

    _highlightConnection(connIndex) {
        const conn = this._connections[connIndex];
        if (!conn) return;

        this._activeConnIdx = connIndex;

        // Highlight the active row in the table
        const rows = this.container.querySelectorAll('#conn-results tbody tr');
        rows.forEach((tr, i) => {
            if (i === connIndex) {
                tr.style.background = '#dbeafe';
                tr.style.outline = '2px solid #3b82f6';
                tr.style.outlineOffset = '-2px';
            } else {
                tr.style.background = '';
                tr.style.outline = '';
                tr.style.outlineOffset = '';
            }
        });

        if (!this._viewer) return;

        const nodeA = conn.solid_a?.node_id;
        const nodeB = conn.solid_b?.node_id;
        const solidKeyA = `${nodeA}:s${conn.solid_a?.solid_index ?? 0}`;
        const solidKeyB = `${nodeB}:s${conn.solid_b?.solid_index ?? 0}`;

        // Reset all compound meshes to neutral grey / semi-transparent
        const meshCount = this._stlFiles.filter(f => f.url).length;
        for (let i = 0; i < meshCount; i++) {
            this._viewer.setMeshColor(i, 0xaaaaaa, 0.25);
        }

        // Use the tree-based child→compound mapping to find the mesh indices.
        // Fall back to _solidIndexMap for the "(All parts)" case where the
        // compound map is not built (no single selected node).
        const idxA = this._childToCompoundIdx.get(solidKeyA) ?? this._solidIndexMap.get(solidKeyA);
        if (idxA != null) {
            this._viewer.setMeshColor(idxA, 0x4a90d9, 1.0);  // blue
        }

        const idxB = this._childToCompoundIdx.get(solidKeyB) ?? this._solidIndexMap.get(solidKeyB);
        if (idxB != null) {
            this._viewer.setMeshColor(idxB, 0xe8860c, 1.0);  // orange
        }

        // Draw weld paths
        this._viewer.clearLines();
        if (conn.weld_paths && conn.weld_paths.length > 0) {
            this._viewer.addLines(conn.weld_paths, 0xff3333);
        }
    }

    // ---------------------------------------------------------------
    // Load cached results
    // ---------------------------------------------------------------

    async _loadCached(filename, nodeId = null, scope = null) {
        try {
            // Pass nodeId as-is: null = no node selected (fetch latest), "" = whole-file scope
            const data = await this.api.getConnectionResult(filename, nodeId, scope);
            // Only render if results match the current selection
            if (nodeId && data.node_id && data.node_id !== nodeId) {
                this._hideResults();
                return false;
            }
            // Sync scope dropdown if backend returned a different scope
            if (data.scope && data.scope !== this._scope) {
                this._scope = data.scope;
                const scopeSelect = this.container.querySelector('#conn-scope-select');
                if (scopeSelect) scopeSelect.value = data.scope;
            }
            this._renderResults(data);
            return (data.connections?.length > 0);
        } catch {
            this._hideResults();
            return false;
        }
    }

    // ---------------------------------------------------------------
    // Detection
    // ---------------------------------------------------------------

    async _startDetection(force = false) {
        // If cached results are already displayed, skip re-detection
        if (!force && this._connections.length > 0) {
            this._showStatus('Connection results already loaded from cache.');
            setTimeout(() => this._hideStatus(), 3000);
            return;
        }

        const btn = this.container.querySelector('#conn-detect-btn');
        btn.disabled = true;
        this._showStatus('Starting connection detection...');
        this._hideResults();

        try {
            const resp = await this.api.startConnectionDetection(
                this._currentFilename, this._selectedNodeId, this._scope,
            );
            this._taskId = resp.task_id;
            this._pollTimer = setInterval(() => this._pollStatus(), 2000);
        } catch (err) {
            const msg = err.message || err.detail?.error || err.error || 'Unknown error';
            this._showStatus(`Error: ${msg}`, true);
            btn.disabled = false;
        }
    }

    async _pollStatus() {
        if (!this._taskId) return;

        try {
            const resp = await this.api.getConnectionStatus(this._taskId);

            if (resp.status === 'completed') {
                // Only clear the poll timer — do NOT dispose the viewer
                if (this._pollTimer) {
                    clearInterval(this._pollTimer);
                    this._pollTimer = null;
                }
                this._hideStatus();
                this._renderResults(resp.results);
                this.container.querySelector('#conn-detect-btn').disabled = false;
                // Refresh tree badges
                this._refreshTreeStatus();
            } else if (resp.status === 'failed') {
                if (this._pollTimer) {
                    clearInterval(this._pollTimer);
                    this._pollTimer = null;
                }
                this._showStatus(`Failed: ${resp.error}`, true);
                this.container.querySelector('#conn-detect-btn').disabled = false;
            } else {
                this._showStatus('Detecting connections... (this may take a while)');
            }
        } catch {
            // Network blip — keep polling
        }
    }

    // ---------------------------------------------------------------
    // Status display
    // ---------------------------------------------------------------

    _showStatus(msg, isError = false) {
        const el = this.container.querySelector('#conn-status');
        const text = el.querySelector('.conn-status-text');
        const progress = el.querySelector('#conn-progress');
        el.style.display = 'flex';
        el.style.gap = '0.5rem';
        el.style.alignItems = 'center';
        text.textContent = msg;
        text.style.color = isError ? 'var(--pico-color-red-500, #dc2626)' : '';
        progress.style.display = isError ? 'none' : '';
    }

    _hideStatus() {
        this.container.querySelector('#conn-status').style.display = 'none';
    }

    _hideResults() {
        this._connections = [];
        this.container.querySelector('#conn-summary').style.display = 'none';
        this.container.querySelector('#conn-results').style.display = 'none';
    }

    // ---------------------------------------------------------------
    // Render results
    // ---------------------------------------------------------------

    _renderResults(data) {
        const summary = data.summary || {};
        const timing = data.timing || {};
        const connections = data.connections || [];

        // Store for viewer highlighting and re-renders
        this._connections = connections;
        this._lastSummary = summary;
        this._lastTiming = timing;

        // Summary bar
        const summaryEl = this.container.querySelector('#conn-summary');
        const welded = summary.welded_connections || 0;
        const bolted = summary.bolted_connections || 0;
        const unclass = summary.unclassified_contacts || 0;
        const weldLen = summary.total_weld_length_mm || 0;
        const boltCount = summary.total_bolt_count || 0;
        const totalMs = timing.total_ms || 0;

        summaryEl.innerHTML = `
            <div style="
                background: var(--pico-card-background-color, #f8fafc);
                border: 1px solid var(--pico-muted-border-color, #e2e8f0);
                border-radius: 4px;
                padding: 0.3rem 0.5rem;
                display: flex; gap: 0.8rem; flex-wrap: wrap; align-items: center;
                font-size: 0.75rem;
            ">
                <span><strong>${summary.total_solids || 0}</strong> solids</span>
                <span><strong>${summary.candidate_pairs || 0}</strong> pairs</span>
                <span style="color: #d97706;"><strong>${welded}</strong> welded (${(weldLen / 1000).toFixed(1)}m)</span>
                <span style="color: #2563eb;"><strong>${bolted}</strong> bolted (${boltCount})</span>
                ${unclass ? `<span style="color: #6b7280;"><strong>${unclass}</strong> unclass.</span>` : ''}
                <span style="color: #9ca3af;">${(totalMs / 1000).toFixed(1)}s</span>
            </div>
        `;
        summaryEl.style.display = 'block';

        // Results table
        const resultsEl = this.container.querySelector('#conn-results');
        if (connections.length === 0) {
            resultsEl.innerHTML = '<p><em>No connections detected.</em></p>';
            resultsEl.style.display = 'block';
            return;
        }

        const rows = connections.map((c, i) => {
            const a = c.solid_a || {};
            const b = c.solid_b || {};
            const nameA = a.name + (a.solid_index > 0 ? ` [s${a.solid_index}]` : '');
            const nameB = b.name + (b.solid_index > 0 ? ` [s${b.solid_index}]` : '');

            const displayType = this._effectiveType(c);
            const typeBadge = this._typeBadge(displayType, c);
            const detail = this._typeDetail(displayType, c);
            const vState = this._verificationState(c);

            return `
                <tr data-conn-idx="${i}" style="cursor:pointer;${vState.rowStyle}" title="Click to highlight in viewer">
                    <td style="text-align:right; color:#9ca3af; padding:1px 3px;">${i + 1}</td>
                    <td style="padding:1px 3px; max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${a.node_id}&#10;${nameA}">${nameA}</td>
                    <td style="padding:1px 3px; max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${b.node_id}&#10;${nameB}">${nameB}</td>
                    <td style="padding:1px 3px;">${typeBadge}</td>
                    <td style="padding:1px 3px;">${detail}</td>
                    <td style="white-space:nowrap; padding:1px 3px;">
                        <select class="conn-reclassify" data-conn-idx="${i}"
                                style="padding:1px 2px;font-size:0.7rem;border:1px solid #d1d5db;border-radius:3px;cursor:pointer;"
                                title="Change type">
                            <option value="welded" ${displayType === 'welded' ? 'selected' : ''}>W</option>
                            <option value="bolted" ${displayType === 'bolted' ? 'selected' : ''}>B</option>
                            <option value="contact" ${displayType === 'contact' ? 'selected' : ''}>C</option>
                        </select>
                        <button class="conn-verify-btn" data-action="accept" data-conn-idx="${i}"
                                title="Verify" style="padding:1px 4px;font-size:0.7rem;cursor:pointer;background:none;border:1px solid #d1d5db;border-radius:3px;${vState.verifyActive}">&#10003;</button>
                    </td>
                </tr>
            `;
        }).join('');

        resultsEl.innerHTML = `
            <div style="max-height: calc(100vh - 520px); overflow-y: auto; border: 1px solid var(--pico-muted-border-color, #e2e8f0); border-radius: 4px;">
                <table role="grid" style="margin:0; font-size:0.73rem;">
                    <thead>
                        <tr>
                            <th style="width:28px; padding:2px 4px;">#</th>
                            <th style="padding:2px 4px;">Part A</th>
                            <th style="padding:2px 4px;">Part B</th>
                            <th style="width:60px; padding:2px 4px;">Type</th>
                            <th style="padding:2px 4px;">Detail</th>
                            <th style="width:100px; padding:2px 4px;">Verify</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
        resultsEl.style.display = 'block';

        // Enable export button
        const exportBtn = this.container.querySelector('#conn-export-btn');
        if (exportBtn) exportBtn.disabled = false;

        this._bindResultEvents(resultsEl);
    }

    // ---------------------------------------------------------------
    // Verification helpers
    // ---------------------------------------------------------------

    _effectiveType(conn) {
        const v = conn.verification;
        if (v && v.action === 'reclassify' && v.new_type) return v.new_type;
        return conn.type;
    }

    _typeBadge(type, conn) {
        const reclassified = conn.verification?.action === 'reclassify';
        const prefix = reclassified ? '<span title="Reclassified" style="font-size:0.7rem;">*</span>' : '';
        if (type === 'welded') {
            return prefix + '<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;font-size:0.8rem;">Welded</span>';
        } else if (type === 'bolted') {
            return prefix + '<span style="background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:4px;font-size:0.8rem;">Bolted</span>';
        }
        return prefix + '<span style="background:#f3f4f6;color:#6b7280;padding:2px 8px;border-radius:4px;font-size:0.8rem;">Contact</span>';
    }

    _typeDetail(type, conn) {
        if (type === 'welded' && conn.weld_length_mm) {
            const auto = conn.auto_classified ? ' <span style="color:#6b7280;font-size:0.75rem;">(auto)</span>' : '';
            const faces = conn.contact_faces ? ` (${conn.contact_faces} face${conn.contact_faces > 1 ? 's' : ''})` : '';
            return `${conn.weld_length_mm.toFixed(1)}mm weld${faces}${auto}`;
        } else if (type === 'bolted' && conn.bolt_count) {
            return `${conn.bolt_count}x &empty;${conn.bolt_diameter_mm}mm`;
        }
        const gap = conn.min_distance_mm != null ? `${conn.min_distance_mm}mm gap` : '';
        const subtype = conn.contact_subtype ? ` <span style="color:#9ca3af;font-size:0.75rem;">${conn.contact_subtype}</span>` : '';
        return (gap || '\u2014') + subtype;
    }

    _verificationState(conn) {
        const v = conn.verification;
        if (!v) {
            return { html: '', rowStyle: '', verifyActive: '' };
        }
        // Any verification action (accept or reclassify) = verified
        return {
            html: '',
            rowStyle: 'background:rgba(34,197,94,0.08);',
            verifyActive: 'background:#dcfce7;border-color:#22c55e;color:#16a34a;',
        };
    }

    _bindResultEvents(resultsEl) {
        // Remove any previous delegated handlers
        if (this._onResultsClick) {
            resultsEl.removeEventListener('click', this._onResultsClick);
        }
        if (this._onResultsChange) {
            resultsEl.removeEventListener('change', this._onResultsChange);
        }

        // Single delegated click handler for rows + verify buttons
        this._onResultsClick = (e) => {
            // Accept/reject button
            const btn = e.target.closest('.conn-verify-btn');
            if (btn) {
                e.stopPropagation();
                const idx = parseInt(btn.dataset.connIdx, 10);
                const action = btn.dataset.action;
                this._verifyConnection(idx, action);
                return;
            }

            // Skip if clicking a select
            if (e.target.closest('.conn-reclassify')) return;

            // Row click → highlight
            const tr = e.target.closest('tr[data-conn-idx]');
            if (tr) {
                const idx = parseInt(tr.dataset.connIdx, 10);
                this._highlightConnection(idx);
            }
        };

        // Single delegated change handler for type dropdowns
        this._onResultsChange = (e) => {
            const sel = e.target.closest('.conn-reclassify');
            if (!sel) return;
            e.stopPropagation();
            const idx = parseInt(sel.dataset.connIdx, 10);
            const newType = sel.value;
            if (!newType) return;
            const conn = this._connections[idx];
            if (!conn) return;
            // If selecting the original type, just verify as-is
            if (newType === conn.type) {
                this._verifyConnection(idx, 'accept');
            } else {
                this._verifyConnection(idx, 'reclassify', newType);
            }
        };

        resultsEl.addEventListener('click', this._onResultsClick);
        resultsEl.addEventListener('change', this._onResultsChange);
    }

    _verifyConnection(connIdx, action, newType = null) {
        const conn = this._connections[connIdx];
        if (!conn || !conn.id) return;

        if (action === 'reclassify' && newType) {
            // Changing type auto-verifies
            conn.verification = { action: 'reclassify', new_type: newType };
        } else {
            // Accept: preserve any existing type override
            const existing = conn.verification;
            if (existing && existing.action === 'reclassify' && existing.new_type) {
                // Already reclassified — keep the override, just ensure verified
                conn.verification = { action: 'reclassify', new_type: existing.new_type };
            } else {
                conn.verification = { action: 'accept' };
            }
        }

        // Update only the affected row (no full re-render)
        this._updateRowVerification(connIdx);

        // Fire API call in background
        this.api.verifyConnection(
            this._currentFilename, conn.id,
            conn.verification.action,
            conn.verification.new_type || null,
        ).catch(err => {
            console.error('Verification failed:', err);
        });
    }

    _updateRowVerification(connIdx) {
        const conn = this._connections[connIdx];
        if (!conn) return;

        const tr = this.container.querySelector(`tr[data-conn-idx="${connIdx}"]`);
        if (!tr) return;

        const displayType = this._effectiveType(conn);
        const typeBadge = this._typeBadge(displayType, conn);
        const detail = this._typeDetail(displayType, conn);
        const vState = this._verificationState(conn);

        // Update row style
        tr.style.background = '';
        tr.style.opacity = '';
        if (vState.rowStyle) {
            // Parse inline style pairs from rowStyle string
            vState.rowStyle.split(';').forEach(s => {
                const [k, v] = s.split(':').map(x => x.trim());
                if (k && v) tr.style[k] = v;
            });
        }

        const cells = tr.children;
        // cells: [0]=#, [1]=PartA, [2]=PartB, [3]=Type, [4]=Detail, [5]=Verify
        if (cells[3]) cells[3].innerHTML = typeBadge;
        if (cells[4]) cells[4].innerHTML = detail;
        if (cells[5]) {
            const verifyStyle = `padding:1px 4px;font-size:0.7rem;cursor:pointer;background:none;border:1px solid #d1d5db;border-radius:3px;${vState.verifyActive}`;
            cells[5].innerHTML = `
                <select class="conn-reclassify" data-conn-idx="${connIdx}"
                        style="padding:1px 2px;font-size:0.7rem;border:1px solid #d1d5db;border-radius:3px;cursor:pointer;"
                        title="Change type">
                    <option value="welded" ${displayType === 'welded' ? 'selected' : ''}>W</option>
                    <option value="bolted" ${displayType === 'bolted' ? 'selected' : ''}>B</option>
                    <option value="contact" ${displayType === 'contact' ? 'selected' : ''}>C</option>
                </select>
                <button class="conn-verify-btn" data-action="accept" data-conn-idx="${connIdx}"
                        title="Verify" style="${verifyStyle}">&#10003;</button>
            `;
        }
    }

    // ---------------------------------------------------------------
    // Batch detection
    // ---------------------------------------------------------------

    async _startBatchDetection(force = false) {
        const btn = this.container.querySelector('#conn-detect-all-btn');
        btn.disabled = true;
        this._showStatus(force ? 'Re-detecting all connections...' : 'Planning batch detection...');

        try {
            const opts = { force };
            if (this._selectedNodeId) opts.nodeId = this._selectedNodeId;
            const resp = await this.api.startBatchDetection(this._currentFilename, opts);

            if (resp.status === 'completed' && resp.planned_units === 0) {
                this._showStatus('All connections already detected — select a node to view results.', false);
                btn.disabled = false;
                return;
            }

            const extra = resp.total_available > resp.planned_units
                ? ` (${resp.planned_units} of ${resp.total_available} units)` : '';
            this._batchTaskId = resp.task_id;
            this._showBatchProgress({
                completed: 0, total: resp.planned_units, current_unit: `Starting...${extra}`, percent: 0,
            });

            this._batchPollTimer = setInterval(() => this._pollBatchStatus(), 2000);
        } catch (err) {
            const msg = err.message || err.detail?.error || err.error || 'Unknown error';
            this._showStatus(`Batch error: ${msg}`, true);
            btn.disabled = false;
        }
    }

    async _pollBatchStatus() {
        if (!this._batchTaskId) return;

        try {
            const resp = await this.api.getBatchDetectionStatus(this._batchTaskId);

            if (resp.status === 'completed') {
                clearInterval(this._batchPollTimer);
                this._batchPollTimer = null;

                const completed = resp.completed_units?.length || 0;
                const failed = resp.failed_units?.length || 0;
                const msg = `Batch complete: ${completed} detected` +
                    (failed ? `, ${failed} failed` : '');
                this._showStatus(msg, false);
                this.container.querySelector('#conn-detect-all-btn').disabled = false;

                // Refresh tree status badges
                this._refreshTreeStatus();
            } else if (resp.status === 'failed') {
                clearInterval(this._batchPollTimer);
                this._batchPollTimer = null;
                this._showStatus(`Batch failed: ${resp.error}`, true);
                this.container.querySelector('#conn-detect-all-btn').disabled = false;
            } else {
                this._showBatchProgress(resp.progress || {});
                // Periodically refresh tree badges during batch
                this._refreshTreeStatus();
            }
        } catch {
            // Network blip — keep polling
        }
    }

    _showBatchProgress(progress) {
        const el = this.container.querySelector('#conn-status');
        const text = el.querySelector('.conn-status-text');
        const progressBar = el.querySelector('#conn-progress');
        el.style.display = 'flex';
        el.style.gap = '0.5rem';
        el.style.alignItems = 'center';

        const completed = progress.completed || 0;
        const total = progress.total || 0;
        const current = progress.current_unit || '';
        text.textContent = `Batch: ${completed}/${total} — ${current}`;
        text.style.color = '';
        progressBar.style.display = '';
        if (total > 0) {
            progressBar.value = completed;
            progressBar.max = total;
        }
    }

    // ---------------------------------------------------------------
    // Tree status badges
    // ---------------------------------------------------------------

    async _refreshTreeStatus() {
        if (!this._currentFilename) return;

        try {
            const data = await this.api.getConnectionTreeStatus(this._currentFilename);
            this._treeStatus = data.node_status || {};
            this._applyTreeBadges();
        } catch {
            // Not critical — badges just won't update
        }
    }

    _applyTreeBadges() {
        if (!this._treeStatus) return;
        const panel = this.container.querySelector('#conn-tree-panel');
        if (!panel) return;

        panel.querySelectorAll('.conn-tree-row').forEach(row => {
            const nodeId = row.dataset.nodeId;
            if (!nodeId) return;

            // Remove existing badge
            const existing = row.querySelector('.conn-status-badge');
            if (existing) existing.remove();

            // Find status for this node (check all scopes)
            let bestStatus = null;
            let totalConns = 0;
            let verifiedConns = 0;
            for (const [key, info] of Object.entries(this._treeStatus)) {
                if (info.node_id !== nodeId) continue;
                const counts = info.counts || {};
                totalConns += counts.total || 0;
                verifiedConns += (counts.accepted || 0) + (counts.rejected || 0);
                if (!bestStatus || _statusPriority(info.verification) > _statusPriority(bestStatus)) {
                    bestStatus = info.verification;
                }
            }

            if (!bestStatus || bestStatus === 'none') return;

            const badge = document.createElement('span');
            badge.className = 'conn-status-badge';

            if (bestStatus === 'complete') {
                badge.dataset.status = 'detected-complete';
                badge.textContent = ` \u2713 ${totalConns}`;
                badge.title = `All ${totalConns} connections verified`;
            } else if (bestStatus === 'partial') {
                badge.dataset.status = 'detected-partial';
                badge.textContent = ` ${verifiedConns}/${totalConns}`;
                badge.title = `${verifiedConns} of ${totalConns} verified`;
            } else {
                badge.dataset.status = 'detected-unverified';
                badge.textContent = ` \u25CF ${totalConns}`;
                badge.title = `${totalConns} connections — not yet verified`;
            }

            row.appendChild(badge);
        });
    }

}


function _statusPriority(status) {
    switch (status) {
        case 'complete': return 3;
        case 'partial': return 2;
        case 'unverified': return 1;
        default: return 0;
    }
}
