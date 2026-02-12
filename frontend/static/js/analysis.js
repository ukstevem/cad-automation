/**
 * Analysis page - assembly tree viewer and node classification
 */

export class AnalysisPage {
    constructor(api) {
        this.api = api;
        this.container = null;
        /** @type {Map<string, string>} node id -> classification */
        this.classifications = new Map();
    }

    render(container) {
        this.container = container;
        container.innerHTML = this._template();
        this._bindEvents();
        this._loadFiles();
    }

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
                    <p aria-busy="true">Analysing assembly structure... <span id="elapsed-timer"></span></p>
                </div>
            </section>

            <section id="tree-results" hidden>
                <div id="analysis-summary"></div>
                <div id="assembly-tree-container" class="assembly-tree"></div>
            </section>
        `;
    }

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

    async _analyze(filename) {
        const errorEl = this.container.querySelector('#analysis-error');
        const loadingEl = this.container.querySelector('#analysis-loading');
        const resultsEl = this.container.querySelector('#tree-results');
        const btn = this.container.querySelector('#analyze-btn');

        errorEl.hidden = true;
        resultsEl.hidden = true;
        loadingEl.hidden = false;
        btn.disabled = true;
        this.classifications.clear();

        // Elapsed timer so user knows it's still working
        const timerEl = loadingEl.querySelector('#elapsed-timer');
        const start = Date.now();
        const timer = setInterval(() => {
            const secs = Math.floor((Date.now() - start) / 1000);
            if (timerEl) timerEl.textContent = `${secs}s elapsed`;
        }, 1000);

        try {
            const data = await this.api.getAssemblyTree(filename);
            clearInterval(timer);
            loadingEl.hidden = true;
            btn.disabled = false;
            this._renderTree(data);
        } catch (err) {
            clearInterval(timer);
            loadingEl.hidden = true;
            btn.disabled = false;
            const msg = err?.detail?.message || err?.message || 'Analysis failed.';
            errorEl.hidden = false;
            errorEl.innerHTML = `<article class="result-card fail"><header><strong>Error</strong></header><div class="result-card-body"><p>${this._esc(msg)}</p></div></article>`;
        }
    }

    _renderTree(data) {
        const resultsEl = this.container.querySelector('#tree-results');
        const summaryEl = this.container.querySelector('#analysis-summary');
        const treeEl = this.container.querySelector('#assembly-tree-container');

        resultsEl.hidden = false;

        // Summary
        const s = data.summary || {};
        summaryEl.innerHTML = `
            <div class="analysis-summary">
                <div class="summary-stat"><strong>${s.total_assemblies || 0}</strong> Assemblies</div>
                <div class="summary-stat"><strong>${s.total_parts || 0}</strong> Parts</div>
                <div class="summary-stat"><strong>${s.total_solids || 0}</strong> Solids</div>
            </div>
        `;

        // Tree
        const nodes = data.assembly_tree || [];
        treeEl.innerHTML = '<ul>' + nodes.map(n => this._renderNode(n)).join('') + '</ul>';

        // Bind tree interactions
        this._bindTreeEvents(treeEl);
    }

    _renderNode(node) {
        const hasChildren = node.children && node.children.length > 0;
        const toggleClass = hasChildren ? 'expanded' : 'leaf';
        const childrenHtml = hasChildren
            ? '<ul>' + node.children.map(c => this._renderNode(c)).join('') + '</ul>'
            : '';

        const badgeLabel = this._badgeLabel(node.node_type);
        const solidInfo = node.node_type !== 'assembly' && node.solid_count !== undefined
            ? `<span class="node-solid-count">${node.solid_count} solid${node.solid_count !== 1 ? 's' : ''}</span>`
            : '';

        const actions = this._actionsHtml(node);

        // Show instance ref as a subtle label when it differs from the name
        const instanceRef = node.instance_ref && node.instance_ref !== node.name
            ? `<span class="node-instance-ref" title="Instance reference">${this._esc(node.instance_ref)}</span>`
            : '';

        return `
            <li class="tree-node" data-node-id="${this._esc(node.id)}" data-node-type="${this._esc(node.node_type)}">
                <div class="tree-node-row">
                    <button class="tree-toggle ${toggleClass}" aria-label="Toggle">\u25B6</button>
                    <span class="tree-node-name">${this._esc(node.name)}</span>
                    ${instanceRef}
                    <span class="node-type-badge ${this._esc(node.node_type)}">${badgeLabel}</span>
                    ${solidInfo}
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
        };
        return labels[nodeType] || nodeType;
    }

    _actionsHtml(node) {
        const btns = [];
        switch (node.node_type) {
            case 'assembly':
                // drill-down is handled by expand/collapse toggle
                break;
            case 'part_multi_solid':
                btns.push('<button class="btn-explode" data-action="explode">Explode</button>');
                btns.push('<button class="btn-bought-out" data-action="bought-out">Bought-out</button>');
                break;
            case 'part_single_solid':
                btns.push('<button class="btn-cnc" data-action="cnc">CNC</button>');
                btns.push('<button class="btn-bought-out" data-action="bought-out">Bought-out</button>');
                break;
            case 'part_no_solid':
                btns.push('<button class="btn-bought-out" data-action="bought-out">Bought-out</button>');
                break;
        }
        return btns.length ? `<span class="node-actions">${btns.join('')}</span>` : '';
    }

    _bindTreeEvents(treeEl) {
        // Expand / collapse
        treeEl.addEventListener('click', (e) => {
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

            // Action buttons
            const actionBtn = e.target.closest('[data-action]');
            if (actionBtn) {
                const li = actionBtn.closest('.tree-node');
                const nodeId = li.dataset.nodeId;
                const action = actionBtn.dataset.action;
                this._classifyNode(li, nodeId, action);
            }
        });
    }

    _classifyNode(li, nodeId, action) {
        this.classifications.set(nodeId, action);
        li.classList.add('node-classified');
        li.dataset.classification = action;

        // Hide action buttons once classified
        const actions = li.querySelector('.node-actions');
        if (actions) actions.hidden = true;
    }

    _esc(str) {
        const el = document.createElement('span');
        el.textContent = str;
        return el.innerHTML;
    }
}
