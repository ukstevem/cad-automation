/**
 * API client - all server communication in one place
 */

export class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        /** @type {string|null} cached nesting service base URL */
        this._nestingBase = null;
    }

    /**
     * Return the nesting service base URL (fetched once from /api/v1/config).
     */
    async getNestingBase() {
        if (!this._nestingBase) {
            const cfg = await this._get('/api/v1/config');
            this._nestingBase = cfg.nesting_base_url;
        }
        return this._nestingBase;
    }

    async getHealth() {
        return this._get('/health');
    }

    async getServiceInfo() {
        return this._get('/');
    }

    async uploadFile(file, parseGeometry = true, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);
        const url = `${this.baseUrl}/api/v1/upload/?parse_geometry=${parseGeometry}`;

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', url);

            if (onProgress) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        onProgress(Math.round((e.loaded / e.total) * 100));
                    }
                });
            }

            xhr.addEventListener('load', () => {
                try {
                    const data = JSON.parse(xhr.responseText);
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(data);
                    } else {
                        reject({ status: xhr.status, data });
                    }
                } catch {
                    reject({ status: xhr.status, message: 'Invalid response from server' });
                }
            });

            xhr.addEventListener('error', () => {
                reject({ message: 'Network error - could not reach server' });
            });

            xhr.send(formData);
        });
    }

    async validateFile(filename) {
        return this._get(`/api/v1/validate/${encodeURIComponent(filename)}`);
    }

    async parseFile(filename) {
        return this._post(`/api/v1/parse/${encodeURIComponent(filename)}`);
    }

    async listFiles() {
        return this._get('/api/v1/analysis/files');
    }

    async getAssemblyTree(filename) {
        return this._get(`/api/v1/analysis/assembly/${encodeURIComponent(filename)}`);
    }

    async getAnalysisStatus(taskId) {
        return this._get(`/api/v1/analysis/status/${encodeURIComponent(taskId)}`);
    }

    async generateSTL(filename) {
        return this._post(`/api/v1/stl/generate/${encodeURIComponent(filename)}`);
    }

    async getSTLStatus(taskId) {
        return this._get(`/api/v1/stl/status/${encodeURIComponent(taskId)}`);
    }

    async listSTLFiles(filename) {
        return this._get(`/api/v1/stl/files/${encodeURIComponent(filename)}`);
    }

    async generateSTLChildren(filename, parentId) {
        return this._post(
            `/api/v1/stl/generate-children/${encodeURIComponent(filename)}?parent_id=${encodeURIComponent(parentId)}`
        );
    }

    async generateSTLSolids(filename, nodeId) {
        return this._post(
            `/api/v1/stl/generate-solids/${encodeURIComponent(filename)}?node_id=${encodeURIComponent(nodeId)}`
        );
    }

    async getPartsList(filename) {
        return this._get(`/api/v1/analysis/parts-list/${encodeURIComponent(filename)}`);
    }

    async saveProjectState(filename, state) {
        return this._put(
            `/api/v1/analysis/project-state/${encodeURIComponent(filename)}`,
            state,
        );
    }

    async getConsolidation(filename) {
        return this._get(`/api/v1/analysis/consolidate/${encodeURIComponent(filename)}`);
    }

    async startConsolidation(filename) {
        return this._post(`/api/v1/analysis/consolidate/${encodeURIComponent(filename)}`);
    }

    async getConsolidationStatus(taskId) {
        return this._get(`/api/v1/analysis/consolidate-status/${encodeURIComponent(taskId)}`);
    }

    async getCncResult(filename) {
        return this._get(`/api/v1/cnc-analysis/result/${encodeURIComponent(filename)}`);
    }

    async getCncState(filename) {
        return this._get(`/api/v1/cnc-analysis/state/${encodeURIComponent(filename)}`);
    }

    async startCncAnalysis(filename, refIds, memberIds, parentNames = {}, projectNumber = '', steelGrade = '', force = false) {
        return this._postJson(
            `/api/v1/cnc-analysis/analyse/${encodeURIComponent(filename)}`,
            {
                ref_ids: refIds,
                member_ids: memberIds,
                parent_names: parentNames,
                project_number: projectNumber,
                steel_grade: steelGrade,
                force: force,
            },
        );
    }

    async getCncStatus(taskId) {
        return this._get(`/api/v1/cnc-analysis/status/${encodeURIComponent(taskId)}`);
    }

    // ── Projects ───────────────────────────────────────────────────

    async listProjects() {
        return this._get('/api/v1/projects/');
    }

    async createProject(projectNumber) {
        return this._postJson('/api/v1/projects/', { project_number: projectNumber });
    }

    async getProject(projectNumber) {
        return this._get(`/api/v1/projects/${encodeURIComponent(projectNumber)}`);
    }

    async updateProjectAnalyses(projectNumber, analyses) {
        return this._put(
            `/api/v1/projects/${encodeURIComponent(projectNumber)}/analyses`,
            { analyses },
        );
    }

    async deleteProject(projectNumber) {
        return this._delete(`/api/v1/projects/${encodeURIComponent(projectNumber)}`);
    }

    async getProjectNestingItems(projectNumber) {
        return this._get(`/api/v1/projects/${encodeURIComponent(projectNumber)}/nesting-items`);
    }

    getProjectNestingPdfUrl(projectNumber) {
        return `${this.baseUrl}/api/v1/projects/${encodeURIComponent(projectNumber)}/nesting-pdf`;
    }

    async updateProjectNestingTask(projectNumber, nestingTaskId, nestingStartedAt) {
        return this._put(
            `/api/v1/projects/${encodeURIComponent(projectNumber)}/nesting-task`,
            { nesting_task_id: nestingTaskId, nesting_started_at: nestingStartedAt },
        );
    }

    // ── Connection detection ──────────────────────────────────────

    async startConnectionDetection(filename, nodeId = null, scope = 'all') {
        const body = {};
        if (nodeId) body.node_id = nodeId;
        if (scope !== 'all') body.scope = scope;
        return this._postJson(
            `/api/v1/connections/detect/${encodeURIComponent(filename)}`,
            body,
        );
    }

    async getConnectionStatus(taskId) {
        return this._get(`/api/v1/connections/status/${encodeURIComponent(taskId)}`);
    }

    getConnectionExportXlsxUrl(filename, nodeId = null, scope = null) {
        let url = `/api/v1/connections/export-xlsx/${encodeURIComponent(filename)}`;
        const params = [];
        if (nodeId != null) params.push(`node_id=${encodeURIComponent(nodeId)}`);
        if (scope != null) params.push(`scope=${encodeURIComponent(scope)}`);
        if (params.length) url += '?' + params.join('&');
        return url;
    }

    async getConnectionResult(filename, nodeId = null, scope = null) {
        let url = `/api/v1/connections/result/${encodeURIComponent(filename)}`;
        const params = [];
        if (nodeId != null) params.push(`node_id=${encodeURIComponent(nodeId)}`);
        if (scope != null) params.push(`scope=${encodeURIComponent(scope)}`);
        if (params.length) url += '?' + params.join('&');
        return this._get(url);
    }

    async verifyConnection(filename, connectionId, action, newType = null) {
        const body = { connection_id: connectionId, action };
        if (newType) body.new_type = newType;
        return this._put(
            `/api/v1/connections/verify/${encodeURIComponent(filename)}`,
            body,
        );
    }

    getConnectionExportUrl(filename) {
        return `${this.baseUrl}/api/v1/connections/export/${encodeURIComponent(filename)}`;
    }

    async startBatchDetection(filename, { force = false, nodeId = null, maxUnits = 200 } = {}) {
        const body = { force, max_units: maxUnits };
        if (nodeId) body.node_id = nodeId;
        return this._postJson(
            `/api/v1/connections/detect-all/${encodeURIComponent(filename)}`,
            body,
        );
    }

    async getBatchDetectionStatus(taskId) {
        return this._get(`/api/v1/connections/detect-all-status/${encodeURIComponent(taskId)}`);
    }

    async getConnectionTreeStatus(filename) {
        return this._get(`/api/v1/connections/tree-status/${encodeURIComponent(filename)}`);
    }

    async startConnectionPreview(filename, nodeId) {
        return this._postJson(
            `/api/v1/connections/preview/${encodeURIComponent(filename)}`,
            { node_id: nodeId },
        );
    }

    async startConnectionPreviewBatch(filename, nodeIds) {
        return this._postJson(
            `/api/v1/connections/preview-batch/${encodeURIComponent(filename)}`,
            { node_ids: nodeIds },
        );
    }

    async getConnectionPreviewStatus(taskId) {
        return this._get(`/api/v1/connections/preview-status/${encodeURIComponent(taskId)}`);
    }

    async _get(path) {
        const res = await fetch(`${this.baseUrl}${path}`);
        if (!res.ok) throw await this._handleError(res);
        return res.json();
    }

    async _post(path) {
        const res = await fetch(`${this.baseUrl}${path}`, { method: 'POST' });
        if (!res.ok) throw await this._handleError(res);
        return res.json();
    }

    async _postJson(path, body) {
        const res = await fetch(`${this.baseUrl}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) throw await this._handleError(res);
        return res.json();
    }

    async _delete(path) {
        const res = await fetch(`${this.baseUrl}${path}`, { method: 'DELETE' });
        if (!res.ok) throw await this._handleError(res);
        return res.json();
    }

    async _put(path, body) {
        const res = await fetch(`${this.baseUrl}${path}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) throw await this._handleError(res);
        return res.json();
    }

    async _handleError(res) {
        try {
            const data = await res.json();
            return { status: res.status, ...data };
        } catch {
            return { status: res.status, message: res.statusText };
        }
    }
}
