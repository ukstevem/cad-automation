/**
 * API client - all server communication in one place
 */

export class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
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
