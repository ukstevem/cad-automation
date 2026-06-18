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

    /** Lightweight refined-class for a set of refs (no NC1/DXF) — drives triage. */
    async classifyParts(filename, refIds, memberIds = {}, steelGrade = '') {
        return this._postJson(
            `/api/v1/cnc-analysis/classify/${encodeURIComponent(filename)}`,
            { ref_ids: refIds, member_ids: memberIds, steel_grade: steelGrade },
        );
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

    // ── Camera calibration (ChArUco) ──────────────────────────────

    async getBoardDefaults() {
        return this._get('/api/v1/calibration/board-defaults');
    }

    /** URL for the printable ChArUco board PNG (board params as query string). */
    getBoardImageUrl(board) {
        const p = new URLSearchParams({
            squares_x: board.squares_x,
            squares_y: board.squares_y,
            square_mm: board.square_mm,
            marker_mm: board.marker_mm,
            dictionary: board.dictionary,
        });
        return `${this.baseUrl}/api/v1/calibration/board?${p.toString()}`;
    }

    /** Single-frame detection feedback. `frameBlob` is a Blob/File of a JPEG/PNG. */
    async detectBoard(frameBlob, board) {
        const fd = new FormData();
        fd.append('frame', frameBlob, 'frame.jpg');
        this._appendBoard(fd, board);
        return this._postForm('/api/v1/calibration/detect', fd);
    }

    /** Run calibration over many frames. `frames` is an array of Blob/File. */
    async computeCalibration(frames, name, board, cameraLabel = '') {
        const fd = new FormData();
        frames.forEach((f, i) => fd.append('frames', f, f.name || `frame_${i}.jpg`));
        fd.append('name', name);
        fd.append('camera_label', cameraLabel);
        this._appendBoard(fd, board);
        return this._postForm('/api/v1/calibration/compute', fd);
    }

    async listCalibrationProfiles() {
        return this._get('/api/v1/calibration/profiles');
    }

    async getCalibrationProfile(name) {
        return this._get(`/api/v1/calibration/profiles/${encodeURIComponent(name)}`);
    }

    async deleteCalibrationProfile(name) {
        return this._delete(`/api/v1/calibration/profiles/${encodeURIComponent(name)}`);
    }

    _appendBoard(fd, board) {
        fd.append('squares_x', board.squares_x);
        fd.append('squares_y', board.squares_y);
        fd.append('square_mm', board.square_mm);
        fd.append('marker_mm', board.marker_mm);
        fd.append('dictionary', board.dictionary);
    }

    // ── AR alignment (Phase 0) ────────────────────────────────────

    async getArGeometry(filename, nodeId = null) {
        let url = `/api/v1/ar/geometry/${encodeURIComponent(filename)}`;
        if (nodeId) url += `?node_id=${encodeURIComponent(nodeId)}`;
        return this._get(url);
    }

    async solveAr(filename, profile, correspondences, nodeId = null, ransac = false) {
        return this._postJson(`/api/v1/ar/solve/${encodeURIComponent(filename)}`, {
            profile, correspondences, node_id: nodeId, ransac,
        });
    }

    async detectMarkers(filename, photoBlob, { profile, dictionary = 'DICT_APRILTAG_36h11', markerSizeMm = 100 }) {
        const fd = new FormData();
        fd.append('photo', photoBlob, photoBlob.name || 'photo.jpg');
        fd.append('profile', profile);
        fd.append('dictionary', dictionary);
        fd.append('marker_size_mm', String(markerSizeMm));
        return this._postForm(`/api/v1/ar/detect-markers/${encodeURIComponent(filename)}`, fd);
    }

    async registerMarker(filename, photoBlob, { profile, modelRvec, modelTvec, dictionary = 'DICT_APRILTAG_36h11', markerSizeMm = 100, markerId = -1 }) {
        const fd = new FormData();
        fd.append('photo', photoBlob, photoBlob.name || 'photo.jpg');
        fd.append('profile', profile);
        fd.append('model_rvec', JSON.stringify(modelRvec));
        fd.append('model_tvec', JSON.stringify(modelTvec));
        fd.append('dictionary', dictionary);
        fd.append('marker_size_mm', String(markerSizeMm));
        fd.append('marker_id', String(markerId));
        return this._postForm(`/api/v1/ar/register-marker/${encodeURIComponent(filename)}`, fd);
    }

    async calibrateConstellation(filename, photoBlobs, { profile, dictionary = 'DICT_APRILTAG_36h11', markerSizeMm = 100 }) {
        const fd = new FormData();
        photoBlobs.forEach((b, i) => fd.append('photos', b, b.name || `p${i}.jpg`));
        fd.append('profile', profile);
        fd.append('dictionary', dictionary);
        fd.append('marker_size_mm', String(markerSizeMm));
        return this._postForm(`/api/v1/ar/calibrate-constellation/${encodeURIComponent(filename)}`, fd);
    }

    async anchorConstellation(filename, photoBlob, { profile, modelRvec, modelTvec, dictionary = 'DICT_APRILTAG_36h11', markerSizeMm = 100 }) {
        const fd = new FormData();
        fd.append('photo', photoBlob, photoBlob.name || 'photo.jpg');
        fd.append('profile', profile);
        fd.append('model_rvec', JSON.stringify(modelRvec));
        fd.append('model_tvec', JSON.stringify(modelTvec));
        fd.append('dictionary', dictionary);
        fd.append('marker_size_mm', String(markerSizeMm));
        return this._postForm(`/api/v1/ar/anchor-constellation/${encodeURIComponent(filename)}`, fd);
    }

    async autoSolve(filename, photoBlob, { profile, dictionary = 'DICT_APRILTAG_36h11', nodeId = '' }) {
        const fd = new FormData();
        fd.append('photo', photoBlob, photoBlob.name || 'photo.jpg');
        fd.append('profile', profile);
        fd.append('dictionary', dictionary);
        fd.append('node_id', nodeId || '');
        return this._postForm(`/api/v1/ar/auto-solve/${encodeURIComponent(filename)}`, fd);
    }

    async saveCapture(filename, photoBlob, { profile, correspondences, pose, reprojRms, nodeId = '' }) {
        const fd = new FormData();
        fd.append('photo', photoBlob, photoBlob.name || 'photo.jpg');
        fd.append('profile', profile);
        fd.append('correspondences', JSON.stringify(correspondences));
        fd.append('pose', JSON.stringify(pose));
        fd.append('reproj_rms', String(reprojRms));
        fd.append('node_id', nodeId || '');
        return this._postForm(`/api/v1/ar/capture/${encodeURIComponent(filename)}`, fd);
    }

    async _postForm(path, formData) {
        const res = await fetch(`${this.baseUrl}${path}`, { method: 'POST', body: formData });
        if (!res.ok) throw await this._handleError(res);
        return res.json();
    }

    async _get(path) {
        // no-store: API responses are dynamic — never serve a stale cached copy
        // (a cached /analysis/assembly response once hid the newly-added
        // classification field even after a hard refresh).
        const res = await fetch(`${this.baseUrl}${path}`, { cache: 'no-store' });
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
