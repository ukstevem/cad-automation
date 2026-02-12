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

    async _handleError(res) {
        try {
            const data = await res.json();
            return { status: res.status, ...data };
        } catch {
            return { status: res.status, message: res.statusText };
        }
    }
}
