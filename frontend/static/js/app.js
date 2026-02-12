/**
 * Application entry point - page router and health check
 */
import { ApiClient } from './api.js';
import { UploadPage } from './upload.js';
import { AnalysisPage } from './analysis.js';

const api = new ApiClient();

const pages = {
    upload: new UploadPage(api),
    analysis: new AnalysisPage(api),
};

class App {
    constructor() {
        this.contentEl = document.getElementById('app-content');
        this.statusEl = document.getElementById('api-status');
        this.versionEl = document.getElementById('api-version');
        this.init();
    }

    async init() {
        this._checkApiStatus();
        this._setupRouting();
        this._route();
    }

    _setupRouting() {
        window.addEventListener('hashchange', () => this._route());

        document.querySelectorAll('[data-page]').forEach(link => {
            link.addEventListener('click', (e) => {
                if (link.classList.contains('disabled')) {
                    e.preventDefault();
                    return;
                }
                e.preventDefault();
                window.location.hash = link.dataset.page;
            });
        });
    }

    _route() {
        const hash = window.location.hash.slice(1) || 'upload';
        const page = pages[hash];

        if (page) {
            this.contentEl.innerHTML = '';
            page.render(this.contentEl);
        }

        document.querySelectorAll('[data-page]').forEach(link => {
            link.classList.toggle('active', link.dataset.page === hash);
        });
    }

    async _checkApiStatus() {
        try {
            const health = await api.getHealth();
            this.statusEl.textContent = `API: ${health.status}`;
            this.statusEl.className = 'status-healthy';

            const info = await api.getServiceInfo();
            this.versionEl.textContent = `v${info.version}`;
        } catch {
            this.statusEl.textContent = 'API: unreachable';
            this.statusEl.className = 'status-error';
        }
    }
}

new App();
