/**
 * Reusable UI component factories
 */

export function createStatusBadge(passed, label) {
    const cls = passed ? 'badge pass' : 'badge fail';
    const icon = passed ? '\u2713' : '\u2717';
    return `<span class="${cls}">${icon} ${label}</span>`;
}

export function createCheckRow(label, passed, detail) {
    const icon = passed ? '\u2713' : '\u2717';
    const cls = passed ? 'check-pass' : 'check-fail';
    return `
        <div class="check-row ${cls}">
            <span class="check-icon">${icon}</span>
            <span class="check-label">${label}</span>
            ${detail ? `<span class="check-detail">${detail}</span>` : ''}
        </div>
    `;
}

export function createResultCard(title, content, status) {
    const statusClass = status === 'pass' ? 'pass' : status === 'fail' ? 'fail' : 'neutral';
    return `
        <article class="result-card ${statusClass}">
            <header>
                <strong>${title}</strong>
            </header>
            <div class="result-card-body">
                ${content}
            </div>
        </article>
    `;
}

export function createVerdictBanner(passed, subtitle) {
    const cls = passed ? 'verdict-pass' : 'verdict-fail';
    const icon = passed ? '\u2713' : '\u2717';
    const label = passed ? 'WORKABLE' : 'NOT WORKABLE';
    return `
        <div class="verdict-banner ${cls}">
            <span class="verdict-icon">${icon}</span>
            <div>
                <div class="verdict-label">${label}</div>
                <div class="verdict-subtitle">${subtitle}</div>
            </div>
        </div>
    `;
}
