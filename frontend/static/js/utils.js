/**
 * Shared utility functions
 */

export function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(i > 0 ? 2 : 0)} ${units[i]}`;
}

export function formatDimension(value, decimals = 2) {
    if (value == null || isNaN(value)) return '-';
    return Number(value).toFixed(decimals);
}

export function formatVolume(value) {
    if (value == null || isNaN(value) || value === 0) return '-';
    if (value > 1e6) return `${(value / 1e6).toFixed(2)} cm\u00B3`;
    return `${formatDimension(value)} mm\u00B3`;
}

export function formatArea(value) {
    if (value == null || isNaN(value) || value === 0) return '-';
    if (value > 1e6) return `${(value / 1e6).toFixed(2)} cm\u00B2`;
    return `${formatDimension(value)} mm\u00B2`;
}

export function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

export function pluralize(count, singular, plural) {
    return count === 1 ? singular : (plural || singular + 's');
}
