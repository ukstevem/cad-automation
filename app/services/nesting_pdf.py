"""
Generate a PDF cutting-list report from nesting results.

Renders an HTML template that mirrors the on-screen layout (synopsis KPIs,
summary table, bar visuals, per-section cut tables) and converts it to PDF
via WeasyPrint.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone


def render_cutting_list_pdf(cutting_list: dict, project_number: str = "") -> bytes:
    """Return PDF bytes for the given cutting-list payload."""
    from weasyprint import HTML  # lazy import — heavy library

    markup = _build_html(cutting_list, project_number)
    return HTML(string=markup).write_pdf()


# ── Colour palette (matches frontend CSS) ────────────────────────

_PRIMARY = "#2563eb"
_GREEN = "#16a34a"
_AMBER = "#d97706"
_RED = "#dc2626"
_SLATE = "#64748b"
_BORDER = "#e2e8f0"
_SURFACE = "#f8fafc"

# Rotating hues for cut blocks so adjacent cuts are distinguishable
_CUT_COLOURS = [
    "#93c5fd", "#a5b4fc", "#86efac", "#fde68a",
    "#fca5a5", "#c4b5fd", "#67e8f9", "#fdba74",
]


def _esc(val) -> str:
    return html.escape(str(val)) if val else ""


def _build_html(cl: dict, project_number: str) -> str:
    totals = cl.get("totals", {})
    sections = cl.get("sections", [])
    job_label = cl.get("job_label", project_number) or project_number
    run_at = cl.get("run_at", "")

    # ── Overall utilisation ──
    total_stock_len = 0
    total_used_len = 0
    for sec in sections:
        for bar in sec.get("bars", []):
            total_stock_len += bar.get("stock_length_mm", 0)
            total_used_len += bar.get("used_length_mm", 0)
    overall_util = round(total_used_len / total_stock_len * 100) if total_stock_len else 0

    # ── Format timestamp ──
    run_date = ""
    if run_at:
        try:
            dt = datetime.fromisoformat(run_at)
            run_date = dt.strftime("%d %b %Y  %H:%M")
        except Exception:
            run_date = run_at

    now_str = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")

    # ── Synopsis table rows ──
    synopsis_rows = ""
    for sec in sections:
        summ = sec.get("summary", {})
        sec_stock = sum(b.get("stock_length_mm", 0) for b in sec.get("bars", []))
        sec_used = sum(b.get("used_length_mm", 0) for b in sec.get("bars", []))
        sec_util = round(sec_used / sec_stock * 100) if sec_stock else 0
        waste_mm = summ.get("total_waste_mm") or (sec_stock - sec_used)
        waste_str = f"{waste_mm / 1000:.1f} m" if waste_mm > 0 else "—"

        status = sec.get("phase1_status", "?")
        status_colour = _GREEN if status == "optimal" else _AMBER if status == "feasible" else _SLATE

        synopsis_rows += f"""
        <tr>
            <td>{_esc(sec.get('designation', '?'))}</td>
            <td style="text-align:center">{summ.get('items_placed', '?')}</td>
            <td style="text-align:center">{summ.get('stocks_used', '?')}</td>
            <td>
                <div style="display:inline-block;width:60px;height:8px;background:{_BORDER};
                            border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:4px">
                    <div style="width:{sec_util}%;height:100%;background:{_PRIMARY};border-radius:4px"></div>
                </div>
                <span style="font-weight:600;font-size:0.8em">{sec_util}%</span>
            </td>
            <td>{waste_str}</td>
            <td><span style="background:{status_colour};color:#fff;padding:1px 6px;
                            border-radius:3px;font-size:0.75em">{_esc(status)}</span></td>
        </tr>"""

    # ── Per-section detail ──
    section_detail_html = ""
    for sec in sections:
        summ = sec.get("summary", {})
        designation = _esc(sec.get("designation", "?"))

        status = sec.get("phase1_status", "?")
        status_colour = _GREEN if status == "optimal" else _AMBER if status == "feasible" else _SLATE

        section_detail_html += f"""
        <div class="section-block">
            <h3 style="margin:0 0 6px 0;font-size:1em">
                {designation}
                &mdash; {summ.get('items_placed','?')} placed, {summ.get('stocks_used','?')} bars
                <span style="background:{status_colour};color:#fff;padding:1px 6px;
                             border-radius:3px;font-size:0.8em;margin-left:4px">{_esc(status)}</span>
            </h3>
        """

        for bar in sec.get("bars", []):
            stock_len = bar.get("stock_length_mm", 0)
            used_len = bar.get("used_length_mm", 0)
            waste_mm = bar.get("waste_mm", 0)
            use_pct = round(used_len / stock_len * 100) if stock_len else 0

            # Bar header
            section_detail_html += f"""
            <div class="bar-block">
                <div class="bar-header">
                    <span style="font-weight:600">{_esc(bar.get('bar_label',''))}</span>
                    <span>{stock_len} mm</span>
                    <span>{use_pct}% used</span>
                    <span>waste: {waste_mm} mm</span>
                </div>
                <div class="bar-visual">
            """

            # Cut blocks
            for i, cut in enumerate(bar.get("cuts", [])):
                w_pct = (cut.get("length_mm", 0) / stock_len * 100) if stock_len else 0
                colour = _CUT_COLOURS[i % len(_CUT_COLOURS)]
                label = _esc(cut.get("member") or cut.get("ref_id") or f"Cut {cut.get('cut_no','')}")
                section_detail_html += (
                    f'<div class="cut-block" style="width:{w_pct:.1f}%;background:{colour}"'
                    f' title="{label}: {cut.get("length_mm",0)} mm">'
                    f'{cut.get("length_mm","")}</div>'
                )

            # Waste block
            if waste_mm > 0 and stock_len > 0:
                w_pct = waste_mm / stock_len * 100
                section_detail_html += (
                    f'<div class="waste-block" style="width:{w_pct:.1f}%">{waste_mm}</div>'
                )

            section_detail_html += "</div>"  # bar-visual

            # Cut table
            section_detail_html += """
                <table class="cuts-table">
                    <thead><tr><th>#</th><th>Member</th><th>Parent</th><th>Length</th></tr></thead>
                    <tbody>
            """
            for cut in bar.get("cuts", []):
                section_detail_html += f"""
                    <tr>
                        <td>{cut.get('cut_no','')}</td>
                        <td>{_esc(cut.get('member') or cut.get('ref_id') or '—')}</td>
                        <td>{_esc(cut.get('parent') or '—')}</td>
                        <td>{cut.get('length_mm','')} mm</td>
                    </tr>"""
            section_detail_html += "</tbody></table></div>"  # bar-block

        # Unassigned
        unassigned = sec.get("unassigned", [])
        if unassigned:
            section_detail_html += f"""
            <div class="unassigned">
                <strong>Unassigned ({len(unassigned)})</strong>
                <ul>{"".join(
                    f'<li>{_esc(u.get("member_name") or u.get("ref_id") or "?")} — {u.get("length","")} mm</li>'
                    for u in unassigned
                )}</ul>
            </div>"""

        section_detail_html += "</div>"  # section-block

    # ── Unassigned total ──
    unassigned_total = totals.get("total_items_unassigned", 0)
    unassigned_kpi = ""
    if unassigned_total > 0:
        unassigned_kpi = f"""
        <div class="kpi" style="border-color:{_RED}">
            <span class="kpi-value" style="color:{_RED}">{unassigned_total}</span>
            <span class="kpi-label">Unassigned</span>
        </div>"""

    waste_total = totals.get("total_waste_mm")
    waste_str = f"{waste_total / 1000:.1f} m" if waste_total is not None else "?"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4 landscape;
    margin: 15mm 12mm;
    @bottom-right {{
        content: "Page " counter(page) " of " counter(pages);
        font-size: 8pt;
        color: {_SLATE};
    }}
}}
body {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 9pt;
    color: #1e293b;
    line-height: 1.4;
}}
h1 {{
    font-size: 14pt;
    margin: 0 0 2px 0;
}}
.subtitle {{
    color: {_SLATE};
    font-size: 8pt;
    margin-bottom: 10px;
}}

/* KPI cards */
.kpi-row {{
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
}}
.kpi {{
    display: flex;
    flex-direction: column;
    align-items: center;
    border: 1px solid {_BORDER};
    border-radius: 5px;
    padding: 6px 18px;
    background: {_SURFACE};
    white-space: nowrap;
}}
.kpi-value {{
    font-size: 14pt;
    font-weight: 700;
    color: {_PRIMARY};
    line-height: 1.2;
}}
.kpi-label {{
    font-size: 7pt;
    text-transform: uppercase;
    color: {_SLATE};
    letter-spacing: 0.03em;
}}

/* Synopsis table */
.synopsis-table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 14px;
    font-size: 8.5pt;
}}
.synopsis-table th {{
    text-align: left;
    font-size: 7pt;
    text-transform: uppercase;
    color: {_SLATE};
    border-bottom: 1px solid {_BORDER};
    padding: 3px 6px;
}}
.synopsis-table td {{
    padding: 3px 6px;
    border-bottom: 1px solid {_SURFACE};
}}

/* Section detail */
.section-block {{
    page-break-inside: avoid;
    margin-bottom: 14px;
    border: 1px solid {_BORDER};
    border-radius: 5px;
    padding: 8px 10px;
}}
.bar-block {{
    margin-bottom: 8px;
}}
.bar-header {{
    display: flex;
    gap: 12px;
    font-size: 8pt;
    color: {_SLATE};
    margin-bottom: 2px;
}}
.bar-visual {{
    display: flex;
    height: 18px;
    background: {_BORDER};
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 4px;
}}
.cut-block {{
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 6.5pt;
    font-weight: 600;
    color: #1e293b;
    border-right: 1px solid #fff;
    min-width: 0;
    overflow: hidden;
}}
.waste-block {{
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 6.5pt;
    color: {_SLATE};
    background: repeating-linear-gradient(
        45deg, {_BORDER}, {_BORDER} 2px, #f1f5f9 2px, #f1f5f9 4px
    );
}}
.cuts-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7.5pt;
    margin-bottom: 4px;
}}
.cuts-table th {{
    text-align: left;
    font-size: 7pt;
    text-transform: uppercase;
    color: {_SLATE};
    border-bottom: 1px solid {_BORDER};
    padding: 2px 4px;
}}
.cuts-table td {{
    padding: 2px 4px;
    border-bottom: 1px solid {_SURFACE};
}}

.unassigned {{
    background: #fef3c7;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 8pt;
    margin-top: 4px;
}}
.unassigned ul {{
    margin: 2px 0 0 16px;
    padding: 0;
}}
</style>
</head>
<body>

<h1>Cutting List — {_esc(job_label)}</h1>
<div class="subtitle">
    {f'Nesting run: {run_date} &nbsp;|&nbsp; ' if run_date else ''}
    Report generated: {now_str}
</div>

<div class="kpi-row">
    <div class="kpi">
        <span class="kpi-value">{totals.get('total_items_placed', '?')}</span>
        <span class="kpi-label">Placed</span>
    </div>
    <div class="kpi">
        <span class="kpi-value">{totals.get('total_stocks_used', '?')}</span>
        <span class="kpi-label">Bars Used</span>
    </div>
    <div class="kpi">
        <span class="kpi-value">{overall_util}%</span>
        <span class="kpi-label">Utilisation</span>
    </div>
    <div class="kpi">
        <span class="kpi-value">{waste_str}</span>
        <span class="kpi-label">Total Waste</span>
    </div>
    {unassigned_kpi}
</div>

<table class="synopsis-table">
    <thead><tr>
        <th>Section</th><th>Placed</th><th>Bars</th>
        <th>Utilisation</th><th>Waste</th><th>Status</th>
    </tr></thead>
    <tbody>{synopsis_rows}</tbody>
</table>

{section_detail_html}

</body>
</html>"""
