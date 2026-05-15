
from __future__ import annotations

import json
import html
from pathlib import Path
from typing import Any


def read_json_safe(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_text_safe(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def percent(value: Any) -> str:
    try:
        value = float(value)
        if value <= 1:
            value = value * 100
        return f"{value:.1f}%"
    except Exception:
        return "N/A"


def bool_badge(value: Any) -> str:
    if value is True:
        return '<span class="badge good">Passed</span>'
    if value is False:
        return '<span class="badge bad">Failed</span>'
    return '<span class="badge neutral">N/A</span>'


def short_text(value: str, limit: int = 420) -> str:
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def create_visual_report(output_dir: str | Path, report_path: str | Path) -> Path:
    output_dir = Path(output_dir)
    report_path = Path(report_path)

    evaluation = read_json_safe(output_dir / "evaluation_report.json", {})
    hallucination = read_json_safe(output_dir / "hallucination_report.json", {})
    hallucination_after = read_json_safe(
        output_dir / "hallucination_report_after_edit_learning.json",
        {}
    )
    retrieval_trace = read_json_safe(output_dir / "retrieval_trace.json", [])
    structured_fields = read_json_safe(output_dir / "structured_fields.json", [])

    guarded_draft = read_text_safe(output_dir / "draft_guarded.md", "")
    final_draft = read_text_safe(output_dir / "draft_after_edit_learning.md", "")

    recall = evaluation.get("retrieval_recall_at_k", "N/A")
    grounding = evaluation.get("grounding_coverage", "N/A")
    unsupported_control = evaluation.get("unsupported_claim_control")
    edit_learning = evaluation.get("edit_learning_applied")

    claims_checked = hallucination.get("total_claims_checked", 0)
    claims_removed = hallucination.get("removed_or_quarantined_claims", 0)

    after_claims_checked = hallucination_after.get("total_claims_checked", 0)
    after_claims_removed = hallucination_after.get("removed_or_quarantined_claims", 0)

    evidence_cards = ""

    for item in retrieval_trace:
        chunk_id = html.escape(str(item.get("chunk_id", "unknown")))
        source = html.escape(str(item.get("source_path", "unknown")))
        page = html.escape(str(item.get("page_number", "N/A")))
        method = html.escape(str(item.get("retrieval_method", "N/A")))
        score = item.get("score", 0)
        text = html.escape(short_text(item.get("text", "")))

        evidence_cards += f"""
        <div class="evidence-card">
            <div class="evidence-top">
                <span class="rank">Rank {item.get("rank", "N/A")}</span>
                <span class="chip">{method}</span>
                <span class="score">Score: {float(score):.3f}</span>
            </div>
            <div class="chunk-id">{chunk_id}</div>
            <div class="source">Source: {source} | Page: {page}</div>
            <p>{text}</p>
        </div>
        """

    if not evidence_cards:
        evidence_cards = '<p class="muted">No retrieval evidence found.</p>'

    structured_html = ""

    for doc in structured_fields:
        doc_id = html.escape(str(doc.get("doc_id", "unknown")))

        parties = doc.get("parties", {})
        dates = doc.get("dates", {})
        amounts = doc.get("amounts", {})
        property_address = html.escape(str(doc.get("property_address", "N/A")))
        issues = ", ".join(doc.get("issues", [])) or "N/A"
        issues = html.escape(issues)

        structured_html += f"""
        <div class="structured-card">
            <h3>{doc_id}</h3>
            <table>
                <tr><th>Property</th><td>{property_address}</td></tr>
                <tr><th>Parties</th><td><pre>{html.escape(json.dumps(parties, indent=2))}</pre></td></tr>
                <tr><th>Dates</th><td><pre>{html.escape(json.dumps(dates, indent=2))}</pre></td></tr>
                <tr><th>Amounts</th><td><pre>{html.escape(json.dumps(amounts, indent=2))}</pre></td></tr>
                <tr><th>Issues</th><td>{issues}</td></tr>
            </table>
        </div>
        """

    if not structured_html:
        structured_html = '<p class="muted">No structured fields found.</p>'

    removed_claims = hallucination.get("removed_claims", [])
    removed_claims_html = ""

    for item in removed_claims:
        claim = html.escape(str(item.get("claim", "")))
        reason = html.escape(str(item.get("reason", "")))
        support_score = item.get("support_score", "N/A")

        removed_claims_html += f"""
        <div class="removed-claim">
            <p><strong>Removed claim:</strong> {claim}</p>
            <p><strong>Reason:</strong> {reason}</p>
            <p><strong>Support score:</strong> {support_score}</p>
        </div>
        """

    if not removed_claims_html:
        removed_claims_html = '<p class="muted">No unsupported claims were removed in the first guarded draft.</p>'

    html_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Legal Document AI Assessment Report</title>
<style>
    body {{
        margin: 0;
        background: #f4f7fb;
        color: #172033;
        font-family: Arial, Helvetica, sans-serif;
    }}

    .page {{
        max-width: 1180px;
        margin: 0 auto;
        padding: 34px 22px 60px;
    }}

    .hero {{
        background: linear-gradient(135deg, #101828, #1d4ed8);
        color: white;
        padding: 34px;
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(16, 24, 40, 0.18);
        margin-bottom: 28px;
    }}

    .hero h1 {{
        margin: 0 0 10px;
        font-size: 34px;
        letter-spacing: -0.7px;
    }}

    .hero p {{
        margin: 0;
        max-width: 860px;
        color: #dbeafe;
        font-size: 16px;
        line-height: 1.6;
    }}

    .grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 28px;
    }}

    .metric-card {{
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(16, 24, 40, 0.08);
        border: 1px solid #e7eef8;
    }}

    .metric-label {{
        color: #667085;
        font-size: 13px;
        margin-bottom: 8px;
    }}

    .metric-value {{
        font-size: 28px;
        font-weight: 800;
        color: #101828;
    }}

    .section {{
        background: white;
        padding: 26px;
        border-radius: 22px;
        box-shadow: 0 10px 30px rgba(16, 24, 40, 0.07);
        border: 1px solid #e7eef8;
        margin-bottom: 24px;
    }}

    .section h2 {{
        margin: 0 0 16px;
        font-size: 23px;
        color: #101828;
    }}

    .pipeline {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 12px;
    }}

    .stage {{
        background: #f8fbff;
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 16px;
        min-height: 120px;
    }}

    .stage-number {{
        width: 28px;
        height: 28px;
        border-radius: 999px;
        background: #2563eb;
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-bottom: 10px;
    }}

    .stage h3 {{
        margin: 0 0 8px;
        font-size: 15px;
    }}

    .stage p {{
        margin: 0;
        color: #667085;
        font-size: 13px;
        line-height: 1.45;
    }}

    .badge {{
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
    }}

    .good {{
        background: #dcfce7;
        color: #166534;
    }}

    .bad {{
        background: #fee2e2;
        color: #991b1b;
    }}

    .neutral {{
        background: #e5e7eb;
        color: #374151;
    }}

    .evidence-card, .structured-card, .removed-claim {{
        background: #f8fbff;
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 14px;
    }}

    .evidence-top {{
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }}

    .rank {{
        font-weight: 800;
        color: #1d4ed8;
    }}

    .chip {{
        background: #e0f2fe;
        color: #075985;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }}

    .score {{
        color: #667085;
        font-size: 13px;
    }}

    .chunk-id {{
        font-family: Consolas, monospace;
        font-size: 13px;
        color: #344054;
        margin-bottom: 6px;
    }}

    .source {{
        color: #667085;
        font-size: 13px;
        margin-bottom: 8px;
    }}

    .evidence-card p {{
        margin: 0;
        line-height: 1.6;
        color: #344054;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }}

    th, td {{
        border-bottom: 1px solid #e5e7eb;
        padding: 12px;
        text-align: left;
        vertical-align: top;
    }}

    th {{
        width: 180px;
        color: #475467;
        background: #f9fafb;
    }}

    pre {{
        white-space: pre-wrap;
        margin: 0;
        font-family: Consolas, monospace;
        font-size: 13px;
    }}

    .draft-box {{
        background: #0b1220;
        color: #e5e7eb;
        padding: 22px;
        border-radius: 16px;
        overflow-x: auto;
        white-space: pre-wrap;
        line-height: 1.6;
        font-family: Consolas, monospace;
        font-size: 13px;
        max-height: 560px;
        overflow-y: auto;
    }}

    .muted {{
        color: #667085;
    }}

    .two-col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }}

    .footer {{
        text-align: center;
        color: #667085;
        font-size: 13px;
        margin-top: 28px;
    }}

    @media (max-width: 950px) {{
        .grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
        .pipeline {{
            grid-template-columns: repeat(2, 1fr);
        }}
        .two-col {{
            grid-template-columns: 1fr;
        }}
    }}

    @media (max-width: 600px) {{
        .grid {{
            grid-template-columns: 1fr;
        }}
        .pipeline {{
            grid-template-columns: 1fr;
        }}
    }}
</style>
</head>
<body>
<div class="page">

    <div class="hero">
        <h1>Legal Document AI Assessment Report</h1>
        <p>
            A complete visual summary of the document-processing, grounded retrieval,
            hallucination-removal, draft-generation, edit-learning, and evaluation workflow.
            This report is generated directly from the pipeline outputs.
        </p>
    </div>

    <div class="grid">
        <div class="metric-card">
            <div class="metric-label">Retrieval Recall@K</div>
            <div class="metric-value">{percent(recall)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Grounding Coverage</div>
            <div class="metric-value">{percent(grounding)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Unsupported Claim Control</div>
            <div class="metric-value">{bool_badge(unsupported_control)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Edit Learning Applied</div>
            <div class="metric-value">{bool_badge(edit_learning)}</div>
        </div>
    </div>

    <div class="section">
        <h2>Pipeline Overview</h2>
        <div class="pipeline">
            <div class="stage">
                <div class="stage-number">1</div>
                <h3>Document Processing</h3>
                <p>Extracts embedded text, runs OCR fallback, normalizes text, and captures confidence warnings.</p>
            </div>
            <div class="stage">
                <div class="stage-number">2</div>
                <h3>Structured Extraction</h3>
                <p>Pulls parties, dates, amounts, property address, issues, and uncertainty notes.</p>
            </div>
            <div class="stage">
                <div class="stage-number">3</div>
                <h3>Hybrid Retrieval</h3>
                <p>Combines BM25 keyword search with FAISS semantic search to retrieve grounded evidence.</p>
            </div>
            <div class="stage">
                <div class="stage-number">4</div>
                <h3>Draft Generation</h3>
                <p>Creates a first-pass notice-related case fact summary using only retrieved evidence.</p>
            </div>
            <div class="stage">
                <div class="stage-number">5</div>
                <h3>Hallucination Guard</h3>
                <p>Checks claims against evidence and removes unsupported or high-risk legal claims.</p>
            </div>
            <div class="stage">
                <div class="stage-number">6</div>
                <h3>Edit Learning</h3>
                <p>Learns cautious drafting preferences from operator edits and applies them to future drafts.</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Hallucination Removal Summary</h2>
        <div class="grid">
            <div class="metric-card">
                <div class="metric-label">Claims Checked Before Edit Learning</div>
                <div class="metric-value">{claims_checked}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Claims Removed Before Edit Learning</div>
                <div class="metric-value">{claims_removed}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Claims Checked After Edit Learning</div>
                <div class="metric-value">{after_claims_checked}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Claims Removed After Edit Learning</div>
                <div class="metric-value">{after_claims_removed}</div>
            </div>
        </div>
        <h3>Removed or Quarantined Claims</h3>
        {removed_claims_html}
    </div>

    <div class="section">
        <h2>Structured Fields Extracted</h2>
        {structured_html}
    </div>

    <div class="section">
        <h2>Retrieved Evidence</h2>
        {evidence_cards}
    </div>

    <div class="section">
        <h2>Final Draft After Hallucination Removal and Edit Learning</h2>
        <div class="draft-box">{html.escape(final_draft or guarded_draft or "No draft found.")}</div>
    </div>

    <div class="footer">
        Generated from pipeline artifacts in outputs/sample_run.
    </div>

</div>
</body>
</html>
"""

    report_path.write_text(html_report, encoding="utf-8")
    return report_path
