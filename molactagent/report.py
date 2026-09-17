"""HTML / Markdown report renderers for MolActAgent (multi-class).

Layout per sample:
  Hero            — verdict, confidence, evidence basis caption
  Top Drivers     — derived from weighted driver attribution; each row
                    shows dominant class, citation count, and net
                    strength
  Narrative       — Markdown report (h2/h3, tables, lists, etc.)
  Robustness      — vote consistency bar with one segment per class
                    plus an Inconclusive segment
  Tech Details    — collapsed JSON appendix

Aggregate "interpretability" reports follow the same colour palette
so the whole output directory feels coherent.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime

__all__ = [
    "render_html_report",
    "render_interpretability_text",
    "render_interpretability_md",
    "render_interpretability_html",
]


# ── colour palette (deterministic, colourblind-friendlier) ──────────
#
# The first two entries deliberately match the previous binary
# palette (green / red) so a typical 2-class run produces visually
# similar reports to before. Class N is mapped to ``_PALETTE[N % len]``;
# "Inconclusive" / "Mixed" / unknown classes always render as grey.

_PALETTE: tuple[str, ...] = (
    "#34c759",  # green
    "#ff3b30",  # red
    "#007aff",  # blue
    "#ff9500",  # orange
    "#af52de",  # purple
    "#5ac8fa",  # cyan
    "#ffcc00",  # yellow
    "#ff2d55",  # pink
)
_INCONCLUSIVE_COLOR = "#8e8e93"


def _class_color(cls: str | None, class_labels) -> str:
    """Return a deterministic colour for ``cls``."""
    if cls is None or cls in ("Inconclusive", "Unknown", "Mixed"):
        return _INCONCLUSIVE_COLOR
    try:
        idx = list(class_labels).index(cls)
    except (ValueError, TypeError):
        return _INCONCLUSIVE_COLOR
    return _PALETTE[idx % len(_PALETTE)]


def _arrow_for(favored_class: str | None, class_labels) -> str:
    """Render the per-driver direction arrow: ``->ClassName``."""
    if favored_class is None or favored_class in ("Inconclusive", "Unknown"):
        return "mixed"
    if favored_class == "Mixed":
        return "mixed"
    return f"-> {favored_class}"


# ── Markdown → HTML (extended subset) ────────────────────────────────

def _esc(text) -> str:
    return html.escape(str(text))


def _md_to_html(md: str) -> str:
    """Convert an expanded Markdown subset to HTML.

    Supported:
      - ATX headings ##, ###, ####
      - **bold**, *italic*, `code`
      - Numbered lists (1. ...)
      - Unordered lists (- ... or * ...)
      - Pipe tables: | a | b | / |---|---|
      - Horizontal rule (---)
      - Paragraphs separated by blank lines
    """
    if not md or not md.strip():
        return ""

    lines = md.strip().split("\n")
    out: list[str] = []
    i = 0
    in_para = False
    in_ol = False
    in_ul = False

    def _close_para():
        nonlocal in_para
        if in_para:
            out.append("</p>")
            in_para = False

    def _close_lists():
        nonlocal in_ol, in_ul
        if in_ol:
            out.append("</ol>")
            in_ol = False
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def _close_all():
        _close_para()
        _close_lists()

    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if s == "---":
            _close_all()
            out.append('<hr class="rpt-hr">')
            i += 1
            continue

        # Headings (most specific first)
        if s.startswith("#### "):
            _close_all()
            out.append(f'<h4 class="rpt-h4">{_inline(s[5:])}</h4>')
            i += 1
            continue
        if s.startswith("### "):
            _close_all()
            out.append(f'<h3 class="rpt-h3">{_inline(s[4:])}</h3>')
            i += 1
            continue
        if s.startswith("## "):
            _close_all()
            out.append(f'<h2 class="rpt-h2">{_inline(s[3:])}</h2>')
            i += 1
            continue
        if s.startswith("# ") and not s.startswith("## "):
            # Demote to h2 for visual consistency.
            _close_all()
            out.append(f'<h2 class="rpt-h2">{_inline(s[2:])}</h2>')
            i += 1
            continue

        # Pipe table — needs at least 2 lines, with separator on the second.
        if (
            s.startswith("|")
            and s.endswith("|")
            and i + 1 < len(lines)
            and _is_table_separator(lines[i + 1])
        ):
            _close_all()
            j = _render_table(lines, i, out)
            i = j
            continue

        # Numbered list
        ol_match = re.match(r"^(\d+)\.\s+(.+)$", s)
        if ol_match:
            _close_para()
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append('<ol class="rpt-ol">')
                in_ol = True
            out.append(f"<li>{_inline(ol_match.group(2))}</li>")
            i += 1
            continue

        # Unordered list
        ul_match = re.match(r"^[-*]\s+(.+)$", s)
        if ul_match:
            _close_para()
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append('<ul class="rpt-ul">')
                in_ul = True
            out.append(f"<li>{_inline(ul_match.group(1))}</li>")
            i += 1
            continue

        if in_ol or in_ul:
            _close_lists()

        # Blank line ends paragraph
        if s == "":
            _close_para()
            i += 1
            continue

        if not in_para:
            out.append('<p class="rpt-p">')
            in_para = True
        else:
            out.append(" ")
        out.append(_inline(s))
        i += 1

    _close_all()
    return "\n".join(out)


def _is_table_separator(line: str) -> bool:
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    if not cells:
        return False
    return all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def _render_table(lines: list[str], start: int, out: list[str]) -> int:
    header_cells = _split_row(lines[start])
    sep_cells = _split_row(lines[start + 1])
    aligns = []
    for c in sep_cells:
        c = c.strip()
        if c.startswith(":") and c.endswith(":"):
            aligns.append("center")
        elif c.endswith(":"):
            aligns.append("right")
        else:
            aligns.append("left")

    out.append('<div class="rpt-table-wrap"><table class="rpt-table">')
    out.append("<thead><tr>")
    for cell, a in zip(header_cells, aligns + ["left"] * len(header_cells)):
        out.append(f'<th style="text-align:{a}">{_inline(cell)}</th>')
    out.append("</tr></thead>")
    out.append("<tbody>")

    j = start + 2
    while j < len(lines):
        s = lines[j].strip()
        if not s.startswith("|") or not s.endswith("|"):
            break
        if _is_table_separator(s):
            j += 1
            continue
        cells = _split_row(s)
        out.append("<tr>")
        for k, cell in enumerate(cells):
            a = aligns[k] if k < len(aligns) else "left"
            out.append(f'<td style="text-align:{a}">{_inline(cell)}</td>')
        out.append("</tr>")
        j += 1

    out.append("</tbody></table></div>")
    return j


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _inline(text: str) -> str:
    """Inline formatting: code, bold, italic. Order matters."""
    code_blocks: list[str] = []

    def _stash_code(m):
        code_blocks.append(m.group(1))
        return f"\u0000CODE{len(code_blocks) - 1}\u0000"

    raw = re.sub(r"`([^`]+)`", _stash_code, text)

    t = _esc(raw)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)

    for idx, code in enumerate(code_blocks):
        t = t.replace(
            f"\u0000CODE{idx}\u0000",
            f"<code class='rpt-code'>{_esc(code)}</code>",
        )
    return t


# ── Top-drivers card (derived from driver_attribution) ───────────────

def _top_drivers_card(attribution, class_labels, max_rows=8):
    if not attribution:
        return ""
    cited = [fa for fa in attribution if fa.get("n_cited", 0) > 0]
    if not cited:
        return ""
    rows = cited[:max_rows]
    max_strength = max(fa.get("abs_strength", 0) for fa in rows) or 1.0

    body = []
    for fa in rows:
        dom = fa.get("dominant_class") or fa.get("direction") or "Mixed"
        color = _class_color(dom, class_labels)
        label = "Mixed" if dom in (None, "Mixed") else dom

        net = fa.get("weighted_score", 0.0)
        bar_w = abs(fa.get("abs_strength", 0)) / max_strength * 100
        desc = fa.get("description") or ""
        n_cited = fa.get("n_cited", 0)
        why = (fa.get("rationales") or [""])[0]
        why_html = (
            f'<div class="drv-why">{_esc(why)}</div>' if why else ""
        )

        body.append(
            f'<div class="drv-row">'
            f'<div class="drv-name">{_esc(fa["feature"])}'
            f'<span class="drv-desc">{_esc(desc)}</span></div>'
            f'<div class="drv-bar-w">'
            f'<div class="drv-bar" style="width:{bar_w:.0f}%;background:{color}"></div>'
            f'</div>'
            f'<div class="drv-meta">'
            f'<span class="drv-dir" style="color:{color}">-> {_esc(label)}</span>'
            f'<span class="drv-num">cited in {n_cited} tuple(s)</span>'
            f'<span class="drv-num">net strength {net:+.2f}</span>'
            f'</div>'
            f'{why_html}'
            f'</div>'
        )
    return (
        '<div class="drv-card">'
        '<div class="drv-title">Top Drivers '
        '<span class="drv-sub">cited by per-tuple analyses, weighted by stated strength</span></div>'
        + "\n".join(body)
        + "</div>"
    )


# ── Robustness card (vote consistency) ──────────────────────────────

def _robustness_card(vote_summary, class_labels):
    """N+1-segment vote bar: one per class plus an Inconclusive segment."""
    per_class = vote_summary.get("per_class", {}) or {}
    n_unk = vote_summary.get("num_unknown", 0)
    n_degen = vote_summary.get("n_degenerate", 0)
    total = sum(int(v.get("votes", 0)) for v in per_class.values()) + int(n_unk)
    if total == 0:
        return ""

    # Render bar segments in canonical class_labels order so colours
    # are deterministic; classes outside class_labels (defensive
    # fallback) appear after the named ones.
    ordered_classes: list[str] = list(class_labels) if class_labels else []
    extras = [c for c in per_class.keys() if c not in ordered_classes]
    ordered_classes.extend(extras)

    seg_html = []
    legend_html = []
    for cls in ordered_classes:
        votes = int((per_class.get(cls) or {}).get("votes", 0))
        if votes == 0 and cls not in (class_labels or []):
            continue
        color = _class_color(cls, class_labels)
        pct = (votes / total) * 100
        seg_html.append(
            f'<div class="rb-seg" style="width:{pct:.1f}%;background:{color}"></div>'
        )
        legend_html.append(
            f'<span><span class="dot" style="background:{color}"></span>'
            f'{_esc(cls)}: {votes} ({pct:.0f}%)</span>'
        )
    # Inconclusive segment
    pct_unk = (n_unk / total) * 100
    seg_html.append(
        f'<div class="rb-seg" style="width:{pct_unk:.1f}%;'
        f'background:{_INCONCLUSIVE_COLOR}"></div>'
    )
    legend_html.append(
        f'<span><span class="dot" style="background:{_INCONCLUSIVE_COLOR}"></span>'
        f'Inconclusive: {n_unk} ({pct_unk:.0f}%)</span>'
    )

    degen_note = (
        f' <span class="rb-degen">({n_degen} degenerate tuple(s) excluded)</span>'
        if n_degen else ""
    )
    return (
        '<div class="rb-card">'
        f'<div class="rb-title">Vote Consistency Across Tuples{degen_note}</div>'
        '<div class="rb-bar">'
        + "".join(seg_html)
        + '</div>'
        '<div class="rb-leg">'
        + "".join(legend_html)
        + '</div>'
        '</div>'
    )


# ── Regression-specific cards ───────────────────────────────────────

def _regression_color(value, target_range, refused: bool = False) -> str:
    """Return a colour for the regression hero badge.

    Refused predictions render as the grey "Inconclusive" colour.
    Otherwise we map the predicted value to a position within the
    observed target range and pick from a small low-to-high palette
    (cool to warm) so the badge gives a visual sense of "low" vs
    "high" without needing to read the number.
    """
    if refused or value is None:
        return _INCONCLUSIVE_COLOR
    try:
        lo, hi = float(target_range[0]), float(target_range[1])
    except (TypeError, ValueError):
        return _PALETTE[0]
    if hi <= lo:
        return _PALETTE[0]
    pos = (float(value) - lo) / (hi - lo)
    pos = max(0.0, min(1.0, pos))
    # Five-stop gradient: blue (low) to green to yellow to orange to red (high)
    stops = ["#007aff", "#34c759", "#ffcc00", "#ff9500", "#ff3b30"]
    idx = int(round(pos * (len(stops) - 1)))
    return stops[idx]


def _regression_drivers_card(attribution, max_rows: int = 8) -> str:
    """Top-drivers card for regression: per-feature direction +
    weighted strength, with up/down arrows."""
    if not attribution:
        return ""
    cited = [fa for fa in attribution if fa.get("n_cited", 0) > 0]
    if not cited:
        return ""
    rows = cited[:max_rows]
    max_strength = max(fa.get("abs_strength", 0) for fa in rows) or 1.0

    body = []
    for fa in rows:
        direction = fa.get("dominant_direction", "mixed")
        if direction == "increases":
            color = "#ff3b30"  # warm = pushes up
            arrow = "↑ increases"
        elif direction == "decreases":
            color = "#007aff"  # cool = pushes down
            arrow = "↓ decreases"
        else:
            color = _INCONCLUSIVE_COLOR
            arrow = "~ mixed"

        bar_w = abs(fa.get("abs_strength", 0)) / max_strength * 100
        net = fa.get("weighted_score", 0.0)
        desc = fa.get("description") or ""
        n_cited = fa.get("n_cited", 0)
        why = (fa.get("rationales") or [""])[0]
        why_html = (
            f'<div class="drv-why">{_esc(why)}</div>' if why else ""
        )
        body.append(
            f'<div class="drv-row">'
            f'<div class="drv-name">{_esc(fa["feature"])}'
            f'<span class="drv-desc">{_esc(desc)}</span></div>'
            f'<div class="drv-bar-w">'
            f'<div class="drv-bar" style="width:{bar_w:.0f}%;background:{color}"></div>'
            f'</div>'
            f'<div class="drv-meta">'
            f'<span class="drv-dir" style="color:{color}">{_esc(arrow)}</span>'
            f'<span class="drv-num">cited in {n_cited} tuple(s)</span>'
            f'<span class="drv-num">net strength {net:+.2f}</span>'
            f'</div>'
            f'{why_html}'
            f'</div>'
        )
    return (
        '<div class="drv-card">'
        '<div class="drv-title">Top Drivers '
        '<span class="drv-sub">'
        'cited by per-tuple analyses; arrows show whether the feature '
        'pushes the prediction up or down'
        '</span></div>'
        + "\n".join(body)
        + "</div>"
    )


def _regression_robustness_card(result: dict) -> str:
    """Show each tuple's predicted value as a dot on a horizontal
    target-range scale, with the median (=final prediction) marked.
    Communicates tuple agreement at a glance."""
    predictions = result.get("predictions") or []
    if not predictions:
        return ""
    ri = result.get("reference_info", {}) or {}
    target_range = ri.get("target_range")
    final_value = result.get("predicted_value")
    if target_range is None:
        target_range = (min(predictions), max(predictions))
    try:
        lo, hi = float(target_range[0]), float(target_range[1])
    except (TypeError, ValueError):
        return ""
    span = max(hi - lo, 1e-9)

    def _pos_pct(v: float) -> float:
        return max(0.0, min(100.0, (float(v) - lo) / span * 100.0))

    dots = []
    for p in predictions:
        x = _pos_pct(p)
        dots.append(
            f'<div class="rb-dot" '
            f'style="left:{x:.1f}%;background:#86868b" '
            f'title="tuple prediction {float(p):.4g}"></div>'
        )
    if final_value is not None:
        x_final = _pos_pct(final_value)
        dots.append(
            f'<div class="rb-dot rb-dot-final" '
            f'style="left:{x_final:.1f}%;background:#1d1d1f" '
            f'title="final prediction {float(final_value):.4g}"></div>'
        )

    n_extrap = sum(
        1 for t in result.get("tuple_results", [])
        if t.get("extrapolation_flag")
    )
    extrap_note = (
        f' <span class="rb-degen">({n_extrap} tuple(s) flagged extrapolation)</span>'
        if n_extrap else ""
    )
    return (
        '<div class="rb-card">'
        f'<div class="rb-title">Tuple Predictions Across the Target Range{extrap_note}</div>'
        '<div class="rb-axis">'
        '<div class="rb-track"></div>'
        + "".join(dots)
        + '</div>'
        '<div class="rb-axlabels">'
        f'<span>{_fmt_target_value(lo)}</span>'
        f'<span>{_fmt_target_value(hi)}</span>'
        '</div>'
        '</div>'
    )


def _fmt_target_value(v) -> str:
    """Compact numeric formatting for axis labels and badges."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f) and abs(f) < 1e8:
        return str(int(f))
    return f"{f:.4g}"


# ── Main entry ───────────────────────────────────────────────────────

def render_html_report(
    result: dict,
    explanation: str = "",
    class_labels=None,
    title: str = "MolActAgent Report",
) -> str:
    """Render a per-sample HTML report. Branches on ``result['task']``
    (defaults to classification for back-compat)."""
    if result.get("task") == "regression":
        return _render_regression_html_report(result, explanation, title)
    return _render_classification_html_report(result, explanation, class_labels, title)


def _render_classification_html_report(
    result: dict,
    explanation: str,
    class_labels,
    title: str,
) -> str:
    """Render a per-sample HTML report for a classification prediction."""
    cls_list = list(class_labels) if class_labels else (
        result.get("candidate_classes") or []
    )

    verdict = result.get("final_label", "Inconclusive")
    conf = result.get("confidence", 0.0) * 100
    lc = _class_color(verdict, cls_list)
    tid = result.get("test_index", "?")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    cb = result.get("confidence_basis", {})
    basis_text = cb.get("basis_text", "")
    basis_rationale = cb.get("rationale", "")

    ri = result.get("reference_info", {})
    n_total = ri.get("total_reference_samples", "?")
    per_class_ref = ri.get("class_reference_counts", {}) or {}
    n_comp = ri.get("n_comparisons", "?")
    per_class_ref_text = ", ".join(
        f"{n} {cls}" for cls, n in per_class_ref.items()
    ) or "—"

    attribution = result.get("driver_attribution", [])
    vote_summary = result.get("vote_summary", {})

    drivers_card = _top_drivers_card(attribution, cls_list)
    robustness_card = _robustness_card(vote_summary, cls_list)
    report_body = (
        _md_to_html(explanation) if explanation and explanation.strip()
        else '<p class="rpt-p"><em>No analysis narrative available.</em></p>'
    )
    raw_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} — Sample {tid}</title>
<style>
:root{{
  --bg:#f5f5f7;--card:#fff;--tx:#1d1d1f;--tx2:#6e6e73;--tx3:#86868b;
  --bd:#e8e8ed;--bd2:#f0f0f3;
  --r:18px;--r2:12px;--sh:0 2px 12px rgba(0,0,0,.06);
  --f:-apple-system,"SF Pro Display",BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;
  --fmono:"SF Mono","Menlo","Fira Code",Consolas,monospace
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:var(--f);background:var(--bg);color:var(--tx);line-height:1.6;-webkit-font-smoothing:antialiased}}
.c{{max-width:880px;margin:0 auto;padding:40px 24px 60px}}

/* Hero */
.hero{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:40px 36px 32px;text-align:center;margin-bottom:20px}}
.hero-t{{font-size:24px;font-weight:700}}
.hero-s{{font-size:13px;color:var(--tx2);margin:4px 0 22px}}
.badge{{display:inline-block;font-size:20px;font-weight:700;color:#fff;padding:9px 36px;border-radius:14px;letter-spacing:.03em}}
.conf{{margin-top:18px}}
.conf-l{{font-size:12px;color:var(--tx2);margin-bottom:6px;text-transform:uppercase;letter-spacing:.06em}}
.conf-tr{{height:7px;background:var(--bd);border-radius:4px;overflow:hidden;max-width:300px;margin:0 auto}}
.conf-f{{height:100%;border-radius:4px;animation:g .8s ease}}
.conf-n{{font-size:32px;font-weight:700;margin-top:8px}}
@keyframes g{{from{{width:0}}}}
.basis{{font-size:13px;color:var(--tx2);margin-top:14px;line-height:1.5;max-width:540px;margin-left:auto;margin-right:auto}}
.basis-rationale{{font-size:12px;color:var(--tx3);margin-top:6px;font-style:italic}}
.ts{{font-size:11px;color:var(--tx3);margin-top:14px;letter-spacing:.04em}}

/* Generic card chrome */
.card{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:28px 32px;margin-bottom:20px}}
.card-title{{font-size:13px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:16px}}

/* Top Drivers card */
.drv-card{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:24px 28px;margin-bottom:20px}}
.drv-title{{font-size:14px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}}
.drv-sub{{font-size:11px;font-weight:500;text-transform:none;letter-spacing:0;color:var(--tx3);margin-left:8px}}
.drv-row{{padding:12px 0;border-bottom:1px solid var(--bd2)}}
.drv-row:last-child{{border-bottom:none}}
.drv-name{{font-size:14px;font-weight:600;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:6px}}
.drv-desc{{font-weight:400;font-size:12px;color:var(--tx2)}}
.drv-bar-w{{height:6px;background:var(--bd);border-radius:3px;overflow:hidden;margin:6px 0}}
.drv-bar{{height:100%;border-radius:3px;transition:width .6s ease}}
.drv-meta{{display:flex;gap:14px;font-size:12px;color:var(--tx2);flex-wrap:wrap;margin-top:4px}}
.drv-dir{{font-weight:600}}
.drv-num{{color:var(--tx3)}}
.drv-why{{font-size:13px;color:#3a3a3c;margin-top:6px;line-height:1.5;background:#fafafa;padding:8px 12px;border-radius:8px;border-left:3px solid var(--bd)}}

/* Report body (Markdown narrative) */
.rpt{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:36px 40px;margin-bottom:20px}}
.rpt-h2{{font-size:19px;font-weight:700;margin:30px 0 12px;color:var(--tx);border-bottom:1px solid var(--bd);padding-bottom:6px}}
.rpt-h2:first-child{{margin-top:0}}
.rpt-h3{{font-size:15px;font-weight:700;margin:22px 0 8px;color:var(--tx)}}
.rpt-h4{{font-size:13px;font-weight:700;margin:18px 0 6px;color:var(--tx2);text-transform:uppercase;letter-spacing:.05em}}
.rpt-p{{font-size:15px;line-height:1.78;color:#333;margin-bottom:12px}}
.rpt-ol,.rpt-ul{{padding-left:24px;margin:8px 0 14px}}
.rpt-ol li,.rpt-ul li{{font-size:15px;line-height:1.78;color:#333;margin-bottom:6px}}
.rpt-hr{{border:none;border-top:1px solid var(--bd);margin:24px 0}}
.rpt-code{{font-family:var(--fmono);font-size:13px;background:#f0f0f3;padding:1px 6px;border-radius:4px;color:#333}}

.rpt-table-wrap{{overflow-x:auto;margin:14px 0 18px}}
.rpt-table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff}}
.rpt-table thead{{background:#fafafa}}
.rpt-table th{{font-weight:700;color:var(--tx);padding:10px 12px;border-bottom:2px solid var(--bd);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
.rpt-table td{{padding:9px 12px;border-bottom:1px solid var(--bd2);color:#333;line-height:1.5}}
.rpt-table tbody tr:last-child td{{border-bottom:none}}
.rpt-table tbody tr:hover{{background:#fafafa}}

/* Robustness card */
.rb-card{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:22px 28px;margin-bottom:20px}}
.rb-title{{font-size:13px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px}}
.rb-degen{{font-weight:500;text-transform:none;letter-spacing:0;color:var(--tx3);font-size:11px}}
.rb-bar{{display:flex;height:22px;border-radius:8px;overflow:hidden}}
.rb-seg{{transition:width .6s ease}}
.rb-leg{{display:flex;gap:18px;font-size:12px;font-weight:500;margin-top:10px;flex-wrap:wrap;color:var(--tx2)}}
.rb-leg span{{display:flex;align-items:center;gap:5px}}
.dot{{width:9px;height:9px;border-radius:50%;display:inline-block}}

/* Tech details */
.det{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);margin-bottom:20px;overflow:hidden}}
.det summary{{padding:16px 32px;font-size:13px;font-weight:600;cursor:pointer;user-select:none;list-style:none;display:flex;align-items:center;gap:8px;color:var(--tx2);text-transform:uppercase;letter-spacing:.06em}}
.det summary::-webkit-details-marker{{display:none}}
.det summary::before{{content:"▸";font-size:12px;transition:transform .2s}}
.det[open] summary::before{{transform:rotate(90deg)}}
.det-b{{padding:0 32px 22px}}
.json{{font-family:var(--fmono);font-size:11px;line-height:1.55;background:#fafafa;border-radius:10px;padding:14px;overflow-x:auto;max-height:420px;overflow-y:auto;white-space:pre;color:#424245}}

.ft{{text-align:center;font-size:11px;color:var(--tx3);margin-top:24px;letter-spacing:.05em}}
</style>
</head>
<body>
<div class="c">

<div class="hero">
  <div class="hero-t">{_esc(title)}</div>
  <div class="hero-s">Sample {tid}</div>
  <span class="badge" style="background:{lc}">{_esc(verdict)}</span>
  <div class="conf">
    <div class="conf-l">Confidence</div>
    <div class="conf-tr"><div class="conf-f" style="width:{conf:.1f}%;background:{lc}"></div></div>
    <div class="conf-n" style="color:{lc}">{conf:.1f}%</div>
  </div>
  <div class="basis">{_esc(basis_text)}</div>
  {f'<div class="basis-rationale">{_esc(basis_rationale)}</div>' if basis_rationale else ""}
  <div class="ts">{ts}</div>
</div>

{drivers_card}

<div class="rpt">
{report_body}
</div>

{robustness_card}

<details class="det">
  <summary>Technical Details (raw evidence)</summary>
  <div class="det-b">
    <pre class="json">{_esc(raw_json)}</pre>
  </div>
</details>

<div class="ft">MolActAgent · {ts} · sample {tid} · {n_comp} tuple comparisons against {n_total} reference samples ({per_class_ref_text})</div>
</div>
</body>
</html>"""


def _render_regression_html_report(
    result: dict,
    explanation: str,
    title: str,
) -> str:
    """Render a per-sample HTML report for a regression prediction."""
    pv = result.get("predicted_value")
    iv = result.get("interval") or (None, None)
    refused = pv is None
    conf = (result.get("confidence") or 0.0) * 100
    tid = result.get("test_index", "?")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    cb = result.get("confidence_basis", {}) or {}
    basis_text = cb.get("basis_text", "")
    basis_rationale = cb.get("rationale", "")

    ri = result.get("reference_info", {}) or {}
    n_total = ri.get("total_reference_samples", "?")
    target_range = ri.get("target_range")
    n_comp = ri.get("n_comparisons", "?")

    badge_color = _regression_color(pv, target_range, refused=refused)
    if refused:
        verdict_text = "REFUSED"
        interval_text = "evidence too thin to predict"
    else:
        verdict_text = _fmt_target_value(pv)
        if iv[0] is not None and iv[1] is not None:
            interval_text = (
                f"interval [{_fmt_target_value(iv[0])}, "
                f"{_fmt_target_value(iv[1])}]"
            )
        else:
            interval_text = ""

    target_range_text = ""
    if target_range is not None:
        target_range_text = (
            f"observed target range "
            f"[{_fmt_target_value(target_range[0])}, "
            f"{_fmt_target_value(target_range[1])}]"
        )

    attribution = result.get("driver_attribution", []) or []
    drivers_card = _regression_drivers_card(attribution)
    robustness_card = _regression_robustness_card(result)
    report_body = (
        _md_to_html(explanation) if explanation and explanation.strip()
        else '<p class="rpt-p"><em>No analysis narrative available.</em></p>'
    )
    raw_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} — Sample {tid}</title>
<style>
:root{{
  --bg:#f5f5f7;--card:#fff;--tx:#1d1d1f;--tx2:#6e6e73;--tx3:#86868b;
  --bd:#e8e8ed;--bd2:#f0f0f3;
  --r:18px;--r2:12px;--sh:0 2px 12px rgba(0,0,0,.06);
  --f:-apple-system,"SF Pro Display",BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;
  --fmono:"SF Mono","Menlo","Fira Code",Consolas,monospace
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:var(--f);background:var(--bg);color:var(--tx);line-height:1.6;-webkit-font-smoothing:antialiased}}
.c{{max-width:880px;margin:0 auto;padding:40px 24px 60px}}

.hero{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:40px 36px 32px;text-align:center;margin-bottom:20px}}
.hero-t{{font-size:24px;font-weight:700}}
.hero-s{{font-size:13px;color:var(--tx2);margin:4px 0 22px}}
.badge{{display:inline-block;font-size:24px;font-weight:700;color:#fff;padding:9px 36px;border-radius:14px;letter-spacing:.03em;font-family:var(--fmono)}}
.iv{{font-size:13px;color:var(--tx2);margin-top:10px}}
.conf{{margin-top:18px}}
.conf-l{{font-size:12px;color:var(--tx2);margin-bottom:6px;text-transform:uppercase;letter-spacing:.06em}}
.conf-tr{{height:7px;background:var(--bd);border-radius:4px;overflow:hidden;max-width:300px;margin:0 auto}}
.conf-f{{height:100%;border-radius:4px;animation:g .8s ease}}
.conf-n{{font-size:32px;font-weight:700;margin-top:8px}}
@keyframes g{{from{{width:0}}}}
.basis{{font-size:13px;color:var(--tx2);margin-top:14px;line-height:1.5;max-width:540px;margin-left:auto;margin-right:auto}}
.basis-rationale{{font-size:12px;color:var(--tx3);margin-top:6px;font-style:italic}}
.range-text{{font-size:12px;color:var(--tx3);margin-top:8px}}
.ts{{font-size:11px;color:var(--tx3);margin-top:14px;letter-spacing:.04em}}

.drv-card{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:24px 28px;margin-bottom:20px}}
.drv-title{{font-size:14px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}}
.drv-sub{{font-size:11px;font-weight:500;text-transform:none;letter-spacing:0;color:var(--tx3);margin-left:8px}}
.drv-row{{padding:12px 0;border-bottom:1px solid var(--bd2)}}
.drv-row:last-child{{border-bottom:none}}
.drv-name{{font-size:14px;font-weight:600;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:6px}}
.drv-desc{{font-weight:400;font-size:12px;color:var(--tx2)}}
.drv-bar-w{{height:6px;background:var(--bd);border-radius:3px;overflow:hidden;margin:6px 0}}
.drv-bar{{height:100%;border-radius:3px;transition:width .6s ease}}
.drv-meta{{display:flex;gap:14px;font-size:12px;color:var(--tx2);flex-wrap:wrap;margin-top:4px}}
.drv-dir{{font-weight:600}}
.drv-num{{color:var(--tx3)}}
.drv-why{{font-size:13px;color:#3a3a3c;margin-top:6px;line-height:1.5;background:#fafafa;padding:8px 12px;border-radius:8px;border-left:3px solid var(--bd)}}

.rpt{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:36px 40px;margin-bottom:20px}}
.rpt-h2{{font-size:19px;font-weight:700;margin:30px 0 12px;color:var(--tx);border-bottom:1px solid var(--bd);padding-bottom:6px}}
.rpt-h2:first-child{{margin-top:0}}
.rpt-h3{{font-size:15px;font-weight:700;margin:22px 0 8px;color:var(--tx)}}
.rpt-h4{{font-size:13px;font-weight:700;margin:18px 0 6px;color:var(--tx2);text-transform:uppercase;letter-spacing:.05em}}
.rpt-p{{font-size:15px;line-height:1.78;color:#333;margin-bottom:12px}}
.rpt-ol,.rpt-ul{{padding-left:24px;margin:8px 0 14px}}
.rpt-ol li,.rpt-ul li{{font-size:15px;line-height:1.78;color:#333;margin-bottom:6px}}
.rpt-hr{{border:none;border-top:1px solid var(--bd);margin:24px 0}}
.rpt-code{{font-family:var(--fmono);font-size:13px;background:#f0f0f3;padding:1px 6px;border-radius:4px;color:#333}}
.rpt-table-wrap{{overflow-x:auto;margin:14px 0 18px}}
.rpt-table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff}}
.rpt-table thead{{background:#fafafa}}
.rpt-table th{{font-weight:700;color:var(--tx);padding:10px 12px;border-bottom:2px solid var(--bd);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
.rpt-table td{{padding:9px 12px;border-bottom:1px solid var(--bd2);color:#333;line-height:1.5}}
.rpt-table tbody tr:last-child td{{border-bottom:none}}
.rpt-table tbody tr:hover{{background:#fafafa}}

.rb-card{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:22px 28px;margin-bottom:20px}}
.rb-title{{font-size:13px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}}
.rb-degen{{font-weight:500;text-transform:none;letter-spacing:0;color:var(--tx3);font-size:11px}}
.rb-axis{{position:relative;height:30px;margin:8px 0}}
.rb-track{{position:absolute;top:14px;left:0;right:0;height:2px;background:var(--bd2)}}
.rb-dot{{position:absolute;top:8px;width:14px;height:14px;border-radius:50%;transform:translateX(-50%);border:2px solid var(--card);box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.rb-dot-final{{width:18px;height:18px;top:6px;border:2px solid var(--card)}}
.rb-axlabels{{display:flex;justify-content:space-between;font-size:11px;color:var(--tx3);font-family:var(--fmono);margin-top:6px}}

.det{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);margin-bottom:20px;overflow:hidden}}
.det summary{{padding:16px 32px;font-size:13px;font-weight:600;cursor:pointer;user-select:none;list-style:none;display:flex;align-items:center;gap:8px;color:var(--tx2);text-transform:uppercase;letter-spacing:.06em}}
.det summary::-webkit-details-marker{{display:none}}
.det summary::before{{content:"▸";font-size:12px;transition:transform .2s}}
.det[open] summary::before{{transform:rotate(90deg)}}
.det-b{{padding:0 32px 22px}}
.json{{font-family:var(--fmono);font-size:11px;line-height:1.55;background:#fafafa;border-radius:10px;padding:14px;overflow-x:auto;max-height:420px;overflow-y:auto;white-space:pre;color:#424245}}

.ft{{text-align:center;font-size:11px;color:var(--tx3);margin-top:24px;letter-spacing:.05em}}
</style>
</head>
<body>
<div class="c">

<div class="hero">
  <div class="hero-t">{_esc(title)}</div>
  <div class="hero-s">Sample {tid}</div>
  <span class="badge" style="background:{badge_color}">{_esc(verdict_text)}</span>
  {f'<div class="iv">{_esc(interval_text)}</div>' if interval_text else ""}
  <div class="conf">
    <div class="conf-l">Confidence</div>
    <div class="conf-tr"><div class="conf-f" style="width:{conf:.1f}%;background:{badge_color}"></div></div>
    <div class="conf-n" style="color:{badge_color}">{conf:.1f}%</div>
  </div>
  <div class="basis">{_esc(basis_text)}</div>
  {f'<div class="basis-rationale">{_esc(basis_rationale)}</div>' if basis_rationale else ""}
  {f'<div class="range-text">{_esc(target_range_text)}</div>' if target_range_text else ""}
  <div class="ts">{ts}</div>
</div>

{drivers_card}

<div class="rpt">
{report_body}
</div>

{robustness_card}

<details class="det">
  <summary>Technical Details (raw evidence)</summary>
  <div class="det-b">
    <pre class="json">{_esc(raw_json)}</pre>
  </div>
</details>

<div class="ft">MolActAgent · {ts} · sample {tid} · regression · {n_comp} tuple(s) against {n_total} reference samples</div>
</div>
</body>
</html>"""


# ── Aggregate interpretability rendering ────────────────────────────
#
# All three sister renderers turn a list of pipeline outcomes into a
# self-contained "why did MolActAgent decide that?" artefact in three
# different formats:
#
#   * text  — fixed-width banner for the terminal
#   * md    — Markdown file (interpretability.md)
#   * html  — styled HTML file (interpretability.html)
#
# All three are derived from the SAME data already in
# ``outcome.raw_result`` (no extra LLM calls needed) so they're free —
# users get them by default in every run.


def _truncate_text(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def _successful_outcomes(outcomes) -> list:
    return [
        o for o in outcomes
        if not getattr(o, "error", None) and getattr(o, "raw_result", None)
    ]


def _per_class_votes_text(vote_summary: dict, class_labels) -> str:
    """Render '<cls>: N (W.WW)' triples for the vote summary."""
    per = vote_summary.get("per_class", {}) or {}
    parts = []
    for cls in (class_labels or []):
        entry = per.get(cls, {}) or {}
        parts.append(
            f"{cls}={entry.get('votes', 0)} "
            f"({entry.get('weighted', 0):.2f})"
        )
    # Include extras not in class_labels (defensive fallback)
    for cls, entry in per.items():
        if cls not in (class_labels or []):
            parts.append(
                f"{cls}={entry.get('votes', 0)} "
                f"({entry.get('weighted', 0):.2f})"
            )
    return "  ".join(parts)


def _is_regression_outcomes(outcomes) -> bool:
    """Detect whether a list of outcomes is from a regression run."""
    for o in outcomes:
        raw = getattr(o, "raw_result", {}) or {}
        task = raw.get("task") or getattr(o, "task", None)
        if task == "regression":
            return True
        if task == "classification":
            return False
    return False


def render_interpretability_text(
    outcomes,
    *,
    class_labels=None,
) -> str:
    """Build the fixed-width banner shown in the terminal after a run.

    Returns ``""`` if there are no successful outcomes to summarise.
    Branches on the outcomes' task field.
    """
    succ = _successful_outcomes(outcomes)
    if not succ:
        return ""
    if _is_regression_outcomes(succ):
        return _render_regression_interpretability_text(succ)
    cls_list = list(class_labels) if class_labels else []
    cls_text = ", ".join(cls_list) if cls_list else "(class set unavailable)"
    lines: list[str] = [
        "=" * 60,
        "INTERPRETABILITY — WHY MOLACTAGENT SAID WHAT IT SAID",
        "=" * 60,
        f"Candidate classes: {cls_text}",
        (
            "For every test sample MolActAgent shows the top features "
            "that drove the decision (weighted by how many and how "
            "strongly the per-tuple analyses cited them), the per-class "
            "vote tally, and how the final confidence was calibrated."
        ),
        "",
    ]
    for o in succ:
        r = o.raw_result
        lines.append(
            f"── {o.sample_id}  →  {o.predicted_label}  "
            f"({o.confidence * 100:.0f}% confidence)"
        )
        reason = r.get("reason") or ""
        if reason:
            lines.append(f"   Verdict rationale: {reason}")
        vs = r.get("vote_summary") or {}
        if vs:
            lines.append(
                "   Tuple votes: "
                + _per_class_votes_text(vs, cls_list)
                + f"  Unknown={vs.get('num_unknown', 0)}"
            )
        cb = r.get("confidence_basis") or {}
        if cb:
            lines.append(
                "   Confidence basis: "
                f"margin={cb.get('margin_score', 0):.2f}  "
                f"sample_size={cb.get('sample_size_score', 0):.2f}  "
                f"coverage={cb.get('coverage_score', 0):.2f}"
            )
            if cb.get("rationale"):
                lines.append(f"                     ({cb['rationale']})")
        drivers = (r.get("driver_attribution") or [])[:5]
        if drivers:
            lines.append(
                "   Top drivers (feature · favoured class · strength · citations):"
            )
            for fa in drivers:
                dom = fa.get("dominant_class") or "Mixed"
                arrow = _arrow_for(dom, cls_list)
                weighted = fa.get("weighted_score", 0)
                n_cited = fa.get("n_cited", 0)
                feat = fa.get("feature", "?")
                rationales = fa.get("rationales") or []
                why = (
                    f" — {_truncate_text(str(rationales[0]), 120)}"
                    if rationales else ""
                )
                lines.append(
                    f"     • {feat}  {arrow}  "
                    f"weighted={weighted:+.2f}  cited={n_cited}{why}"
                )
        else:
            lines.append("   Top drivers: (none cited; verdict is Inconclusive)")
        sup = r.get("supporting_evidence") or ""
        opp = r.get("opposing_evidence") or ""
        if sup:
            lines.append(f"   Supporting evidence: {_truncate_text(sup, 220)}")
        if opp:
            lines.append(f"   Opposing evidence:   {_truncate_text(opp, 220)}")
        ref_info = r.get("reference_info") or {}
        if ref_info:
            per_cls_nbr = ref_info.get("class_neighbor_counts", {}) or {}
            nbr_text = ", ".join(f"{n} {cls}" for cls, n in per_cls_nbr.items())
            lines.append(
                "   Reference context: "
                f"{nbr_text} neighbours from "
                f"{ref_info.get('total_reference_samples', '?')} ref. samples "
                f"({ref_info.get('n_comparisons', '?')} tuple comparisons)"
            )
        if getattr(o, "explanation_html_path", None):
            lines.append(f"   Full HTML report: {o.explanation_html_path}")
        lines.append("")
    lines.append("=" * 60)
    lines.append(
        "Tip: run with `--per-sample` to also produce the per-sample "
        "HTML / Markdown / JSON reports (one extra LLM call per sample)."
    )
    lines.append("=" * 60)
    return "\n".join(lines)


def render_interpretability_md(
    outcomes,
    *,
    class_labels=None,
    title: str = "MolActAgent — Interpretability Report",
) -> str:
    """Markdown file saved alongside summary.csv. Branches on task."""
    succ = _successful_outcomes(outcomes)
    if not succ:
        return ""
    if _is_regression_outcomes(succ):
        return _render_regression_interpretability_md(succ, title)
    cls_list = list(class_labels) if class_labels else []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"# {title}",
        "",
        f"_Generated {ts}._  ",
        "Candidate classes: " + ", ".join(f"**{c}**" for c in cls_list) + ".",
        "",
        (
            "MolActAgent's verdicts are aggregations of per-tuple "
            "multi-way comparisons against the reference set. For each "
            "test sample below we surface the per-class vote tally, "
            "confidence calibration, and the top features that drove "
            "the decision — every column is grounded in the raw "
            "evidence inside `summary.json` / per-sample `*.json`."
        ),
        "",
        "## Verdicts at a glance",
        "",
    ]

    # Header row: Sample | Verdict | Confidence | <one column per class> | Inconclusive
    header_cells = ["Sample", "Verdict", "Confidence"]
    header_cells.extend(f"{cls} votes" for cls in cls_list)
    header_cells.append("Inconclusive")
    sep_cells = [":---" for _ in header_cells]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(sep_cells) + " |")
    for o in succ:
        vs = (o.raw_result.get("vote_summary") or {})
        per = vs.get("per_class", {}) or {}
        cells = [
            f"`{o.sample_id}`",
            f"**{o.predicted_label}**",
            f"{o.confidence * 100:.0f}%",
        ]
        for cls in cls_list:
            entry = per.get(cls, {}) or {}
            cells.append(
                f"{entry.get('votes', 0)} ({entry.get('weighted', 0):.2f})"
            )
        cells.append(str(vs.get("num_unknown", 0)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "---", ""])

    for o in succ:
        r = o.raw_result
        lines.append(
            f"## {o.sample_id} → {o.predicted_label} "
            f"({o.confidence * 100:.0f}% confidence)"
        )
        lines.append("")
        reason = r.get("reason") or ""
        if reason:
            lines.append(f"**Verdict rationale.** {reason}")
            lines.append("")

        # Vote + confidence
        vs = r.get("vote_summary") or {}
        cb = r.get("confidence_basis") or {}
        if vs or cb:
            lines.append("**Voting and confidence.**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| :--- | :--- |")
            if vs:
                per = vs.get("per_class", {}) or {}
                tally = " · ".join(
                    f"{cls}: {(per.get(cls) or {}).get('votes', 0)}"
                    for cls in cls_list
                )
                weighted = " · ".join(
                    f"{cls}: {(per.get(cls) or {}).get('weighted', 0):.2f}"
                    for cls in cls_list
                )
                lines.append(
                    f"| Tuple votes | {tally} · Unknown: "
                    f"{vs.get('num_unknown', 0)} |"
                )
                lines.append(f"| Weighted votes | {weighted} |")
            if cb:
                lines.append(
                    f"| Margin score | {cb.get('margin_score', 0):.2f} |"
                )
                lines.append(
                    f"| Sample-size score | {cb.get('sample_size_score', 0):.2f} |"
                )
                lines.append(
                    f"| Coverage score | {cb.get('coverage_score', 0):.2f} |"
                )
                if cb.get("rationale"):
                    lines.append(
                        f"| Calibration note | {cb['rationale']} |"
                    )
            lines.append("")

        # Top drivers
        drivers = r.get("driver_attribution") or []
        top = drivers[:5]
        if top:
            lines.append("**Top drivers.**")
            lines.append("")
            lines.append(
                "| # | Feature | Favoured class | Weighted strength "
                "| Citations | First rationale |"
            )
            lines.append("| --- | :--- | :--- | ---: | ---: | :--- |")
            for i, fa in enumerate(top, 1):
                dom = fa.get("dominant_class") or "Mixed"
                rationales = fa.get("rationales") or []
                why = _truncate_text(str(rationales[0]), 240) if rationales else ""
                lines.append(
                    f"| {i} | `{fa.get('feature', '?')}` | {dom} "
                    f"| {fa.get('weighted_score', 0):+.2f} "
                    f"| {fa.get('n_cited', 0)} | {why} |"
                )
            lines.append("")
        else:
            lines.append("_Top drivers: none cited (verdict is Inconclusive)._")
            lines.append("")

        sup = r.get("supporting_evidence") or ""
        opp = r.get("opposing_evidence") or ""
        if sup or opp:
            if sup:
                lines.append(f"**Supporting evidence.** {_truncate_text(sup, 600)}")
                lines.append("")
            if opp:
                lines.append(f"**Opposing evidence.** {_truncate_text(opp, 600)}")
                lines.append("")

        ref_info = r.get("reference_info") or {}
        if ref_info:
            per_cls_nbr = ref_info.get("class_neighbor_counts", {}) or {}
            nbr_text = ", ".join(f"{n} {cls}" for cls, n in per_cls_nbr.items())
            lines.append(
                f"**Reference context.** "
                f"{nbr_text} neighbours selected from "
                f"{ref_info.get('total_reference_samples', '?')} reference samples "
                f"across {ref_info.get('n_comparisons', '?')} tuple comparisons."
            )
            lines.append("")

        if getattr(o, "explanation_html_path", None):
            lines.append(
                f"**Per-sample HTML report:** `{o.explanation_html_path}`"
            )
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(
        "_Tip: pass `--per-sample` (CLI) or `per_sample_reports=True` "
        "(Python API) to also generate full HTML / Markdown / JSON "
        "reports per test sample. Each per-sample report costs one "
        "additional LLM call._"
    )
    return "\n".join(lines).rstrip() + "\n"


def render_interpretability_html(
    outcomes,
    *,
    class_labels=None,
    title: str = "MolActAgent — Interpretability Report",
) -> str:
    """Self-contained HTML version of the aggregate interpretability
    report. Branches on task.
    """
    succ = _successful_outcomes(outcomes)
    if not succ:
        return ""
    if _is_regression_outcomes(succ):
        return _render_regression_interpretability_html(succ, title)
    cls_list = list(class_labels) if class_labels else []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Summary table — columns: Sample | Verdict | Confidence | one per class | Unknown
    header_cells = ["<th>Sample</th>", "<th>Verdict</th>", "<th>Confidence</th>"]
    for cls in cls_list:
        header_cells.append(f"<th>{_esc(cls)} votes (weighted)</th>")
    header_cells.append("<th>Inconclusive</th>")

    summary_rows: list[str] = []
    for o in succ:
        vs = o.raw_result.get("vote_summary") or {}
        per = vs.get("per_class", {}) or {}
        color = _class_color(o.predicted_label, cls_list)
        cells = [
            f'<td><code>{_esc(o.sample_id)}</code></td>',
            f'<td><span class="ip-verdict" style="background:{color}">'
            f'{_esc(o.predicted_label)}</span></td>',
            f'<td>{o.confidence * 100:.0f}%</td>',
        ]
        for cls in cls_list:
            entry = per.get(cls, {}) or {}
            cells.append(
                f'<td>{entry.get("votes", 0)} '
                f'<span class="ip-w">({entry.get("weighted", 0):.2f})</span></td>'
            )
        cells.append(f'<td>{vs.get("num_unknown", 0)}</td>')
        summary_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Per-sample sections
    sample_sections: list[str] = []
    for o in succ:
        r = o.raw_result
        color = _class_color(o.predicted_label, cls_list)
        head = (
            f'<div class="ip-sample">'
            f'<div class="ip-sample-head">'
            f'<div class="ip-sample-id"><code>{_esc(o.sample_id)}</code></div>'
            f'<div class="ip-sample-verdict">'
            f'<span class="ip-badge" style="background:{color}">{_esc(o.predicted_label)}</span>'
            f'<span class="ip-conf">{o.confidence * 100:.0f}% confidence</span>'
            f'</div>'
            f'</div>'
        )
        body: list[str] = []

        reason = r.get("reason") or ""
        if reason:
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Verdict rationale</div>'
                f'<div class="ip-row-r">{_esc(reason)}</div></div>'
            )
        vs = r.get("vote_summary") or {}
        if vs:
            per = vs.get("per_class", {}) or {}
            tally_html = " · ".join(
                f'<strong>{_esc(cls)}:</strong> '
                f'{(per.get(cls) or {}).get("votes", 0)} '
                f'(weighted {(per.get(cls) or {}).get("weighted", 0):.2f})'
                for cls in cls_list
            )
            tally_html += (
                f' · <strong>Unknown:</strong> {vs.get("num_unknown", 0)}'
            )
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Tuple votes</div>'
                f'<div class="ip-row-r">{tally_html}</div></div>'
            )
        cb = r.get("confidence_basis") or {}
        if cb:
            sub = (
                f'margin <strong>{cb.get("margin_score", 0):.2f}</strong> · '
                f'sample-size <strong>{cb.get("sample_size_score", 0):.2f}</strong> · '
                f'coverage <strong>{cb.get("coverage_score", 0):.2f}</strong>'
            )
            note = (
                f'<div class="ip-rationale">{_esc(cb["rationale"])}</div>'
                if cb.get("rationale") else ""
            )
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Confidence basis</div>'
                f'<div class="ip-row-r">{sub}{note}</div></div>'
            )

        drivers = (r.get("driver_attribution") or [])[:5]
        if drivers:
            rows = []
            for i, fa in enumerate(drivers, 1):
                dom = fa.get("dominant_class") or "Mixed"
                arrow = _arrow_for(dom, cls_list)
                d_color = _class_color(dom, cls_list)
                rationales = fa.get("rationales") or []
                why = _truncate_text(str(rationales[0]), 240) if rationales else ""
                rows.append(
                    f'<tr>'
                    f'<td>{i}</td>'
                    f'<td><code>{_esc(fa.get("feature", "?"))}</code></td>'
                    f'<td><span class="ip-arrow" style="color:{d_color}">{_esc(arrow)}</span></td>'
                    f'<td class="ip-num">{fa.get("weighted_score", 0):+.2f}</td>'
                    f'<td class="ip-num">{fa.get("n_cited", 0)}</td>'
                    f'<td class="ip-why">{_esc(why)}</td>'
                    f'</tr>'
                )
            body.append(
                '<div class="ip-section-title">Top drivers</div>'
                '<table class="ip-table">'
                '<thead><tr>'
                '<th>#</th><th>Feature</th><th>Favoured class</th>'
                '<th>Weighted</th><th>Citations</th><th>First rationale</th>'
                '</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table>'
            )
        sup = r.get("supporting_evidence") or ""
        opp = r.get("opposing_evidence") or ""
        if sup:
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Supporting</div>'
                f'<div class="ip-row-r">{_esc(_truncate_text(sup, 600))}</div></div>'
            )
        if opp:
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Opposing</div>'
                f'<div class="ip-row-r">{_esc(_truncate_text(opp, 600))}</div></div>'
            )
        ref_info = r.get("reference_info") or {}
        if ref_info:
            per_cls_nbr = ref_info.get("class_neighbor_counts", {}) or {}
            nbr_text = ", ".join(
                f"{n} {_esc(cls)}" for cls, n in per_cls_nbr.items()
            )
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Reference context</div>'
                f'<div class="ip-row-r">'
                f'{nbr_text} neighbours from '
                f'{ref_info.get("total_reference_samples", "?")} reference samples '
                f'({ref_info.get("n_comparisons", "?")} tuple comparisons)'
                f'</div></div>'
            )
        if getattr(o, "explanation_html_path", None):
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Per-sample report</div>'
                f'<div class="ip-row-r"><a href="{_esc(o.explanation_html_path)}">'
                f'{_esc(o.explanation_html_path)}</a></div></div>'
            )

        sample_sections.append(
            head + '<div class="ip-sample-body">' + "".join(body) + "</div></div>"
        )

    cls_list_html = ", ".join(f"<strong>{_esc(c)}</strong>" for c in cls_list)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>
:root{{
  --bg:#f5f5f7;--card:#fff;--tx:#1d1d1f;--tx2:#6e6e73;--tx3:#86868b;
  --bd:#e8e8ed;--bd2:#f0f0f3;
  --r:18px;--r2:12px;--sh:0 2px 12px rgba(0,0,0,.06);
  --f:-apple-system,"SF Pro Display",BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;
  --fmono:"SF Mono","Menlo","Fira Code",Consolas,monospace
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:var(--f);background:var(--bg);color:var(--tx);line-height:1.6;-webkit-font-smoothing:antialiased}}
.c{{max-width:980px;margin:0 auto;padding:36px 24px 60px}}
.hero{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:36px 36px 28px;margin-bottom:22px}}
.hero h1{{font-size:24px;font-weight:700}}
.hero .sub{{color:var(--tx2);font-size:13px;margin-top:6px}}
.hero .lede{{color:var(--tx2);font-size:13px;margin-top:18px;line-height:1.65}}

.summary-card{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:24px 32px;margin-bottom:22px}}
.summary-card h2{{font-size:13px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}}
.summary-table{{width:100%;border-collapse:collapse;font-size:13px}}
.summary-table th{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--tx2);text-align:left;padding:8px 10px;border-bottom:2px solid var(--bd)}}
.summary-table td{{padding:10px;border-bottom:1px solid var(--bd2);color:#333;vertical-align:middle}}
.summary-table tbody tr:last-child td{{border-bottom:none}}
.summary-table tbody tr:hover{{background:#fafafa}}
.summary-table .ip-w{{color:var(--tx3);font-size:11px;margin-left:2px}}
.ip-verdict{{display:inline-block;color:#fff;font-weight:600;font-size:11px;padding:3px 11px;border-radius:8px;letter-spacing:.03em}}

.ip-sample{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:24px 32px;margin-bottom:18px}}
.ip-sample-head{{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:12px;border-bottom:1px solid var(--bd2);padding-bottom:14px}}
.ip-sample-id{{font-size:18px;font-weight:700}}
.ip-sample-id code{{font-family:var(--fmono)}}
.ip-sample-verdict{{display:flex;align-items:center;gap:10px}}
.ip-badge{{display:inline-block;color:#fff;font-weight:600;font-size:13px;padding:5px 14px;border-radius:8px;letter-spacing:.03em}}
.ip-conf{{color:var(--tx2);font-size:13px}}

.ip-row{{display:flex;align-items:flex-start;gap:14px;padding:8px 0;border-bottom:1px dashed var(--bd2)}}
.ip-row:last-child{{border-bottom:none}}
.ip-row-l{{flex:0 0 160px;color:var(--tx2);font-size:11px;text-transform:uppercase;letter-spacing:.06em;font-weight:700;padding-top:2px}}
.ip-row-r{{flex:1;color:#333;font-size:13.5px;line-height:1.55}}
.ip-rationale{{color:var(--tx3);font-style:italic;font-size:12px;margin-top:4px}}

.ip-section-title{{font-size:11px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin:20px 0 10px}}
.ip-table{{width:100%;border-collapse:collapse;font-size:12.5px}}
.ip-table th{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--tx2);text-align:left;padding:9px 10px;border-bottom:2px solid var(--bd);background:#fafafa}}
.ip-table td{{padding:8px 10px;border-bottom:1px solid var(--bd2);color:#333;vertical-align:top}}
.ip-table tbody tr:last-child td{{border-bottom:none}}
.ip-table .ip-num{{font-family:var(--fmono);text-align:right;white-space:nowrap}}
.ip-table .ip-why{{color:var(--tx2);font-size:12px}}
.ip-arrow{{font-weight:700;font-family:var(--fmono);font-size:12px}}

.ft{{text-align:center;color:var(--tx3);font-size:11px;margin-top:30px;padding-top:18px;border-top:1px solid var(--bd2)}}
</style>
</head>
<body>
<div class="c">
<div class="hero">
  <h1>{_esc(title)}</h1>
  <div class="sub">Generated {ts} · candidate classes: {cls_list_html}</div>
  <div class="lede">MolActAgent's verdicts are aggregations of per-tuple multi-way comparisons against the reference set. The tables below surface the per-class vote tally, confidence calibration, and the top features that drove each decision — every value is grounded in the raw evidence stored in <code>summary.json</code> and the per-sample <code>*.json</code> files.</div>
</div>

<div class="summary-card">
  <h2>Verdicts at a glance</h2>
  <table class="summary-table">
    <thead><tr>
      {"".join(header_cells)}
    </tr></thead>
    <tbody>
      {"".join(summary_rows)}
    </tbody>
  </table>
</div>

{"".join(sample_sections)}

<div class="ft">MolActAgent · {ts} · pass <code>--per-sample</code> for full per-sample HTML / Markdown reports (one extra LLM call per sample).</div>
</div>
</body>
</html>"""


# ── Regression interpretability renderers ───────────────────────────


def _abs_error(predicted: float | None, true_str: str | None) -> float | None:
    """Compute absolute error from a numeric prediction and a
    string-typed truth value (which may be a number or a label)."""
    if predicted is None or true_str is None:
        return None
    try:
        true_v = float(true_str)
    except (TypeError, ValueError):
        return None
    return abs(float(predicted) - true_v)


def _render_regression_interpretability_text(succ) -> str:
    """Fixed-width terminal banner for a regression run."""
    lines: list[str] = [
        "=" * 60,
        "INTERPRETABILITY — REGRESSION RUN",
        "=" * 60,
        (
            "For every test sample MolActAgent shows the predicted "
            "value and interval, the agreement across tuples, and "
            "the top features whose values pushed the prediction up "
            "or down."
        ),
        "",
    ]
    for o in succ:
        r = o.raw_result or {}
        pv = r.get("predicted_value")
        iv = r.get("interval") or (None, None)
        true = o.true_label
        err = _abs_error(pv, true)
        err_text = f"  abs_err={err:.4g}" if err is not None else ""
        if pv is None:
            lines.append(
                f"── {o.sample_id}  →  REFUSED   "
                f"(true={true if true is not None else '?'}{err_text})"
            )
        else:
            lo = iv[0] if iv[0] is not None else pv
            hi = iv[1] if iv[1] is not None else pv
            lines.append(
                f"── {o.sample_id}  →  {_fmt_target_value(pv)}  "
                f"interval [{_fmt_target_value(lo)}, "
                f"{_fmt_target_value(hi)}]  "
                f"({o.confidence * 100:.0f}% confidence)  "
                f"true={true if true is not None else '?'}{err_text}"
            )
        reason = r.get("reason") or ""
        if reason:
            lines.append(f"   Verdict rationale: {reason}")
        cb = r.get("confidence_basis") or {}
        if cb:
            lines.append(
                "   Confidence basis: "
                f"agreement={cb.get('agreement_score', 0):.2f}  "
                f"avg_tuple_conf={cb.get('avg_tuple_confidence', 0):.2f}"
            )
            if cb.get("rationale"):
                lines.append(f"                     ({cb['rationale']})")
        drivers = (r.get("driver_attribution") or [])[:5]
        if drivers:
            lines.append(
                "   Top drivers (feature · direction · strength · citations):"
            )
            for fa in drivers:
                direction = fa.get("dominant_direction") or "mixed"
                arrow = (
                    "↑" if direction == "increases"
                    else "↓" if direction == "decreases"
                    else "~"
                )
                weighted = fa.get("weighted_score", 0)
                n_cited = fa.get("n_cited", 0)
                feat = fa.get("feature", "?")
                rationales = fa.get("rationales") or []
                why = (
                    f" — {_truncate_text(str(rationales[0]), 120)}"
                    if rationales else ""
                )
                lines.append(
                    f"     • {feat}  {arrow}  {direction}  "
                    f"weighted={weighted:+.2f}  cited={n_cited}{why}"
                )
        else:
            lines.append("   Top drivers: (none cited)")
        if getattr(o, "explanation_html_path", None):
            lines.append(f"   Full HTML report: {o.explanation_html_path}")
        lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def _render_regression_interpretability_md(
    succ, title: str = "MolActAgent — Interpretability Report",
) -> str:
    """Markdown interpretability for a regression run."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"# {title}",
        "",
        f"_Generated {ts}._  ",
        "Regression run — predicted continuous values, intervals, "
        "and per-feature directional drivers.",
        "",
        "## Predictions at a glance",
        "",
        "| Sample | Predicted | Interval | Confidence | True | Abs error |",
        "| :--- | ---: | :--- | ---: | ---: | ---: |",
    ]
    for o in succ:
        r = o.raw_result or {}
        pv = r.get("predicted_value")
        iv = r.get("interval") or (None, None)
        true = o.true_label
        err = _abs_error(pv, true)
        err_str = f"{err:.4g}" if err is not None else "—"
        true_str = true if true is not None else "—"
        if pv is None:
            pv_str = "REFUSED"
            iv_str = "—"
        else:
            pv_str = _fmt_target_value(pv)
            if iv[0] is not None and iv[1] is not None:
                iv_str = f"[{_fmt_target_value(iv[0])}, {_fmt_target_value(iv[1])}]"
            else:
                iv_str = "—"
        lines.append(
            f"| `{o.sample_id}` | **{pv_str}** | {iv_str} "
            f"| {o.confidence * 100:.0f}% | {true_str} | {err_str} |"
        )
    lines.extend(["", "---", ""])

    for o in succ:
        r = o.raw_result or {}
        pv = r.get("predicted_value")
        pv_disp = "REFUSED" if pv is None else _fmt_target_value(pv)
        lines.append(
            f"## {o.sample_id} → {pv_disp} "
            f"({o.confidence * 100:.0f}% confidence)"
        )
        lines.append("")
        reason = r.get("reason") or ""
        if reason:
            lines.append(f"**Verdict rationale.** {reason}")
            lines.append("")
        cb = r.get("confidence_basis") or {}
        if cb:
            lines.append("**Confidence basis.**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| :--- | :--- |")
            lines.append(
                f"| Agreement score | {cb.get('agreement_score', 0):.2f} |"
            )
            lines.append(
                f"| Avg per-tuple confidence | "
                f"{cb.get('avg_tuple_confidence', 0):.2f} |"
            )
            if cb.get("rationale"):
                lines.append(f"| Calibration note | {cb['rationale']} |")
            lines.append("")
        drivers = (r.get("driver_attribution") or [])[:5]
        if drivers:
            lines.append("**Top drivers.**")
            lines.append("")
            lines.append(
                "| # | Feature | Direction | Weighted strength "
                "| Citations | First rationale |"
            )
            lines.append("| --- | :--- | :--- | ---: | ---: | :--- |")
            for i, fa in enumerate(drivers, 1):
                direction = fa.get("dominant_direction") or "mixed"
                arrow = (
                    "↑" if direction == "increases"
                    else "↓" if direction == "decreases" else "~"
                )
                rationales = fa.get("rationales") or []
                why = _truncate_text(str(rationales[0]), 240) if rationales else ""
                lines.append(
                    f"| {i} | `{fa.get('feature', '?')}` "
                    f"| {arrow} {direction} "
                    f"| {fa.get('weighted_score', 0):+.2f} "
                    f"| {fa.get('n_cited', 0)} | {why} |"
                )
            lines.append("")
        if getattr(o, "explanation_html_path", None):
            lines.append(
                f"**Per-sample HTML report:** `{o.explanation_html_path}`"
            )
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_regression_interpretability_html(
    succ, title: str = "MolActAgent — Interpretability Report",
) -> str:
    """Self-contained HTML interpretability for a regression run."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary_rows: list[str] = []
    for o in succ:
        r = o.raw_result or {}
        pv = r.get("predicted_value")
        iv = r.get("interval") or (None, None)
        ri = r.get("reference_info") or {}
        target_range = ri.get("target_range")
        color = _regression_color(pv, target_range, refused=pv is None)
        true = o.true_label
        err = _abs_error(pv, true)
        err_str = f"{err:.4g}" if err is not None else "—"
        true_str = true if true is not None else "—"
        if pv is None:
            pv_html = (
                f'<span class="ip-verdict" style="background:{color}">REFUSED</span>'
            )
            iv_str = "—"
        else:
            pv_html = (
                f'<span class="ip-verdict" style="background:{color}">'
                f'{_esc(_fmt_target_value(pv))}</span>'
            )
            if iv[0] is not None and iv[1] is not None:
                iv_str = (
                    f"[{_fmt_target_value(iv[0])}, "
                    f"{_fmt_target_value(iv[1])}]"
                )
            else:
                iv_str = "—"
        summary_rows.append(
            f'<tr>'
            f'<td><code>{_esc(o.sample_id)}</code></td>'
            f'<td>{pv_html}</td>'
            f'<td>{_esc(iv_str)}</td>'
            f'<td>{o.confidence * 100:.0f}%</td>'
            f'<td>{_esc(true_str)}</td>'
            f'<td>{_esc(err_str)}</td>'
            f'</tr>'
        )

    sample_sections: list[str] = []
    for o in succ:
        r = o.raw_result or {}
        pv = r.get("predicted_value")
        ri = r.get("reference_info") or {}
        target_range = ri.get("target_range")
        color = _regression_color(pv, target_range, refused=pv is None)
        head_label = "REFUSED" if pv is None else _fmt_target_value(pv)
        head = (
            f'<div class="ip-sample">'
            f'<div class="ip-sample-head">'
            f'<div class="ip-sample-id"><code>{_esc(o.sample_id)}</code></div>'
            f'<div class="ip-sample-verdict">'
            f'<span class="ip-badge" style="background:{color}">{_esc(head_label)}</span>'
            f'<span class="ip-conf">{o.confidence * 100:.0f}% confidence</span>'
            f'</div>'
            f'</div>'
        )
        body: list[str] = []
        reason = r.get("reason") or ""
        if reason:
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Verdict rationale</div>'
                f'<div class="ip-row-r">{_esc(reason)}</div></div>'
            )
        cb = r.get("confidence_basis") or {}
        if cb:
            sub = (
                f'agreement <strong>{cb.get("agreement_score", 0):.2f}</strong> · '
                f'avg-tuple-conf <strong>{cb.get("avg_tuple_confidence", 0):.2f}</strong>'
            )
            note = (
                f'<div class="ip-rationale">{_esc(cb["rationale"])}</div>'
                if cb.get("rationale") else ""
            )
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Confidence basis</div>'
                f'<div class="ip-row-r">{sub}{note}</div></div>'
            )

        drivers = (r.get("driver_attribution") or [])[:5]
        if drivers:
            rows = []
            for i, fa in enumerate(drivers, 1):
                direction = fa.get("dominant_direction") or "mixed"
                if direction == "increases":
                    d_color = "#ff3b30"; arrow_str = "↑ increases"
                elif direction == "decreases":
                    d_color = "#007aff"; arrow_str = "↓ decreases"
                else:
                    d_color = _INCONCLUSIVE_COLOR; arrow_str = "~ mixed"
                rationales = fa.get("rationales") or []
                why = _truncate_text(str(rationales[0]), 240) if rationales else ""
                rows.append(
                    f'<tr>'
                    f'<td>{i}</td>'
                    f'<td><code>{_esc(fa.get("feature", "?"))}</code></td>'
                    f'<td><span class="ip-arrow" style="color:{d_color}">{_esc(arrow_str)}</span></td>'
                    f'<td class="ip-num">{fa.get("weighted_score", 0):+.2f}</td>'
                    f'<td class="ip-num">{fa.get("n_cited", 0)}</td>'
                    f'<td class="ip-why">{_esc(why)}</td>'
                    f'</tr>'
                )
            body.append(
                '<div class="ip-section-title">Top drivers</div>'
                '<table class="ip-table">'
                '<thead><tr>'
                '<th>#</th><th>Feature</th><th>Direction</th>'
                '<th>Weighted</th><th>Citations</th><th>First rationale</th>'
                '</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table>'
            )
        if getattr(o, "explanation_html_path", None):
            body.append(
                f'<div class="ip-row"><div class="ip-row-l">Per-sample report</div>'
                f'<div class="ip-row-r"><a href="{_esc(o.explanation_html_path)}">'
                f'{_esc(o.explanation_html_path)}</a></div></div>'
            )
        sample_sections.append(
            head + '<div class="ip-sample-body">' + "".join(body) + "</div></div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>
:root{{
  --bg:#f5f5f7;--card:#fff;--tx:#1d1d1f;--tx2:#6e6e73;--tx3:#86868b;
  --bd:#e8e8ed;--bd2:#f0f0f3;
  --r:18px;--sh:0 2px 12px rgba(0,0,0,.06);
  --f:-apple-system,"SF Pro Display",BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;
  --fmono:"SF Mono","Menlo","Fira Code",Consolas,monospace
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:var(--f);background:var(--bg);color:var(--tx);line-height:1.6}}
.c{{max-width:980px;margin:0 auto;padding:36px 24px 60px}}
.hero{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:36px 36px 28px;margin-bottom:22px}}
.hero h1{{font-size:24px;font-weight:700}}
.hero .sub{{color:var(--tx2);font-size:13px;margin-top:6px}}
.hero .lede{{color:var(--tx2);font-size:13px;margin-top:18px;line-height:1.65}}
.summary-card{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:24px 32px;margin-bottom:22px}}
.summary-card h2{{font-size:13px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}}
.summary-table{{width:100%;border-collapse:collapse;font-size:13px}}
.summary-table th{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--tx2);text-align:left;padding:8px 10px;border-bottom:2px solid var(--bd)}}
.summary-table td{{padding:10px;border-bottom:1px solid var(--bd2);color:#333;vertical-align:middle}}
.ip-verdict{{display:inline-block;color:#fff;font-weight:600;font-size:11px;padding:3px 11px;border-radius:8px;letter-spacing:.03em;font-family:var(--fmono)}}
.ip-sample{{background:var(--card);border-radius:var(--r);box-shadow:var(--sh);padding:24px 32px;margin-bottom:18px}}
.ip-sample-head{{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:12px;border-bottom:1px solid var(--bd2);padding-bottom:14px}}
.ip-sample-id{{font-size:18px;font-weight:700}}
.ip-sample-id code{{font-family:var(--fmono)}}
.ip-sample-verdict{{display:flex;align-items:center;gap:10px}}
.ip-badge{{display:inline-block;color:#fff;font-weight:600;font-size:13px;padding:5px 14px;border-radius:8px;letter-spacing:.03em;font-family:var(--fmono)}}
.ip-conf{{color:var(--tx2);font-size:13px}}
.ip-row{{display:flex;align-items:flex-start;gap:14px;padding:8px 0;border-bottom:1px dashed var(--bd2)}}
.ip-row:last-child{{border-bottom:none}}
.ip-row-l{{flex:0 0 160px;color:var(--tx2);font-size:11px;text-transform:uppercase;letter-spacing:.06em;font-weight:700;padding-top:2px}}
.ip-row-r{{flex:1;color:#333;font-size:13.5px;line-height:1.55}}
.ip-rationale{{color:var(--tx3);font-style:italic;font-size:12px;margin-top:4px}}
.ip-section-title{{font-size:11px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin:20px 0 10px}}
.ip-table{{width:100%;border-collapse:collapse;font-size:12.5px}}
.ip-table th{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--tx2);text-align:left;padding:9px 10px;border-bottom:2px solid var(--bd);background:#fafafa}}
.ip-table td{{padding:8px 10px;border-bottom:1px solid var(--bd2);color:#333;vertical-align:top}}
.ip-table .ip-num{{font-family:var(--fmono);text-align:right;white-space:nowrap}}
.ip-table .ip-why{{color:var(--tx2);font-size:12px}}
.ip-arrow{{font-weight:700;font-family:var(--fmono);font-size:12px}}
.ft{{text-align:center;color:var(--tx3);font-size:11px;margin-top:30px;padding-top:18px;border-top:1px solid var(--bd2)}}
</style>
</head>
<body>
<div class="c">
<div class="hero">
  <h1>{_esc(title)}</h1>
  <div class="sub">Generated {ts} · regression run</div>
  <div class="lede">For each test sample below, MolActAgent shows the predicted continuous value, an honest uncertainty interval, the model's confidence (combining tuple agreement and per-tuple LLM confidence), and the per-feature drivers that pushed the prediction up or down.</div>
</div>

<div class="summary-card">
  <h2>Predictions at a glance</h2>
  <table class="summary-table">
    <thead><tr>
      <th>Sample</th><th>Predicted</th><th>Interval</th>
      <th>Confidence</th><th>True</th><th>Abs error</th>
    </tr></thead>
    <tbody>
      {"".join(summary_rows)}
    </tbody>
  </table>
</div>

{"".join(sample_sections)}

<div class="ft">MolActAgent · {ts} · pass <code>--per-sample</code> for full per-sample HTML / Markdown reports (one extra LLM call per sample).</div>
</div>
</body>
</html>"""
