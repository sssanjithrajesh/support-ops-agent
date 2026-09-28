"""Render a Scorecard as a terminal table, a Markdown table, and an SVG bar chart (no matplotlib).

Regenerated on every eval run and committed, so the chart in the README is always the current number.
Lifted from contract-compliance-reviewer/evals/report.py with the columns changed.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals.scoring import HeldoutMetrics, Metrics, Scorecard

COLUMNS = [
    ("queue_accuracy", "queue acc", "#2f80ed"),
    ("priority_accuracy", "priority acc", "#9b59b6"),
    ("action_accuracy", "action acc", "#27ae60"),
    ("escalation_recall", "escalation recall", "#f2994a"),
    ("kb_top1", "kb top-1", "#16a085"),
]


def print_terminal(card: Scorecard, heldout: HeldoutMetrics | None = None) -> None:
    console = Console()
    table = Table(title=f"Scorecard: {card.planner} planner, {card.overall.n} hand-labelled cases")
    table.add_column("expected action")
    table.add_column("n", justify="right")
    for _, label, _ in COLUMNS:
        table.add_column(label, justify="right")
    table.add_column("unsafe auto-send", justify="right")

    def row(label: str, m: Metrics, bold: bool = False) -> None:
        table.add_row(
            label,
            str(m.n),
            *[f"{getattr(m, key):.0%}" for key, _, _ in COLUMNS],
            f"{m.unsafe_autosend_rate:.0%}",
            style="bold" if bold else "",
        )

    for action, m in sorted(card.by_action.items()):
        row(action, m)
    table.add_section()
    row("OVERALL", card.overall, bold=True)
    console.print(table)
    console.print(
        f"dataset label agrees with the human label on {card.overall.dataset_label_agreement:.0%} of dataset cases"
    )
    if heldout:
        console.print(
            f"held-out {heldout.n} tickets vs dataset labels: queue {heldout.queue_accuracy:.0%} "
            f"(majority baseline {heldout.majority_queue_baseline:.0%}), priority {heldout.priority_accuracy:.0%}; "
            f"actions {heldout.actions}"
        )


def to_markdown(card: Scorecard, heldout: HeldoutMetrics | None = None) -> str:
    head = "| expected action | n | " + " | ".join(label for _, label, _ in COLUMNS) + " | unsafe auto-send |"
    sep = "|---|---:|" + "---:|" * len(COLUMNS) + "---:|"
    lines = [f"### Scorecard: `{card.planner}` planner", "", head, sep]

    def cells(m: Metrics) -> str:
        return " | ".join(f"{getattr(m, key):.0%}" for key, _, _ in COLUMNS) + f" | {m.unsafe_autosend_rate:.0%}"

    for action, m in sorted(card.by_action.items()):
        lines.append(f"| {action} | {m.n} | {cells(m)} |")
    m = card.overall
    lines.append(f"| **overall** | **{m.n}** | {cells(m)} |")
    lines += [
        "",
        f"Dataset label agrees with the human label on {m.dataset_label_agreement:.0%} of the dataset-sourced cases.",
    ]
    if heldout:
        lines += [
            "",
            f"Held-out {heldout.n} tickets against the dataset's own labels: queue accuracy "
            f"{heldout.queue_accuracy:.0%} (majority-class baseline {heldout.majority_queue_baseline:.0%}), "
            f"priority accuracy {heldout.priority_accuracy:.0%}. Actions: "
            + ", ".join(f"{k} {v}" for k, v in sorted(heldout.actions.items()))
            + ".",
        ]
    return "\n".join(lines) + "\n"


def to_svg(card: Scorecard) -> str:
    rows = sorted(card.by_action.items()) + [("OVERALL", card.overall)]
    left, top, right, bottom = 150, 62, 30, 40
    row_h, bar_h, gap = 8 * len(COLUMNS) + 8, 6, 1
    width = 760
    plot_w = width - left - right
    height = top + row_h * len(rows) + bottom
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>',
        f'<text x="{left}" y="24" font-size="15" font-weight="600" fill="#1a1a1a">Eval scorecard | {card.planner} '
        f"planner | {card.overall.n} hand-labelled cases | unsafe auto-send "
        f"{card.overall.unsafe_autosend_rate:.0%}</text>",
    ]
    lx = left
    for _, label, colour in COLUMNS:
        out.append(f'<rect x="{lx}" y="36" width="10" height="10" fill="{colour}" rx="2"/>')
        out.append(f'<text x="{lx + 14}" y="45" fill="#444">{label}</text>')
        lx += 118
    for pct in (0, 25, 50, 75, 100):
        x = left + plot_w * pct / 100
        out.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{height - bottom + 4}" stroke="#e6e6e6"/>')
        out.append(f'<text x="{x:.1f}" y="{height - bottom + 18}" text-anchor="middle" fill="#888">{pct}%</text>')
    for i, (label, m) in enumerate(rows):
        y = top + i * row_h
        weight = ' font-weight="600"' if label == "OVERALL" else ""
        out.append(
            f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" fill="#1a1a1a"{weight}>{label} (n={m.n})</text>'
        )
        for j, (key, _, colour) in enumerate(COLUMNS):
            w = plot_w * getattr(m, key)
            by = y + 2 + j * (bar_h + gap)
            out.append(f'<rect x="{left}" y="{by}" width="{w:.1f}" height="{bar_h}" fill="{colour}" rx="1"/>')
    out.append("</svg>")
    return "\n".join(out)


def write_reports(card: Scorecard, heldout: HeldoutMetrics | None, svg_path: Path, md_path: Path) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(to_svg(card), encoding="utf-8")
    md_path.write_text(to_markdown(card, heldout), encoding="utf-8")
