"""Dependency-free SVG charts for the README and docs (no matplotlib)."""

from __future__ import annotations

from xml.sax.saxutils import escape

FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"
COLOURS = {"escalate": "#b03a2e", "review": "#d9a21b", "auto_send": "#2b7a4b"}


def _legend(x: int, colour: str, text: str) -> str:
    return (
        f"<rect x='{x}' y='8' width='12' height='12' fill='{colour}'/><text x='{x + 16}' y='18'>{escape(text)}</text>"
    )


def actions_by_queue(rows: list[tuple[str, dict[str, int]]], width: int = 640) -> str:
    """Stacked horizontal bars per queue: escalate / review / auto_send counts."""
    if not rows:
        return ""
    row_h, top, left = 28, 36, 210
    mx = max(sum(c.values()) for _, c in rows) or 1
    scale = (width - left - 70) / mx
    h = top + row_h * len(rows) + 10
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{h}' {FONT} font-size='12'>"]
    out.append(
        _legend(left, COLOURS["escalate"], "escalate")
        + _legend(left + 90, COLOURS["review"], "review (human)")
        + _legend(left + 210, COLOURS["auto_send"], "auto-send")
    )
    for i, (label, counts) in enumerate(rows):
        y = top + i * row_h
        out.append(f"<text x='{left - 6}' y='{y + 14}' text-anchor='end' font-weight='600'>{escape(label)}</text>")
        x = left
        for action in ("escalate", "review", "auto_send"):
            w = counts.get(action, 0) * scale
            if w:
                out.append(f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='18' fill='{COLOURS[action]}'/>")
            x += w
        total = sum(counts.values())
        out.append(f"<text x='{x + 4:.1f}' y='{y + 14}' fill='#333'>{total}</text>")
    out.append("</svg>")
    return "".join(out)


def signal_bars(rows: list[tuple[str, int, int]], width: int = 640) -> str:
    """Per signal: how many tickets fired it, and how many of those ended in escalation."""
    if not rows:
        return ""
    row_h, top, left = 26, 36, 130
    mx = max(n for _, n, _ in rows) or 1
    scale = (width - left - 80) / mx
    h = top + row_h * len(rows) + 10
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{h}' {FONT} font-size='12'>"]
    out.append(
        _legend(left, "#2f6db3", "tickets with signal") + _legend(left + 150, COLOURS["escalate"], "of which escalated")
    )
    for i, (label, n, esc) in enumerate(rows):
        y = top + i * row_h
        out.append(f"<text x='{left - 6}' y='{y + 14}' text-anchor='end' font-weight='600'>{escape(label)}</text>")
        out.append(f"<rect x='{left}' y='{y}' width='{n * scale:.1f}' height='16' fill='#2f6db3'/>")
        out.append(f"<rect x='{left}' y='{y}' width='{esc * scale:.1f}' height='16' fill='{COLOURS['escalate']}'/>")
        out.append(f"<text x='{left + n * scale + 4:.1f}' y='{y + 13}' fill='#333'>{esc} / {n}</text>")
    out.append("</svg>")
    return "".join(out)
