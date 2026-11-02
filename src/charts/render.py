"""Safe SVG previews calculated solely from the canonical dataset (no model geometry)."""
from __future__ import annotations

import math
from decimal import Decimal
from xml.sax.saxutils import escape

from .schema import ChartDocument, ChartView, GraphSuggestion


def render_svg(chart: ChartDocument) -> str:
    points = chart.data.points
    kind = chart.presentation.chart_type
    width = max(900, 180 + 80 * len(points)) if kind == "line" else 900
    height = 400 if kind == "line" else max(320, 125 + 40 * len(points))
    title, unit = escape(chart.title), escape(chart.data.unit)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{chart.chart_id}_title {chart.chart_id}_desc">',
                f'<title id="{chart.chart_id}_title">{title}</title>',
                f'<desc id="{chart.chart_id}_desc">' + escape("; ".join(f"{p.label}: {format(p.value, 'f')} {chart.data.unit}" for p in points)) + '</desc>',
                '<rect width="100%" height="100%" fill="#FFFFFF"/>',
                f'<text x="24" y="32" fill="#182434" font-family="Arial,sans-serif" font-size="22">{title}</text>',
                f'<text x="24" y="56" fill="#536174" font-family="Arial,sans-serif" font-size="14">{unit}</text>']

    def color(p):
        return chart.presentation.point_colors.get(p.id, chart.presentation.color)

    def text(x, y, content, size=13):
        return f'<text x="{x:.3f}" y="{y:.3f}" fill="#182434" font-family="Arial,sans-serif" font-size="{size}">{escape(content)}</text>'

    if kind in ("bar", "line"):
        low = min(Decimal(0), min(p.value for p in points))
        high = max(Decimal(0), max(p.value for p in points))
        span = high - low or Decimal(1)
        if kind == "bar":
            x = lambda v: 190 + float((v - low) / span) * 530
            zero = x(Decimal(0))
            elements.append(f'<path d="M {zero:.3f} 76 V {85 + 40 * len(points)}" stroke="#9CA3AF"/>')
            elements.append(text(zero, height - 14, "0"))
            for i, p in enumerate(points):
                y, end = 82 + 40 * i, x(p.value)
                label = p.label if len(p.label) <= 23 else p.label[:22] + "…"
                elements += [text(24, y + 18, label),
                             f'<rect data-point-id="{p.id}" x="{min(zero, end):.3f}" y="{y}" width="{abs(end - zero):.3f}" height="26" fill="{color(p)}"/>',
                             text(740, y + 18, format(p.value, "f"))]
        else:
            coords = [(100 + i * (width - 200) / (len(points) - 1), 90 + float((high - p.value) / span) * (height - 180)) for i, p in enumerate(points)]
            zero = 90 + float(high / span) * (height - 180)
            elements += [f'<path d="M 100 {zero:.3f} H {width - 100}" stroke="#9CA3AF"/>', text(64, zero + 4, "0"),
                         f'<polyline points="' + " ".join(f"{x:.3f},{y:.3f}" for x, y in coords) + f'" fill="none" stroke="{chart.presentation.color}" stroke-width="3"/>']
            for p, (x, y) in zip(points, coords):
                elements += [f'<circle data-point-id="{p.id}" cx="{x:.3f}" cy="{y:.3f}" r="5" fill="{color(p)}"/>',
                             text(x, y - 12, format(p.value, "f")), text(x - 20, height - 52, p.label if len(p.label) <= 12 else p.label[:11] + "…")]
    else:
        total = sum(p.value for p in points)
        angle, cx, cy, radius = -math.pi / 2, 240, height / 2 + 25, min(140, (height - 105) / 2)
        for i, p in enumerate(points):
            fraction = float(p.value / total)
            end = angle + fraction * 2 * math.pi
            if fraction >= 1:
                elements.append(f'<circle data-point-id="{p.id}" cx="{cx}" cy="{cy}" r="{radius}" fill="{color(p)}" stroke="#FFFFFF"/>')
            elif fraction > 0:
                x1, y1 = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
                x2, y2 = cx + radius * math.cos(end), cy + radius * math.sin(end)
                elements.append(f'<path data-point-id="{p.id}" d="M {cx} {cy} L {x1:.3f} {y1:.3f} A {radius} {radius} 0 {int(fraction > .5)} 1 {x2:.3f} {y2:.3f} Z" fill="{color(p)}" stroke="#FFFFFF"/>')
            y = 92 + 40 * i
            elements += [f'<rect x="470" y="{y - 12}" width="14" height="14" fill="{color(p)}"/>',
                         text(500, y, p.label), text(740, y, format(p.value, "f"))]
            angle = end
    return "".join(elements) + "</svg>"


def chart_view(chart: ChartDocument) -> ChartView:
    return ChartView(chart=chart, preview_svg=render_svg(chart))


def graph_suggestion(chart: ChartDocument) -> GraphSuggestion:
    return GraphSuggestion(chart=chart, preview_svg=render_svg(chart))
