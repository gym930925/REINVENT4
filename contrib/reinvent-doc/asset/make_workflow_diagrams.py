"""Generate one SVG per workflow: left = step flow, right = pros/cons panel."""

import textwrap

W, H = 820, 300

ARROW = """
<defs>
  <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L0,6 L8,3 z" fill="#555"/>
  </marker>
</defs>"""

def svg_wrap(content, w=W, h=H):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Arial,sans-serif">\n'
        + content + "\n</svg>"
    )

def rect(x, y, w, h, fill, stroke, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'

def text(x, y, s, size=12, anchor="middle", weight="normal", fill="#222"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}">{s}</text>'

def arrow_right(x1, y, x2):
    return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>'

def arrow_down(x, y1, y2):
    return f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>'

def flow_box(x, y, w, h, fill, stroke, label, sublabel=""):
    lines = [rect(x, y, w, h, fill, stroke)]
    cy = y + h // 2 + (0 if not sublabel else -8)
    lines.append(text(x + w // 2, cy, label, size=12, weight="bold"))
    if sublabel:
        lines.append(text(x + w // 2, cy + 17, sublabel, size=10, fill="#555"))
    return "\n".join(lines)

def pros_cons(x, y, pros, cons, accent):
    lines = []
    # panel background
    lines.append(rect(x, y, 290, H - 2*y, "#fafafa", "#ccc"))
    lines.append(text(x + 145, y + 22, "Pros / Cons", size=13, weight="bold", fill="#333"))
    # divider
    lines.append(f'<line x1="{x+10}" y1="{y+32}" x2="{x+280}" y2="{y+32}" stroke="#ddd" stroke-width="1"/>')

    py = y + 50
    lines.append(text(x + 12, py, "✓", size=12, anchor="start", fill="#2e7d32"))
    lines.append(text(x + 28, py, "Pros", size=11, anchor="start", weight="bold", fill="#2e7d32"))
    py += 18
    for pro in pros:
        lines.append(text(x + 22, py, f"+ {pro}", size=10.5, anchor="start", fill="#333"))
        py += 16

    py += 6
    lines.append(text(x + 12, py, "✗", size=12, anchor="start", fill="#c62828"))
    lines.append(text(x + 28, py, "Cons", size=11, anchor="start", weight="bold", fill="#c62828"))
    py += 18
    for con in cons:
        lines.append(text(x + 22, py, f"– {con}", size=10.5, anchor="start", fill="#333"))
        py += 16

    return "\n".join(lines)

def divider(x):
    return f'<line x1="{x}" y1="20" x2="{x}" y2="{H-20}" stroke="#ddd" stroke-width="1" stroke-dasharray="4,3"/>'


# ── Workflow 1: Sample & Filter ──────────────────────────────────────────────
BX, BW, BH = 18, 110, 54   # box x-start, width, height
GY = H // 2 - BH // 2      # vertical centre
GAP = 18

steps1 = [
    ("Prior", "#e3f2fd", "#90caf9"),
    ("Sampling\n(large N)", "#e3f2fd", "#90caf9"),
    ("Property\nFilter", "#e3f2fd", "#90caf9"),
    ("Cluster &\nSelect", "#e3f2fd", "#90caf9"),
]

def build_flow(steps, bx=BX, bw=BW, bh=BH, gy=GY, gap=GAP):
    parts = []
    x = bx
    for i, (label, fill, stroke) in enumerate(steps):
        lines = label.split("\n")
        cy = gy + bh // 2
        parts.append(rect(x, gy, bw, bh, fill, stroke))
        if len(lines) == 1:
            parts.append(text(x + bw // 2, cy + 5, lines[0], size=11, weight="bold"))
        else:
            parts.append(text(x + bw // 2, cy - 5, lines[0], size=11, weight="bold"))
            parts.append(text(x + bw // 2, cy + 12, lines[1], size=10, fill="#555"))
        if i < len(steps) - 1:
            ax1 = x + bw
            ax2 = x + bw + gap
            parts.append(arrow_right(ax1 + 2, gy + bh // 2, ax2 - 2))
        x += bw + gap
    return "\n".join(parts), x  # x = right edge after last box

body1, fx1 = build_flow(steps1)
pc1 = pros_cons(
    fx1 + 20, 20,
    pros=["No labelled data required", "Simple to set up", "Good for initial exploration"],
    cons=["No active optimisation", "Low hit rate for specific targets", "Requires large sampling N"],
    accent="#1976d2",
)

svg1 = svg_wrap(
    ARROW
    + f'<rect width="{W}" height="{H}" fill="#f8f9fa"/>'
    + rect(10, 10, W - 20, H - 20, "white", "#e0e0e0")
    + text(W // 2 - 80, 30, "Workflow 1 — Sample &amp; Filter", size=13, weight="bold", fill="#1565c0")
    + body1
    + divider(fx1 + 15)
    + pc1
)


# ── Workflow 2: Transfer Learning ────────────────────────────────────────────
steps2 = [
    ("Prior", "#e8f5e9", "#a5d6a7"),
    ("Known\nMolecules", "#e8f5e9", "#a5d6a7"),
    ("Transfer\nLearning", "#c8e6c9", "#66bb6a"),
    ("Focused\nAgent", "#c8e6c9", "#66bb6a"),
    ("Sampling", "#e8f5e9", "#a5d6a7"),
]

body2, fx2 = build_flow(steps2)
# add a "+" join indicator between Prior and Known Molecules
# draw small "+" between box 0 and box 1 — actually show two inputs merging into TL
# simpler: annotate with an input arrow from top for known molecules

pc2 = pros_cons(
    fx2 + 20, 20,
    pros=["No explicit scoring function needed", "Captures implicit SAR from data", "Works with small datasets (~20+)"],
    cons=["Quality depends on training data", "Can overfit — no novelty guarantee", "Does not optimise a property explicitly"],
    accent="#2e7d32",
)

svg2 = svg_wrap(
    ARROW
    + f'<rect width="{W}" height="{H}" fill="#f8f9fa"/>'
    + rect(10, 10, W - 20, H - 20, "white", "#e0e0e0")
    + text(W // 2 - 80, 30, "Workflow 2 — Transfer Learning", size=13, weight="bold", fill="#2e7d32")
    + body2
    + divider(fx2 + 15)
    + pc2
)


# ── Workflow 3: RL from Prior ────────────────────────────────────────────────
steps3 = [
    ("Prior", "#fce8e6", "#ef9a9a"),
    ("Scoring\nFunction", "#fce8e6", "#ef9a9a"),
    ("RL\n(staged)", "#ffcdd2", "#e57373"),
    ("Optimised\nAgent", "#ffcdd2", "#e57373"),
    ("Sampling", "#fce8e6", "#ef9a9a"),
]

body3, fx3 = build_flow(steps3)
pc3 = pros_cons(
    fx3 + 20, 20,
    pros=["No known actives required", "Actively optimises toward target", "Curriculum learning for complex objectives"],
    cons=["Scoring function must be well-designed", "Risk of mode collapse without diversity filter", "Computationally heavier than TL"],
    accent="#c62828",
)

svg3 = svg_wrap(
    ARROW
    + f'<rect width="{W}" height="{H}" fill="#f8f9fa"/>'
    + rect(10, 10, W - 20, H - 20, "white", "#e0e0e0")
    + text(W // 2 - 80, 30, "Workflow 3 — RL from Prior", size=13, weight="bold", fill="#c62828")
    + body3
    + divider(fx3 + 15)
    + pc3
)


# ── Workflow 4: TL then RL ───────────────────────────────────────────────────
steps4 = [
    ("Prior", "#fff8e1", "#ffe082"),
    ("TL on\nActives", "#fff9c4", "#ffd54f"),
    ("TL\nAgent", "#fff9c4", "#ffd54f"),
    ("RL\n(staged)", "#ffe0b2", "#ffb74d"),
    ("Optimised\nAgent", "#ffe0b2", "#ffb74d"),
    ("Sampling", "#fff8e1", "#ffe082"),
]

body4, fx4 = build_flow(steps4)
pc4 = pros_cons(
    fx4 + 20, 20,
    pros=["Fastest RL convergence", "Fewer wasted evaluations", "Best chemical quality", "★ Most recommended"],
    cons=["Requires both actives and scoring fn", "TL can over-constrain if data is biased", "More steps to configure"],
    accent="#e65100",
)

svg4 = svg_wrap(
    ARROW
    + f'<rect width="{W}" height="{H}" fill="#f8f9fa"/>'
    + rect(10, 10, W - 20, H - 20, "white", "#e0e0e0")
    + text(W // 2 - 80, 30, "Workflow 4 — TL → RL  ★", size=13, weight="bold", fill="#e65100")
    + body4
    + divider(fx4 + 15)
    + pc4
)


# ── Write files ──────────────────────────────────────────────────────────────
for fname, content in [
    ("doc/asset/workflow1_sample.svg", svg1),
    ("doc/asset/workflow2_tl.svg",     svg2),
    ("doc/asset/workflow3_rl.svg",     svg3),
    ("doc/asset/workflow4_tl_rl.svg",  svg4),
]:
    with open(fname, "w") as f:
        f.write(content)
    print(f"Written: {fname}")
