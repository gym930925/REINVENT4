"""Generate workflow decision figure as SVG."""

W, H = 780, 520

svg_lines = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Arial,sans-serif">',

    # ── background ──────────────────────────────────────────────────────────
    f'<rect width="{W}" height="{H}" fill="#f8f9fa"/>',

    # ── axis labels ─────────────────────────────────────────────────────────
    # x-axis: Scoring function (No → Yes)
    '<text x="390" y="32" text-anchor="middle" font-size="15" font-weight="bold" fill="#333">Scoring function available?</text>',
    '<text x="225" y="52" text-anchor="middle" font-size="13" fill="#666">No</text>',
    '<text x="565" y="52" text-anchor="middle" font-size="13" fill="#666">Yes</text>',
    # arrow x
    '<line x1="140" y1="45" x2="640" y2="45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>',

    # y-axis: Known molecules (No → Yes, bottom to top)
    '<text x="22" y="290" text-anchor="middle" font-size="15" font-weight="bold" fill="#333" transform="rotate(-90,22,290)">Known molecules?</text>',
    '<text x="52" y="420" text-anchor="middle" font-size="13" fill="#666">No</text>',
    '<text x="52" y="175" text-anchor="middle" font-size="13" fill="#666">Yes</text>',
    # arrow y
    '<line x1="65" y1="460" x2="65" y2="60" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>',

    # ── arrow marker def ────────────────────────────────────────────────────
    '<defs>',
    '  <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">',
    '    <path d="M0,0 L0,6 L8,3 z" fill="#999"/>',
    '  </marker>',
    '  <marker id="flow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">',
    '    <path d="M0,0 L0,6 L8,3 z" fill="#aaa"/>',
    '  </marker>',
    '</defs>',

    # ── grid lines ──────────────────────────────────────────────────────────
    '<line x1="80" y1="270" x2="700" y2="270" stroke="#ddd" stroke-width="1" stroke-dasharray="4,4"/>',
    '<line x1="390" y1="60" x2="390" y2="490" stroke="#ddd" stroke-width="1" stroke-dasharray="4,4"/>',

    # ══ QUADRANT 1: bottom-left — Sample & Filter ══════════════════════════
    # box
    '<rect x="85" y="280" width="295" height="195" rx="10" fill="#e8f4fd" stroke="#90caf9" stroke-width="1.5"/>',
    # badge
    '<rect x="95" y="290" width="22" height="22" rx="4" fill="#1976d2"/>',
    '<text x="106" y="306" text-anchor="middle" font-size="13" font-weight="bold" fill="white">1</text>',
    # title
    '<text x="230" y="308" text-anchor="middle" font-size="14" font-weight="bold" fill="#1976d2">Sample &amp; Filter</text>',
    # body
    '<text x="103" y="330" font-size="11.5" fill="#333">Prior → large-scale sampling</text>',
    '<text x="103" y="348" font-size="11.5" fill="#333">Post-hoc filter by MW, QED, alerts</text>',
    '<text x="103" y="366" font-size="11.5" fill="#333">Cluster &amp; select</text>',
    # tag
    '<rect x="103" y="430" width="110" height="22" rx="5" fill="#bbdefb"/>',
    '<text x="158" y="445" text-anchor="middle" font-size="11" fill="#1565c0">Early exploration</text>',
    '<rect x="222" y="430" width="140" height="22" rx="5" fill="#bbdefb"/>',
    '<text x="292" y="445" text-anchor="middle" font-size="11" fill="#1565c0">No prior knowledge needed</text>',

    # ══ QUADRANT 2: bottom-right — RL from Prior ═══════════════════════════
    '<rect x="395" y="280" width="295" height="195" rx="10" fill="#fce8e6" stroke="#ef9a9a" stroke-width="1.5"/>',
    '<rect x="405" y="290" width="22" height="22" rx="4" fill="#c62828"/>',
    '<text x="416" y="306" text-anchor="middle" font-size="13" font-weight="bold" fill="white">3</text>',
    '<text x="545" y="308" text-anchor="middle" font-size="14" font-weight="bold" fill="#c62828">RL from Prior</text>',
    '<text x="413" y="330" font-size="11.5" fill="#333">Prior → RL with scoring function</text>',
    '<text x="413" y="348" font-size="11.5" fill="#333">Iterative reward-guided optimisation</text>',
    '<text x="413" y="366" font-size="11.5" fill="#333">Diversity filter prevents collapse</text>',
    '<rect x="413" y="430" width="130" height="22" rx="5" fill="#ffcdd2"/>',
    '<text x="478" y="445" text-anchor="middle" font-size="11" fill="#b71c1c">Property-driven</text>',
    '<rect x="552" y="430" width="120" height="22" rx="5" fill="#ffcdd2"/>',
    '<text x="612" y="445" text-anchor="middle" font-size="11" fill="#b71c1c">No actives needed</text>',

    # ══ QUADRANT 3: top-left — TL from Known Molecules ═════════════════════
    '<rect x="85" y="70" width="295" height="195" rx="10" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>',
    '<rect x="95" y="80" width="22" height="22" rx="4" fill="#2e7d32"/>',
    '<text x="106" y="96" text-anchor="middle" font-size="13" font-weight="bold" fill="white">2</text>',
    '<text x="230" y="98" text-anchor="middle" font-size="14" font-weight="bold" fill="#2e7d32">Transfer Learning</text>',
    '<text x="103" y="120" font-size="11.5" fill="#333">Prior → TL on known molecules</text>',
    '<text x="103" y="138" font-size="11.5" fill="#333">Agent learns the chemical series</text>',
    '<text x="103" y="156" font-size="11.5" fill="#333">Sample focused analogues</text>',
    '<rect x="103" y="222" width="120" height="22" rx="5" fill="#c8e6c9"/>',
    '<text x="163" y="237" text-anchor="middle" font-size="11" fill="#1b5e20">Focused generation</text>',
    '<rect x="232" y="222" width="120" height="22" rx="5" fill="#c8e6c9"/>',
    '<text x="292" y="237" text-anchor="middle" font-size="11" fill="#1b5e20">No scoring needed</text>',

    # ══ QUADRANT 4: top-right — TL then RL (STAR) ══════════════════════════
    '<rect x="395" y="70" width="295" height="195" rx="10" fill="#fff8e1" stroke="#ffe082" stroke-width="2"/>',
    '<rect x="405" y="80" width="22" height="22" rx="4" fill="#f57f17"/>',
    '<text x="416" y="96" text-anchor="middle" font-size="13" font-weight="bold" fill="white">4</text>',
    '<text x="545" y="98" text-anchor="middle" font-size="14" font-weight="bold" fill="#e65100">TL → RL</text>',
    '<text x="413" y="120" font-size="11.5" fill="#333">TL scopes to relevant chemistry</text>',
    '<text x="413" y="138" font-size="11.5" fill="#333">RL optimises within that space</text>',
    '<text x="413" y="156" font-size="11.5" fill="#333">Faster convergence, less wasted</text>',
    '<text x="413" y="172" font-size="11.5" fill="#333">compute on expensive scoring</text>',
    '<rect x="413" y="222" width="130" height="22" rx="5" fill="#fff9c4"/>',
    '<text x="478" y="237" text-anchor="middle" font-size="11" fill="#e65100">Full design campaign</text>',
    '<rect x="552" y="222" width="120" height="22" rx="5" fill="#fff9c4"/>',
    '<text x="612" y="237" text-anchor="middle" font-size="11" fill="#e65100">★ Recommended</text>',

    # ══ PROGRESSION ARROWS ══════════════════════════════════════════════════
    # 1 → 3  (add scoring function)
    '<line x1="380" y1="385" x2="395" y2="385" stroke="#aaa" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#flow)"/>',
    # 1 → 2  (add known molecules)
    '<line x1="230" y1="278" x2="230" y2="266" stroke="#aaa" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#flow)"/>',
    # 2 → 4  (add scoring function)
    '<line x1="380" y1="175" x2="395" y2="175" stroke="#aaa" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#flow)"/>',
    # 3 → 4  (add known molecules)
    '<line x1="540" y1="278" x2="540" y2="266" stroke="#aaa" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#flow)"/>',

    '</svg>',
]

with open("doc/asset/workflows.svg", "w") as f:
    f.write("\n".join(svg_lines))

print("Written: doc/asset/workflows.svg")
