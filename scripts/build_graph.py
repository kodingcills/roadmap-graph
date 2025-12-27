from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

@dataclass(frozen=True)
class Node:
    id: str
    kind: str  # layer | lesson | concept | artifact
    title: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class Edge:
    type: str
    src: str
    dst: str
    meta: Dict[str, Any]


# -----------------------------
# DOT helpers
# -----------------------------
def dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def node_label(n: Node) -> str:
    if n.kind == "layer":
        scope = n.meta.get("scope", [])
        scope_txt = "\\n".join([f"• {dot_escape(x)}" for x in scope[:4]])
        return f"{dot_escape(n.title)}\\n\\n{scope_txt}" if scope_txt else dot_escape(n.title)

    if n.kind == "lesson":
        obj = n.meta.get("mechanism_objective", "")
        return f"{dot_escape(n.id)}: {dot_escape(n.title)}\\n\\n{dot_escape(obj)}"

    if n.kind == "concept":
        return f"{dot_escape(n.id.replace('CONCEPT:', ''))}\\n\\n{dot_escape(n.meta.get('definition',''))}"

    if n.kind == "artifact":
        return f"{dot_escape(n.id.replace('ARTIFACT:', ''))}\\n\\n{dot_escape(n.meta.get('interface_summary',''))}"

    return dot_escape(n.title)


def node_style(n: Node) -> str:
    # Node styling rules from your spec
    if n.kind == "layer":
        return 'shape=box style="rounded,filled" penwidth=2'
    if n.kind == "lesson":
        return 'shape=box style="rounded" penwidth=1.5'
    if n.kind == "concept":
        return 'shape=ellipse style="filled" penwidth=1.2'
    if n.kind == "artifact":
        # document-ish: note works well in dot
        return 'shape=note style="rounded" penwidth=1.2'
    return 'shape=box'


def edge_style(e: Edge) -> str:
    t = e.type
    if t == "PREREQ_HARD":
        return 'penwidth=2.5 style="solid"'
    if t == "PREREQ_SOFT":
        return 'penwidth=1.5 style="dashed"'
    if t == "REINFORCES":
        return 'penwidth=1.0 style="dotted"'
    if t == "PRODUCES":
        return 'penwidth=1.5 style="solid" arrowhead=vee'
    if t == "USES_ARTIFACT":
        return 'penwidth=1.5 style="solid" arrowhead=normal'
    return 'penwidth=1.0 style="solid"'


def edge_label(e: Edge) -> str:
    t = e.type
    if t == "PREREQ_HARD":
        checks = e.meta.get("gate_checks", [])
        if checks:
            # keep label short; details go to report
            return "gate"
        return ""
    if t == "PREREQ_SOFT":
        return "soft"
    if t == "PRODUCES":
        return "+"
    if t == "USES_ARTIFACT":
        return "uses"
    if t == "REINFORCES":
        return "reinforces"
    return ""


def render_dot(dot_path: Path, out_dir: Path, formats: List[str]) -> None:
    for fmt in formats:
        out_path = out_dir / (dot_path.stem + f".{fmt}")
        cmd = ["dot", f"-T{fmt}", str(dot_path), "-o", str(out_path)]
        subprocess.run(cmd, check=True)


def load_graph(path: Path) -> Tuple[Dict[str, Node], List[Edge], Dict[str, Any]]:
    data = yaml.safe_load(path.read_text())

    nodes: Dict[str, Node] = {}
    meta = data.get("meta", {})

    # layers
    for layer in data.get("layers", []):
        lid = layer["id"]
        nodes[lid] = Node(
            id=lid,
            kind="layer",
            title=layer["title"],
            meta={"scope": layer.get("scope", []), "capstone": layer.get("capstone")},
        )

    # lessons
    for lesson in data.get("lessons", []):
        sid = lesson["id"]
        nodes[sid] = Node(
            id=sid,
            kind="lesson",
            title=lesson["title"],
            meta={
                "layer": lesson["layer"],
                "mechanism_objective": lesson.get("mechanism_objective", ""),
                "timebox_minutes": lesson.get("timebox_minutes", 0),
                "outputs": lesson.get("outputs", []),
            },
        )

    # concepts
    for concept in data.get("concepts", []):
        cid = concept["id"]
        nodes[cid] = Node(
            id=cid,
            kind="concept",
            title=cid,
            meta={
                "definition": concept.get("definition", ""),
                "why_load_bearing": concept.get("why_load_bearing", ""),
            },
        )

    # artifacts
    for art in data.get("artifacts", []):
        aid = art["id"]
        nodes[aid] = Node(
            id=aid,
            kind="artifact",
            title=aid,
            meta={
                "interface_summary": art.get("interface_summary", ""),
                "tests_required": art.get("tests_required", []),
            },
        )

    edges: List[Edge] = []
    for e in data.get("edges", []):
        edges.append(
            Edge(
                type=e["type"],
                src=e["from"],
                dst=e["to"],
                meta={k: v for k, v in e.items() if k not in ["type", "from", "to"]},
            )
        )

    return nodes, edges, meta


def build_macro_dot(nodes: Dict[str, Node], edges: List[Edge], meta: Dict[str, Any]) -> str:
    title = meta.get("title", "Roadmap")
    lines = [
        "digraph G {",
        '  rankdir=LR;',
        '  graph [fontsize=18 labelloc="t" label="%s"];' % dot_escape(title + " — Macro View"),
        "  node [fontname=Helvetica];",
        "  edge [fontname=Helvetica];",
    ]

    # Only layers
    layer_ids = [n.id for n in nodes.values() if n.kind == "layer"]
    for lid in layer_ids:
        n = nodes[lid]
        lines.append(f'  "{n.id}" [{node_style(n)} label="{node_label(n)}"];')

    # Layer-to-layer edges (derive from lesson prereqs if desired; for now: only explicit edges that connect layers)
    # Minimal and safe: if an edge connects nodes in different layers, collapse to layer edge.
    layer_edges = set()
    lesson_layer = {nid: nodes[nid].meta.get("layer") for nid in nodes if nodes[nid].kind == "lesson"}

    for e in edges:
        # only consider prereq edges
        if e.type not in ("PREREQ_HARD", "PREREQ_SOFT"):
            continue
        src_layer = lesson_layer.get(e.src) or (e.src if e.src.startswith("LAYER:") else None)
        dst_layer = lesson_layer.get(e.dst) or (e.dst if e.dst.startswith("LAYER:") else None)

        # if src/dst are lessons, map to their layer nodes
        if e.src in lesson_layer:
            src_layer = lesson_layer[e.src]
        if e.dst in lesson_layer:
            dst_layer = lesson_layer[e.dst]

        if src_layer and dst_layer and src_layer != dst_layer:
            layer_edges.add((src_layer, dst_layer, e.type))

    for (src, dst, etype) in sorted(layer_edges):
        style = 'penwidth=2.5 style="solid"' if etype == "PREREQ_HARD" else 'penwidth=1.5 style="dashed"'
        lines.append(f'  "{src}" -> "{dst}" [{style}];')

    lines.append("}")
    return "\n".join(lines)


def build_layer_dot(layer_id: str, nodes: Dict[str, Node], edges: List[Edge], meta: Dict[str, Any]) -> str:
    layer = nodes[layer_id]
    title = f"{meta.get('title','Roadmap')} — {layer.title} — Layer View"
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  graph [fontsize=18 labelloc="t" label="%s"];' % dot_escape(title),
        "  node [fontname=Helvetica];",
        "  edge [fontname=Helvetica];",
        "",
        f'  subgraph "cluster_{layer_id}" {{',
        '    style="rounded";',
        f'    label="{dot_escape(layer.title)}";',
    ]

    # Nodes in layer
    lesson_ids = [n.id for n in nodes.values() if n.kind == "lesson" and n.meta.get("layer") == layer_id]
    for lid in lesson_ids:
        n = nodes[lid]
        lines.append(f'    "{n.id}" [{node_style(n)} label="{node_label(n)}"];')

    lines.append("  }")

    # Include only edges where both endpoints are lessons in this layer (plus concept/artifact edges connected to them)
    include = set(lesson_ids)

    # optionally pull in concept/artifact nodes connected to these lessons to reduce clutter:
    # We'll include concept/artifact nodes if they directly connect as prereqs/produces/uses.
    for e in edges:
        if e.src in include and nodes.get(e.dst) and nodes[e.dst].kind in ("concept", "artifact"):
            include.add(e.dst)
        if e.dst in include and nodes.get(e.src) and nodes[e.src].kind in ("concept", "artifact"):
            include.add(e.src)

    # Emit included non-layer nodes (concept/artifact)
    for nid in include:
        if nid in lesson_ids:
            continue
        n = nodes[nid]
        lines.append(f'  "{n.id}" [{node_style(n)} label="{node_label(n)}"];')

    # Emit edges among included nodes
    for e in edges:
        if e.src in include and e.dst in include:
            style = edge_style(e)
            lbl = edge_label(e)
            lbl_clause = f' label="{dot_escape(lbl)}"' if lbl else ""
            lines.append(f'  "{e.src}" -> "{e.dst}" [{style}{lbl_clause}];')

    lines.append("}")
    return "\n".join(lines)


def build_lesson_dot(lesson_id: str, nodes: Dict[str, Node], edges: List[Edge], meta: Dict[str, Any]) -> str:
    lesson = nodes[lesson_id]
    title = f"{meta.get('title','Roadmap')} — Lesson Focus: {lesson_id}"

    # Collect neighborhood: prereqs (in), reinforces(out), artifacts produced/used
    include = {lesson_id}
    incoming = [e for e in edges if e.dst == lesson_id and e.type in ("PREREQ_HARD", "PREREQ_SOFT")]
    outgoing_reinf = [e for e in edges if e.src == lesson_id and e.type == "REINFORCES"]
    produces = [e for e in edges if e.src == lesson_id and e.type == "PRODUCES"]
    uses = [e for e in edges if e.src == lesson_id and e.type == "USES_ARTIFACT"]

    for e in incoming + outgoing_reinf + produces + uses:
        include.add(e.src)
        include.add(e.dst)

    # Also include one-hop prereqs of prereqs (hard only) to make “what am I missing?” clearer (small expansion).
    prereq_ids = [e.src for e in incoming]
    for pid in prereq_ids:
        for e2 in edges:
            if e2.dst == pid and e2.type == "PREREQ_HARD":
                include.add(e2.src)
                include.add(e2.dst)

    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  graph [fontsize=18 labelloc="t" label="%s"];' % dot_escape(title),
        "  node [fontname=Helvetica];",
        "  edge [fontname=Helvetica];",
        "",
    ]

    # Emit nodes with a visual emphasis for the focal lesson
    for nid in include:
        n = nodes[nid]
        style = node_style(n)
        if nid == lesson_id:
            style += ' color="black" penwidth=3'
        lines.append(f'  "{n.id}" [{style} label="{node_label(n)}"];')

    # Emit relevant edges among included nodes
    for e in edges:
        if e.src in include and e.dst in include:
            style = edge_style(e)
            lbl = edge_label(e)
            lbl_clause = f' label="{dot_escape(lbl)}"' if lbl else ""
            lines.append(f'  "{e.src}" -> "{e.dst}" [{style}{lbl_clause}];')

    lines.append("}")
    return "\n".join(lines)


def lesson_report_md(lesson_id: str, nodes: Dict[str, Node], edges: List[Edge]) -> str:
    lesson = nodes[lesson_id]
    incoming = [e for e in edges if e.dst == lesson_id and e.type in ("PREREQ_HARD", "PREREQ_SOFT")]
    produces = [e for e in edges if e.src == lesson_id and e.type == "PRODUCES"]
    uses = [e for e in edges if e.src == lesson_id and e.type == "USES_ARTIFACT"]
    reinforces = [e for e in edges if e.src == lesson_id and e.type == "REINFORCES"]

    def node_title(nid: str) -> str:
        n = nodes[nid]
        if n.kind == "lesson":
            return f"{nid} — {n.title}"
        if n.kind == "concept":
            return nid.replace("CONCEPT:", "Concept: ")
        if n.kind == "artifact":
            return nid.replace("ARTIFACT:", "Artifact: ")
        if n.kind == "layer":
            return nid.replace("LAYER:", "Layer: ")
        return nid

    lines = []
    lines.append(f"# Dependency Report — {lesson_id}")
    lines.append("")
    lines.append(f"**Title:** {lesson.title}")
    lines.append("")
    lines.append(f"**Mechanism objective:** {lesson.meta.get('mechanism_objective','')}")
    lines.append("")
    lines.append("## Outputs")
    for out in lesson.meta.get("outputs", []):
        lines.append(f"- {out}")
    lines.append("")

    lines.append("## Hard prerequisites")
    hard = [e for e in incoming if e.type == "PREREQ_HARD"]
    if not hard:
        lines.append("- (none)")
    else:
        for e in hard:
            lines.append(f"- {node_title(e.src)}")
            checks = e.meta.get("gate_checks", [])
            if checks:
                lines.append("  - Gate checks:")
                for c in checks:
                    lines.append(f"    - {c}")
    lines.append("")

    lines.append("## Soft prerequisites")
    soft = [e for e in incoming if e.type == "PREREQ_SOFT"]
    if not soft:
        lines.append("- (none)")
    else:
        for e in soft:
            lines.append(f"- {node_title(e.src)}")
            note = e.meta.get("recovery_note") or e.meta.get("recovery_note", "")
            if note:
                lines.append(f"  - Recovery: {note}")
    lines.append("")

    lines.append("## Reinforcement targets")
    if not reinforces:
        lines.append("- (none)")
    else:
        for e in reinforces:
            lines.append(f"- Strengthens {node_title(e.dst)}")
            note = e.meta.get("note", "")
            if note:
                lines.append(f"  - Note: {note}")
    lines.append("")

    lines.append("## Produces artifacts")
    if not produces:
        lines.append("- (none)")
    else:
        for e in produces:
            lines.append(f"- {node_title(e.dst)}")
            contract = e.meta.get("artifact_contract", "")
            if contract:
                lines.append(f"  - Contract: {contract}")
    lines.append("")

    lines.append("## Uses artifacts")
    if not uses:
        lines.append("- (none)")
    else:
        for e in uses:
            lines.append(f"- {node_title(e.dst)}")
            why = e.meta.get("why", "")
            if why:
                lines.append(f"  - Why: {why}")
    lines.append("")

    return "\n".join(lines)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="graph/graph.yaml", help="Path to graph.yaml")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--render", action="store_true", help="Render PNG/SVG via graphviz dot")
    ap.add_argument("--formats", default="png,svg", help="Comma-separated render formats (png,svg,...)")
    ap.add_argument("--layer", default="", help="Generate meso view for this layer id (e.g., LAYER:ML-Quant). If empty, generate all layers.")
    ap.add_argument("--lesson", default="", help="Generate micro view for this lesson id (e.g., L10.7). If empty, generate all lessons.")
    args = ap.parse_args()

    graph_path = Path(args.graph)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges, meta = load_graph(graph_path)

    # Macro
    macro_dot = build_macro_dot(nodes, edges, meta)
    macro_path = out_dir / "macro.dot"
    macro_path.write_text(macro_dot)

    # Meso (layer)
    layer_ids = [n.id for n in nodes.values() if n.kind == "layer"]
    if args.layer:
        layer_ids = [args.layer]

    for lid in layer_ids:
        if lid not in nodes or nodes[lid].kind != "layer":
            raise ValueError(f"Unknown layer id: {lid}")
        dot_text = build_layer_dot(lid, nodes, edges, meta)
        (out_dir / f"layer_{lid.replace(':','_')}.dot").write_text(dot_text)

    # Micro (lesson)
    lesson_ids = [n.id for n in nodes.values() if n.kind == "lesson"]
    if args.lesson:
        lesson_ids = [args.lesson]

    for sid in lesson_ids:
        if sid not in nodes or nodes[sid].kind != "lesson":
            raise ValueError(f"Unknown lesson id: {sid}")
        dot_text = build_lesson_dot(sid, nodes, edges, meta)
        (out_dir / f"lesson_{sid}.dot").write_text(dot_text)

        report = lesson_report_md(sid, nodes, edges)
        (out_dir / f"lesson_{sid}_dependency_report.md").write_text(report)

    # Render
    if args.render:
        fmts = [x.strip() for x in args.formats.split(",") if x.strip()]
        # Render all dot files in out/
        for dot_file in out_dir.glob("*.dot"):
            render_dot(dot_file, out_dir, fmts)

    print(f"Done. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
