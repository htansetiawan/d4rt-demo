#!/usr/bin/env python3
"""Execute either notebook and export a standalone HTML presentation.

Run without flags for the real GPU demo; use --illustration for CPU rehearsal.
"""

import argparse
import copy
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor

ROOT = Path(__file__).resolve().parents[1]


def render(illustration=False, soccer=False, simple=False):
    stem = "D4RT_simple" if simple else ("D4RT_soccer" if soccer else ("D4RT_rehearsal" if illustration else "D4RT_demo"))
    path = ROOT / f"{stem}.ipynb"
    nb = nbformat.read(path, as_version=4)
    # Kernel widgets need a running server. All main figures/player are exported.
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.strip().startswith("viz.query_explorer"):
            cell.source = 'print("Live query explorer: open the notebook with a Python kernel.")'
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata":{"path":str(ROOT)}})
    client.execute()
    executed_path = ROOT / "artifacts" / f"{stem}_executed.ipynb"
    executed_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, executed_path)
    # Keep executed results in the notebook the presenter opens; retain the live
    # widget source so Run All still provides the explorer in Jupyter.
    original = nbformat.read(path, as_version=4)
    presented = copy.deepcopy(nb)
    for a, b in zip(original.cells, presented.cells):
        if a.cell_type == "code" and a.source.strip().startswith("viz.query_explorer"):
            b.source = a.source
            b.outputs = []
            b.execution_count = None
    nbformat.write(presented, path)
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    exporter.embed_images = True
    exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={"hide-input"}), enabled=True)
    body, _ = exporter.from_notebook_node(nb)
    html_path = ROOT / "artifacts" / f"{stem}.html"
    html_path.write_text(body)
    print("Executed:", executed_path)
    print("Offline presentation:", html_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--illustration", action="store_true")
    mode.add_argument("--soccer", action="store_true")
    mode.add_argument("--simple", action="store_true")
    args = parser.parse_args()
    render(args.illustration, args.soccer, args.simple)
