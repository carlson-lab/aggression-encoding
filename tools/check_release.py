"""Check notebook schemas and enforce this repository's source-only release policy."""

import ast
import json
from pathlib import Path
import posixpath
import re
import subprocess
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {".py", ".ipynb", ".m", ".sh", ".md", ".yml", ".yaml"}
ALLOWED_NAMES = {"LICENSE", ".gitignore", "requirements.txt"}
ALLOWED_JSON = {"docs/source_manifest.json"}


def main():
    result = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    names = sorted({name.decode() for name in result.split(b"\0") if name})
    errors = []
    notebooks = 0
    manifest_path = ROOT / "docs/source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["sources"]:
            if entry["path"] not in names:
                errors.append(f"{entry['path']}: manifest path does not match Git spelling/case")
    for document in ["README.md", "analysis/mediation/README.md", "docs/analysis_index.md"]:
        for link in re.findall(r"\]\(([^)]+)\)", (ROOT / document).read_text()):
            if link.startswith(("https:", "http:", "#", "mailto:")):
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(document), link))
            if resolved not in names and not any(n.startswith(resolved + "/") for n in names):
                errors.append(f"{document}: unresolved relative link {link}")
    for name in names:
        path = ROOT / name
        if not path.exists():
            continue
        if path.is_symlink():
            errors.append(f"{name}: symlinks are not part of this source-only release")
            continue
        if not (
            path.suffix in ALLOWED_SUFFIXES
            or path.name in ALLOWED_NAMES
            or name in ALLOWED_JSON
        ):
            errors.append(f"{name}: data, model, result, or unreviewed file type")
        if path.suffix != ".ipynb":
            continue
        notebooks += 1
        try:
            raw = json.loads(path.read_text())
            nbformat.validate(raw)
        except Exception as exc:
            errors.append(f"{name}: invalid notebook: {exc}")
            continue
        if set(raw.get("metadata", {})) - {"kernelspec", "language_info"}:
            errors.append(f"{name}: nonessential notebook metadata")
        if raw.get("metadata", {}).get("language_info") != {"name": "python"}:
            errors.append(f"{name}: unexpected language metadata")
        if raw.get("metadata", {}).get("kernelspec") != {
            "display_name": "Python 3", "language": "python", "name": "python3"
        }:
            errors.append(f"{name}: unexpected kernel metadata")
        for i, cell in enumerate(raw["cells"]):
            if cell.get("metadata") or cell.get("attachments"):
                errors.append(f"{name}: cell {i} contains metadata or attachments")
            if cell.get("outputs") or cell.get("execution_count") is not None:
                errors.append(f"{name}: cell {i} contains a saved output or execution count")
            source = "".join(cell.get("source", []))
            if "data:image/" in source or "data:application/" in source:
                errors.append(f"{name}: cell {i} contains embedded data")
            if name.startswith("analysis/mediation/") and cell["cell_type"] == "code":
                try:
                    ast.parse(source)
                except SyntaxError as exc:
                    errors.append(f"{name}: cell {i} has invalid Python syntax: {exc}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {notebooks} notebooks are valid and output-free; {len(names)} files checked.")
    print("This check does not execute analyses or establish numerical reproducibility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
