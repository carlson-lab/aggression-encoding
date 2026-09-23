"""Remove saved results and execution metadata from repository notebooks."""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def repository_files():
    result = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    return sorted({ROOT / name.decode() for name in result.split(b"\0") if name})


def main():
    count = 0
    for path in repository_files():
        if path.suffix != ".ipynb" or not path.exists():
            continue
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            cell["metadata"] = {}
            cell.pop("attachments", None)
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        notebook["metadata"] = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        count += 1
    print(f"Cleared outputs and execution metadata in {count} notebooks.")


if __name__ == "__main__":
    main()
