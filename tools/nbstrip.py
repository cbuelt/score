"""Git clean filter that strips outputs from Jupyter notebooks.

Notebooks keep their outputs in the working tree, but are committed without
them (no figures, no machine-specific paths in tracebacks). Enable once per
clone:

    git config filter.nbstrip.clean "python3 tools/nbstrip.py"
    git config filter.nbstrip.smudge cat

``.gitattributes`` assigns the filter to ``*.ipynb``. Standard library only.
"""

import json
import sys


def strip(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        meta = cell.get("metadata", {})
        for key in ("execution", "collapsed", "scrolled", "ExecuteTime"):
            meta.pop(key, None)
    nb.get("metadata", {}).pop("widgets", None)
    return nb


if __name__ == "__main__":
    nb = json.load(sys.stdin)
    sys.stdout.write(json.dumps(strip(nb), indent=1, ensure_ascii=False) + "\n")
