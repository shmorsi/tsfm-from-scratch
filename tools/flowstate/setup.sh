#!/usr/bin/env bash
# Isolated environment for the published FlowState-9.1M checkpoint (Colab, GPU).
# granite-tsfm's GIFT-Eval wrapper lives on its `gift-flowstate` branch and upstream's
# notebook downgrades pandas with pip (granite-tsfm itself needs pandas>=2.2, so we
# keep 2.2+); its dependency set still conflicts with our main environment - hence a
# separate venv. Versions are pinned to what was current on 2026-10-14.
set -euo pipefail
GRANITE_COMMIT=640f7c80a21a4e2253b6d6140d067452bb93e7d4   # branch gift-flowstate
VENV=${FS_VENV:-/content/fs-venv}
SRC=${FS_SRC:-/content/granite-tsfm}
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.11
if [ ! -d "$SRC" ]; then git clone -q https://github.com/ibm-granite/granite-tsfm "$SRC"; fi
git -C "$SRC" fetch -q origin gift-flowstate
git -C "$SRC" checkout -q "$GRANITE_COMMIT"
VIRTUAL_ENV="$VENV" uv pip install -q "$SRC[notebooks]" "pandas>=2.2,<3" "numpy<2" tabulate \
  "gluonts~=0.15.1" "datasets~=2.17.1" "pyarrow<21" python-dotenv orjson pyyaml
echo "FlowState env ready: $VENV"
