set shell := ["bash", "-cu"]

default:
    @just --list

setup SPIKEFORGE_DIR="$HOME/Projects/spikeforge/spikeforge":
    uv venv .venv
    uv pip install --python .venv/bin/python -r requirements.txt
    SITE=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])"); echo "{{SPIKEFORGE_DIR}}" > "$SITE/spikeforge-src.pth"
    @echo "spikeforge source pinned via spikeforge-src.pth"

test:
    .venv/bin/python -m unittest discover -s tests -t . -v

train:
    .venv/bin/python scripts/train.py --seed 0 --epochs 50
