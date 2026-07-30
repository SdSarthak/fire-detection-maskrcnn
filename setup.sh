#!/usr/bin/env bash
# Set up a local development environment for the fire detection project.
set -euo pipefail

cd "$(dirname "$0")"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { printf "${BLUE}%s${NC}\n" "$*"; }
ok()    { printf "${GREEN}  %s${NC}\n" "$*"; }
warn()  { printf "${YELLOW}%s${NC}\n" "$*"; }

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
command -v "$PYTHON" >/dev/null 2>&1 || { warn "Python 3.9+ is required but was not found."; exit 1; }

info "Python: $("$PYTHON" --version 2>&1)"

info "1/4 Creating the virtual environment..."
if [ ! -d venv ]; then
  "$PYTHON" -m venv venv
fi
# shellcheck disable=SC1091
if [ -f venv/bin/activate ]; then . venv/bin/activate; else . venv/Scripts/activate; fi
ok "venv ready"

info "2/4 Installing dependencies..."
python -m pip install --quiet --upgrade pip setuptools wheel
python -m pip install --quiet -r requirements-dev.txt
ok "dependencies installed"

info "3/4 Creating project directories..."
mkdir -p segmentation/data/train segmentation/data/val segmentation/data/test \
         segmentation/data/annotations segmentation/weights segmentation/outputs
ok "directories ready"

info "4/4 Preparing configuration..."
if [ ! -f .env ]; then
  cp .env.example .env
  ok ".env created from .env.example -- edit it before deploying"
else
  ok ".env already exists, left untouched"
fi

cat <<'EOF'

Setup complete.

Next steps
  1. Add images to segmentation/data/{train,val,test}/ and export VIA
     annotations to segmentation/data/annotations/.
  2. Train:      cd segmentation && python train.py --epochs 30
  3. Infer:      cd segmentation && python infer.py --input data/test
  4. Serve:      MODEL_PATH=segmentation/weights/fire_detection_model.pt \
                   python mlops/flask_app/app.py
  5. Test:       pytest
  6. Deploy:     see mlops/DEPLOYMENT_GUIDE.md

EOF
