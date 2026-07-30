#!/bin/bash
# Setup script for Mask R-CNN Fire Detection project
# Initializes project structure, installs dependencies, and prepares for deployment

set -e

echo "=================================="
echo "Mask R-CNN Fire Detection - Setup"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Step 1: Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}Warning: Python not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Create virtual environment for segmentation
echo -e "${BLUE}Step 2: Setting up segmentation environment...${NC}"
cd segmentation

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo -e "${GREEN}✓ Virtual environment created${NC}"

# Install segmentation dependencies
echo -e "${BLUE}Step 3: Installing segmentation dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo -e "${BLUE}Step 4: Creating project directories...${NC}"
mkdir -p data/train data/val data/test data/annotations
mkdir -p weights
mkdir -p outputs
echo -e "${GREEN}✓ Directories created${NC}"

cd ..

# Setup Flask application
echo -e "${BLUE}Step 5: Setting up Flask application...${NC}"
cd mlops/flask_app

if [ ! -d "uploads" ]; then
    mkdir -p uploads
    echo -e "${GREEN}✓ Flask upload directory created${NC}"
fi

cd ../..

# Display setup summary
echo ""
echo -e "${GREEN}=================================="
echo "Setup Complete!"
echo "==================================${NC}"
echo ""
echo "Project structure:"
echo "  segmentation/        - Model training and inference"
echo "  mlops/               - Deployment infrastructure"
echo ""
echo "Next steps:"
echo -e "${YELLOW}1. Prepare training data:${NC}"
echo "   - Place images in segmentation/data/train/, val/, test/"
echo "   - Create annotations using VGG Annotator"
echo "   - Place JSON files in segmentation/data/annotations/"
echo ""
echo -e "${YELLOW}2. Train the model:${NC}"
echo "   cd segmentation"
echo "   python train.py"
echo ""
echo -e "${YELLOW}3. Run inference:${NC}"
echo "   python infer.py"
echo ""
echo -e "${YELLOW}4. Deploy to GCP:${NC}"
echo "   Read mlops/DEPLOYMENT_GUIDE.md for detailed instructions"
echo ""
echo -e "${YELLOW}5. Local Docker testing:${NC}"
echo "   docker build -t fire-detection -f mlops/flask_app/Dockerfile ."
echo "   docker run -p 5000:5000 fire-detection"
echo ""
echo "Documentation:"
echo "  - README.md               - Project overview"
echo "  - segmentation/README.md  - Model training guide"
echo "  - mlops/README.md         - Deployment guide"
echo "  - mlops/DEPLOYMENT_GUIDE.md - Complete GCP setup"
echo ""
