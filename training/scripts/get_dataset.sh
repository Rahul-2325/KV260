#!/bin/bash
set -e
PROJECT=~/wildfire_project

echo "=== Step 1: Verify Kaggle API ==="
kaggle --version

echo ""
echo "=== Step 2: Download D-Fire dataset (~3 GB) ==="
mkdir -p $PROJECT/dataset
cd $PROJECT/dataset

if [ ! -f "smoke-fire-detection-yolo.zip" ]; then
    kaggle datasets download -d sayedgamal99/smoke-fire-detection-yolo
fi

echo ""
echo "=== Step 3: Extract test images ==="
if [ ! -d "data/test/images" ]; then
    unzip -q smoke-fire-detection-yolo.zip "data/test/images/*"
fi
echo "Test images: $(ls data/test/images/ 2>/dev/null | wc -l)"

echo ""
echo "=== Step 4: Select 500 calibration images ==="
mkdir -p $PROJECT/calibration

if [ -z "$(ls -A $PROJECT/calibration 2>/dev/null)" ]; then
    cd data/test/images/
    ls *.jpg | shuf -n 500 | xargs -I{} cp {} $PROJECT/calibration/
fi
echo "Calibration images: $(ls $PROJECT/calibration/ | wc -l)"

echo ""
echo "=== Step 5: Disk usage ==="
du -sh $PROJECT/dataset/ $PROJECT/calibration/

echo ""
echo "✓ Setup complete!"
echo "  Calibration: $PROJECT/calibration/ (500 images)"
echo "  Test set:    $PROJECT/dataset/data/test/images/ (4306 images)"
