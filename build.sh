#!/bin/bash
echo "Installing dependencies..."
pip install ultralytics opencv-python pillow pyinstaller

echo "Building .app..."
pyinstaller --noconfirm --onefile --windowed \
  --add-data "best.pt:." \
  --name "CrackScan" \
  app.py

echo ""
echo "Done! Your app is in the dist/ folder."