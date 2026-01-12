#!/bin/bash
# Build script for Background Remover macOS Application
# This creates a standalone .app that can be placed in /Applications

set -e

echo "=========================================="
echo "Background Remover - macOS App Builder"
echo "=========================================="
echo ""

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "📦 Installing build dependencies..."
pip install pyinstaller pillow --quiet

echo ""
echo "🔨 Building application with PyInstaller..."
echo "(This may take several minutes)"
echo ""

# Clean previous builds
rm -rf build dist "Background Remover.spec" 2>/dev/null || true

# Use PyInstaller - simplified build without moviepy
pyinstaller \
    --name "Background Remover" \
    --windowed \
    --onedir \
    --noconfirm \
    --clean \
    --add-data "backgroundremover/u2net:backgroundremover/u2net" \
    --add-data "backgroundremover/github.py:backgroundremover" \
    --add-data "backgroundremover/__init__.py:backgroundremover" \
    --hidden-import "PIL._tkinter_finder" \
    --hidden-import "torch" \
    --hidden-import "torchvision" \
    --hidden-import "torchvision.transforms" \
    --hidden-import "scipy.spatial" \
    --hidden-import "scipy.ndimage" \
    --hidden-import "scipy.ndimage._morphology" \
    --hidden-import "pymatting" \
    --hidden-import "pymatting.alpha" \
    --hidden-import "pymatting.alpha.estimate_alpha_cf" \
    --hidden-import "pymatting.foreground" \
    --hidden-import "pymatting.foreground.estimate_foreground_ml" \
    --hidden-import "pymatting.util" \
    --hidden-import "pymatting.util.util" \
    --hidden-import "hsh" \
    --hidden-import "hsh.library" \
    --hidden-import "hsh.library.hash" \
    --hidden-import "pillow_heif" \
    --hidden-import "requests" \
    --hidden-import "tqdm" \
    --hidden-import "numba" \
    --collect-all "torch" \
    --collect-all "torchvision" \
    --collect-all "scipy" \
    --collect-all "pymatting" \
    --collect-all "hsh" \
    --collect-all "numba" \
    --collect-all "llvmlite" \
    --collect-all "pillow_heif" \
    --exclude-module "moviepy" \
    --exclude-module "imageio" \
    --exclude-module "imageio_ffmpeg" \
    --exclude-module "ffmpeg" \
    --exclude-module "matplotlib" \
    --exclude-module "PyQt5" \
    --exclude-module "PyQt6" \
    --exclude-module "PySide2" \
    --exclude-module "PySide6" \
    --exclude-module "wx" \
    --exclude-module "IPython" \
    --exclude-module "jupyter" \
    --exclude-module "notebook" \
    --exclude-module "tensorboard" \
    app_gui.py

echo ""
echo "✅ Build complete!"
echo ""

# Check if build was successful
if [ -d "dist/Background Remover.app" ]; then
    APP_PATH="dist/Background Remover.app"
    APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)
    
    echo "=========================================="
    echo "📱 Application Built Successfully!"
    echo "=========================================="
    echo ""
    echo "📍 Location: $SCRIPT_DIR/$APP_PATH"
    echo "📦 Size: $APP_SIZE"
    echo ""
    echo "To install to Applications folder, run:"
    echo "  cp -R \"$SCRIPT_DIR/$APP_PATH\" /Applications/"
    echo ""
    echo "Or drag the app from dist/ folder to Applications"
    echo ""
    
    # Optionally open the dist folder
    open "$SCRIPT_DIR/dist"
else
    echo "❌ Build failed. Check the output above for errors."
    exit 1
fi
