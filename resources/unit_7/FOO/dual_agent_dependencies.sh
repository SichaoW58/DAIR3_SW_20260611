#!/bin/bash
# install_dependencies.sh
# Installation script for grant_review_v2.py multi-modal dependencies

echo "Installing dependencies for grant_review_v2.py..."

# Core dependencies
echo "Installing core dependencies..."
pip install PyQt5

# PDF processing
echo "Installing PDF processing..."
pip install PyPDF2

# Image processing and OCR
echo "Installing image processing..."
pip install pillow

# Note: pytesseract requires Tesseract-OCR to be installed separately
echo "Installing pytesseract (Python wrapper)..."
pip install pytesseract

echo ""
echo "NOTE: For OCR to work, you also need to install Tesseract-OCR:"
echo "  - Ubuntu/Debian: sudo apt-get install tesseract-ocr"
echo "  - macOS: brew install tesseract"
echo "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"

# Audio processing
echo ""
echo "Installing audio processing..."
pip install SpeechRecognition pydub

# Additional audio dependencies
echo ""
echo "NOTE: For audio processing, you may also need:"
echo "  - Ubuntu/Debian: sudo apt-get install python3-pyaudio portaudio19-dev"
echo "  - macOS: brew install portaudio"
echo "  - Windows: PyAudio wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio"

echo ""
echo "Installation complete!"
echo "Some features may require additional system-level installations as noted above."
