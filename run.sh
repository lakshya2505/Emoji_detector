#!/bin/bash
# Emoji Detector Run Script
# Make sure to give this file execute permission with:
# chmod +x run.sh

echo " Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo " Installing dependencies..."
pip install -r requirements.txt

echo "Starting Emoji Detector..."
python3 emoji_detector.py
