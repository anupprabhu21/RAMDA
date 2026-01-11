#!/bin/bash

echo "🔧 Setting up Resource-Aware Edge AI Project..."

# Remove old venv if broken
if [ -d "venv" ]; then
    echo "⚠️ Removing existing virtual environment..."
    rm -rf venv
fi

# Create new venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ Setup completed successfully"
echo "➡️ Activate environment using: source venv/bin/activate"

