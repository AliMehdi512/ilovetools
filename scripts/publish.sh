#!/bin/bash

# Quick Publish Script for ilovetools to PyPI
# This script automates the publishing process

set -e  # Exit on error

echo "=========================================="
echo "ilovetools PyPI Publishing Script"
echo "=========================================="
echo ""

# Get current version from setup.py
CURRENT_VERSION=$(grep "version=" setup.py | cut -d'"' -f2)
echo "Current version: $CURRENT_VERSION"
echo ""

# Ask for confirmation
read -p "Do you want to publish version $CURRENT_VERSION to PyPI? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Publishing cancelled."
    exit 1
fi

echo ""
echo "Step 1: Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
echo "✓ Cleaned"

echo ""
echo "Step 2: Installing/upgrading build tools..."
pip install --upgrade build twine setuptools wheel
echo "✓ Build tools ready"

echo ""
echo "Step 3: Building package..."
python -m build
echo "✓ Package built"

echo ""
echo "Step 4: Checking package..."
twine check dist/*
echo "✓ Package checked"

echo ""
echo "Step 5: Publishing to PyPI..."
echo "Note: You'll need to enter your PyPI API token"
twine upload dist/*
echo "✓ Published to PyPI"

echo ""
echo "Step 6: Creating git tag..."
git tag "v$CURRENT_VERSION"
git push origin "v$CURRENT_VERSION"
echo "✓ Tag created and pushed"

echo ""
echo "=========================================="
echo "✓ Successfully published version $CURRENT_VERSION"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify on PyPI: https://pypi.org/project/ilovetools/$CURRENT_VERSION/"
echo "2. Test installation: pip install ilovetools==$CURRENT_VERSION"
echo "3. Run verification: python tests/test_pypi_installation.py"
echo ""
