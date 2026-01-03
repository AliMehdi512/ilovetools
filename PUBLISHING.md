# Publishing to PyPI Guide

This guide explains how to publish the ilovetools package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org
2. **API Token**: Generate an API token from PyPI account settings
3. **GitHub Secret**: Add the token as `PYPI_API_TOKEN` in repository secrets

## Method 1: Automated Publishing (Recommended)

### Using GitHub Actions

The repository has an automated workflow that publishes to PyPI when you create a tag.

```bash
# 1. Ensure version is updated in setup.py (already done: 0.2.22)

# 2. Commit all changes
git add .
git commit -m "Release version 0.2.22"
git push

# 3. Create and push a tag
git tag v0.2.22
git push origin v0.2.22
```

The GitHub Actions workflow will automatically:
- Build the package
- Run checks
- Publish to PyPI
- Create a GitHub release

### Manual Trigger

You can also manually trigger the workflow from GitHub Actions tab:
1. Go to Actions → Publish to PyPI
2. Click "Run workflow"
3. Enter version number (e.g., 0.2.22)
4. Click "Run workflow"

## Method 2: Manual Publishing

If you prefer to publish manually:

```bash
# 1. Install build tools
pip install build twine

# 2. Build the package
python -m build

# 3. Check the package
twine check dist/*

# 4. Upload to TestPyPI (optional, for testing)
twine upload --repository testpypi dist/*

# 5. Upload to PyPI
twine upload dist/*
```

## After Publishing

### Verify Installation

```bash
# Install from PyPI
pip install ilovetools==0.2.22

# Run verification test
python tests/test_pypi_installation.py
```

### Test the Package

```python
# Test imports
from ilovetools.ml.normalization import BatchNorm1d, LayerNorm

# Test functionality
import numpy as np

bn = BatchNorm1d(num_features=128)
x = np.random.randn(32, 128)
output = bn.forward(x, training=True)
print(f"BatchNorm1d works! Output shape: {output.shape}")

ln = LayerNorm(normalized_shape=512)
x = np.random.randn(32, 10, 512)
output = ln.forward(x)
print(f"LayerNorm works! Output shape: {output.shape}")
```

## Version 0.2.22 Changes

This release includes:

✅ **Enhanced Batch Normalization**
- BatchNorm1d for fully connected layers
- BatchNorm2d for convolutional layers
- Full training/inference mode support
- Running statistics tracking
- Learnable affine parameters

✅ **Enhanced Layer Normalization**
- LayerNorm for RNNs and Transformers
- Per-sample normalization
- Backward pass support

✅ **Additional Normalizations**
- GroupNorm for small batch sizes
- InstanceNorm for style transfer

✅ **Comprehensive Testing**
- 18+ test cases
- Functional API tests
- Edge case coverage

✅ **Documentation**
- Complete examples
- Best practices guide
- Usage documentation

## Troubleshooting

### Build Fails
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Rebuild
python -m build
```

### Upload Fails
```bash
# Check credentials
twine check dist/*

# Verify API token is correct
# Update token in GitHub secrets if needed
```

### Import Errors After Installation
```bash
# Reinstall with --force-reinstall
pip install --force-reinstall ilovetools==0.2.22

# Or install from source
pip install git+https://github.com/AliMehdi512/ilovetools.git
```

## Quick Publish Checklist

- [ ] Version bumped in setup.py (✓ 0.2.22)
- [ ] All changes committed and pushed
- [ ] Tests passing locally
- [ ] CHANGELOG updated (if exists)
- [ ] Tag created and pushed
- [ ] GitHub Actions workflow completed
- [ ] Package visible on PyPI
- [ ] Installation verified
- [ ] Imports working correctly

## Links

- **PyPI Package**: https://pypi.org/project/ilovetools/
- **GitHub Repository**: https://github.com/AliMehdi512/ilovetools
- **Documentation**: https://github.com/AliMehdi512/ilovetools/blob/main/README.md
- **Issues**: https://github.com/AliMehdi512/ilovetools/issues

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review GitHub Actions logs
3. Open an issue on GitHub
4. Contact: ali.mehdi.dev579@gmail.com
