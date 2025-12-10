# 🚀 Deployment Guide - ilovetools to PyPI

This guide will help you deploy the `ilovetools` package to PyPI using GitHub Actions.

## Prerequisites

1. **PyPI Account**: You need a PyPI account. Create one at [https://pypi.org/account/register/](https://pypi.org/account/register/)
2. **GitHub Repository**: You already have this set up at [https://github.com/AliMehdi512/ilovetools](https://github.com/AliMehdi512/ilovetools)

## Step 1: Generate PyPI API Token

1. Log in to your PyPI account at [https://pypi.org](https://pypi.org)
2. Go to **Account Settings** → **API tokens**
3. Click **Add API token**
4. Enter a token name (e.g., "ilovetools-github-actions")
5. Select scope:
   - If the package already exists on PyPI: Select **"Scope to project: ilovetools"**
   - If this is the first time: Select **"Entire account (all projects)"** (you can change this later)
6. Click **Create token**
7. **IMPORTANT**: Copy the token immediately! It starts with `pypi-` and you won't be able to see it again.

## Step 2: Add PyPI Token to GitHub Secrets

1. Go to your GitHub repository: [https://github.com/AliMehdi512/ilovetools](https://github.com/AliMehdi512/ilovetools)
2. Click on **Settings** (repository settings, not your profile)
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Name: `PYPI_API_TOKEN`
6. Value: Paste the PyPI token you copied (starts with `pypi-`)
7. Click **Add secret**

## Step 3: Verify Workflow Configuration

The GitHub Actions workflow is already configured in `.github/workflows/publish-to-pypi.yml`. It will:

- ✅ Trigger on pushes to `main` branch when:
  - `setup.py` is modified
  - `pyproject.toml` is modified
  - Any file in `ilovetools/` is modified
- ✅ Can be triggered manually from the Actions tab
- ✅ Build the package
- ✅ Check the package for errors
- ✅ Upload to PyPI

## Step 4: Publishing Options

### Option A: Automatic Publishing (Recommended for regular updates)

When you push changes to the `main` branch that affect package files, it will automatically publish:

```bash
git add .
git commit -m "Release version 0.2.6"
git push origin main
```

### Option B: Manual Publishing (Recommended for controlled releases)

1. Go to [https://github.com/AliMehdi512/ilovetools/actions](https://github.com/AliMehdi512/ilovetools/actions)
2. Click on **"Publish to PyPI"** workflow
3. Click **"Run workflow"** button
4. Select branch (usually `main`)
5. Click **"Run workflow"**

### Option C: Local Publishing (For testing)

Build and test locally before pushing:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the package
twine check dist/*

# Test upload to TestPyPI (optional)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Step 5: Verify Publication

After successful deployment:

1. Check your package on PyPI: [https://pypi.org/project/ilovetools/](https://pypi.org/project/ilovetools/)
2. Test installation:
   ```bash
   pip install ilovetools==0.2.6
   ```
3. Verify version:
   ```python
   import ilovetools
   print(ilovetools.__version__)  # Should print: 0.2.6
   ```

## Version Management

Before each release:

1. ✅ Update version in **all three files**:
   - `setup.py` → `version="X.Y.Z"`
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `ilovetools/__init__.py` → `__version__ = "X.Y.Z"`

2. ✅ Update `CHANGELOG.md` with new changes

3. ✅ Commit and push:
   ```bash
   git add setup.py pyproject.toml ilovetools/__init__.py CHANGELOG.md
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```

## Current Version Status

✅ **Version 0.2.6** - All files synchronized
- `setup.py`: 0.2.6
- `pyproject.toml`: 0.2.6
- `ilovetools/__init__.py`: 0.2.6
- `CHANGELOG.md`: Updated with 0.2.6 entry

## Troubleshooting

### Error: "File already exists"
This means the version is already published on PyPI. Increment the version number in all three files.

### Error: "Invalid credentials"
Check that `PYPI_API_TOKEN` secret is correctly set in GitHub repository settings.

### Error: "Package name already claimed"
The package `ilovetools` should already be yours. Make sure you're using the correct PyPI account.

### Workflow doesn't trigger
- Check that you're pushing to the `main` branch
- Ensure changes are in `setup.py`, `pyproject.toml`, or `ilovetools/` directory
- Check GitHub Actions tab for any errors

## Security Best Practices

- ✅ Never commit PyPI tokens to the repository
- ✅ Use GitHub Secrets for sensitive data
- ✅ Use scoped tokens (project-specific) when possible
- ✅ Rotate tokens periodically
- ✅ Review the package contents before publishing

## Additional Resources

- [PyPI Documentation](https://pypi.org/help/)
- [Python Packaging Guide](https://packaging.python.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Twine Documentation](https://twine.readthedocs.io/)

## Support

For issues or questions:
- GitHub Issues: [https://github.com/AliMehdi512/ilovetools/issues](https://github.com/AliMehdi512/ilovetools/issues)
- Email: ali.mehdi.dev579@gmail.com
