# Automatic Versioning with setuptools-scm

## How It Works

Versions are **automatically derived from git tags** - you never manually edit version numbers!

## Workflow

### 1. Make Changes
```bash
# Make your changes
git add .
git commit -m "Add new feature"
```

### 2. Create Release Tag
```bash
# Tag a release (this becomes the version)
git tag v1.0.0
git push origin v1.0.0
```

### 3. Version is Automatic
```python
import trading_bot
print(trading_bot.__version__)  # "1.0.0"
```

## Version Examples

| Git State | Version |
|-----------|---------|
| `git tag v1.2.3` | `1.2.3` |
| `v1.2.3` + 5 commits | `1.2.4.dev5+g1234abc` |
| Uncommitted changes | `1.2.4.dev5+g1234abc.d20260105` |
| No tags | `0.0.0.dev0+unknown` |

## Benefits

✅ **No manual editing** - version comes from git  
✅ **Dev versions** include commit hash for traceability  
✅ **Always unique** - every commit has unique version  
✅ **PEP 440 compliant** - works with PyPI, pip, etc.

## Release Workflow

```bash
# 1. Update CHANGELOG
echo "## [1.1.0] - 2026-01-05\n- Added persistence features" >> CHANGELOG.md
git add CHANGELOG.md
git commit -m "Update changelog for v1.1.0"

# 2. Create tag
git tag -a v1.1.0 -m "Release v1.1.0: Persistence features"

# 3. Push (includes tag)
git push origin main --follow-tags

# 4. (Optional) Publish to PyPI
python -m build
python -m twine upload dist/*
```

## Semantic Versioning Guide

```
MAJOR.MINOR.PATCH
  │      │     │
  │      │     └─ Bug fixes (1.0.1)
  │      └─────── New features, backward compatible (1.1.0)
  └────────────── Breaking changes (2.0.0)
```

**Examples:**
- `v1.0.0` → `v1.0.1` - Fixed bug in trade_history
- `v1.0.1` → `v1.1.0` - Added new DataCurator features
- `v1.1.0` → `v2.0.0` - Changed ExchangeClient interface (breaking)

## Checking Current Version

```bash
# In Python
python -c "from trading_bot import __version__; print(__version__)"

# Or use git
git describe --tags --always
```

## Pre-releases

```bash
# Alpha release
git tag v1.2.0a1

# Beta release
git tag v1.2.0b1

# Release candidate
git tag v1.2.0rc1

# Final release
git tag v1.2.0
```

## Troubleshooting

### "Version shows 0.0.0.dev0+unknown"
**Cause:** No git tags exist  
**Fix:** Create initial tag
```bash
git tag v1.0.0
```

### "Version file not generated"
**Cause:** Package not installed with setuptools-scm  
**Fix:** Install in editable mode
```bash
pip install -e .
```

### "Version not updating"
**Cause:** Need to reinstall after creating new tag  
**Fix:** 
```bash
pip install -e . --force-reinstall --no-deps
# Or just: import importlib; importlib.reload(trading_bot)
```

## How to Create First Release

```bash
# 1. Make sure you're in the trading_bot directory with .git
cd r:\repo\trading_bot

# 2. Commit any pending changes
git add .
git commit -m "Initial release preparation"

# 3. Create v1.0.0 tag
git tag -a v1.0.0 -m "Initial release"

# 4. Push with tags
git push origin main --tags

# 5. Reinstall to pick up new version
cd py
pip install -e . --force-reinstall --no-deps

# 6. Verify
python -c "from trading_bot import __version__; print(__version__)"
# Should print: 1.0.0
```

Done! Version now updates automatically from git tags. 🎉
