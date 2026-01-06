# Security Checklist for Trading Bot

## ✅ Before Committing to Git

Always verify you're not committing sensitive data:

```bash
# Check what will be committed
git status

# Look for these patterns
git status | grep -E "\.db$|\.joblib$|\.csv$|data/|models/"
```

## 🔒 Files That Should NEVER Be Committed

### Financial Data
- [ ] `data/trades.db` - Your trading history
- [ ] `data/*.db` - Any database files
- [ ] `*.db-journal` - SQLite journal files
- [ ] `*_backup.db` - Database backups
- [ ] `exported_data.csv` - Exported analysis
- [ ] `my_data.csv` - User exports
- [ ] Any `.csv` files with actual trading data

### Trained Models (Proprietary)
- [ ] `models/*.joblib` - Trained ML models
- [ ] `models/*.pkl` - Pickle files
- [ ] `models/*.h5` - Keras/TensorFlow models
- [ ] Any serialized model weights

### Credentials & API Keys
- [ ] `.env` files with API keys
- [ ] `config.json` with credentials
- [ ] Any files containing API keys or secrets
- [ ] Exchange adapter files with hardcoded credentials

## ✅ What IS Safe to Commit

### Code
- [x] `*.py` - Python source files
- [x] `requirements.txt` - Dependencies
- [x] `setup.py` - Package configuration

### Documentation
- [x] `README.md` files
- [x] `*.md` documentation
- [x] Example files (without real data)

### Configuration Templates
- [x] `config_template.py` (without API keys)
- [x] Example configurations
- [x] `.gitignore` file itself

### Empty Directories
- [x] `data/README.md` (with warning)
- [x] `models/README.md` (with warning)

## 🛡️ .gitignore Verification

Verify these patterns are in your `.gitignore`:

```bash
# Check .gitignore contains sensitive patterns
grep -E "data/|models/|\.db|\.joblib" .gitignore
```

Should see:
```
data/                   # SQLite databases with all trading data
*.db                    # All database files
*.db-journal            # SQLite journal files
models/                 # Trained ML models
*.joblib                # Serialized models
```

## 🚨 If You Accidentally Commit Sensitive Data

### Option 1: Just pushed to private repo
If it's a private repository and you just pushed:
```bash
# Remove file from Git (keeps local copy)
git rm --cached data/trades.db
git commit -m "Remove sensitive data"
git push
```

### Option 2: Pushed sensitive data publicly
If you pushed to a **public** repository, you need to:

1. **Immediately change all API keys/credentials**
2. **Remove from Git history:**
   ```bash
   # Use BFG Repo Cleaner (easier) or git filter-branch
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch data/trades.db" \
     --prune-empty --tag-name-filter cat -- --all
   
   # Force push (WARNING: destructive!)
   git push origin --force --all
   ```

3. **Assume data is compromised** - rotate all credentials

### Option 3: Nuclear option (start fresh)
```bash
# Create new repo without history
git checkout --orphan new-branch
git add -A
git commit -m "Fresh start with cleaned history"
git branch -D main
git branch -m main
git push -f origin main
```

## 📋 Pre-Commit Checklist

Before every `git push`:

1. [ ] Run `git status` and review all files
2. [ ] Check for `.db`, `.joblib`, `.csv` files
3. [ ] Verify no hardcoded API keys or credentials
4. [ ] Confirm `.gitignore` is working
5. [ ] Use `git diff --cached` to review changes
6. [ ] When in doubt, don't commit!

## 🔐 Additional Security Measures

### Encrypt Sensitive Data
If you need to backup sensitive data in the cloud:
```bash
# Encrypt database before uploading
gpg --symmetric --cipher-algo AES256 data/trades.db
# Creates trades.db.gpg (encrypted)
```

### Use Environment Variables
Never hardcode API keys:
```python
# ❌ Bad
api_key = "sk-1234567890abcdef"

# ✅ Good
import os
api_key = os.getenv("EXCHANGE_API_KEY")
```

### Separate Credentials Repo
Keep a separate **private** repo for:
- API keys and credentials
- Production configurations
- Deployment scripts

## 📞 What If Data Gets Leaked?

1. **Immediately revoke API keys**
2. **Change all passwords**
3. **Review account for unauthorized access**
4. **Consider the data compromised permanently**
5. **Learn from it and improve security practices**

---

Remember: **When in doubt, don't commit!** It's easier to add files later than to remove them from Git history.
