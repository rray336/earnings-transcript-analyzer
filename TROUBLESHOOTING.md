# 🔧 Git Workflow Troubleshooting

## Common Issues and Solutions

### ❌ **Error: "'Push' is not recognized as an internal or external command"**

**Problem:** Git is not installed or not in your system PATH.

**Solution:**
1. **Run the diagnostic script first:**
   ```
   Double-click: check-git.bat
   ```

2. **If Git is not installed:**
   - Go to: https://git-scm.com/download/win
   - Download "Git for Windows"
   - During installation, select "Git from the command line and also from 3rd-party software"
   - Restart your computer after installation

3. **If Git is installed but not working:**
   - Open Command Prompt as Administrator
   - Type: `where git`
   - If nothing appears, add Git to PATH:
     - Go to System Properties → Environment Variables
     - Add Git installation path (usually `C:\Program Files\Git\bin`)

### ❌ **Error: "Not in a Git repository"**

**Problem:** The folder is not initialized as a Git repository.

**Solution:**
```bash
cd "C:\Users\rahul\OneDrive\IMP_DOCS\AI\EarningsAnalyzer"
git init
git remote add origin https://github.com/YOUR_USERNAME/earnings-transcript-analyzer.git
```

### ❌ **Error: "Push failed" or "Authentication failed"**

**Problem:** GitHub authentication not set up.

**Solutions:**

**Option 1 - Personal Access Token (Recommended):**
1. Go to GitHub.com → Settings → Developer settings → Personal access tokens
2. Generate new token with "repo" permissions
3. Use token as password when prompted

**Option 2 - GitHub CLI:**
```bash
# Install GitHub CLI first, then:
gh auth login
```

**Option 3 - SSH Keys:**
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your@email.com"`
2. Add to GitHub: Settings → SSH and GPG keys
3. Use SSH URL instead of HTTPS

### ❌ **Error: "Nothing to commit"**

**Problem:** No changes detected.

**Solution:** This is normal! It means your repository is up to date.

### ❌ **Error: "Remote repository not found"**

**Problem:** GitHub repository URL is incorrect.

**Solution:**
```bash
git remote -v                    # Check current remote
git remote remove origin         # Remove incorrect remote
git remote add origin https://github.com/YOUR_USERNAME/CORRECT_REPO_NAME.git
```

## 🛠️ Alternative Solutions

### **1. Use Manual Sync Script**
If automated scripts don't work:
```
Double-click: manual-sync.bat
```
This opens a command prompt where you can run Git commands manually.

### **2. Use Git GUI Tools**
- **GitHub Desktop:** https://desktop.github.com/
- **SourceTree:** https://www.sourcetreeapp.com/
- **GitKraken:** https://www.gitkraken.com/

### **3. Use VS Code Git Integration**
- Install VS Code
- Open your project folder
- Use the Source Control panel (Ctrl+Shift+G)

## 🔍 Diagnostic Steps

### **Step 1: Run Diagnostic Script**
```
Double-click: check-git.bat
```

### **Step 2: Manual Testing**
Open Command Prompt and test:
```bash
git --version                    # Should show Git version
cd "YOUR_PROJECT_PATH"          # Navigate to project
git status                      # Should show repository status
```

### **Step 3: Check Repository Setup**
```bash
git remote -v                   # Should show GitHub URL
git config user.name            # Should show your name
git config user.email           # Should show your email
```

## 📞 Getting Help

### **Quick Fixes:**
1. **Restart your computer** after installing Git
2. **Run as Administrator** if you get permission errors
3. **Check internet connection** for push/pull operations
4. **Update Git** to the latest version

### **When All Else Fails:**
Use the manual sync script (`manual-sync.bat`) and run commands individually:

```bash
git status
git add .
git commit -m "Your message here"
git push origin main
```

### **Emergency Backup:**
If scripts completely fail, you can always:
1. Copy your project files
2. Re-clone from GitHub
3. Copy your files back
4. Commit and push manually

## ✅ Success Checklist

After fixing issues, verify everything works:
- [ ] `check-git.bat` shows all ✅
- [ ] `quick-commit.bat` works without errors
- [ ] `sync.bat` can pull from GitHub
- [ ] Changes appear on GitHub.com

---

**Remember:** When in doubt, run `check-git.bat` first to diagnose the issue! 🎯