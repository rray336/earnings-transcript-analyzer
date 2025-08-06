# Git Workflow Automation

This project includes automated scripts to handle Git synchronization and development workflow.

## 🚀 Quick Start Scripts

### **1. `sync.bat` - Daily Sync Script**
**Use this every morning before starting development:**
```bash
# Double-click sync.bat or run in terminal:
sync.bat
```

**What it does:**
- Pulls latest changes from GitHub
- Shows current repository status
- Provides menu for common Git operations
- Interactive commit and push functionality

### **2. `quick-commit.bat` - Fast Commit & Push**
**Use this to quickly save and upload your changes:**
```bash
# With message:
quick-commit.bat "Add new feature"

# Interactive (will prompt for message):
quick-commit.bat
```

**What it does:**
- Shows current changes
- Adds all changes to Git
- Commits with your message
- Pushes to GitHub automatically

### **3. `dev-start.bat` - Development Environment**
**Use this to start your development session:**
```bash
# Double-click dev-start.bat or run in terminal:
dev-start.bat
```

**What it does:**
- Syncs with GitHub first
- Checks Python environment
- Starts the Flask development server
- Opens at http://localhost:5000

## 📅 Daily Workflow

### **Morning Routine:**
1. Double-click `sync.bat`
2. Check for any remote changes
3. Start development

### **During Development:**
1. Make your code changes
2. Test your changes locally
3. Run `quick-commit.bat "Description of changes"`

### **End of Day:**
1. Run `quick-commit.bat "End of day commit"`
2. Ensure all changes are backed up to GitHub

## 🎯 Script Details

### **sync.bat Commands:**
- `commit` - Interactive commit and push
- `status` - Check Git status
- `log` - View recent commits
- Enter - Continue to development

### **Error Handling:**
All scripts include error handling for:
- Network connectivity issues
- Git command failures
- Missing Python environment
- Invalid commit messages

## 🔧 Customization

### **Change Project Path:**
Edit the path in each script:
```batch
cd /d "C:\Users\rahul\OneDrive\IMP_DOCS\AI\EarningsAnalyzer"
```

### **Change Default Branch:**
Replace `main` with your branch name:
```batch
git pull origin main
git push origin main
```

### **Add More Commands:**
Extend `sync.bat` with additional Git operations as needed.

## 🚨 Important Notes

1. **Always run `sync.bat` before starting development**
2. **Use descriptive commit messages**
3. **Test your changes before committing**
4. **Scripts handle basic error cases but check output**
5. **Internet connection required for GitHub sync**

## 📝 Manual Git Commands (Backup)

If scripts fail, use these manual commands:

```bash
# Navigate to project
cd "C:\Users\rahul\OneDrive\IMP_DOCS\AI\EarningsAnalyzer"

# Daily sync
git pull origin main

# Commit changes
git add .
git commit -m "Your commit message"
git push origin main

# Check status
git status
git log --oneline -5
```

## 🎉 Benefits

- ✅ **No more forgetting to sync**
- ✅ **Automated error handling**  
- ✅ **One-click development startup**
- ✅ **Consistent commit workflow**
- ✅ **Visual feedback and status**
- ✅ **Backup of manual commands**

---

**Just double-click the .bat files and let them handle the Git workflow for you!** 🚀