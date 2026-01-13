# Python Setup Guide for Cursor IDE

## Current Issue
- ✅ Python 3.14.2 is installed
- ✅ Basic packages (pandas, numpy, scikit-learn, scipy, matplotlib) are installed
- ❌ TensorFlow doesn't support Python 3.14 yet (supports up to Python 3.12)

## Solution Options

### Option 1: Install Python 3.12 (Recommended)
1. Download Python 3.12 from: https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Install Python 3.12 alongside 3.14 (they can coexist)
4. In Cursor IDE:
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose Python 3.12
5. Install packages: `python -m pip install -r requirements.txt`

### Option 2: Use Virtual Environment with Python 3.12
1. Install Python 3.12 (if not already installed)
2. Create virtual environment:
   ```powershell
   py -3.12 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. In Cursor IDE, select the Python interpreter from `venv\Scripts\python.exe`

## Configure Cursor IDE

1. **Select Python Interpreter:**
   - Press `Ctrl+Shift+P` (or `F1`)
   - Type: `Python: Select Interpreter`
   - Choose the correct Python version (3.12 for TensorFlow)

2. **Verify Installation:**
   - Open a Python file
   - Check bottom-right corner for Python version
   - Run: `python -m pip list` to see installed packages

## Quick Fix Commands

```powershell
# Check current Python version
python --version

# Install all packages (except TensorFlow for now)
python -m pip install pandas numpy scikit-learn scipy matplotlib

# After switching to Python 3.12, install TensorFlow:
python -m pip install tensorflow
```

## Current Package Status
- ✅ pandas
- ✅ numpy
- ✅ scikit-learn
- ✅ scipy
- ✅ matplotlib
- ❌ tensorflow (requires Python 3.12 or earlier)
