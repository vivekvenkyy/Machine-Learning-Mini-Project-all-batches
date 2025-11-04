# 🔧 Installation Guide

Complete installation instructions for Windows, macOS, and Linux.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Linux Installation](#linux-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Optional Dependencies](#optional-dependencies)

---

## Prerequisites

### Required Software

- **Python**: Version 3.8 or higher
- **pip**: Python package installer (usually comes with Python)
- **Internet connection**: For downloading packages

### Checking Your Python Version

**Windows (PowerShell/CMD)**:

```powershell
python --version
```

**macOS/Linux (Terminal)**:

```bash
python3 --version
```

**Expected Output**: `Python 3.8.x` or higher

If Python is not installed, download from: https://www.python.org/downloads/

---

## Windows Installation

### Method 1: Using PowerShell (Recommended)

1. **Open PowerShell**:

   - Press `Win + X`
   - Select "Windows PowerShell" or "Terminal"

2. **Navigate to project folder**:

   ```powershell
   cd D:\AIML\ML_mini
   ```

3. **Install dependencies**:

   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the application**:

   ```powershell
   streamlit run app.py
   ```

5. **Open browser**:
   - Automatically opens at `http://localhost:8501`
   - Or manually navigate to the URL shown in terminal

### Method 2: Using Batch File (Easiest)

1. **Double-click** `run_app.bat` in the project folder
2. **Wait** for installation (if first time)
3. **Browser opens automatically**

### Method 3: Using Virtual Environment (Best Practice)

1. **Create virtual environment**:

   ```powershell
   cd D:\AIML\ML_mini
   python -m venv venv
   ```

2. **Activate virtual environment**:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   If you get an execution policy error:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:

   ```powershell
   pip install -r requirements.txt
   ```

4. **Run application**:

   ```powershell
   streamlit run app.py
   ```

5. **Deactivate when done** (optional):
   ```powershell
   deactivate
   ```

---

## macOS Installation

### Method 1: Using Terminal (Standard)

1. **Open Terminal**:

   - Press `Cmd + Space`
   - Type "Terminal" and press Enter

2. **Navigate to project folder**:

   ```bash
   cd ~/Downloads/ML_mini  # Adjust path as needed
   ```

3. **Install dependencies**:

   ```bash
   pip3 install -r requirements.txt
   ```

   Or with user flag (if permission issues):

   ```bash
   pip3 install --user -r requirements.txt
   ```

4. **Run the application**:

   ```bash
   streamlit run app.py
   ```

5. **Open browser**:
   - Should open automatically
   - Or go to `http://localhost:8501`

### Method 2: Using Virtual Environment (Recommended)

1. **Create virtual environment**:

   ```bash
   cd ~/Downloads/ML_mini
   python3 -m venv venv
   ```

2. **Activate virtual environment**:

   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run application**:

   ```bash
   streamlit run app.py
   ```

5. **Deactivate when done**:
   ```bash
   deactivate
   ```

### Method 3: Using Homebrew Python

If you installed Python via Homebrew:

1. **Ensure Homebrew Python is used**:

   ```bash
   which python3  # Should show /usr/local/bin/python3 or /opt/homebrew/bin/python3
   ```

2. **Install dependencies**:

   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run application**:
   ```bash
   streamlit run app.py
   ```

---

## Linux Installation

### Ubuntu/Debian

1. **Update package list**:

   ```bash
   sudo apt update
   ```

2. **Install Python and pip** (if not installed):

   ```bash
   sudo apt install python3 python3-pip python3-venv
   ```

3. **Navigate to project**:

   ```bash
   cd ~/ML_mini  # Adjust path
   ```

4. **Create virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

6. **Run application**:
   ```bash
   streamlit run app.py
   ```

### Fedora/CentOS/RHEL

1. **Install Python and pip**:

   ```bash
   sudo dnf install python3 python3-pip
   ```

2. **Follow steps 3-6 from Ubuntu section above**

### Arch Linux

1. **Install Python and pip**:

   ```bash
   sudo pacman -S python python-pip
   ```

2. **Follow steps 3-6 from Ubuntu section above**

---

## Verification

### Check Installation

After installation, verify everything works:

```bash
# Check Streamlit
streamlit --version
# Expected: Streamlit, version 1.28.0 or higher

# Check scikit-learn
python -c "import sklearn; print(sklearn.__version__)"
# Expected: 1.3.0 or higher

# Check pandas
python -c "import pandas; print(pandas.__version__)"
# Expected: 2.0.0 or higher

# Check plotly
python -c "import matplotlib; print(matplotlib.__version__)"
# Expected: 3.7.0 or higher

# Check seaborn
python -c "import seaborn; print(seaborn.__version__)"
# Expected: 0.12.0 or higher
```

### Test Run

1. **Start the app**:

   ```bash
   streamlit run app.py
   ```

2. **Check output**:

   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

3. **Verify in browser**:
   - App loads without errors
   - Sidebar shows "Configuration" section
   - Three tabs are visible
   - Dataset info shows "California Housing"

---

## Troubleshooting

### Issue: "Python not found"

**Windows**:

```powershell
# Check if Python is in PATH
where python
# If not found, reinstall Python and check "Add to PATH"
```

**macOS/Linux**:

```bash
# Try python3 instead
which python3
# If not found, install Python
```

### Issue: "pip not found"

**Install pip**:

```bash
# Windows
python -m ensurepip --upgrade

# macOS/Linux
python3 -m ensurepip --upgrade
```

### Issue: "Permission denied"

**Solution 1 - User install**:

```bash
pip install --user -r requirements.txt
```

**Solution 2 - Virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

**Solution 3 - Sudo** (Linux only, not recommended):

```bash
sudo pip install -r requirements.txt
```

### Issue: "Module not found" after installation

**Check installation location**:

```bash
pip show streamlit
# Note the "Location" path
```

**Ensure Python uses correct path**:

```bash
python -c "import sys; print(sys.path)"
# Should include the location from above
```

**Reinstall in correct location**:

```bash
python -m pip install -r requirements.txt
```

### Issue: XGBoost installation fails

**Windows - Install Visual C++**:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Retry: `pip install xgboost`

**macOS - Install Xcode tools**:

```bash
xcode-select --install
pip install xgboost
```

**Linux - Install build tools**:

```bash
sudo apt install build-essential  # Ubuntu/Debian
sudo dnf install gcc gcc-c++       # Fedora
pip install xgboost
```

**Alternative - Skip XGBoost**:
The app works without XGBoost. You'll see a warning but other models still work.

### Issue: Streamlit won't start

**Check port 8501**:

```bash
# Windows
netstat -ano | findstr :8501

# macOS/Linux
lsof -i :8501
```

**Kill process if needed**:

```bash
# Windows (replace PID)
taskkill /PID <PID> /F

# macOS/Linux
kill -9 <PID>
```

**Use different port**:

```bash
streamlit run app.py --server.port 8502
```

### Issue: Browser doesn't open automatically

**Manually open browser**:

1. Copy URL from terminal (e.g., `http://localhost:8501`)
2. Open browser
3. Paste URL in address bar

**Enable auto-open**:

```bash
streamlit run app.py --server.headless false
```

### Issue: Slow performance / App hangs

**Reduce dataset size**:

- Use smaller test set percentage
- Train fewer models

**Increase memory**:

- Close other applications
- Use more powerful machine

**Check logs**:

```bash
streamlit run app.py --logger.level=debug
```

### Issue: CSV upload fails

**Check CSV format**:

- ✅ Has column headers
- ✅ UTF-8 encoding
- ✅ Comma-separated (not semicolon)
- ✅ No special characters in headers

**Convert encoding**:

```bash
# Windows (PowerShell)
Get-Content input.csv | Set-Content -Encoding utf8 output.csv

# macOS/Linux
iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
```

---

## Optional Dependencies

### Hugging Face Datasets

To enable Hugging Face dataset support:

```bash
pip install datasets
```

**Note**: This adds ~500MB of dependencies, so it's optional.

### CUDA Support for XGBoost (GPU acceleration)

If you have an NVIDIA GPU:

1. **Install CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads

2. **Install XGBoost with GPU**:

   ```bash
   pip install xgboost[gpu]
   ```

3. **Verify**:
   ```python
   import xgboost as xgb
   print(xgb.__version__)
   ```

### Jupyter Notebook Integration

To use the code in Jupyter:

```bash
pip install jupyter notebook
jupyter notebook
```

Then create a new notebook and import functions from `app.py`.

---

## Updating Dependencies

### Update all packages

```bash
pip install --upgrade -r requirements.txt
```

### Update specific package

```bash
pip install --upgrade streamlit
pip install --upgrade scikit-learn
```

### Check for outdated packages

```bash
pip list --outdated
```

---

## Uninstallation

### Remove all packages

```bash
pip uninstall -r requirements.txt -y
```

### Remove virtual environment

```bash
# Windows
rmdir /s venv

# macOS/Linux
rm -rf venv
```

---

## System Requirements

### Minimum Requirements

- **CPU**: Dual-core processor
- **RAM**: 4 GB
- **Storage**: 500 MB free space
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements

- **CPU**: Quad-core processor
- **RAM**: 8 GB
- **Storage**: 2 GB free space
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+

### Performance Notes

- **Small datasets** (<1K rows): Runs on minimum specs
- **Medium datasets** (1K-10K): Recommended specs
- **Large datasets** (>10K): 16GB RAM + SSD recommended

---

## Docker Installation (Advanced)

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY sample_data.csv .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t ml-demo .

# Run container
docker run -p 8501:8501 ml-demo
```

### Access app

Open browser to `http://localhost:8501`

---

## Cloud Deployment

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect GitHub repository
4. Deploy!

### Heroku

1. Create `Procfile`:

   ```
   web: streamlit run app.py --server.port $PORT
   ```

2. Deploy:
   ```bash
   heroku create
   git push heroku main
   ```

### AWS EC2

1. Launch Ubuntu instance
2. SSH into instance
3. Follow Linux installation steps
4. Configure security group (port 8501)
5. Access via public IP

---

## Getting Help

### Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **scikit-learn Docs**: https://scikit-learn.org
- **Stack Overflow**: Tag questions with `streamlit` and `scikit-learn`

### Common Commands Reference

```bash
# Install
pip install -r requirements.txt

# Run
streamlit run app.py

# Update
pip install --upgrade -r requirements.txt

# Check version
streamlit --version

# Clear cache
streamlit cache clear

# Run with options
streamlit run app.py --server.port 8502 --server.headless true
```

---

**Installation complete! Ready to explore machine learning! 🚀**
