# 📘 User Guide — Agent AI State Management Workshop

Welcome! This guide will help you set up your environment and follow along with the hands-on demo.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install Python](#2-install-python)
3. [Install VS Code](#3-install-vs-code)
4. [Install VS Code Extensions](#4-install-vs-code-extensions)
5. [Set Up Python Environment](#5-set-up-python-environment)
6. [Configure API Key](#6-configure-api-key)
7. [Open the Demo Notebook](#7-open-the-demo-notebook)
8. [Running the Demo](#8-running-the-demo)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

Before starting, make sure you have:

- [ ] A computer (macOS, Windows, or Linux)
- [ ] Internet connection
- [ ] An OpenAI API key (or compatible provider like Azure OpenAI)

**Time needed**: ~15 minutes for first-time setup

---

## 2. Install Python

### Check if Python is Installed

Open a terminal and run:

```bash
python3 --version
```

If you see **Python 3.10** or higher, skip to [Section 3](#3-install-vs-code).

### macOS

**Option A: Homebrew (recommended)**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify
python3.11 --version
```

**Option B: Official Installer**
1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download Python 3.11+ for macOS
3. Run the `.pkg` installer
4. Open a **new terminal** and verify: `python3 --version`

### Windows

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download Python 3.11+ for Windows
3. Run the installer
4. ✅ **Check "Add Python to PATH"** (important!)
5. Click "Install Now"
6. Open a **new Command Prompt** and verify: `python --version`

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 --version
```

### Linux (Fedora)

```bash
sudo dnf install python3.11
python3.11 --version
```

---

## 3. Install VS Code

If you don't have VS Code installed:

1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download for your operating system
3. Run the installer
4. Launch VS Code

---

## 4. Install VS Code Extensions

You need these extensions to run Jupyter notebooks in VS Code:

### Required Extensions

Open VS Code and install these extensions:

#### 1. Python Extension

1. Click the **Extensions** icon (left sidebar) or press `Cmd+Shift+X` (macOS) / `Ctrl+Shift+X` (Windows/Linux)
2. Search for **"Python"**
3. Install **"Python"** by Microsoft (the one with millions of downloads)

![Python Extension](https://ms-python.gallerycdn.vsassets.io/extensions/ms-python/python/2024.0.1/1704394355648/Microsoft.VisualStudio.Services.Icons.Default)

#### 2. Jupyter Extension

1. In Extensions, search for **"Jupyter"**
2. Install **"Jupyter"** by Microsoft
3. This also installs:
   - Jupyter Keymap
   - Jupyter Notebook Renderers
   - Jupyter Cell Tags
   - Jupyter Slide Show

#### 3. (Optional) Pylance

For better Python IntelliSense:
1. Search for **"Pylance"**
2. Install **"Pylance"** by Microsoft

### Quick Install via Command Line

You can also install extensions from the terminal:

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance
```

### Verify Extensions

1. Open VS Code
2. Go to Extensions (`Cmd+Shift+X` / `Ctrl+Shift+X`)
3. Click "Installed" tab
4. You should see:
   - ✅ Python
   - ✅ Jupyter

---

## 5. Set Up Python Environment

### Create Virtual Environment

> ⚠️ **Always use a virtual environment!** This keeps your project dependencies isolated.

Open a terminal in the folder containing `agent-state-management-demo.ipynb`:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
# macOS/Linux:
source .venv/bin/activate

# Windows (Command Prompt):
.venv\Scripts\activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal prompt — this means the virtual environment is active.

### Install Dependencies

```bash
# Make sure (.venv) is in your prompt!
pip install numpy sentence-transformers tiktoken openai matplotlib python-dotenv
```

**What gets installed:**

| Package | Purpose |
|---------|---------|
| `numpy` | Math operations |
| `sentence-transformers` | Local text embeddings |
| `tiktoken` | Token counting |
| `openai` | LLM API calls |
| `matplotlib` | Charts |
| `python-dotenv` | Load API key from file |

This may take a few minutes (downloads ~500MB for the embedding model).

---

## 6. Configure API Key

### Create .env File

In the project root, create a file named `.env`:

```bash
# macOS/Linux
echo 'OPENAI_API_KEY=sk-your-key-here' > .env

# Or use any text editor
```

Replace `sk-your-key-here` with your actual OpenAI API key.

### Get an API Key

If you don't have an OpenAI API key:

1. Go to [platform.openai.com](https://platform.openai.com/)
2. Sign up or log in
3. Go to **API Keys** section
4. Click **Create new secret key**
5. Copy the key (starts with `sk-`)

> ⚠️ The `.env` file is in `.gitignore` — your key won't be committed to git.

---

## 7. Open the Demo Notebook

### In VS Code

1. Open VS Code
2. **File** → **Open Folder** → select the folder containing the demo files
3. In the file explorer, click `agent-state-management-demo.ipynb`

### Select Python Kernel

When you open the notebook:

1. Click **"Select Kernel"** (top right of notebook)
2. Choose **"Python Environments"**
3. Select the `.venv` environment (should show `Python 3.x.x ('.venv': venv)`)

If you don't see `.venv`:
1. Press `Cmd+Shift+P` (macOS) / `Ctrl+Shift+P` (Windows/Linux)
2. Type **"Python: Select Interpreter"**
3. Click **"Enter interpreter path"**
4. Navigate to `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows)

---

## 8. Running the Demo

### Cell Execution Basics

| Action | Shortcut |
|--------|----------|
| Run current cell | `Shift + Enter` |
| Run current cell, stay in place | `Ctrl + Enter` |
| Run all cells | Click **"Run All"** button |
| Add cell below | `B` (in command mode) |
| Delete cell | `D D` (press D twice) |

### Demo Structure

The notebook has three phases:

1. **Phase 1: Sliding Window** — Shows the problem (lost context)
2. **Phase 2: Summarization** — Shows a partial fix (compressed history)
3. **Phase 3: Tiered Memory** — Shows the full solution (self-organizing memory)

### Running Order

Run cells in order from top to bottom:

1. **Setup & Dependencies** — Installs packages, loads models
2. **Helper Functions** — Defines utilities
3. **Scenario** — Loads the 25-turn conversation
4. **Runner Function** — Defines how to process turns
5. **Phase 1** → **Phase 2** → **Phase 3**
6. **Comparison Charts** — Visualizes the differences

### Expected First Run

When you run the setup cell for the first time:
- The embedding model downloads (~90MB)
- This only happens once

```
LLM: gpt-4o-mini
Embeddings: all-MiniLM-L6-v2 (local)
Tokenizer: o200k_base
Ready.
```

---

## 9. Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Check `.venv` is active, reinstall packages |
| `AuthenticationError` | Check `.env` has correct API key |
| Kernel not found | Reselect Python interpreter in VS Code |
| Charts not showing | Add `%matplotlib inline` at top of notebook |

### "No Module Named openai"

```bash
# Make sure venv is active
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Reinstall
pip install openai
```

### Kernel Keeps Dying

1. Close VS Code
2. Delete `.venv` folder
3. Recreate: `python3 -m venv .venv`
4. Reinstall packages
5. Reopen VS Code

### Can't Find .venv Kernel

1. `Cmd+Shift+P` → **"Python: Clear Cache and Reload Window"**
2. Close and reopen VS Code
3. Try selecting kernel again

### Rate Limit Errors

If you see `RateLimitError`:
- You've hit OpenAI's rate limit
- Wait a minute and try again
- Or add `time.sleep(1)` between API calls

---

## Quick Reference

### Key Files

| File | Purpose |
|------|---------|
| `agent-state-management-demo.ipynb` | Main demo notebook |
| `.env` | Your API key (create this) |
| `USER-GUIDE.md` | This guide |

### Useful Commands

```bash
# Activate venv
source .venv/bin/activate

# Check Python location (should show .venv)
which python

# Install a missing package
pip install <package-name>

# Open VS Code in current folder
code .
```

---

## Need Help?

- Check the [Troubleshooting](#9-troubleshooting) section above
- Ask during the workshop!

---

*Enjoy the workshop!*
