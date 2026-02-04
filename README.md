# Google Agent Development Kit (ADK) with Ollama

A multi-agent system for Cardio-Kidney-Metabolic (CKM) condition assessment using Google's Agent Development Kit (ADK) and Ollama's **qwen2.5:14b** model. This project demonstrates a sophisticated multi-agent pattern where specialist agents (cardiologist, nephrologist, and diabetologist) work in parallel, followed by a mediator agent that synthesizes their recommendations.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Windows](#windows)
- [Ollama Setup](#ollama-setup)
- [Project Setup](#project-setup)
- [Verification](#verification)
- [Running the Project](#running-the-project)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## Overview

This project implements a CKM (Cardio-Kidney-Metabolic) Syndrome multi-agent consultation pattern with a **streamlined clinical intake experience**:

### New UX Features

- **Welcome Flow** with two intake modes:
  - **Guided Intake (recommended)**: 3–5 decision-critical questions per turn
  - **Paste Mode**: Paste full case (free text or JSON) for automatic structuring

- **Decision-First Branching**: For peri-operative cases, asks procedure details before CKM essentials

- **Consultation Snapshot Output (≤250 words)**:
  - A) One-line problem
  - B) 5 key facts
  - C) 5 key risks
  - D) Decisions needed today
  - E) Next steps with owner + timing

- **Expandable Details**: Reply A (medication table), B (specialty rationale), C (citations)

- **De-duplication**: No repeated summaries across specialties

- **Missing Data Flags**: Explicit statements like "HF phenotype unclear; EF not provided"

### Core Architecture

1. **Intake Agent** handles user interaction and case collection
2. **Three specialist agents run in parallel**:
   - Cardiologist (HFrEF/HFpEF management, ESC 2023/AHA 2024 guidelines)
   - Nephrologist (CKD management, KDIGO 2024 guidelines)
   - Diabetologist (Diabetes management, ADA 2024 guidelines)
3. **Mediator agent** synthesizes recommendations into Consultation Snapshot format
4. **Root agent** coordinates the flow and handles expansion requests

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or higher** (Python 3.11+ strongly recommended)
- **pip** (Python package installer)
- **Ollama** (for running local LLM models)
- **Git** (for cloning the repository)

## Installation Guide

### macOS

#### Step 1: Install Python

1. **Check if Python is installed:**
   ```bash
   python3 --version
   ```

2. **If Python is not installed**, install it using Homebrew:
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install Python
   brew install python3
   ```

3. **Verify installation:**
   ```bash
   python3 --version
   python3 -m pip --version
   ```

#### Step 2: Install Ollama

1. **Download and install Ollama:**
   ```bash
   # Using Homebrew (recommended)
   brew install ollama
   
   # OR download from [https://ollama.com/download/mac](https://ollama.com/download/mac)
   ```

2. **Start Ollama service:**
   ```bash
   ollama serve
   ```
   
   **Note:** If you receive an error like `Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address`, it means the Ollama software is already running in the background. You can skip this step and proceed.

3. **Verify Ollama is running** (in a new terminal):
   ```bash
   curl http://localhost:11434/api/tags
   ```

#### Step 3: Clone & Set Up Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fpesce81/Google-CKM-Agent-Ollama.git
   ```

2. **Navigate to project directory:**
   ```bash
   cd Google-CKM-Agent-Ollama
   ```

3. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

4. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

5. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

6. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `google-adk[all]>=1.0.0`
   - `litellm>=1.0.0`
   - `requests>=2.31.0`

### Linux

#### Step 1: Install Python

1. **Update package manager:**
   ```bash
   # For Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   
   # For Fedora/RHEL/CentOS
   sudo dnf install python3 python3-pip
   
   # For Arch Linux
   sudo pacman -S python python-pip
   ```

2. **Verify installation:**
   ```bash
   python3 --version
   python3 -m pip --version
   ```

#### Step 2: Install Ollama

1. **Install Ollama:**
   ```bash
   # Using the official installer
   curl -fsSL https://ollama.com/install.sh | sh
   
   # OR download from [https://ollama.com/download/linux](https://ollama.com/download/linux)
   ```

2. **Start Ollama service:**
   ```bash
   ollama serve
   ```
   
   **Note:** If you receive an error like `Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address`, it means the Ollama software is already running in the background. You can skip this step and proceed.

3. **Verify Ollama is running** (in a new terminal):
   ```bash
   curl http://localhost:11434/api/tags
   ```

#### Step 3: Clone & Set Up Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fpesce81/Google-CKM-Agent-Ollama.git
   ```

2. **Navigate to project directory:**
   ```bash
   cd Google-CKM-Agent-Ollama
   ```

3. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

4. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

5. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

6. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Windows

#### Step 1: Install Python

1. **Download Python:**
   - Visit https://www.python.org/downloads/
   - Download Python 3.10 or higher (3.11+ recommended)
   - **Important:** Check "Add Python to PATH" during installation

2. **Verify installation:**
   ```powershell
   python --version
   python -m pip --version
   ```

   If the commands don't work, you may need to restart your terminal or add Python to your PATH manually.

#### Step 2: Install Ollama

1. **Download Ollama:**
   - Visit https://ollama.com/download/windows
   - Download and run the Windows installer
   - Follow the installation wizard

2. **Start Ollama:**
   - Ollama should start automatically as a Windows service.
   - If not, open Command Prompt or PowerShell and run:
     ```powershell
     ollama serve
     ```
   
   **Note:** If you receive an error like `Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address`, it means the Ollama software is already running in the background (check your system tray). You can skip this step and proceed.

3. **Verify Ollama is running** (in a new terminal):
   ```powershell
   curl http://localhost:11434/api/tags
   ```
   
   If `curl` is not available, use PowerShell's `Invoke-WebRequest`:
   ```powershell
   Invoke-WebRequest -Uri http://localhost:11434/api/tags
   ```

#### Step 3: Clone & Set Up Project

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/fpesce81/Google-CKM-Agent-Ollama.git
   ```

2. **Open Command Prompt or PowerShell** and navigate to project directory:
   ```powershell
   cd Google-CKM-Agent-Ollama
   ```

3. **Create a virtual environment:**
   ```powershell
   python -m venv venv
   ```

4. **Activate the virtual environment:**
   ```powershell
   # Command Prompt
   venv\Scripts\activate.bat
   
   # PowerShell
   venv\Scripts\Activate.ps1
   ```
   
   If you get an execution policy error in PowerShell, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

5. **Upgrade pip:**
   ```powershell
   python -m pip install --upgrade pip
   ```

6. **Install project dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Ollama Setup

### Pull the Required Model

After Ollama is installed and running, you need to download the `qwen2.5:14b` model:

```bash
# macOS/Linux
ollama pull qwen2.5:14b

# Windows (Command Prompt or PowerShell)
ollama pull qwen2.5:14b
```

This will download the model (approximately 8-10 GB). The download time depends on your internet connection.

### Verify Model Installation

```bash
# List all installed models
ollama list

# Test the model
ollama run qwen2.5:14b "Hello, how are you?"
```

### Using Other Ollama Models

This project is configured and tested to use **`qwen2.5:14b`** by default, but you can use other Ollama models. Here are some popular alternatives:

**Popular Ollama Models:**
- `qwen2.5:32b` - More capable, requires more RAM (approx 20GB+)
- `llama3.1:8b` - Balanced performance and speed
- `mistral:7b` - Fast and efficient
- `gemma2:9b` - Google's Gemma model

**To use a different model:**

1. **Pull the desired model:**
   ```bash
   ollama pull llama3.2:3b
   ```

2. **Update the model in your code:**
   
   Edit `src/agent.py`, `src/mediator.py`, and `src/specialists.py` to change the model name:
   
   ```python
   # Change from:
   model=LiteLlm(model="ollama_chat/qwen2.5:14b")
   
   # To (example):
   model=LiteLlm(model="ollama_chat/llama3.2:3b")
   ```
   
   You'll need to update the model in:
   - `src/agent.py` (root_agent)
   - `src/mediator.py` (mediator_agent)
   - `src/specialists.py` (all three specialist agents)

3. **Verify the model works:**
   ```bash
   ollama run <model-name> "Test message"
   ```

**For additional model options and configuration details**, refer to the [official Google ADK documentation](https://github.com/google/adk-python) and [Ollama model library](https://ollama.com/library).

### Configure Ollama API Base (Optional)

By default, Ollama runs on `http://localhost:11434`. If you need to use a different address or port, set the environment variable:

**macOS/Linux:**
```bash
export OLLAMA_API_BASE=http://localhost:11434
```

**Windows (Command Prompt):**
```cmd
set OLLAMA_API_BASE=http://localhost:11434
```

**Windows (PowerShell):**
```powershell
$env:OLLAMA_API_BASE="http://localhost:11434"
```

To make this permanent, add it to your shell profile:
- **macOS/Linux:** Add to `~/.bashrc`, `~/.zshrc`, or `~/.profile`
- **Windows:** Add via System Properties > Environment Variables

## Project Setup

### Verify Installation

Ensure you are inside the `Google-CKM-Agent-Ollama` directory before running the verification.

Run the verification script to check that everything is set up correctly:

```bash
# macOS/Linux
python verify_setup.py

# Windows
python verify_setup.py
```

This script checks:
- ✅ Python dependencies are installed
- ✅ Ollama server is accessible
- ✅ Required model is available
- ✅ Agent configuration files exist

### Expected Output

If everything is set up correctly, you should see:

```
============================================================
ADK Ollama Demo - Setup Verification
============================================================

1. Checking Python dependencies...
   ✓ All dependencies installed

2. Checking environment variables...
   ⚠ OLLAMA_API_BASE not set (using default: http://localhost:11434)

3. Checking Ollama server connection...
   ✓ Ollama server is running
   Available models: qwen2.5:14b

4. Checking model availability...
   ✓ Model found: exact match

5. Checking agent configuration...
   ✓ Agent file found: src/agent.py

============================================================
✓ All checks passed! You're ready to use the agent.

Try running:
   adk web         # Start web interface
   adk run .       # Start CLI interface
============================================================
```

## Running the Project

### Method 1: ADK Web Interface (Recommended)

The ADK web interface provides an interactive UI for testing and debugging your agents.

1. **Ensure Ollama is running:**
   ```bash
   # In a separate terminal
   ollama serve
   ```
   *(Skip if already running in system tray)*

2. **Navigate to the project directory** (if not already there):
   ```bash
   cd Google-CKM-Agent-Ollama
   ```

3. **Activate your virtual environment** (if not already active):
   ```bash
   # macOS/Linux
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

4. **Start the ADK web interface:**
   ```bash
   adk web .
   ```

5. **Access the web interface:**
   - Open your browser and navigate to: **http://localhost:8000/dev-ui/?app=src**
   - **IMPORTANT:** The conversation will not start automatically. You must type a greeting like **"Hi"**, **"Start"**, or **"Good morning"** in the chat box to trigger the Welcome Message.
   
   You should see:
   > Welcome to the Cardio-Kidney-Metabolic (CKM) Syndrome Multi-Specialist Consultation portal.
   > I help clinicians prepare and synthesize complex CKM cases involving the interplay of heart failure, chronic kidney disease, and metabolic conditions (diabetes, obesity). Recommendations follow current guidelines from cardiology (ESC/AHA), nephrology (KDIGO), and endocrinology (ADA).
   > Choose your intake mode:
   > 1. Guided intake (recommended) — I'll ask 3–5 high-yield questions step by step
   > 2. Paste mode — Paste the full case (free text or JSON) and I'll structure it
   > Reply 1 or 2 to begin.

6. **To Stop the Server:**
   - Go back to your terminal window.
   - Press **Ctrl + C** to stop the process.

**⚠️ CRITICAL NOTE: ONE CASE PER SESSION**
To ensure patient safety and data integrity, you **MUST start a New Session**  before entering a new patient case.

The agents maintain context ("memory") during a conversation. Entering a new patient in an active session will cause **data contamination** (e.g., mixing up age, sex, or medications from the previous patient).

### Method 2: ADK CLI Interface

For command-line interaction:

1. **Ensure Ollama is running** (in a separate terminal)

2. **Navigate to the project folder and activate your virtual environment**

3. **Run the agent:**
   ```bash
   adk run .
   ```

   This will start an interactive CLI session where you can input patient cases.

## Usage Examples

### Example 1: Basic Patient Case

Once the web interface is running, you can input a patient case like:

```
65-year-old male with:
- Type 2 diabetes (HbA1c 8.2%)
- Chronic kidney disease Stage 3a (eGFR 58 mL/min/1.73m², creatinine 1.4 mg/dL)
- Heart failure with reduced ejection fraction (EF 35%, BNP 450 pg/mL)
- Current medications: metformin 1000mg BID, lisinopril 10mg daily, metoprolol 50mg BID
- BP: 142/88 mmHg, HR: 78 bpm
- Chief complaint: Increasing fatigue and shortness of breath over past 2 weeks
```

The system will:
1. Process the case through the root agent
2. Delegate to three specialist agents (running in parallel)
3. Synthesize recommendations through the mediator agent
4. Return a comprehensive treatment plan

### Example 2: Structured Case

You can also provide structured free-form text:

```
72-year-old female patient

Demographics:
- Age: 72 years
- Sex: Female
- Weight: 85 kg
- Height: 165 cm

Medical History:
- Type 2 diabetes, diagnosed 2015
- Essential hypertension
- CKD Stage 3b

Vital Signs:
- Blood pressure: 138/82 mmHg
- Heart rate: 82 bpm

Laboratory Results:
- HbA1c: 7.8%
- eGFR: 45 mL/min/1.73m²
- Creatinine: 1.6 mg/dL
- Glucose: 165 mg/dL
- Ejection fraction: 42%
- NT-proBNP: 320 pg/mL

Current Medications:
- Metformin 500mg BID
- Glipizide 5mg daily
- Losartan 50mg daily

Chief Complaint:
Progressive lower extremity edema and weight gain of 5kg over 1 month
```

See `examples.md` for more detailed examples.

## Troubleshooting

### Issue: "Command 'ollama' not found"

**Solution:**
- Ensure Ollama is installed and added to your PATH
- On macOS/Linux, restart your terminal after installation
- On Windows, restart your computer or manually add Ollama to PATH

### Issue: "Cannot connect to Ollama server"

**Solutions:**
1. **Check if Ollama is running:**
   ```bash
   # macOS/Linux
   curl http://localhost:11434/api/tags
   
   # Windows
   Invoke-WebRequest -Uri http://localhost:11434/api/tags
   ```

2. **Start Ollama if not running:**
   ```bash
   ollama serve
   ```
   *Note: If you get a "bind: Only one usage" error, Ollama is already running.*

3. **Check if port 11434 is in use:**
   ```bash
   # macOS/Linux
   lsof -i :11434
   
   # Windows
   netstat -ano | findstr :11434
   ```

### Issue: "Model 'qwen2.5:14b' not found"

**Solution:**
```bash
ollama pull qwen2.5:14b
```

Wait for the download to complete (this may take several minutes depending on your internet speed).

### Issue: "ModuleNotFoundError: No module named 'google.adk'"

**Solutions:**
1. **Ensure virtual environment is activated:**
   ```bash
   # macOS/Linux
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Upgrade pip and retry:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Issue: "adk: command not found"

**Solution:**
The `adk` command should be available after installing `google-adk[all]`. If it's not found:

1. **Ensure virtual environment is activated**

2. **Reinstall with the 'all' extra:**
   ```bash
   pip install --upgrade "google-adk[all]"
   ```

3. **Verify installation:**
   ```bash
   pip show google-adk
   adk --version
   ```

### Issue: Port 8000 already in use (ADK Web)

**Solution:**
1. **Find the process using port 8000:**
   ```bash
   # macOS/Linux
   lsof -i :8000
   
   # Windows
   netstat -ano | findstr :8000
   ```

2. **Kill the process** or use a different port (if ADK supports it)

3. **Or wait** for the previous instance to finish

### Issue: Slow model responses

**Possible causes and solutions:**
1. **Model is still downloading** - Wait for download to complete
2. **Insufficient system resources** - Close other applications
3. **Large model size** - Consider using a smaller model for testing
4. **Network issues** (if using remote Ollama) - Check network connection

### Issue: Virtual environment activation fails (Windows PowerShell)

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:
```powershell
venv\Scripts\Activate.ps1
```

## Project Structure

```
Google-Agent-Ollama/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── examples.md              # Usage examples
├── verify_setup.py          # Setup verification script
└── src/
    ├── __init__.py
    ├── agent.py             # Root agent and orchestration
    ├── mediator.py          # Mediator agent
    ├── specialists.py       # Specialist agents (cardiologist, nephrologist, diabetologist)
    └── utils.py             # Utility functions
```

## Additional Resources

- [Google ADK Documentation](https://github.com/google/adk-python)
- [Ollama Documentation](https://ollama.com/docs)
- [LiteLLM Documentation](https://docs.litellm.ai/)

## Support

If you encounter issues not covered in this guide:

1. Run the verification script: `python verify_setup.py`
2. Check the [Troubleshooting](#troubleshooting) section
3. Review the project's example cases in `examples.md`
4. Consult the ADK and Ollama documentation

## License

This project is provided as-is for demonstration purposes.

---

**Note:** This project uses the `qwen2.5:14b` model via Ollama by default. Ensure you have sufficient disk space (approximately 8-10 GB) and system resources to run the model effectively. You can use other Ollama models by updating the model configuration in the source files (see [Using Other Ollama Models](#using-other-ollama-models) section). For additional model options and advanced configuration, refer to the [official Google ADK documentation](https://github.com/google/adk-python).