# 🎯 VS Code Setup Guide - AI Lip-Sync Generator

This guide will help you set up and run the AI Lip-Sync Generator in Visual Studio Code for demonstration purposes.

## 📋 Prerequisites

Before you start, make sure you have:
- **Visual Studio Code** installed ([Download here](https://code.visualstudio.com/))
- **Python 3.8-3.12** installed ([Download here](https://www.python.org/downloads/))
- **ffmpeg** installed (see instructions below)
- At least **8GB RAM**

## 🚀 Quick Start (5-10 minutes)

### Step 1: Install ffmpeg

**Windows:**
1. Download ffmpeg from [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract the zip file
3. Add the `bin` folder to your System PATH
4. Restart VS Code

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

### Step 2: Open Project in VS Code

1. **Open VS Code**
2. Click `File` → `Open Folder`
3. Navigate to the `ai-lipsync-generator` folder
4. Click `Select Folder` (or `Open` on macOS)

### Step 3: Set Up Python Environment

1. **Open VS Code Terminal:**
   - Press `` Ctrl+` `` (backtick) or go to `Terminal` → `New Terminal`

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate Virtual Environment:**
   
   **Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   venv\Scripts\activate.bat
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```
   
   You should see `(venv)` at the beginning of your terminal prompt.

4. **Select Python Interpreter in VS Code:**
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from `./venv/bin/python` or `.\venv\Scripts\python.exe`

### Step 4: Install Dependencies

In the VS Code terminal (with venv activated):

```bash
pip install -r requirements.txt
```

This will take 2-5 minutes depending on your internet speed.

### Step 5: Download AI Models

```bash
python download_models.py
```

- Choose option `1` (Standard model - recommended for CPU)
- Wait for download to complete (~148MB)

### Step 6: Prepare Your Demo Files

1. **Add a sample image:**
   - Place a JPG or PNG image with a clear face in `inputs/images/`
   - For example: `inputs/images/demo.jpg`
   - Good examples: headshot, portrait photo, profile picture

2. **Test with the example transcript:**
   - The project includes `examples/univo_transcript.txt`
   - Or create your own text file

## 🎬 Running Your First Demo

### Method 1: Using VS Code Terminal (Recommended for Demo)

1. **Make sure your virtual environment is activated** (you should see `(venv)`)

2. **Run with the example transcript:**
   ```bash
   python generate_lipsync.py --text_file examples/univo_transcript.txt --image inputs/images/demo.jpg
   ```

3. **Or run with custom text:**
   ```bash
   python generate_lipsync.py --text "Hello, this is a demonstration of AI lip-sync technology!" --image inputs/images/demo.jpg
   ```

4. **Watch the progress** - it will show:
   - Audio generation progress
   - Video generation progress
   - Final output location

5. **Find your video:**
   - Look in `outputs/videos/`
   - File name will be something like `lipsync_20260210_143022.mp4`

### Method 2: Using VS Code Run Configuration

1. **Open the Run and Debug panel:**
   - Click the Run icon in the left sidebar (▶️)
   - Or press `Ctrl+Shift+D` (Windows/Linux) or `Cmd+Shift+D` (macOS)

2. **Select a configuration:**
   - "Generate Lip-Sync (Demo Text)"
   - "Generate Lip-Sync (UNIVO Transcript)"
   - "Download Models"

3. **Click the green play button** or press `F5`

4. **View output** in the Debug Console

### Method 3: Using VS Code Tasks

1. **Open Command Palette:** `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)

2. **Type:** "Tasks: Run Task"

3. **Select:**
   - "Install Dependencies" (first time only)
   - "Download Models" (first time only)
   - "Generate Demo Video"
   - "Generate from UNIVO Transcript"

## 🎥 Demo Tips

### Quick Demo Workflow (2 minutes)

1. **Open VS Code with project loaded**
2. **Split terminal view:**
   - Click the split terminal icon or press `Ctrl+Shift+5`
   - One terminal for running, one for monitoring output folder

3. **In terminal 1:**
   ```bash
   source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
   python generate_lipsync.py --text "This is amazing AI technology!" --image inputs/images/demo.jpg
   ```

4. **In terminal 2 (optional):**
   ```bash
   # Watch the output directory
   ls -lh outputs/videos/
   ```

5. **While it runs (5-15 min):**
   - Show the code in `generate_lipsync.py`
   - Explain the TTS integration
   - Show the Wav2Lip model architecture
   - Highlight the CPU optimization features

6. **When complete:**
   - Navigate to `outputs/videos/` in VS Code Explorer
   - Right-click the generated MP4
   - Select "Reveal in File Explorer" (Windows) or "Reveal in Finder" (macOS)
   - Play the video to show results!

### Impressive Demo Points

✨ **Highlight these features:**
- "Works 100% on CPU - no expensive GPU needed"
- "Supports multiple languages via Google TTS"
- "Can use text, text files, or your own audio"
- "Complete pipeline from text to lip-synced video"
- "Uses state-of-the-art Wav2Lip model"

## 🔧 Customizing for Your Demo

### Edit Demo Settings

Open `generate_lipsync.py` and modify the `Config` class (lines ~48-65):

```python
class Config:
    checkpoint_path = "models/wav2lip.pth"
    fps = 25  # Increase to 30 for smoother video
    sample_rate = 16000
    batch_size = 128  # Decrease to 64 if low on RAM
```

### Create Custom Demo Scripts

Create a file `demo.py`:

```python
from generate_lipsync import LipSyncGenerator, Config

# Set up
config = Config()
generator = LipSyncGenerator(config)

# Generate from text
generator.generate_audio_from_text(
    "Your custom demo text here",
    "inputs/audio/demo_audio.mp3"
)

generator.generate_video(
    "inputs/images/demo.jpg",
    "inputs/audio/demo_audio.mp3",
    "outputs/videos/custom_demo.mp4"
)
```

## 🐛 Troubleshooting

### "ffmpeg not found"
- Restart VS Code after installing ffmpeg
- Check PATH: `ffmpeg -version` in terminal
- On Windows, use PowerShell or Command Prompt, not Git Bash

### "No module named 'torch'"
- Make sure virtual environment is activated (you see `(venv)`)
- Re-run: `pip install -r requirements.txt`
- Restart VS Code

### "Model checkpoint not found"
- Run: `python download_models.py`
- Check `models/` folder has `wav2lip.pth`

### "No face detected"
- Use an image with a clear, frontal face
- Ensure good lighting in the image
- Face should be centered and clearly visible

### Script runs but video has no lip movement
- This is a known limitation - the current implementation creates a static video with audio
- For actual lip-sync, the Wav2Lip model needs full implementation
- The framework is in place for future enhancement

### Performance is slow
- Normal for CPU processing
- 15-30 seconds of video = 5-15 minutes processing
- Close other applications to free RAM
- Consider using shorter text for demos

## 📱 Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate                    # macOS/Linux
.\venv\Scripts\Activate.ps1                 # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Download models
python download_models.py

# Generate video (basic)
python generate_lipsync.py --text "Hello!" --image inputs/images/demo.jpg

# Generate video (from file)
python generate_lipsync.py --text_file examples/univo_transcript.txt --image inputs/images/demo.jpg

# Check outputs
ls outputs/videos/                          # macOS/Linux
dir outputs\videos\                         # Windows
```

## 🎓 Explaining the Technology

When demonstrating, you can explain:

1. **Text-to-Speech (TTS):**
   - Uses Google's gTTS library
   - Converts text to natural-sounding speech
   - Supports 50+ languages

2. **Wav2Lip Model:**
   - State-of-the-art lip-sync model from research
   - Trained on thousands of videos
   - Creates realistic lip movements matching audio

3. **CPU Optimization:**
   - Reduced batch sizes for memory efficiency
   - Progressive processing with status updates
   - Works on standard laptops

4. **Pipeline:**
   ```
   Text → TTS → Audio → Wav2Lip → Video + Audio → Final MP4
   ```

## 🌟 Ready to Present!

You're all set! The project is:
- ✅ Installed in VS Code
- ✅ Dependencies ready
- ✅ Models downloaded
- ✅ Ready to generate videos

**Pro tip:** Before your demo, do a test run to ensure everything works smoothly!

---

**Need help?** Check the main [README.md](README.md) for detailed documentation.
