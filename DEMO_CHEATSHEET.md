# 📋 VS Code Demo Cheat Sheet

Quick reference for demonstrating the AI Lip-Sync Generator in VS Code.

## 🎬 Demo Commands (Copy & Paste)

### Quick Setup
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_models.py

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python download_models.py
```

### Quick Demo (Short Text)
```bash
python generate_lipsync.py --text "Hello! This is AI-powered lip-sync technology in action!" --image inputs/images/demo.jpg
```

### UNIVO Transcript Demo (Longer)
```bash
python generate_lipsync.py --text_file examples/univo_transcript.txt --image inputs/images/demo.jpg
```

### Custom Audio Demo
```bash
python generate_lipsync.py --audio inputs/audio/speech.wav --image inputs/images/demo.jpg
```

## ⌨️ VS Code Shortcuts

| Action | Shortcut |
|--------|----------|
| Open Terminal | `` Ctrl+` `` |
| Command Palette | `Ctrl+Shift+P` |
| Run & Debug Panel | `Ctrl+Shift+D` |
| Run Selected Config | `F5` |
| Split Terminal | `Ctrl+Shift+5` |
| Toggle Sidebar | `Ctrl+B` |

## 🎯 One-Click Run (Press F5)

1. Press `Ctrl+Shift+D` (Run & Debug)
2. Select from dropdown:
   - ▶️ "Generate Lip-Sync (Demo Text)"
   - ▶️ "Generate Lip-Sync (UNIVO Transcript)"
   - ▶️ "Generate from Custom Audio"
3. Press `F5` or click green play button

## 🔧 Quick Tasks (Ctrl+Shift+P → "Tasks")

- Install Dependencies
- Download Models  
- Generate Demo Video
- Generate from UNIVO Transcript
- Open Output Folder
- Check Python Version
- Check ffmpeg Installation

## 💬 Demo Talking Points

**While Video Generates (5-15 min):**

✅ "This uses Google's Text-to-Speech to create natural audio"
✅ "The Wav2Lip model is trained on thousands of videos"
✅ "It works 100% on CPU - no GPU required"
✅ "Supports 50+ languages"
✅ "Can handle text, files, or custom audio"
✅ "Perfect for creating talking avatars, presentations, or educational content"

**Code Walkthrough:**
- Show `generate_lipsync.py` - main pipeline
- Show `wav2lip_model.py` - neural network architecture
- Show `requirements.txt` - all open-source libraries
- Show `examples/univo_transcript.txt` - example input

## 🎥 Where to Find Output

```
outputs/videos/lipsync_YYYYMMDD_HHMMSS.mp4
```

Right-click in VS Code → "Reveal in Explorer/Finder"

## 🐛 Quick Fixes

| Problem | Solution |
|---------|----------|
| ffmpeg not found | Restart VS Code |
| No module 'torch' | Activate venv: `source venv/bin/activate` |
| Model not found | Run `python download_models.py` |
| No face detected | Use frontal face image |

## 📱 Platform-Specific

### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
python generate_lipsync.py --text "Demo" --image inputs/images/demo.jpg
dir outputs\videos\
```

### Mac/Linux
```bash
source venv/bin/activate
python generate_lipsync.py --text "Demo" --image inputs/images/demo.jpg
ls -lh outputs/videos/
```

## ⏱️ Processing Times

- 15 sec audio = ~3-5 min processing
- 30 sec audio = ~5-10 min processing  
- 60 sec audio = ~10-15 min processing

**Demo Tip:** Use short text (15-30 sec) for quick demos!

## 🌟 Before Demo Checklist

- [ ] ffmpeg installed and working
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Models downloaded
- [ ] Demo image added (inputs/images/demo.jpg)
- [ ] Test run completed successfully
- [ ] Output video plays correctly

---

**Ready to impress!** 🚀
