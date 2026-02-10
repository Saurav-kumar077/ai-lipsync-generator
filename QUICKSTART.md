# 🚀 QUICK START GUIDE

**Want to demo this in VS Code in 5 minutes?** Follow these steps!

## ⚡ Super Quick Setup (For Demonstration)

### 1️⃣ Prerequisites (2 min)
- Install [VS Code](https://code.visualstudio.com/)
- Install [Python 3.8-3.12](https://www.python.org/downloads/)
- Install ffmpeg:
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html), add to PATH
  - **Mac**: `brew install ffmpeg`
  - **Linux**: `sudo apt-get install ffmpeg`

### 2️⃣ Open in VS Code (1 min)
1. Open VS Code
2. `File` → `Open Folder` → Select `ai-lipsync-generator` folder
3. Click "Yes" to install recommended extensions

### 3️⃣ Setup Environment (2 min)
Open VS Code terminal (`` Ctrl+` ``) and run:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_models.py
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python download_models.py
```

When prompted, choose option **1** (Standard model).

### 4️⃣ Add a Demo Image (30 sec)
- Put a JPG/PNG image with a face in `inputs/images/`
- Name it `demo.jpg` (or update the commands below)

### 5️⃣ Run Your First Demo! (5-15 min processing time)

**Option A: Quick Text Demo**
```bash
python generate_lipsync.py --text "Hello! This is AI-powered lip-sync technology!" --image inputs/images/demo.jpg
```

**Option B: UNIVO Transcript Demo**
```bash
python generate_lipsync.py --text_file examples/univo_transcript.txt --image inputs/images/demo.jpg
```

### 6️⃣ View Your Video! 
- Find it in `outputs/videos/`
- Right-click → "Reveal in Explorer/Finder"
- Play the MP4 video!

---

## 🎯 Alternative: Use VS Code GUI

1. **Press** `Ctrl+Shift+D` (Run & Debug panel)
2. **Select** "Generate Lip-Sync (Demo Text)"
3. **Press** `F5` or click ▶️ green play button
4. **Wait** for processing to complete
5. **Find** video in `outputs/videos/` folder

---

## 📚 Need More Details?

- **Full VS Code Guide**: [VSCODE_SETUP.md](VSCODE_SETUP.md)
- **Complete Documentation**: [README.md](README.md)

## 🆘 Troubleshooting

**"ffmpeg not found"**
→ Restart VS Code after installing ffmpeg

**"No module named 'torch'"**
→ Make sure virtual environment is activated (you see `(venv)` in terminal)

**"Model checkpoint not found"**
→ Run `python download_models.py`

**"No face detected"**
→ Use an image with a clear, centered face

---

**Ready to impress!** 🌟 You now have a working AI lip-sync generator in VS Code.
