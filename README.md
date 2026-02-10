# AI Lip-Sync Video Generator

A complete AI-powered tool that converts text to speech and generates realistic lip-synced videos from static images. Optimized for CPU-only processing and handles videos under 1 minute efficiently.

## 🎯 Features

- **Text-to-Speech (TTS)**: Convert any text to natural-sounding speech using Google TTS
- **Lip-Sync Generation**: Animate static images with synchronized lip movements using Wav2Lip
- **CPU Optimized**: Works without GPU for short videos (<1 minute)
- **Simple CLI**: Easy-to-use command-line interface
- **Multiple Input Formats**: Support for text, text files, or pre-recorded audio
- **Flexible Output**: Generate high-quality MP4 videos

## 📋 Requirements

### System Requirements
- Python 3.8-3.12 (recommended)
- ffmpeg (for video processing)
- 8GB+ RAM recommended
- Works on Windows, Linux, and macOS

### Python Dependencies
All required packages are listed in `requirements.txt`:
- PyTorch (CPU version)
- OpenCV
- NumPy
- Librosa (audio processing)
- gTTS (Google Text-to-Speech)
- pydub
- tqdm (progress bars)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Saurav-kumar077/ai-lipsync-generator.git
cd ai-lipsync-generator
```

### 2. Install System Dependencies

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install ffmpeg python3-pip
```

#### On macOS:
```bash
brew install ffmpeg
```

#### On Windows:
1. Download ffmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Add ffmpeg to your system PATH

### 3. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Wav2Lip Models
```bash
python download_models.py
```

This will download the Wav2Lip model checkpoint (~148MB). Choose option 1 for the standard model, which is recommended for CPU processing.

## 📖 Usage

### Basic Examples

#### 1. Generate from Text
```bash
python generate_lipsync.py --text "Hello, welcome to AI lip-sync generation!" --image inputs/images/person.jpg
```

#### 2. Generate from Text File
```bash
python generate_lipsync.py --text_file examples/univo_transcript.txt --image inputs/images/presenter.jpg
```

#### 3. Generate from Existing Audio
```bash
python generate_lipsync.py --audio inputs/audio/speech.wav --image inputs/images/person.jpg
```

#### 4. Specify Output Location
```bash
python generate_lipsync.py --text "Your text here" --image inputs/images/person.jpg --output outputs/videos/my_video.mp4
```

#### 5. Use Different Language for TTS
```bash
python generate_lipsync.py --text "Bonjour le monde" --image inputs/images/person.jpg --lang fr
```

### Command-Line Options

```
Required Arguments:
  --image IMAGE           Path to input image file (JPG, PNG, BMP)
  
Input Options (choose one):
  --text TEXT            Text to convert to speech
  --text_file FILE       Path to text file containing transcript
  --audio AUDIO          Path to audio file (WAV, MP3) - skip TTS
  
Optional Arguments:
  --output OUTPUT        Output video path (default: auto-generated)
  --model MODEL          Path to Wav2Lip model checkpoint
  --lang LANG            Language code for TTS (default: en)
```

### Supported Languages for TTS
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `hi` - Hindi
- And many more (see [gTTS documentation](https://gtts.readthedocs.io/))

## 📁 Project Structure

```
ai-lipsync-generator/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── generate_lipsync.py      # Main script
├── download_models.py       # Model download helper
├── inputs/
│   ├── images/              # Place your input images here
│   └── audio/               # Optional audio inputs (generated audio saved here)
├── outputs/
│   └── videos/              # Generated videos saved here
├── models/                  # Downloaded model checkpoints
└── examples/
    └── univo_transcript.txt # Example transcript
```

## 🎬 Example: UNIVO Education Demo

An example transcript is provided in `examples/univo_transcript.txt`. To generate a video:

1. Place your presenter image in `inputs/images/` (e.g., `presenter.jpg`)
2. Run:
```bash
python generate_lipsync.py --text_file examples/univo_transcript.txt --image inputs/images/presenter.jpg
```

The script will:
- Convert the transcript to speech (~40 seconds)
- Generate a lip-synced video
- Save output to `outputs/videos/`

**Expected processing time**: 5-15 minutes on CPU

## ⚡ CPU Optimization Tips

### Processing Time Expectations
- **15-second video**: ~3-5 minutes
- **30-second video**: ~5-10 minutes
- **60-second video**: ~10-15 minutes

### Performance Tips

1. **Use High-Quality Images**
   - Resolution: 512x512 to 1024x1024 pixels
   - Format: JPG or PNG
   - Clear, frontal face view
   - Good lighting

2. **Audio Optimization**
   - Keep audio under 60 seconds for best CPU performance
   - Use 16kHz sample rate (handled automatically)
   - Clear speech without background noise

3. **System Optimization**
   - Close unnecessary applications
   - Ensure 8GB+ RAM available
   - Use SSD storage for faster I/O

4. **Model Selection**
   - Use `wav2lip.pth` (standard model) for CPU
   - Avoid `wav2lip_gan.pth` on CPU (too slow)

### Quality vs Speed Tradeoffs
- **Fast**: Smaller images (512x512), standard model
- **Balanced**: Medium images (720x720), standard model ✅ (Recommended)
- **Quality**: Larger images (1024x1024), more processing time

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "ffmpeg not found"
**Problem**: ffmpeg is not installed or not in PATH

**Solution**:
- **Linux**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org) and add to PATH

#### 2. "Model checkpoint not found"
**Problem**: Wav2Lip model not downloaded

**Solution**:
```bash
python download_models.py
```

#### 3. "No module named 'torch'"
**Problem**: PyTorch not installed

**Solution**:
```bash
pip install -r requirements.txt
```

#### 4. "CUDA not available" (expected)
**Note**: This is normal! The project is designed for CPU.

#### 5. "Memory Error"
**Problem**: Insufficient RAM

**Solution**:
- Close other applications
- Use shorter audio (<30 seconds)
- Use smaller images

#### 6. "No face detected"
**Problem**: Face detection failed

**Solution**:
- Use clear, frontal face images
- Ensure good lighting in image
- Try a different image with a more visible face

#### 7. Model Download Fails
**Problem**: Automatic download interrupted

**Solution**:
1. Visit [Wav2Lip GitHub](https://github.com/Rudrabha/Wav2Lip)
2. Download `wav2lip.pth` manually from releases
3. Place in `models/` directory

#### 8. Audio File Format Not Supported
**Problem**: Unsupported audio format

**Solution**:
- Convert to WAV or MP3
- Use online converter or: `ffmpeg -i input.m4a output.wav`

## 🎨 Best Practices

### Image Selection
✅ **Good**:
- Clear, high-resolution photos
- Frontal face view
- Neutral expression
- Good lighting
- Minimal background

❌ **Avoid**:
- Blurry or low-resolution images
- Side profiles
- Sunglasses covering eyes
- Heavy shadows on face

### Text/Script Preparation
✅ **Good**:
- Clear, concise sentences
- Proper punctuation
- Natural speaking rhythm
- Under 200 words for <1 minute

❌ **Avoid**:
- Very long paragraphs without breaks
- Special characters that affect TTS
- Overly technical jargon

## 🔒 Privacy & Security

- All processing is done **locally** on your machine
- No data is sent to external servers (except TTS via Google)
- Generated audio and videos stay on your system
- Models are downloaded once and used offline

## 📝 Technical Details

### Wav2Lip Model
- **Purpose**: Lip-sync generation
- **Architecture**: Deep learning model for audio-to-visual synthesis
- **Size**: ~148MB
- **Framework**: PyTorch

### Audio Processing
- **TTS Engine**: Google Text-to-Speech (gTTS)
- **Sample Rate**: 16kHz (optimal for Wav2Lip)
- **Format**: MP3 for TTS output, WAV for processing

### Video Generation
- **FPS**: 25 frames per second
- **Codec**: H.264 (MP4)
- **Resolution**: Matches input image

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project uses the Wav2Lip model, which is licensed under its respective license. Please refer to the [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip) for model licensing details.

## 🙏 Acknowledgments

- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) - Original lip-sync model
- [gTTS](https://github.com/pndurette/gTTS) - Text-to-Speech library
- UNIVO Education Pvt. Ltd. - Example transcript

## 📧 Support

For issues and questions:
1. Check the Troubleshooting section above
2. Search existing [GitHub Issues](https://github.com/Saurav-kumar077/ai-lipsync-generator/issues)
3. Open a new issue with details about your problem

## 🚦 Quick Start Checklist

- [ ] Install Python 3.8-3.12
- [ ] Install ffmpeg
- [ ] Clone this repository
- [ ] Run `pip install -r requirements.txt`
- [ ] Run `python download_models.py`
- [ ] Place your image in `inputs/images/`
- [ ] Run `python generate_lipsync.py --text "Test" --image inputs/images/your_image.jpg`
- [ ] Check `outputs/videos/` for your generated video!

---

**Note**: First-time setup takes ~5-10 minutes. Video generation takes 5-15 minutes depending on audio length.

Happy lip-syncing! 🎥✨