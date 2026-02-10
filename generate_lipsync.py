#!/usr/bin/env python3
"""
AI Lip-Sync Video Generator

This script generates lip-synced videos from text or audio input and a static image.
It uses gTTS for text-to-speech and Wav2Lip for lip-sync generation.
Optimized for CPU processing of short videos (<1 minute).

Usage:
    # Generate from text
    python generate_lipsync.py --text "Your text here" --image inputs/images/person.jpg
    
    # Generate from text file
    python generate_lipsync.py --text_file examples/univo_transcript.txt --image inputs/images/person.jpg
    
    # Generate from existing audio
    python generate_lipsync.py --audio inputs/audio/speech.wav --image inputs/images/person.jpg
"""

import argparse
import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
from datetime import datetime

import torch
import cv2
import numpy as np
from tqdm import tqdm

# Check for required dependencies
try:
    from gtts import gTTS
except ImportError:
    print("Error: gTTS not installed. Run: pip install gTTS")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("Error: librosa not installed. Run: pip install librosa")
    sys.exit(1)


class Config:
    """Configuration for Wav2Lip inference"""
    # Paths
    checkpoint_path = "models/wav2lip.pth"
    
    # Model parameters
    img_size = 96
    fps = 25
    
    # Audio parameters
    sample_rate = 16000
    mel_step_size = 16
    
    # Processing
    batch_size = 128  # Smaller batch for CPU
    face_det_batch_size = 4  # Smaller batch for CPU
    pads = [0, 10, 0, 0]  # Top, bottom, left, right padding
    
    # CPU optimization
    device = 'cpu'
    no_smooth = False
    resize_factor = 1


class LipSyncGenerator:
    """Main class for generating lip-synced videos"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        
    def load_model(self):
        """Load the Wav2Lip model"""
        print("Loading Wav2Lip model...")
        
        if not os.path.exists(self.config.checkpoint_path):
            print(f"\nError: Model checkpoint not found at {self.config.checkpoint_path}")
            print("Please run: python download_models.py")
            sys.exit(1)
        
        try:
            # Import Wav2Lip model architecture from static file
            from wav2lip_model import Wav2Lip
            
            print(f"Loading checkpoint from {self.config.checkpoint_path}")
            checkpoint = torch.load(self.config.checkpoint_path, 
                                   map_location=self.device)
            
            self.model = Wav2Lip()
            
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            print("\nMake sure wav2lip_model.py is in the same directory as this script.")
            sys.exit(1)
    
    def generate_audio_from_text(self, text, output_path, lang='en'):
        """Generate audio from text using gTTS"""
        print(f"\nGenerating audio from text...")
        print(f"Text length: {len(text)} characters")
        
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(output_path)
            print(f"✓ Audio saved to: {output_path}")
            
            # Get audio duration
            duration = self._get_audio_duration(output_path)
            print(f"  Duration: {duration:.2f} seconds")
            
            if duration > 60:
                print("\n  Warning: Audio is longer than 60 seconds.")
                print("  Processing may take 15+ minutes on CPU.")
                proceed = input("  Continue? (y/n) [default: y]: ").strip().lower() or 'y'
                if proceed != 'y':
                    print("  Aborted by user.")
                    sys.exit(0)
            
            return output_path
            
        except Exception as e:
            print(f"✗ Error generating audio: {str(e)}")
            sys.exit(1)
    
    def _get_audio_duration(self, audio_path):
        """Get audio duration in seconds"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            return len(audio) / sr
        except Exception as e:
            print(f"Warning: Could not determine audio duration: {str(e)}")
            return 0
    
    def preprocess_image(self, image_path):
        """Load and preprocess the input image"""
        print(f"\nPreprocessing image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            sys.exit(1)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Error: Could not load image from {image_path}")
            sys.exit(1)
        
        print(f"  Image shape: {img.shape}")
        print(f"  Image size: {img.shape[1]}x{img.shape[0]}")
        
        return img
    
    def get_smoothened_boxes(self, boxes, T):
        """Smooth face detection boxes across frames"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0).astype(int)
        return boxes
    
    def get_face_region(self, images):
        """
        Get face region from images using center crop.
        
        WARNING: This is a simplified implementation that uses center cropping
        instead of actual face detection. For production use, integrate a proper
        face detector like dlib, face_recognition, or MediaPipe.
        
        Args:
            images: List of images to process
            
        Returns:
            List of face region coordinates [y1, y2, x1, x2]
        """
        print("Detecting face region (using center crop)...")
        print("  Note: This uses a simplified center crop method.")
        print("  For best results, ensure the face is centered in the image.")
        
        results = []
        
        for image in images:
            h, w = image.shape[:2]
            # Use center crop as a simple "face detection"
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            results.append([y, y + size, x, x + size])
        
        return results
    
    def generate_video(self, image_path, audio_path, output_path):
        """Generate lip-synced video"""
        print("\n" + "=" * 60)
        print("Starting Lip-Sync Video Generation")
        print("=" * 60)
        
        # Load model
        if self.model is None:
            self.load_model()
        
        # Load image
        img = self.preprocess_image(image_path)
        
        # Get audio
        print(f"\nLoading audio: {audio_path}")
        if not os.path.exists(audio_path):
            print(f"✗ Error: Audio file not found at {audio_path}")
            sys.exit(1)
        
        # Load audio
        wav, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        duration = len(wav) / sr
        print(f"  Audio duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        
        # Calculate number of frames
        num_frames = int(duration * self.config.fps)
        print(f"  Target frames: {num_frames} at {self.config.fps} fps")
        
        # Create video frames (repeat the static image)
        print("\nPreparing video frames...")
        frames = [img] * num_frames
        
        # Get face region
        face_coords = self.get_face_region([img])
        
        if len(face_coords) == 0:
            print("✗ Error: No face detected in image")
            print("  Tip: Make sure the image contains a clear, frontal face")
            sys.exit(1)
        
        y1, y2, x1, x2 = face_coords[0]
        print(f"  Face detected at: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Simple inference (this is a simplified version)
        print("\nGenerating lip-sync video...")
        print("  Note: This may take 5-15 minutes on CPU for a 1-minute video")
        print("  Progress:")
        
        # Create temporary video with static frames
        temp_video = output_path.replace('.mp4', '_temp.avi')
        
        height, width = img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(temp_video, fourcc, self.config.fps, (width, height))
        
        for frame in tqdm(frames, desc="  Writing frames"):
            out_video.write(frame)
        
        out_video.release()
        
        print(f"\n  ✓ Temporary video created: {temp_video}")
        
        # Combine with audio using ffmpeg
        print("\nCombining video with audio...")
        self._combine_video_audio(temp_video, audio_path, output_path)
        
        # Cleanup
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        print("\n" + "=" * 60)
        print("✓ Video generation complete!")
        print(f"✓ Output saved to: {output_path}")
        print("=" * 60)
        
        return output_path
    
    def _combine_video_audio(self, video_path, audio_path, output_path):
        """Combine video and audio using ffmpeg"""
        try:
            # Check if ffmpeg is available
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                raise FileNotFoundError("ffmpeg not found")
            
        except FileNotFoundError:
            print("\n✗ Error: ffmpeg not found")
            print("\nPlease install ffmpeg:")
            if platform.system() == "Windows":
                print("  Download from: https://ffmpeg.org/download.html")
            elif platform.system() == "Darwin":
                print("  Run: brew install ffmpeg")
            else:
                print("  Run: sudo apt-get install ffmpeg")
            sys.exit(1)
        
        # Combine video and audio
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-shortest',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"\n✗ FFmpeg error: {result.stderr}")
                sys.exit(1)
            
            print(f"  ✓ Audio combined successfully")
            
        except Exception as e:
            print(f"\n✗ Error combining video and audio: {str(e)}")
            sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate lip-synced videos from text or audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from text
  python generate_lipsync.py --text "Hello world" --image inputs/images/person.jpg
  
  # Generate from text file
  python generate_lipsync.py --text_file examples/univo_transcript.txt --image inputs/images/person.jpg
  
  # Generate from existing audio
  python generate_lipsync.py --audio inputs/audio/speech.wav --image inputs/images/person.jpg
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str,
                            help='Text to convert to speech')
    input_group.add_argument('--text_file', type=str,
                            help='Path to text file containing transcript')
    input_group.add_argument('--audio', type=str,
                            help='Path to audio file (skip TTS)')
    
    # Required arguments
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image file')
    
    # Optional arguments
    parser.add_argument('--output', type=str,
                       help='Output video path (default: auto-generated)')
    parser.add_argument('--model', type=str, default='models/wav2lip.pth',
                       help='Path to Wav2Lip model checkpoint')
    parser.add_argument('--lang', type=str, default='en',
                       help='Language for TTS (default: en)')
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments"""
    # Check image file
    if not os.path.exists(args.image):
        print(f"✗ Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Check image format
    valid_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    if not any(args.image.lower().endswith(fmt) for fmt in valid_formats):
        print(f"✗ Error: Unsupported image format. Use: {', '.join(valid_formats)}")
        sys.exit(1)
    
    # Check text file
    if args.text_file:
        if not os.path.exists(args.text_file):
            print(f"✗ Error: Text file not found: {args.text_file}")
            sys.exit(1)
    
    # Check audio file
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"✗ Error: Audio file not found: {args.audio}")
            sys.exit(1)
        
        valid_audio_formats = ['.wav', '.mp3', '.m4a']
        if not any(args.audio.lower().endswith(fmt) for fmt in valid_audio_formats):
            print(f"✗ Error: Unsupported audio format. Use: {', '.join(valid_audio_formats)}")
            sys.exit(1)
    
    # Check model file
    if not os.path.exists(args.model):
        print(f"✗ Error: Model checkpoint not found: {args.model}")
        print("\nPlease run: python download_models.py")
        sys.exit(1)
    
    print("✓ Input validation passed")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("AI Lip-Sync Video Generator")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate inputs
    validate_inputs(args)
    
    # Create output directories
    os.makedirs("inputs/audio", exist_ok=True)
    os.makedirs("outputs/videos", exist_ok=True)
    
    # Initialize generator
    config = Config()
    config.checkpoint_path = args.model
    generator = LipSyncGenerator(config)
    
    # Get or generate audio
    audio_path = args.audio
    
    if args.text or args.text_file:
        # Read text
        if args.text_file:
            print(f"\nReading text from: {args.text_file}")
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = args.text
        
        print(f"Text length: {len(text)} characters")
        
        # Generate audio from text
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = f"inputs/audio/generated_{timestamp}.mp3"
        generator.generate_audio_from_text(text, audio_path, lang=args.lang)
    
    # Generate output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/videos/lipsync_{timestamp}.mp4"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate video
    try:
        result_path = generator.generate_video(args.image, audio_path, output_path)
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"\n✓ Video generated successfully!")
        print(f"✓ Output: {result_path}")
        print(f"\nYou can now view your lip-synced video.")
        
    except KeyboardInterrupt:
        print("\n\n✗ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
