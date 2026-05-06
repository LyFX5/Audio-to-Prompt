#!/usr/bin/env python3
"""
Root-level entry point. Run from project root:
python run.py samples/your_track.wav
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import MusicPromptPipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_audio.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    pipeline = MusicPromptPipeline()
    prompt = pipeline.run(
        audio_file, custom_additions=["high fidelity", "mastered"]
    )
    print("\n✅ Ready for Suno/Udio:")
    print(prompt)
