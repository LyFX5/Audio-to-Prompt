import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.pipeline import MusicPromptPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/quickstart.py <audio_file.wav>")
        return

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"❌ Error: File {audio_path} not found.")
        return

    pipeline = MusicPromptPipeline()
    prompt = pipeline.run(
        audio_path, custom_additions=["minor key", "female vocals"]
    )
    print(f"\n🎵 Generated Prompt:\n{prompt}\n")


if __name__ == "__main__":
    main()
