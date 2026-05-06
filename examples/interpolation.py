import sys
import os
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.extraction import AudioExtractor
from src.mapping import SemanticMapper


def slerp(v0, v1, t):
    """Spherical linear interpolation between two normalized vectors"""
    dot = np.dot(v0, v1)
    if dot > 0.9995:
        return v0 + t * (v1 - v0)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (v0 * np.sin((1 - t) * theta) + v1 * np.sin(t * theta)) / np.sin(
        theta
    )


def interpolate_prompts(track_a_path, track_b_path, blend_ratio=0.5):
    extractor = AudioExtractor()
    mapper = SemanticMapper("data/vocabulary.json")

    print("🔊 Extracting embeddings...")
    emb_a = extractor.get_clap_embedding(track_a_path)
    emb_b = extractor.get_clap_embedding(track_b_path)

    print(f"🔄 Blending embeddings (ratio: {blend_ratio})...")
    blended_emb = slerp(emb_a, emb_b, blend_ratio)

    tags_a = mapper.find_nearest_tags(emb_a, top_k=4)
    tags_b = mapper.find_nearest_tags(emb_b, top_k=4)
    tags_blend = mapper.find_nearest_tags(blended_emb, top_k=6)

    print("\n📊 Results:")
    print(f"Track A: {[t['term'] for t in tags_a]}")
    print(f"Track B: {[t['term'] for t in tags_b]}")
    print(f"Blended: {[t['term'] for t in tags_blend]}")


if __name__ == "__main__":
    interpolate_prompts(
        "samples/ref_a.wav", "samples/ref_b.wav", blend_ratio=0.5
    )
