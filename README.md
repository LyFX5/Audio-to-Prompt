# Audio-to-Prompt

# 🎵 Audio-to-Prompt Pipeline

> A developer-centric framework for reverse-engineering musical style from reference tracks and generating optimized prompts for AI music generators (Suno, Udio, etc.). Built for creators who think in systems, not sheet music.
> 

[Python](https://img.shields.io/badge/Python-3.9+-blue)

[License](https://img.shields.io/badge/License-MIT-green)

[Status](https://img.shields.io/badge/Status-Alpha-orange)

---

## 📦 Overview

This project treats music generation as a **signal processing → semantic mapping → control signal** pipeline. It extracts high-level stylistic features from any reference audio, maps them to a controlled vocabulary of genres/instruments/techniques, and outputs structured prompts optimized for modern text-to-audio models.

**Ideal for:**

- Developers building creative tools or audio pipelines
- Researchers exploring semantic audio embeddings
- Producers using AI generation who want reproducible, transparent workflows
- Anyone bridging engineering and creative expression without traditional music theory

---

## 🏗️ Architecture

```
[Reference Audio]
       ↓
[Feature Extraction] → CLAP Embedding (512-dim) + Librosa Acoustic Metrics
       ↓
[Semantic Mapping]   → Cosine Similarity → Nearest Tag Lookup
       ↓
[Prompt Synthesis]   → Rule-Based Formatting + Style Optimization
       ↓
[AI Generator]       → Suno / Udio / Custom API → [Output Track]
```

---

## ✨ Key Features

| Component | Description |
| --- | --- |
| 🔍 **CLAP Integration** | Maps audio & text into a shared embedding space for style inference |
| 📊 **Acoustic Metrics** | Tempo, energy, spectral centroid, zero-crossing rate, chroma vectors |
| 🏷️ **Controlled Vocabulary** | Extensible JSON-based tag library (genres, instruments, production terms) |
| 🤖 **Prompt Optimizer** | Outputs Suno/Udio-ready comma-separated descriptors under 200 chars |
| 📜 **Audit Logging** | Full trace: `input → features → tags → prompt → output` |
| 🧩 **Modular API** | Drop-in classes for extraction, mapping, and generation |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone <https://github.com/yourusername/audio-to-prompt.git>
cd audio-to-prompt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**`requirements.txt`**

```
torch>=2.1.0
transformers>=4.35.0
laion-clap>=1.1.4
librosa>=0.10.1
numpy>=1.24.0
soundfile>=0.12.1
```

### 2. Download CLAP Checkpoint

```bash
# Music-optimized CLAP (required for first run)
mkdir -p ckpts
wget -P ckpts/ <https://huggingface.co/lukewys/laion_clap_htsat_fusion/resolve/main/music_audioset_epoch_15_esc_90.14.pt>
```

### 3. Run the Pipeline

```python
from pipeline import MusicPromptPipeline

# Initialize (GPU recommended, CPU works for batch=1)
pipeline = MusicPromptPipeline(ckpt_path="ckpts/music_audioset_epoch_15_esc_90.14.pt")

# Analyze reference track
analysis = pipeline.analyze_track("samples/reference.wav")

# Generate optimized prompt
prompt = pipeline.generate_suno_prompt(
    analysis,
    custom_additions=["female vocals", "minor key", "tape saturation"]
)
print(f"🎵 Generated Prompt: {prompt}")
```

**Expected Output:**

```
🎵 Generated Prompt: lo-fi hip hop, 92bpm, chill, analog synth, reverb-heavy, tape saturation, female vocals, minor key
```

---

## 📖 Pipeline Details

### Stage 1: Feature Extraction

- **CLAP**: Encodes 48kHz audio into a 512-dimensional semantic vector aligned with text embeddings
- **Librosa**: Computes traditional MIR features for rhythm/energy grounding
- Outputs are cached to avoid recomputation on repeated runs

### Stage 2: Semantic Mapping

- Loads a curated vocabulary (JSON) of ~50-200 terms
- Computes cosine similarity between audio embedding and precomputed text embeddings
- Returns ranked tags with confidence scores

### Stage 3: Prompt Synthesis

- Applies domain-specific formatting rules:
    - Limits to ≤6 components for optimal model parsing
    - Orders: `genre → tempo → mood → instruments → techniques → custom`
    - Filters redundant or conflicting descriptors
- Outputs ready-to-paste strings for Suno/Udio web interfaces or APIs

---

## 🛠️ Configuration & Extensibility

### Custom Vocabulary

Edit `data/vocabulary.json` to add/remove tags:

```json
{
  "genres": ["ambient techno", "neoclassical", "synthwave"],
  "instruments": ["modular bass", "felt piano", "granular pads"],
  "techniques": ["sidechain compression", "bitcrushing", "stereo widening"]
}
```

### Advanced Usage

```python
# Interpolate between two tracks
prompt_a = pipeline.generate_suno_prompt(pipeline.analyze_track("track_a.wav"))
prompt_b = pipeline.generate_suno_prompt(pipeline.analyze_track("track_b.wav"))

# Blend tags manually or via embedding slerp (see examples/interpolation.ipynb)
```

---

## 📝 Prompt Engineering Guidelines

Suno/Udio models respond best to **structured, unambiguous descriptors**. Follow these rules:

| Component | Format | Example |
| --- | --- | --- |
| **Genre** | Primary + optional fusion | `cinematic orchestral`, `dubstep` |
| **Tempo** | `NNN bpm` | `128 bpm` |
| **Mood** | Single adjective | `tense`, `euphoric`, `melancholic` |
| **Instruments** | Concrete nouns | `808 kick`, `analog pad`, `acoustic guitar` |
| **Production** | Technique keywords | `sidechain`, `tape saturation`, `wide stereo` |
| **Structure** *(opt)* | Brief hint | `builds to drop`, `loop-based`, `verse-chorus` |

⚠️ **Avoid**: Contradictory terms (`fast tempo, ambient`), overly long strings, or subjective phrasing (`sounds like my favorite song`).

---

## ⚖️ IP & Licensing Notes

| Component | License | Notes |
| --- | --- | --- |
| **This Code** | MIT | Free to use, modify, distribute |
| **CLAP Weights** | MIT / CC0 (check checkpoint) | Download from LAION/HuggingFace |
| **Extracted Features** | Transformation | Not derivative of source audio |
| **Generated Prompts** | User IP | Your creative output |
| **Suno/Udio Outputs** | Platform ToS | Commercial rights depend on subscription tier |

✅ **Best Practice**: Log input hash → features → prompt → output ID. This creates a reproducible audit trail showing your contribution is in *system design and prompt engineering*, not raw audio sampling.

---

## 📁 Project Structure

```
├── src/
│   ├── extraction.py      # CLAP + Librosa wrappers
│   ├── mapping.py         # Tag similarity & vocabulary loader
│   ├── prompt_gen.py      # Formatting rules & optimizer
│   └── pipeline.py        # Main orchestrator class
├── data/
│   └── vocabulary.json    # Curated tag library
├── ckpts/                 # Model weights (gitignored)
├── examples/
│   ├── quickstart.ipynb
│   └── interpolation.ipynb
├── logs/                  # Auto-generated transformation traces
├── requirements.txt
└── README.md
```

---

## 🐛 Troubleshooting

| Issue | Solution |
| --- | --- |
| `CUDA out of memory` | Set `device="cpu"` or reduce batch size |
| `librosa load error` | Convert audio to 16/24-bit WAV or MP3 |
| `CLAP checkpoint missing` | Run download command or point `ckpt_path` correctly |
| `Empty tag output` | Expand `vocabulary.json` with domain-specific terms |

---

## 🤝 Contributing

1. Fork & create a feature branch
2. Add tests for new extraction/mapping logic
3. Update `data/vocabulary.json` with clear sourcing
4. Submit PR with prompt examples before/after

Please follow [Conventional Commits](https://www.conventionalcommits.org/) and keep audio samples <5MB.

---

## 📜 License

This project is licensed under the **MIT License** – see `LICENSE` for details.

Third-party model weights are governed by their respective licenses.

---

## 🙏 Acknowledgments

- [LAION-CLAP](https://github.com/LAION-AI/CLAP) – Semantic audio-text embeddings
- [Librosa](https://librosa.org/) – Python audio analysis
- [Suno](https://suno.com/) & [Udio](https://udio.com/) – AI music generation platforms
- Open-source MIR community for reproducible audio research

---

> 💡 **Note**: This tool is a creative inference engine, not a replacement for musical craftsmanship. Treat it as a system interface for exploring sonic ideas with precision and reproducibility.
> 

**Star ⭐ this repo if you're building at the intersection of code and creativity.**