import torch
import librosa
import numpy as np
from transformers import ClapModel, ClapProcessor


class AudioExtractor:
    def __init__(self, model_name="laion/clap-htsat-fused", device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def load_audio(self, audio_path, target_sr=48000):
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio, sr

    def get_clap_embedding(self, audio_path):
        audio, sr = self.load_audio(audio_path)
        inputs = self.processor(
            audios=[audio], sampling_rate=sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.audio_embeds.squeeze().cpu().numpy()

    def get_acoustic_features(self, audio_path, sr=22050):
        y, _ = librosa.load(audio_path, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        energy = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=y, sr=sr)
        )
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        chroma_mean = np.mean(
            librosa.feature.chroma_stft(y=y, sr=sr), axis=1
        ).tolist()

        return {
            "tempo_bpm": float(np.atleast_1d(tempo)[0]),
            "energy": float(energy),
            "spectral_centroid_hz": float(spectral_centroid),
            "zero_crossing_rate": float(zcr),
            "chroma_mean": chroma_mean,
        }
