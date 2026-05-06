import os
import json
import logging
from datetime import datetime
from .extraction import AudioExtractor
from .mapping import SemanticMapper
from .prompt_gen import PromptOptimizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MusicPromptPipeline:
    def __init__(
        self,
        vocab_path="data/vocabulary.json",
        model_name="laion/clap-htsat-fused",
        device=None,
    ):
        logger.info("Initializing models (first run downloads ~1.5GB)...")
        self.extractor = AudioExtractor(model_name=model_name, device=device)
        self.mapper = SemanticMapper(
            vocab_path=vocab_path, model_name=model_name, device=device
        )
        self.prompt_opt = PromptOptimizer()
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def analyze_track(self, audio_path):
        logger.info(f"Extracting features from: {audio_path}")
        clap_embedding = self.extractor.get_clap_embedding(audio_path)
        acoustic_features = self.extractor.get_acoustic_features(audio_path)
        tags = self.mapper.find_nearest_tags(clap_embedding)
        return {
            "embedding": clap_embedding.tolist(),
            "acoustic_features": acoustic_features,
            "tags": tags,
        }

    def generate_prompt(self, analysis, custom_additions=None):
        return self.prompt_opt.format_suno_prompt(
            analysis["acoustic_features"], analysis["tags"], custom_additions
        )

    def run(self, audio_path, custom_additions=None):
        analysis = self.analyze_track(audio_path)
        prompt = self.generate_prompt(analysis, custom_additions)
        self._log_run(audio_path, analysis, prompt)
        return prompt

    def _log_run(self, audio_path, analysis, prompt):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_file": audio_path,
            "acoustic_features": analysis["acoustic_features"],
            "top_tags": analysis["tags"][:5],
            "generated_prompt": prompt,
        }
        log_path = os.path.join(
            self.log_dir,
            f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)
        logger.info(f"Transformation log saved to {log_path}")
