import torch
import json
from transformers import ClapProcessor, ClapModel


class SemanticMapper:
    def __init__(
        self, vocab_path, model_name="laion/clap-htsat-fused", device=None
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.flat_vocab = []
        for category, terms in self.vocab.items():
            for term in terms:
                self.flat_vocab.append({"term": term, "category": category})

        self.text_embeddings = self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        texts = [item["term"] for item in self.flat_vocab]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.text_embeds.cpu()

    def find_nearest_tags(self, audio_embedding, top_k=8):
        audio_tensor = torch.tensor(
            audio_embedding, device=self.device
        ).unsqueeze(0)
        similarities = torch.nn.functional.cosine_similarity(
            audio_tensor, self.text_embeddings.to(self.device), dim=1
        )
        top_k_indices = torch.topk(similarities, top_k).indices.cpu().numpy()

        results = []
        for idx in top_k_indices:
            item = self.flat_vocab[idx]
            results.append(
                {
                    "term": item["term"],
                    "category": item["category"],
                    "confidence": float(similarities[idx].cpu().numpy()),
                }
            )
        return results
