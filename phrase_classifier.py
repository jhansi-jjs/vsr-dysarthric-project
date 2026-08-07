"""
Few-shot personalized phrase classification for voice-loss VSR.

Instead of decoding arbitrary open-vocabulary sentences (which needs data
this project doesn't have), this treats each user-enrolled phrase as a
single class. A person enrolls a small personal phrase set (AAC-style --
"I need water", "Call my mother", etc.), records a few examples of each,
and the system recognizes WHICH phrase they just lip-signed.

Two layers:
  1. Prototype matching -- frozen encoder, zero training, cannot collapse.
     Each phrase's prototype = mean embedding of its enrolled examples.
     Recognition = nearest prototype by cosine similarity.
  2. Linear head fine-tune -- optional, trains a small classifier on top
     of the SAME frozen embeddings for a sharper accuracy boost. The
     encoder never updates, so this cannot catastrophically forget either.
"""

from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------
# Embedding extraction (uses your existing pretrained encoder)
# ----------------------------------------------------------------
def extract_embedding(clip: np.ndarray, base_model, device) -> torch.Tensor:
    """
    Run the FROZEN pretrained encoder on a clip and mean-pool over time
    to get a single fixed-length embedding vector.

    clip: np.ndarray shape (75, 50, 100) -- same format your crop_mouth()
          already produces.
    base_model: your existing pretrained LipNet (base.encode must exist).
    """
    base_model.eval()
    with torch.no_grad():
        bx = torch.from_numpy(clip).unsqueeze(0).to(device).float().div_(255.)
        z = base_model.encode(bx)          # (1, T, dim) -- your encoder's features
        emb = z.mean(dim=1).squeeze(0)     # mean-pool over time -> (dim,)
        emb = emb / (emb.norm() + 1e-8)    # L2-normalize for cosine similarity
    return emb.cpu()


# ----------------------------------------------------------------
# Layer 1: Prototype matching (safe default, zero training)
# ----------------------------------------------------------------
class PhraseEnrollment:
    """
    Holds enrolled phrases and their prototype embeddings.
    Add phrases incrementally as the user records examples.
    """

    def __init__(self):
        self.phrases: List[str] = []
        self.examples: Dict[str, List[torch.Tensor]] = {}
        self.prototypes: Dict[str, torch.Tensor] = {}

    def add_example(self, phrase: str, clip: np.ndarray, base_model, device):
        if phrase not in self.examples:
            self.examples[phrase] = []
            self.phrases.append(phrase)
        emb = extract_embedding(clip, base_model, device)
        self.examples[phrase].append(emb)
        self._recompute_prototype(phrase)

    def _recompute_prototype(self, phrase: str):
        stacked = torch.stack(self.examples[phrase])
        proto = stacked.mean(dim=0)
        proto = proto / (proto.norm() + 1e-8)
        self.prototypes[phrase] = proto

    def example_count(self, phrase: str) -> int:
        return len(self.examples.get(phrase, []))

    def remove_phrase(self, phrase: str):
        self.examples.pop(phrase, None)
        self.prototypes.pop(phrase, None)
        if phrase in self.phrases:
            self.phrases.remove(phrase)

    def clear(self):
        self.phrases.clear()
        self.examples.clear()
        self.prototypes.clear()

    def is_ready(self, min_per_phrase: int = 2, min_phrases: int = 2) -> Tuple[bool, str]:
        if len(self.phrases) < min_phrases:
            return False, f"Enroll at least {min_phrases} different phrases."
        under = [p for p in self.phrases if self.example_count(p) < min_per_phrase]
        if under:
            return False, f"Need {min_per_phrase}+ examples each for: {', '.join(under)}"
        return True, ""


def classify_prototype(clip: np.ndarray, enrollment: PhraseEnrollment,
                        base_model, device, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Recognize which enrolled phrase a new clip matches, via nearest
    prototype (cosine similarity). Returns ranked list of
    (phrase, similarity_score), best first.
    """
    query = extract_embedding(clip, base_model, device)
    scores = []
    for phrase, proto in enrollment.prototypes.items():
        sim = torch.dot(query, proto).item()  # both are L2-normalized -> cosine sim
        scores.append((phrase, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ----------------------------------------------------------------
# Layer 2: Linear head fine-tune (optional accuracy boost)
# Encoder stays frozen -- only this small head trains.
# ----------------------------------------------------------------
class PhraseClassifierHead(nn.Module):
    def __init__(self, embedding_dim: int, num_phrases: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_phrases)

    def forward(self, x):
        return self.fc(x)


def train_linear_head(enrollment: PhraseEnrollment, embedding_dim: int,
                       epochs: int = 30, lr: float = 1e-3, on_epoch=None) -> Tuple[nn.Module, List[str]]:
    """
    Trains ONLY a small linear layer on top of frozen embeddings.
    The base encoder is never touched, so this cannot catastrophically
    forget -- worst case, the linear head just underperforms and you
    fall back to prototype matching (Layer 1) for the demo.
    """
    phrase_list = list(enrollment.phrases)
    phrase_to_idx = {p: i for i, p in enumerate(phrase_list)}

    X, y = [], []
    for phrase in phrase_list:
        for emb in enrollment.examples[phrase]:
            X.append(emb)
            y.append(phrase_to_idx[phrase])

    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)

    head = PhraseClassifierHead(embedding_dim, len(phrase_list))
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    head.train()
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = head(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if on_epoch is not None:
            on_epoch(ep + 1, epochs)
    head.eval()

    return head, phrase_list


def classify_linear(clip: np.ndarray, head: nn.Module, phrase_list: List[str],
                     base_model, device, top_k: int = 3) -> List[Tuple[str, float]]:
    query = extract_embedding(clip, base_model, device)
    with torch.no_grad():
        logits = head(query.unsqueeze(0)).squeeze(0)
        probs = torch.softmax(logits, dim=0)
    ranked = sorted(zip(phrase_list, probs.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
