"""The combined GISTEmbed + InfoNCE objective.

InfoNCE (what sentence-transformers calls MultipleNegativesRankingLoss) treats
every other item in the batch as a negative. That assumption is wrong often
enough to hurt: the kernel has many genuinely near-equivalent helpers, so a
batch routinely contains a "negative" that is a perfectly good answer to the
anchor. Pushing it away is a wrong gradient.

GISTEmbed (Solatorio, 2024) fixes that with a frozen *guide* encoder. For anchor
i, the guide scores its true positive; any in-batch candidate the guide likes
*more* than that true positive is presumed a false negative and is masked out of
the softmax rather than treated as a negative.

We keep both terms:

    total = W_GIST * GIST + W_INFONCE * InfoNCE

GIST alone can over-filter — early in training the guide's judgement is noisy,
and masking too aggressively starves the model of negatives. The plain InfoNCE
term keeps full-batch signal flowing as a floor.

Both terms are computed from ONE forward pass over the batch. Calling two
separate sentence-transformers losses would encode every text twice, which does
not fit in 8 GB at a useful batch size.

Guide model note: this repo uses no pretrained weights, so there is no external
guide available. Instead the guide is OUR OWN stage-1 checkpoint, frozen —
self-guided GIST. See train_embed.py.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn


class GISTInfoNCELoss(nn.Module):
    """Weighted sum of guided (GIST) and unguided (InfoNCE) contrastive loss.

    Expects batches of (anchor, positive) or (anchor, positive, negative).
    """

    def __init__(
        self,
        model: SentenceTransformer,
        guide: SentenceTransformer | None = None,
        scale: float = 20.0,
        w_gist: float = 0.6,
        w_infonce: float = 0.4,
    ) -> None:
        super().__init__()
        self.model = model
        self.guide = guide
        self.scale = scale
        self.w_gist = w_gist
        self.w_infonce = w_infonce

        if self.guide is not None:
            # The guide is a fixed reference, never trained.
            self.guide.eval()
            for p in self.guide.parameters():
                p.requires_grad_(False)
            model_vocab = model.tokenizer.get_vocab()
            guide_vocab = self.guide.tokenizer.get_vocab()
            if model_vocab != guide_vocab:
                raise ValueError(
                    "guide and model must share a tokenizer so the same "
                    "tokenized batch can be fed to both; got different vocabs"
                )

    def _embed(self, features: dict[str, Tensor]) -> Tensor:
        return F.normalize(self.model(features)["sentence_embedding"], p=2, dim=-1)

    @torch.no_grad()
    def _guide_embed(self, features: dict[str, Tensor]) -> Tensor:
        return F.normalize(self.guide(features)["sentence_embedding"], p=2, dim=-1)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        features = list(sentence_features)
        embeddings = [self._embed(f) for f in features]
        anchor, positive = embeddings[0], embeddings[1]
        negatives = embeddings[2:]

        # Candidates: every positive in the batch, plus every explicit hard
        # negative. Row i's correct answer is at column i.
        candidates = torch.cat([positive, *negatives], dim=0)
        scores = anchor @ candidates.T * self.scale
        target = torch.arange(scores.size(0), device=scores.device)

        infonce = F.cross_entropy(scores, target)

        if self.guide is None:
            # Stage 1: no guide exists yet, so this is pure InfoNCE.
            return infonce

        guide_anchor = self._guide_embed(features[0])
        guide_cands = torch.cat(
            [self._guide_embed(f) for f in features[1:]], dim=0
        )
        guide_scores = guide_anchor @ guide_cands.T

        # Threshold per anchor = how much the guide likes the TRUE positive.
        batch = anchor.size(0)
        true_pos_sim = guide_scores[torch.arange(batch), torch.arange(batch)].unsqueeze(1)

        # Anything the guide rates above its own true positive is a suspected
        # false negative -> remove it from the softmax entirely.
        mask = guide_scores > true_pos_sim
        mask[torch.arange(batch), torch.arange(batch)] = False  # never mask the answer

        guided_scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        gist = F.cross_entropy(guided_scores, target)

        return self.w_gist * gist + self.w_infonce * infonce

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "w_gist": self.w_gist,
            "w_infonce": self.w_infonce,
            "guided": self.guide is not None,
        }
