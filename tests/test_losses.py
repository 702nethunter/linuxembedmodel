#!/usr/bin/env python3
"""Tests for the GIST + InfoNCE objective.

The masking rule is the whole point of the loss, so it is tested directly
against hand-constructed embeddings rather than only smoke-tested end to end.

    python tests/test_losses.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from linuxembed.losses import GISTInfoNCELoss


class FakeEncoder(torch.nn.Module):
    """Returns a fixed embedding per feature dict, so we control the geometry."""

    def __init__(self, table: dict[str, torch.Tensor], vocab: dict[str, int]):
        super().__init__()
        self.table = table
        self.tokenizer = type("Tok", (), {"get_vocab": lambda self_: vocab})()

    def forward(self, features):
        return {"sentence_embedding": self.table[features["key"]]}


def make_loss(student_emb, guide_emb, **kw):
    vocab = {"a": 0}
    student = FakeEncoder(student_emb, vocab)
    guide = FakeEncoder(guide_emb, vocab) if guide_emb is not None else None
    return GISTInfoNCELoss(model=student, guide=guide, **kw)


def test_infonce_only_when_no_guide():
    """With guide=None the loss must be exactly plain InfoNCE."""
    torch.manual_seed(0)
    a = F.normalize(torch.randn(4, 8), dim=-1)
    p = F.normalize(torch.randn(4, 8), dim=-1)
    loss = make_loss({"anchor": a, "pos": p}, None, scale=20.0)
    got = loss([{"key": "anchor"}, {"key": "pos"}], labels=None)

    scores = a @ p.T * 20.0
    want = F.cross_entropy(scores, torch.arange(4))
    assert torch.allclose(got, want, atol=1e-6), f"{got} != {want}"
    print("  PASS  guide=None reduces to plain InfoNCE")


def test_gist_masks_false_negative():
    """A candidate the guide prefers over the true positive must be masked out.

    Anchor 0's true positive is p0, but p1 is deliberately placed closer to a0.
    Under plain InfoNCE p1 is a (wrong) negative; under GIST the guide should
    recognise it and drop it from the softmax, lowering the loss for row 0.
    """
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # p0 is 45 degrees off a0; p1 sits right on a0 -> guide rates it higher.
    p = torch.tensor([[0.7071, 0.7071], [1.0, 0.0]])
    a, p = F.normalize(a, dim=-1), F.normalize(p, dim=-1)

    guided = make_loss({"anchor": a, "pos": p}, {"anchor": a, "pos": p},
                       scale=20.0, w_gist=1.0, w_infonce=0.0)
    plain = make_loss({"anchor": a, "pos": p}, None, scale=20.0)

    feats = [{"key": "anchor"}, {"key": "pos"}]
    g = guided(feats, labels=None)
    n = plain(feats, labels=None)
    assert g < n, f"GIST ({g}) should be below InfoNCE ({n}) when a false negative is masked"
    print(f"  PASS  GIST masks the false negative (GIST {g:.4f} < InfoNCE {n:.4f})")


def test_true_positive_never_masked():
    """Even if the guide is degenerate, the diagonal must survive.

    A guide that scores everything identically makes `guide_scores > threshold`
    false everywhere on the diagonal only by luck; the loss must force it.
    """
    a = F.normalize(torch.randn(6, 8), dim=-1)
    p = F.normalize(torch.randn(6, 8), dim=-1)
    same = torch.ones(6, 8)  # degenerate guide: every pair scores identically
    loss = make_loss({"anchor": a, "pos": p}, {"anchor": same, "pos": same},
                     scale=20.0, w_gist=1.0, w_infonce=0.0)
    out = loss([{"key": "anchor"}, {"key": "pos"}], labels=None)
    assert torch.isfinite(out), "masking the true positive would give inf/nan"
    print("  PASS  true positive is never masked (loss stays finite)")


def test_weights_are_respected():
    """total must equal w_gist*GIST + w_infonce*InfoNCE."""
    torch.manual_seed(1)
    a = F.normalize(torch.randn(5, 8), dim=-1)
    p = F.normalize(torch.randn(5, 8), dim=-1)
    ge = {"anchor": F.normalize(torch.randn(5, 8), dim=-1),
          "pos": F.normalize(torch.randn(5, 8), dim=-1)}
    feats = [{"key": "anchor"}, {"key": "pos"}]

    only_gist = make_loss({"anchor": a, "pos": p}, ge, w_gist=1.0, w_infonce=0.0)(feats, None)
    only_nce = make_loss({"anchor": a, "pos": p}, ge, w_gist=0.0, w_infonce=1.0)(feats, None)
    mixed = make_loss({"anchor": a, "pos": p}, ge, w_gist=0.6, w_infonce=0.4)(feats, None)

    want = 0.6 * only_gist + 0.4 * only_nce
    assert torch.allclose(mixed, want, atol=1e-6), f"{mixed} != {want}"
    print("  PASS  weighted sum matches its components")


def test_hard_negatives_extend_candidates():
    """An explicit negative column must widen the candidate set, not replace it."""
    torch.manual_seed(2)
    a = F.normalize(torch.randn(3, 8), dim=-1)
    p = F.normalize(torch.randn(3, 8), dim=-1)
    n = F.normalize(torch.randn(3, 8), dim=-1)
    loss = make_loss({"anchor": a, "pos": p, "neg": n}, None, scale=20.0)
    got = loss([{"key": "anchor"}, {"key": "pos"}, {"key": "neg"}], labels=None)

    scores = a @ torch.cat([p, n]).T * 20.0
    assert scores.shape == (3, 6), scores.shape
    want = F.cross_entropy(scores, torch.arange(3))
    assert torch.allclose(got, want, atol=1e-6)
    print("  PASS  hard negatives extend the candidate set to 2N")


def test_guide_tokenizer_mismatch_rejected():
    a = F.normalize(torch.randn(2, 4), dim=-1)
    student = FakeEncoder({"anchor": a}, {"a": 0})
    guide = FakeEncoder({"anchor": a}, {"b": 1})  # different vocab
    try:
        GISTInfoNCELoss(model=student, guide=guide)
    except ValueError:
        print("  PASS  mismatched guide tokenizer is rejected")
        return
    raise AssertionError("expected ValueError for mismatched tokenizers")


if __name__ == "__main__":
    for fn in [
        test_infonce_only_when_no_guide,
        test_gist_masks_false_negative,
        test_true_positive_never_masked,
        test_weights_are_respected,
        test_hard_negatives_extend_candidates,
        test_guide_tokenizer_mismatch_rejected,
    ]:
        fn()
    print("\n  all loss tests passed")
