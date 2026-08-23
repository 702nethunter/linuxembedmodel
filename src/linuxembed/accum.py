"""Workaround for broken gradient-accumulation normalisation in transformers 4.47.

With `gradient_accumulation_steps > 1`, this version of `Trainer` neither
averages the accumulated loss nor the accumulated gradients. Measured on this
setup, three runs at the same effective batch of 64 gave:

    accum=1   loss 10.478   grad_norm  5.43     <- correct
    accum=4   loss 41.997   grad_norm 23.75     <- 4x too large
    accum=16  (extrapolating) 16x too large

The gradients really are inflated, not just the logged number, so the optimizer
sees `accum x` the intended learning rate. At the schedule this project uses
(accum 16 in pretrain phase 2) that is an 8e-3 effective LR against an intended
5e-4, which does not train.

The cause is the `model_accepts_loss_kwargs` path: `BertForMaskedLM.forward`
accepts `**loss_kwargs`, so `Trainer` passes `num_items_in_batch` and then skips
its own division on the assumption the model normalised the loss itself — which
for this model it does not. Setting `trainer.model_accepts_loss_kwargs = False`
after construction does *not* help (verified: byte-identical losses).

Dividing the loss before `backward()` fixes reporting and gradients together.
After the fix the three configurations agree:

    accum=1   loss 10.478   grad_norm 5.43
    accum=4   loss 10.499   grad_norm 5.94
    accum=16  loss 10.494   grad_norm 5.62

Remove this mixin if transformers is upgraded past the fix, and re-run
`scripts/check_accum.py` to confirm it is no longer needed.
"""

from __future__ import annotations


class NormalizeAccumLossMixin:
    """Divide the loss by gradient_accumulation_steps before backward().

    Mix in *before* the Trainer class so it takes precedence:

        class MyTrainer(NormalizeAccumLossMixin, Trainer): ...

    The division applies to the training path ONLY. `Trainer.prediction_step`
    also routes through `compute_loss`, so dividing unconditionally scales
    `eval_loss` down by the accumulation factor as well -- which reads as the
    model generalising far better than it trains (an eval loss of 0.33 against
    a train loss of 1.34) and is pure artifact. Evaluation does no
    accumulation, so it must not be normalised.
    """

    _in_training_step = False

    def training_step(self, *args, **kwargs):
        self._in_training_step = True
        try:
            return super().training_step(*args, **kwargs)
        finally:
            self._in_training_step = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        n = self.args.gradient_accumulation_steps
        if n == 1 or not self._in_training_step:
            return out
        if return_outputs:
            loss, rest = out
            return loss / n, rest
        return out / n
