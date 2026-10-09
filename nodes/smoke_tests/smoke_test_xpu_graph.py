"""XPUGraphStepRunner on CPU: math parity, grad-buffer discipline, refusals.

Capture itself is XPU-only, so on CPU every step is the eager-live
fallback -- which is exactly what these checks want: the fallback must
train on the identical expression the LossPhase uses (same helpers, but
that is a shared import, not a shared result -- verify the numbers), and
the buffer discipline the replay path depends on (in-place zeroing, never
reassigning .grad) must hold on every path, not just under capture.
"""

import sys
import os
sys.path.insert(0, os.environ.get("REPO", "."))
sys.path.insert(0, os.path.join(os.environ.get("REPO", "."), "nodes", "smoke_tests"))

import torch

from nodes.train.loss import UniformLossWeighting, masked_per_sample_mse
from nodes.train.xpu_graph_step import XPUGraphStepRunner


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


class _TinyModel(torch.nn.Module):
    """forward(xc, t, ctx, y) with one trainable matrix, so .grad exists
    after backward and the runner has something to zero in place."""

    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.proj = torch.nn.Conv2d(4, 4, 1)
        self.drop = torch.nn.Dropout(dropout_p) if dropout_p > 0 else None

    def forward(self, xc, t, ctx, y):
        out = self.proj(xc)
        if self.drop is not None:
            out = self.drop(out)
        return out + 0.01 * ctx.mean() + 0.01 * y.float().mean()

    def trainable_parameters(self):
        return [self.proj.weight, self.proj.bias]


def _inputs(batch=2, h=8, w=8, masked=True):
    xc = torch.randn(batch, 4, h, w)
    t = torch.tensor([10.0, 500.0][:batch])
    ctx = torch.randn(batch, 77, 4)
    y = torch.randn(batch, 3)
    target = torch.randn(batch, 4, h, w)
    sigma = torch.tensor([0.1, 1.0][:batch])
    mask = None
    if masked:
        # (B, C, H, W) like the loader emits (vm.shape == x_t.shape) --
        # a (B, 1, H, W) mask would expand to stride-0 and is not what
        # either path receives.
        mask = torch.ones(batch, 4, h, w)
        mask[:, :, h // 2:, :] = 0.0
    return xc, t, ctx, y, target, sigma, mask


def check_eager_live_matches_the_loss_expression():
    print("[CPU fallback trains on the identical expression: masked MSE, "
          "uniform weighting]")
    torch.manual_seed(0)
    model = _TinyModel()
    runner = XPUGraphStepRunner(model, UniformLossWeighting(), "cpu")
    xc, t, ctx, y, target, sigma, mask = _inputs()
    loss, per_sample, how = runner.step(
        micro=0, xc=xc, t=t, ctx_emb=ctx, y=y, target=target,
        sigma=sigma, mask=mask)
    check(how == "eager-live", f"on CPU every step must fall back, got {how}")
    with torch.no_grad():
        pred = model.forward(xc, t, ctx, y)
        want_per = masked_per_sample_mse(pred, target, mask)
        want_loss = want_per.mean()  # uniform weighting
    check(torch.allclose(loss, want_loss, atol=1e-6),
          f"runner loss {loss.item()} != direct {want_loss.item()}")
    check(torch.allclose(per_sample, want_per, atol=1e-6),
          "per-sample vector differs from the direct expression")
    check(model.proj.weight.grad is not None
          and model.proj.weight.grad.abs().sum() > 0,
          "backward did not populate .grad")
    # Unmasked: plain mean, same as the maskless LossPhase branch.
    m1 = _inputs(1, 8, 8, masked=False)
    loss2, _, how2 = runner.step(
        micro=0, xc=m1[0], t=m1[1], ctx_emb=m1[2],
        y=m1[3], target=m1[4], sigma=m1[5], mask=None)
    with torch.no_grad():
        pred2 = model.forward(m1[0], m1[1], m1[2], m1[3])
        want2 = ((pred2.float() - m1[4].float()).pow(2)
                 .view(1, -1).mean(dim=1)).mean()
    check(how2 == "eager-live" and abs(loss2.item() - want2.item()) < 1e-6,
          "maskless fallback differs from the plain-mean expression")
    print("    PASS")


def check_zeroing_is_in_place():
    print("[zero_window keeps .grad objects (replay writes to captured "
          "addresses; reassigning would orphan them)]")
    torch.manual_seed(1)
    model = _TinyModel()
    runner = XPUGraphStepRunner(model, UniformLossWeighting(), "cpu")
    xc, t, ctx, y, target, sigma, mask = _inputs()
    runner.step(micro=0, xc=xc, t=t, ctx_emb=ctx, y=y, target=target,
                sigma=sigma, mask=mask)
    before = model.proj.weight.grad
    check(before is not None, "no grad to zero")
    runner.zero_window()
    check(model.proj.weight.grad is before,
          "zero_window reassigned .grad -- a replay would write past it")
    check(bool((before == 0).all()),
          "zero_window did not zero in place")
    print("    PASS")


def check_backward_scale_is_baked():
    print("[grad_accum scale: grads at 0.5 are half the grads at 1.0]")
    def grads(scale):
        torch.manual_seed(2)
        model = _TinyModel()
        runner = XPUGraphStepRunner(model, UniformLossWeighting(), "cpu",
                                    backward_scale=scale)
        xc, t, ctx, y, target, sigma, mask = _inputs()
        runner.step(micro=0, xc=xc, t=t, ctx_emb=ctx, y=y, target=target,
                    sigma=sigma, mask=mask)
        return model.proj.weight.grad.clone()
    g1, ghalf = grads(1.0), grads(0.5)
    check(torch.allclose(ghalf, 0.5 * g1, atol=1e-6),
          "backward_scale is not applied to the captured expression")
    print("    PASS")


def check_dropout_is_refused():
    print("[any active Dropout: loud refusal, never a frozen RNG]")
    model = _TinyModel(dropout_p=0.1)
    runner = XPUGraphStepRunner(model, UniformLossWeighting(), "cpu")
    try:
        runner.refuse_if_unsupported()
    except ValueError as e:
        check("Dropout" in str(e) or "dropout" in str(e),
              f"refusal must name dropout; got: {e}")
        print("    PASS")
        return
    check(False, "active dropout was not refused")
    _TinyModel(dropout_p=0.0)
    XPUGraphStepRunner(
        _TinyModel(), UniformLossWeighting(),
        "cpu").refuse_if_unsupported()  # must not raise
    print("    PASS")


def check_capture_is_not_ready_on_cpu():
    print("[maybe_capture with no warm shape: not-ready, never an attempt]")
    model = _TinyModel()
    runner = XPUGraphStepRunner(model, UniformLossWeighting(), "cpu")
    check(runner.maybe_capture((2, 8, 8)) == "not-ready",
          "capture must not be attempted without a warmed entry")
    print("    PASS")


class _FacadeModel:
    """A TrainableModel-shaped façade like ComfyUNetTrainableModel: no
    .modules(), inner nn.Module behind .raw.model."""

    def __init__(self, inner):
        self._inner = inner
        self.raw = type("Raw", (), {"model": inner})()

    def forward(self, xc, t, ctx, y):
        return self._inner(xc)

    def trainable_parameters(self):
        return list(self._inner.parameters())


def check_dropout_scan_reaches_through_the_facade():
    print("[dropout scan unwraps the TrainableModel façade; an unscannable "
          "model is refused, not passed]")
    good = _FacadeModel(torch.nn.Sequential(torch.nn.Linear(4, 4)))
    XPUGraphStepRunner(good, UniformLossWeighting(),
                       "cpu").refuse_if_unsupported()  # must not raise
    bad = _FacadeModel(torch.nn.Sequential(torch.nn.Linear(4, 4),
                                           torch.nn.Dropout(0.2)))
    try:
        XPUGraphStepRunner(bad, UniformLossWeighting(),
                           "cpu").refuse_if_unsupported()
    except ValueError:
        pass
    else:
        check(False, "dropout behind the façade was not refused")
    opaque = object()
    try:
        XPUGraphStepRunner(opaque, UniformLossWeighting(),
                           "cpu").refuse_if_unsupported()
    except (ValueError, AttributeError):
        pass
    else:
        check(False, "an unscannable model was passed silently")
    print("    PASS")


def main():
    check_eager_live_matches_the_loss_expression()
    check_zeroing_is_in_place()
    check_backward_scale_is_baked()
    check_dropout_is_refused()
    check_dropout_scan_reaches_through_the_facade()
    check_capture_is_not_ready_on_cpu()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
