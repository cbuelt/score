"""Tests for the sample-based generative baseline (SB).

Unlike every other family here there is no closed form to check against, so the
references are of two kinds:

* the *estimator* identities -- ``SampleBased.energy_score`` must be bit-for-bit
  the score the evaluation protocol reports for the same members, and the
  second-moment accessors must equal a dense sample covariance built
  independently of the class under test;
* the *statistical* property the whole method rests on -- the fair energy score
  is minimized by the true predictive, so minimizing it over a free ensemble
  must recover the target's dispersion rather than collapsing it, which is
  exactly what the biased ``1/M^2`` estimator would do.

The layout tests use ``C == 2`` because with a single channel the group-inner
and group-outer stackings coincide, and the member axis is stacked exactly like
a group axis.
"""

import math

import pytest
import torch

from src.evaluation.metrics import EVAL_METRICS
from src.methods import SampleBased

C = 2  # multi-channel: distinguishes group-inner from group-outer
EPS = 1e-9
N_TRAIN, N_EVAL = 10, 50
GAMMA = 1.0


@pytest.fixture(autouse=True)
def _float64():
    """Run in float64 so the estimator identities can be checked exactly."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _method(loss: str = "kernel_score", out_channels: int = C, **extra) -> SampleBased:
    return SampleBased(
        loss=loss,
        eps=EPS,
        n_train_samples=N_TRAIN,
        n_eval_samples=N_EVAL,
        out_channels=out_channels,
        gamma=GAMMA,
        **extra,
    )


def _pred(batch: int, spatial: tuple, members: int, seed: int = 0) -> torch.Tensor:
    """A processed head output: ``(B, C, *spatial, M)`` of ensemble members."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, C, *spatial, members, generator=g)


def _sample_cov(pred: torch.Tensor) -> torch.Tensor:
    """Dense ``(B, V, V)`` sample covariance, built without the class under test."""
    flat = pred.flatten(start_dim=1, end_dim=-2)  # (B, V, M)
    centred = flat - flat.mean(dim=-1, keepdim=True)
    return centred @ centred.transpose(1, 2) / (flat.shape[-1] - 1)


# ---------------------------------------------------------------------------
# Head layout
# ---------------------------------------------------------------------------


def test_group_widths_is_a_single_mu_group():
    """The head emits one sample per member, so its width does not grow with M."""
    method = _method()
    assert method.group_widths == {"mu": C, "sigma": None, "other": None}
    assert method.out_channels == C


def test_process_output_unstacks_members_group_inner():
    """``(B, C*M, *spatial) -> (B, C, *spatial, M)`` with each channel contiguous."""
    method = _method()
    batch, length, members = 3, 5, 4
    # Channel c, member m carries the value c*100 + m, so a group-outer read
    # would be caught immediately.
    head = torch.zeros(batch, C * members, length)
    for c in range(C):
        for m in range(members):
            head[:, c * members + m] = c * 100 + m
    pred = method.process_output(head)
    assert pred.shape == (batch, C, length, members)
    for c in range(C):
        for m in range(members):
            assert torch.all(pred[:, c, :, m] == c * 100 + m)


def test_process_output_rejects_a_width_that_is_not_a_multiple_of_c():
    method = _method()
    with pytest.raises(ValueError, match="not a multiple"):
        method.process_output(torch.zeros(2, C * 3 + 1, 5))


def test_backbone_facing_flags():
    """The three properties the plumbing dispatches on."""
    method = _method(global_noise_dim=7, local_noise_dim=3, noise_hidden_dim=11)
    assert method.samples_from_backbone is True
    assert method.has_closed_form_pit is False
    assert method.noise == {"global_dim": 7, "local_dim": 3, "hidden_dim": 11}
    # A sample denormalizes like a location, member by member.
    assert method.group_denorm["mu"] == "loc"


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def test_get_samples_returns_the_members_pred_carries():
    method = _method()
    pred = _pred(3, (4,), members=6)
    assert torch.equal(method.get_samples(pred), pred)
    assert torch.equal(method.get_samples(pred, n_samples=2), pred[..., :2])
    flat = method.get_samples(pred, flatten=True)
    assert flat.shape == (3, C * 4, 6)


def test_get_samples_refuses_to_invent_members():
    """A generative member costs a forward pass, so it cannot be conjured here."""
    method = _method()
    pred = _pred(2, (3,), members=5)
    with pytest.raises(ValueError, match="model\\(x, n_samples=9\\)"):
        method.get_samples(pred, n_samples=9)


def test_get_samples_is_differentiable():
    """The training loss runs through the samples, so no ``no_grad`` decorator."""
    method = _method()
    pred = _pred(2, (3,), members=4).requires_grad_(True)
    method.get_samples(pred, flatten=True).sum().backward()
    assert torch.all(pred.grad == 1.0)


def test_get_mean_is_the_ensemble_mean():
    method = _method()
    pred = _pred(3, (4, 5), members=7)
    assert torch.allclose(method.get_mean(pred), pred.mean(dim=-1))


# ---------------------------------------------------------------------------
# Second moments: all three must match a dense sample covariance
# ---------------------------------------------------------------------------


def test_marg_std_matches_the_sample_covariance_diagonal():
    method = _method()
    pred = _pred(3, (4,), members=9)
    expected = _sample_cov(pred).diagonal(dim1=-2, dim2=-1).sqrt()
    assert torch.allclose(
        method.get_marg_std(pred).flatten(start_dim=1), expected, atol=1e-12
    )


def test_pair_diff_var_matches_the_sample_covariance():
    method = _method()
    pred = _pred(3, (4,), members=9)
    cov = _sample_cov(pred)
    var = cov.diagonal(dim1=-2, dim2=-1)
    rows = torch.arange(cov.shape[-1])
    expected = var.unsqueeze(-1) + var.unsqueeze(-2) - 2.0 * cov
    got = method.pair_diff_var(pred, rows)
    # `pair_diff_var` pins the self-pairs at exactly zero.
    expected = expected - torch.diag_embed(expected.diagonal(dim1=-2, dim2=-1))
    assert torch.allclose(got, expected, atol=1e-10)


def test_quadratic_form_matches_the_sample_covariance():
    method = _method()
    pred = _pred(3, (4,), members=9)
    g = torch.Generator().manual_seed(1)
    a = torch.randn(3, C, 4, generator=g)
    flat = a.flatten(start_dim=1)
    expected = torch.einsum("bv,bvw,bw->b", flat, _sample_cov(pred), flat)
    assert torch.allclose(method.quadratic_form(pred, a), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# The energy score
# ---------------------------------------------------------------------------


def test_energy_score_is_the_metric_the_protocol_reports():
    """Training objective and reported metric are the same estimator, exactly.

    ``SampleBased.energy_score`` goes through ``scoringrules`` while the eval
    metric uses the repo's own ``cdist`` formulation for memory reasons; they
    must still agree to floating-point, or the number the model is fitted on is
    not the number the tables show.
    """
    method = _method()
    method.training = False
    pred = _pred(4, (5,), members=8)
    y = _pred(4, (5,), members=1, seed=3).squeeze(-1)
    assert torch.allclose(
        method.energy_score(pred, y), EVAL_METRICS["energy_score"](method, pred, y),
        atol=1e-12,
    )


def test_kernel_score_is_the_metric_the_protocol_reports():
    """The default objective and the reported metric are the same estimator.

    `SampleBased.kernel_score` routes to the base `sample_kernel_score`, which
    rescales into ``scoringrules``' fixed-bandwidth Gaussian kernel; the eval
    metric evaluates the repo's ``exp(-||a-b||^2 / gamma^2)`` directly. They must
    agree to floating point, or this arm is not fitted on the number the tables
    show -- which is the whole reason it defaults to the kernel score.
    """
    method = _method()
    method.training = False
    pred = _pred(4, (5,), members=8)
    y = _pred(4, (5,), members=1, seed=3).squeeze(-1)
    assert torch.allclose(
        method.kernel_score(pred, y), EVAL_METRICS["kernel_score"](method, pred, y),
        atol=1e-12,
    )


def test_kernel_score_uses_the_fair_estimator_and_the_repo_bandwidth():
    """``0.5 (1 + E k(X, X')) - E k(X, y)`` at ``k = exp(-||.||^2 / gamma^2)``.

    Pins both halves of the conversion into `scoringrules`: the ``M(M-1)``
    spread term, and the ``sqrt(2)/gamma`` rescaling that turns its fixed
    ``exp(-||.||^2 / 2)`` kernel into the repo's.
    """
    method = _method()
    pred = _pred(3, (4,), members=6)
    y = _pred(3, (4,), members=1, seed=7).squeeze(-1)
    members = pred.flatten(start_dim=1, end_dim=-2).transpose(1, 2)  # (B, M, V)
    obs = y.flatten(start_dim=1)
    m = members.shape[1]
    g2 = method.gamma2

    cross = torch.exp(-(members - obs.unsqueeze(1)).pow(2).sum(-1) / g2).mean(-1)
    pair = torch.exp(-torch.cdist(members, members).pow(2) / g2)
    off = pair.sum(dim=(-2, -1)) - torch.diagonal(pair, dim1=-2, dim2=-1).sum(-1)
    fair = (0.5 * (1.0 + off / (m * (m - 1))) - cross).mean()
    biased = (0.5 * (1.0 + pair.sum(dim=(-2, -1)) / (m * m)) - cross).mean()

    got = method.kernel_score(pred, y)
    assert torch.allclose(got, fair, atol=1e-12)
    assert not torch.allclose(got, biased, atol=1e-6)


def test_kernel_score_masks_missing_targets_exactly():
    """A missing coordinate must drop out, leaving the score of the observed subset."""
    method = _method()
    pred = _pred(2, (5,), members=6)
    y = _pred(2, (5,), members=1, seed=11).squeeze(-1)

    holed = y.clone()
    holed[0, 0, 2] = float("nan")
    holed[1, 1, 4] = float("nan")

    keep = torch.isfinite(holed).flatten(start_dim=1)
    fct = pred.flatten(start_dim=1, end_dim=-2)
    obs = y.flatten(start_dim=1)
    g2 = method.gamma2
    per_case = []
    for b in range(fct.shape[0]):
        sel = keep[b]
        members = fct[b, sel].T.unsqueeze(0)  # (1, M, V_obs)
        o = obs[b, sel].unsqueeze(0)
        m = members.shape[1]
        cross = torch.exp(-(members - o.unsqueeze(1)).pow(2).sum(-1) / g2).mean(-1)
        pair = torch.exp(-torch.cdist(members, members).pow(2) / g2)
        off = pair.sum(dim=(-2, -1)) - torch.diagonal(pair, dim1=-2, dim2=-1).sum(-1)
        per_case.append(0.5 * (1.0 + off / (m * (m - 1))) - cross)
    expected = torch.cat(per_case).mean()

    assert torch.allclose(method.kernel_score(pred, holed), expected, atol=1e-12)


def test_kernel_score_has_gradient_at_a_collapsed_ensemble():
    """The zero-init failure mode: every member identical must still be trainable.

    At a collapsed ensemble every ``k(x_i, x_j) = 1``, so the spread term sits at
    its maximum -- but that is a maximum, not a flat spot, and the loss must
    carry gradient out of it.
    """
    method = _method()
    collapsed = torch.zeros(1, C, 3, 5, requires_grad=True)
    y = torch.full((1, C, 3), 0.5)
    (grad,) = torch.autograd.grad(method.kernel_score(collapsed, y), collapsed)
    assert torch.isfinite(grad).all()
    assert grad.abs().max() > 0


def test_kernel_score_prefers_the_correctly_dispersed_ensemble():
    """Properness, on the default objective: dispersion is rewarded, not punished.

    Note this is a statement in expectation over the target, which is why it is
    averaged over many draws. Against a single fixed observation a collapsed
    ensemble sitting near it can score better -- the Gaussian kernel is bounded,
    unlike the energy score's norm, so a lucky point mass is not penalized the
    way an unbounded score would penalize it.
    """
    method = _method()
    torch.manual_seed(0)
    batch, members = 400, 16
    y = torch.randn(batch, C, 3)  # target N(0, I)
    collapsed = torch.zeros(batch, C, 3, members)  # point mass at the mean
    matched = torch.randn(batch, C, 3, members)  # correct dispersion
    assert method.kernel_score(matched, y) < method.kernel_score(collapsed, y)


def test_energy_score_uses_the_fair_estimator():
    """The ``1/(M(M-1))`` spread term, not the biased ``1/M^2`` one."""
    method = _method()
    pred = _pred(3, (4,), members=6)
    y = _pred(3, (4,), members=1, seed=7).squeeze(-1)
    members = pred.flatten(start_dim=1, end_dim=-2).transpose(1, 2)  # (B, M, V)
    obs = y.flatten(start_dim=1)
    m = members.shape[1]
    cross = (members - obs.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
    pair = torch.cdist(members, members).sum(dim=(-2, -1))
    fair = (cross - pair / (2 * m * (m - 1))).mean()
    biased = (cross - pair / (2 * m * m)).mean()
    got = method.energy_score(pred, y)
    assert torch.allclose(got, fair, atol=1e-12)
    assert not torch.allclose(got, biased, atol=1e-6)


def test_energy_score_masks_missing_targets_exactly():
    """A missing coordinate must drop out, leaving the score of the observed subset."""
    method = _method()
    pred = _pred(2, (5,), members=6)
    y = _pred(2, (5,), members=1, seed=11).squeeze(-1)

    holed = y.clone()
    holed[0, 0, 2] = float("nan")
    holed[1, 1, 4] = float("nan")

    # The same problem with those coordinates deleted from both the members and
    # the observation: dropping a coordinate is what "the marginal on the
    # observed set" means for a norm-based score.
    keep = torch.isfinite(holed).flatten(start_dim=1)
    fct = pred.flatten(start_dim=1, end_dim=-2)
    obs = y.flatten(start_dim=1)
    per_case = []
    for b in range(fct.shape[0]):
        sel = keep[b]
        members = fct[b, sel].T.unsqueeze(0)  # (1, M, V_obs)
        o = obs[b, sel].unsqueeze(0)
        m = members.shape[1]
        cross = (members - o.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
        pair = torch.cdist(members, members).sum(dim=(-2, -1))
        per_case.append(cross - pair / (2 * m * (m - 1)))
    expected = torch.cat(per_case).mean()

    assert torch.allclose(method.energy_score(pred, holed), expected, atol=1e-12)


def test_energy_score_is_not_blocked_on_missing_targets():
    """``get_train_loss`` must not wrap this method's energy score in the guard.

    The base class bans a masked ``energy_score`` because its *sliced Gaussian*
    estimator cannot express one. The ensemble estimator can, so this method
    clears the ban -- and EUPPBench, whose targets always have gaps, depends on
    it.
    """
    method = _method()
    assert method.MASK_UNSUPPORTED_LOSSES == ()
    pred = _pred(2, (5,), members=6)
    y = _pred(2, (5,), members=1, seed=13).squeeze(-1)
    y[0, 0, 1] = float("nan")
    assert torch.isfinite(method.get_train_loss()(pred, y))


def test_energy_score_gradient_pushes_members_apart_from_a_collapsed_ensemble():
    """The spread term must have gradient even where every member coincides.

    This is the failure mode the backbones' zero-init heads walk into at step 0:
    the whole ensemble is identical, so every ``||x_i - x_j||`` sits at exactly
    zero. The norm's gradient is zero *there*, but it is a unit vector
    everywhere else, so the loss must still fall when the members are separated.
    """
    method = _method()
    collapsed = torch.zeros(1, C, 3, 5)
    y = torch.full((1, C, 3), 0.5)
    base = method.energy_score(collapsed, y)
    spread = collapsed + 0.1 * torch.randn(
        1, C, 3, 5, generator=torch.Generator().manual_seed(2)
    )
    assert method.energy_score(spread, y) < base


#: Optimal scale for a Gaussian sampler fitted to N(0, 1) draws by minimizing
#: ``E||X - y|| - c/2 E||X - X'||`` over ``M`` i.i.d. members. Both expectations
#: are folded normals, so the stationary point solves ``sqrt(2) s / sqrt(s^2+1)
#: = c`` and hence ``s = c / sqrt(2 - c^2)``. The fair estimator has ``c = 1``
#: and recovers ``s = 1``; the naive ``1/M^2`` one has ``c = (M-1)/M`` and is
#: under-dispersed -- badly so at the ``M`` a generative model can afford.
def _optimal_scale(c: float) -> float:
    return c / math.sqrt(2.0 - c * c)


def test_minimizing_the_fair_score_recovers_the_target_dispersion():
    """The point of the fair correction: a small ensemble must not shrink.

    Members here are what they are in training -- i.i.d. draws from the model,
    redrawn every step -- and the only free parameter is their scale. Fitted
    against ``N(0, 1)`` the fair estimator must find scale 1; the naive
    ``1/M^2`` spread term would settle at 0.686 for ``M = 5``, so the two
    optima are far apart and this cannot pass by accident.
    """
    torch.manual_seed(0)
    method = _method(out_channels=1)
    members, batch = 5, 512
    log_scale = torch.tensor([math.log(0.3)], requires_grad=True)
    opt = torch.optim.Adam([log_scale], lr=0.02)
    for _ in range(1500):
        pred = log_scale.exp() * torch.randn(batch, 1, 1, members)
        loss = method.energy_score(pred, torch.randn(batch, 1, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    got = log_scale.detach().exp().item()
    biased = _optimal_scale((members - 1) / members)
    assert biased == pytest.approx(0.686, abs=1e-3)  # pins the contrast
    assert got == pytest.approx(_optimal_scale(1.0), abs=0.06), got


# ---------------------------------------------------------------------------
# What is deliberately unavailable
# ---------------------------------------------------------------------------


def test_nll_reports_infinity_rather_than_raising():
    """No tractable density; the ``Deterministic`` convention, which eval tolerates."""
    method = _method()
    pred = _pred(2, (3,), members=4)
    y = _pred(2, (3,), members=1, seed=5).squeeze(-1)
    assert math.isinf(method.nll(pred, y))
    assert math.isinf(EVAL_METRICS["nll"](method, pred, y))


@pytest.mark.parametrize("loss", ["nll", "marginal_kernel_score"])
def test_losses_needing_a_density_are_rejected_at_construction(loss):
    """Fail at build time, not thirty epochs into a run.

    The *joint* `kernel_score` is not among these: it is the default, and this
    class overrides it with the fair ensemble estimator.
    """
    with pytest.raises(ValueError, match="no density and no closed-form"):
        _method(loss=loss)


@pytest.mark.parametrize("loss", ["kernel_score", "energy_score", "crps", "mse", "mae"])
def test_ensemble_losses_are_selectable(loss):
    """Everything estimable from members must build and produce a finite loss."""
    method = _method(loss=loss)
    pred = _pred(2, (4,), members=6)
    y = _pred(2, (4,), members=1, seed=17).squeeze(-1)
    assert torch.isfinite(method.get_train_loss()(pred, y))


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_members_is_rejected(count):
    """The fair estimator divides by ``M(M-1)``."""
    with pytest.raises(ValueError, match="at least 2 members"):
        SampleBased(
            loss="energy_score",
            eps=EPS,
            n_train_samples=count,
            n_eval_samples=N_EVAL,
            out_channels=C,
            gamma=GAMMA,
        )


def test_crps_loss_is_available_and_matches_the_ensemble_estimator():
    """``crps`` stays selectable as a marginal-only objective."""
    method = _method(loss="crps")
    pred = _pred(3, (4,), members=6)
    y = _pred(3, (4,), members=1, seed=9).squeeze(-1)
    fct = pred.flatten(start_dim=1, end_dim=-2)
    obs = y.flatten(start_dim=1)
    m = fct.shape[-1]
    cross = (fct - obs.unsqueeze(-1)).abs().mean(dim=-1)
    pair = (fct.unsqueeze(-1) - fct.unsqueeze(-2)).abs().sum(dim=(-2, -1))
    expected = (cross - pair / (2 * m * (m - 1))).mean()
    assert torch.allclose(method.crps(pred, y), expected, atol=1e-12)
