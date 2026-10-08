"""Tests for the orthogonal bases themselves, independent of the method.

Every basis must be orthonormal to machine precision at arbitrary lengths --
that is the property the whole construction rests on, since the
renormalization by the implied marginal variances assumes a congruence with an
orthogonal matrix. The bases of the basis ablation (DCT-I, DCT-IV and the random
control) are checked alongside the main ones, so a future builder cannot be
registered without meeting the same bar.
"""

import math

import pytest
import torch

from src.methods.transforms import (
    _MATRIX_BUILDERS,
    dct1_matrix,
    dct4_matrix,
    dct_matrix,
    dst_matrix,
    make_separable_transform,
    random_orthogonal_matrix,
    unbalanced_haar_matrix,
)

# Powers of two and non-powers, even and odd: haar/hadamard are restricted to
# powers of two and are exercised separately.
ANY_LENGTH = ["dct", "dst", "dct1", "dct4", "uhaar", "random"]


@pytest.mark.parametrize("kind", ANY_LENGTH)
@pytest.mark.parametrize("n", [2, 5, 8, 17, 96])
def test_orthonormal_at_arbitrary_length(kind, n):
    matrix = _MATRIX_BUILDERS[kind](n)
    assert matrix.shape == (n, n)
    identity = torch.eye(n, dtype=matrix.dtype)
    assert torch.allclose(matrix @ matrix.T, identity, atol=1e-10)
    assert torch.allclose(matrix.T @ matrix, identity, atol=1e-10)


@pytest.mark.parametrize("kind", ANY_LENGTH)
def test_registered_in_factory(kind):
    transform = make_separable_transform(kind, [8, 4])
    x = torch.randn(2, 1, 8, 4, dtype=torch.float64)
    # A round trip through an orthogonal transform is the identity.
    assert torch.allclose(transform.inverse(transform.forward(x)), x, atol=1e-10)


def test_random_basis_is_deterministic():
    # The basis must be a property of the method, not of the run: otherwise
    # seeds would vary the model class and not just its initialization.
    assert torch.equal(random_orthogonal_matrix(16), random_orthogonal_matrix(16))
    assert not torch.equal(
        random_orthogonal_matrix(16)[0], random_orthogonal_matrix(17)[0][:16]
    )


def test_random_basis_is_not_flat():
    # The point of the control: it carries no frequency or locality structure,
    # and its entry magnitudes vary strongly, so it also exercises the
    # renormalization by the implied marginal variances far from the flat case.
    n = 64
    matrix = random_orthogonal_matrix(n)
    weights = matrix.pow(2) * n
    assert weights.max() > 4.0
    assert not make_separable_transform("random", [n]).is_flat


@pytest.mark.parametrize("kind,expected_flat", [("hadamard", True), ("dct", False),
                                                ("dct1", False), ("dct4", False),
                                                ("random", False)])
def test_flatness_matches_basis(kind, expected_flat):
    assert make_separable_transform(kind, [8]).is_flat is expected_flat


def test_dct_variants_differ_from_each_other():
    # DCT-I, II and IV encode three different boundary conventions; if two of
    # them coincided the basis ablation would silently compare a basis to
    # itself.
    n = 32
    variants = [dct_matrix(n), dct1_matrix(n), dct4_matrix(n)]
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            assert not torch.allclose(variants[i].abs(), variants[j].abs(), atol=1e-6)


def test_dct1_lowest_mode_has_half_weight_endpoints():
    # Unlike the DCT-II, the DCT-I does *not* contain the constant vector: its
    # whole-sample-even extension gives the endpoints half quadrature weight,
    # so the lowest mode is flat in the interior and reduced by 1/sqrt(2) at
    # the two ends. Worth pinning down, because it means a constant field is
    # not a single DCT-I coefficient -- which is exactly the kind of boundary
    # difference the basis ablation is meant to probe.
    row = dct1_matrix(9)[0]
    interior = row[1:-1]
    assert torch.allclose(interior, interior[0].expand_as(interior), atol=1e-12)
    assert math.isclose(float(row[0] * math.sqrt(2.0)), float(interior[0]), rel_tol=1e-12)
    assert math.isclose(float(row[-1] * math.sqrt(2.0)), float(interior[0]), rel_tol=1e-12)
    assert math.isclose(float(row.norm()), 1.0, rel_tol=1e-12)


def test_dct2_lowest_mode_is_constant():
    # The contrast to the DCT-I: the DCT-II *does* carry the constant vector,
    # so a flat field costs one coefficient there and several here.
    row = dct_matrix(9)[0]
    assert torch.allclose(row, row[0].expand_as(row), atol=1e-12)


def test_dst_and_dct_disagree_on_boundary_sign():
    # Sanity check that the odd basis really is odd: the DST's lowest mode is
    # sign-symmetric about the centre, the DCT's constant mode is not.
    n = 16
    assert dct_matrix(n)[0].sum().abs() > 1.0
    assert dst_matrix(n)[-1].sum().abs() < 1.0
