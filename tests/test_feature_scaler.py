"""Tests for the descriptor-matrix feature standardiser."""

import glob
import os

import numpy as np
import pytest

import molscope as ms
from molscope.descriptors import FeatureScaler, standardize_features

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def test_fit_standardises_columns_to_zero_mean_unit_var():
    rng = np.random.RandomState(0)
    X = rng.normal(5.0, 3.0, size=(30, 4))
    scaler = FeatureScaler.fit(X)
    Z = scaler.transform(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-9)


def test_inverse_transform_round_trips():
    rng = np.random.RandomState(1)
    X = rng.normal(0.0, 10.0, size=(15, 6))
    scaler = FeatureScaler.fit(X)
    assert np.allclose(scaler.inverse_transform(scaler.transform(X)), X)


def test_constant_column_does_not_blow_up():
    X = np.column_stack([np.arange(10.0), np.full(10, 7.0)])  # 2nd col constant
    scaler = FeatureScaler.fit(X)
    assert scaler.std[1] == 1.0  # near-constant column gets unit std
    # A *test* row whose constant-on-train column differs stays finite/sane.
    out = scaler.transform([[3.0, 12.0]])
    assert np.all(np.isfinite(out))
    assert out[0, 1] == pytest.approx(5.0)  # (12 - 7) / 1


def test_per_column_statistics_have_feature_width():
    X = np.ones((5, 8))
    X[:, 0] = np.arange(5)
    scaler = FeatureScaler.fit(X)
    assert scaler.mean.shape == (8,)
    assert scaler.std.shape == (8,)


def test_standardize_features_fits_on_train_only():
    # Train and held-out rows are drawn from different distributions, so a
    # train-only fit must differ from a whole-matrix fit (no leakage).
    rng = np.random.RandomState(2)
    train_rows = rng.normal(0.0, 1.0, size=(20, 3))
    test_rows = rng.normal(50.0, 1.0, size=(20, 3))
    X = np.vstack([train_rows, test_rows])
    train_index = list(range(20))

    X_std, scaler = standardize_features(X, train_index)
    # The scaler saw only the train rows.
    assert np.allclose(scaler.mean, train_rows.mean(axis=0))
    # Every row is transformed, and the held-out rows are far from zero-mean
    # precisely because they did not inform the statistics.
    assert X_std.shape == X.shape
    assert np.all(np.abs(X_std[20:].mean(axis=0)) > 10)
    # ...and differs from a leaky whole-matrix fit.
    full = FeatureScaler.fit(X)
    assert not np.allclose(scaler.mean, full.mean)


def test_standardize_features_transforms_every_row():
    X = np.arange(40.0).reshape(10, 4)
    X_std, _ = standardize_features(X, [0, 1, 2, 3, 4])
    assert X_std.shape == (10, 4)
    assert np.all(np.isfinite(X_std))


def test_standardize_features_accepts_numpy_index_array():
    X = np.arange(20.0).reshape(10, 2)
    idx = np.array([0, 2, 4, 6, 8])
    X_std, scaler = standardize_features(X, idx)
    assert np.allclose(scaler.mean, X[idx].mean(axis=0))


def test_empty_train_index_raises():
    with pytest.raises(ValueError, match="train_index is empty"):
        standardize_features(np.ones((4, 2)), [])


def test_fit_rejects_non_2d_matrix():
    with pytest.raises(ValueError, match="2-D feature matrix"):
        FeatureScaler.fit(np.zeros(5))


def test_fit_rejects_empty_matrix():
    with pytest.raises(ValueError, match="empty matrix"):
        FeatureScaler.fit(np.zeros((0, 3)))


def test_integrates_with_featurize_many():
    paths = sorted(glob.glob(os.path.join(DATA, "*.pdb")))[:5]
    X, names = ms.featurize_many(paths, preset="native-basic", return_names=True)
    X_std, scaler = standardize_features(X, [0, 1, 2])
    assert X_std.shape == X.shape
    assert scaler.mean.shape == (len(names),)
    assert np.all(np.isfinite(X_std))


def test_exported_at_top_level():
    assert ms.FeatureScaler is FeatureScaler
    assert ms.standardize_features is standardize_features
