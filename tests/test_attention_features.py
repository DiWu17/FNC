import numpy as np
import pytest

from clickagent.attention_features import attention_feature_names, summarize_image_attention


def values(vector, names):
    return dict(zip(names, vector))


def test_preserves_raw_mass_and_distinguishes_concentration():
    weights = np.array([[0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0.02]])
    vector, names = summarize_image_attention(weights, (2, 2), layer=15)
    features = values(vector, names)
    assert names == attention_feature_names(2, 15)
    assert len(vector) == len(names) == 52
    assert features["layer_15.head_0.image_mass"] == pytest.approx(0.4)
    assert features["layer_15.head_1.image_mass"] == pytest.approx(0.02)
    assert features["layer_15.head_0.entropy"] == pytest.approx(1)
    assert features["layer_15.head_1.entropy"] == pytest.approx(0)
    assert features["layer_15.head_1.peak"] == pytest.approx(1)
    assert features["layer_15.head_1.centroid_x"] == pytest.approx(0.75)
    assert features["layer_15.head_1.centroid_y"] == pytest.approx(0.75)


def test_mass_is_not_removed_by_spatial_normalization():
    strong, names = summarize_image_attention(np.array([[0.2, 0.3]]), (1, 2))
    weak, _ = summarize_image_attention(np.array([[0.002, 0.003]]), (1, 2))
    strong, weak = values(strong, names), values(weak, names)
    assert strong["layer_0.head_0.image_mass"] == pytest.approx(100 * weak["layer_0.head_0.image_mass"])
    assert strong["layer_0.head_0.entropy"] == pytest.approx(weak["layer_0.head_0.entropy"])


@pytest.mark.parametrize("weights,grid", [(np.zeros((2, 0)), (0, 0)), (np.zeros((2, 1)), (1, 1))])
def test_zero_attention_is_finite(weights, grid):
    vector, names = summarize_image_attention(weights, grid)
    assert np.isfinite(vector).all()
    assert not vector.any()
    assert len(names) == len(vector)


@pytest.mark.parametrize("weights,grid", [
    (np.array([[0.1, 0.2]]), (2, 2)),
    (np.array([[-0.1]]), (1, 1)),
    (np.array([[np.nan]]), (1, 1)),
    (np.array([[0.8, 0.8]]), (1, 2)),
    (np.array([0.1]), (1, 1)),
])
def test_rejects_invalid_attention(weights, grid):
    with pytest.raises(ValueError):
        summarize_image_attention(weights, grid)
