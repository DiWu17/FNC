import numpy as np

from clickagent.attention_field_diagnostics import perturb_field, token_pieces


def sample():
    rng = np.random.default_rng(52)
    x = rng.random((8, 3, 96, 6)).astype('float16'); x[..., 5] = 0
    x[2, :, :, 5] = 1
    return x


def test_spatial_and_context_erasures_preserve_other_channels_and_peaks():
    x = sample(); saved = x.copy(); peak = x[..., 5] == 1
    spatial = perturb_field(x, 'nonpeak_spatial_zero')
    context = perturb_field(x, 'nonpeak_context_zero')
    np.testing.assert_array_equal(spatial[..., 3:], x[..., 3:])
    np.testing.assert_array_equal(context[..., :3], x[..., :3])
    np.testing.assert_array_equal(spatial[peak], x[peak]); np.testing.assert_array_equal(context[peak], x[peak])
    assert not spatial[..., :3][~peak].any() and not context[..., 3:5][~peak].any()
    np.testing.assert_array_equal(x, saved)


def test_shuffle_preserves_per_head_marginals_peaks_and_context_is_reproducible():
    x = sample(); y = perturb_field(x, 'nonpeak_spatial_shuffle', 'case1', 4)
    np.testing.assert_array_equal(y[..., 3:], x[..., 3:])
    np.testing.assert_array_equal(y[2], x[2])
    for t in range(3):
        for h in range(96):
            mask = x[:, t, h, 5] == 0
            assert sorted(map(tuple, y[mask, t, h, :3].tolist())) == sorted(map(tuple, x[mask, t, h, :3].tolist()))
            for ch in range(3):
                np.testing.assert_array_equal(np.sort(y[mask, t, h, ch]), np.sort(x[mask, t, h, ch]))
    np.testing.assert_array_equal(y, perturb_field(x, 'nonpeak_spatial_shuffle', 'case1', 4))
    assert not np.array_equal(y[..., :3], x[..., :3])
    assert not np.array_equal(y, perturb_field(x, 'nonpeak_spatial_shuffle', 'case1', 5))


def test_token_labels_follow_one_based_raw_generation_positions():
    meta = dict(generated_prefixes=['TARGET:', 'TARGET: Show', 'TARGET: Show Ink', 'TARGET: Show Ink Attributes'],
                target_raw_indices=[2, 3, 4])
    assert token_pieces(meta) == [' Show', ' Ink', ' Attributes']
