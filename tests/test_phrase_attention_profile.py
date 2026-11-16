"""Diagnostic labels must not enter candidate selection or token alignment."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from profile_phrase_attention_candidates import selections, token_tags


def test_actual_subword_pieces_share_word_without_losing_boundaries():
    tags = token_tags([' Sol', 'itaire', ' Daily', ' Challenges', ' for', ' 12', '!'])
    assert tags[0]['word'] == tags[1]['word'] == 'solitaire'
    assert tags[0]['role'] == 'content_word_start'
    assert tags[1]['role'] == 'content_word_continuation'
    assert tags[4]['role'] == 'function_or_relation_word'
    assert tags[5]['role'] == 'number'
    assert tags[6]['role'] == 'punctuation_or_space'
    assert tags[0]['position'] == 'first' and tags[-1]['position'] == 'last'
    assert token_tags([' Save'])[0]['position'] == 'single'


def test_selectors_ignore_all_target_labels_and_only_propose_actual_peaks():
    rng = np.random.default_rng(21); t, p = 4, 30
    record = dict(available=True, n_tokens=t, peaks=rng.integers(0, p, size=(t, 96)),
                  features=rng.random((t, 96, 14)), target_mask=np.zeros(p, bool),
                  hit=np.zeros((t, 96), bool), center_hit=np.zeros((t, 96), bool))
    order = list(range(96)); weights = np.linspace(.01, .2, 96)
    before = selections(record, order, weights)
    altered = dict(record, target_mask=np.ones(p, bool), hit=np.ones((t, 96), bool), center_hit=np.ones((t, 96), bool))
    after = selections(altered, order, weights)
    for name, parents in before.items():
        np.testing.assert_array_equal(parents, after[name])
        assert set(parents) <= set(record['peaks'].ravel())


def test_duplicate_tokens_do_not_change_train_weighted_vote_distribution():
    rng = np.random.default_rng(83)
    base = dict(available=True, n_tokens=2, peaks=rng.integers(0, 13, size=(2, 96)),
                features=rng.random((2, 96, 14)), target_mask=np.zeros(13, bool))
    doubled = dict(base, n_tokens=4, peaks=np.tile(base['peaks'], (2, 1)), features=np.tile(base['features'], (2, 1, 1)))
    weights = np.linspace(.01, .25, 96); order = list(range(96))
    np.testing.assert_array_equal(selections(base, order, weights)['train_weighted_peak_votes'],
                                  selections(doubled, order, weights)['train_weighted_peak_votes'])
