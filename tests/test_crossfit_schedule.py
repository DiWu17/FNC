"""Matched stochastic training must not depend on candidate-pool length."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from train_crossfit_scorer import epoch_plan, keyed_seed


def test_budget_query_and_dropout_plan_survives_different_candidate_counts():
    ids = ['same_image_instruction_' + str(i) for i in range(80)]
    def consume(sizes):
        observations = []
        for pos, budget in epoch_plan(ids, 20261131, 5):
            permutation = np.random.default_rng(keyed_seed(20261131, 5, ids[pos], 'candidate_permutation')).permutation(sizes[pos])
            assert sorted(permutation.tolist()) == list(range(sizes[pos]))
            torch.manual_seed(keyed_seed(20261131, 5, ids[pos], 'dropout'))
            observations.append((pos, budget, torch.rand(4).tolist()))
        return observations
    assert consume([4] * 80) == consume([1 + (i % 32) for i in range(80)])
    assert sorted(i for i, _, _ in consume([4] * 80)) == list(range(80))
    assert {k for _, k, _ in consume([4] * 80)} == {4, 8, 16, 32}
    assert epoch_plan(ids, 20261131, 5) != epoch_plan(ids, 20261131, 6)
    assert epoch_plan(ids, 20261131, 5) != epoch_plan(ids, 20261132, 5)
