import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from clickagent.coordinate_calibration_v29 import features, fit_ranker, rank_scores, choose, fit_gate, X_NAMES, GATE_NAMES


class CalibrationChecks(unittest.TestCase):
    def record(self):
        r = features([[10, 10], [20, 20], [31, 33]], 2, [2., 0., 1.],
                     {'global': None, 'free': [31, 33], 'locked': None},
                     {'global': None, 'free': 'Save', 'locked': None}, [0, 0, 80, 80], [100, 100])
        r.update(source='source', sample_id='one', y=np.array([False, False, True]))
        return r

    def test_unknown_native_score_never_enters_source_prior(self):
        r = self.record()
        other = features(r['points'], 2, [2., 0., -900.], {'global': None, 'free': [31, 33], 'locked': None},
                         {'global': None, 'free': 'Save', 'locked': None}, [0, 0, 80, 80], [100, 100])
        np.testing.assert_array_equal(r['source_base'], other['source_base'])
        np.testing.assert_array_equal(r['x'], other['x'])
        self.assertEqual(r['source_base'][2], 0.)
        self.assertFalse(np.allclose(r['base'], other['base']))

    def test_old_score_affine_invariance(self):
        r = self.record()
        other = features(r['points'], 2, [27., 7., 17.], {'global': None, 'free': [31, 33], 'locked': None},
                         {'global': None, 'free': 'Save', 'locked': None}, [0, 0, 80, 80], [100, 100])
        np.testing.assert_allclose(r['source_base'], other['source_base'])
        np.testing.assert_allclose(r['x'], other['x'])

    def test_exact_native_coordinate_retention_and_switch(self):
        r = self.record();scores = np.array([4., 3., 0.])
        gate = dict(weight=np.zeros(len(GATE_NAMES)), mean=np.zeros(len(GATE_NAMES)), std=np.ones(len(GATE_NAMES)), bias=0.)
        j, reason = choose(r, scores, gate, .5)
        self.assertEqual(j, 2);self.assertEqual(r['points'][j].tolist(), [31, 33])
        self.assertEqual(reason['reason'], 'keep_exact_free')
        gate['bias'] = 3.
        j, reason = choose(r, scores, gate, .75)
        self.assertEqual(j, 0);self.assertEqual(reason['reason'], 'learned_switch')

    def test_absent_free_and_empty_pool(self):
        r = features([], 0, [], {'global': None, 'free': None, 'locked': None},
                     {'global': None, 'free': None, 'locked': None}, [0, 0, 80, 80], [100, 100])
        self.assertEqual(r['x'].shape, (0, len(X_NAMES)));self.assertIsNone(choose(r, [])[0])
        self.assertTrue(np.isfinite(r['x']).all())

    def test_fitting_can_reduce_wrong_prior_coefficient(self):
        r = self.record();r['x'][:] = 0.;r['source_base'] = np.array([4., 2., -2.])
        model = fit_ranker([r], 'source')
        self.assertLess(float(model['prior_coef']), .01)
        self.assertTrue(np.isfinite(rank_scores(model, r)).all())

    def test_gate_ignores_cases_where_both_choices_have_same_label(self):
        r = self.record();r['y'][:] = True
        model = fit_gate([r], {'one': np.array([4., 3., 0.])})
        self.assertEqual(model['discordant_queries'], 0)
        self.assertEqual(choose(r, np.array([4., 3., 0.]), model, .5)[0], 2)


if __name__ == '__main__':
    unittest.main()
