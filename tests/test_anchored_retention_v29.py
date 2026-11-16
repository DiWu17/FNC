import sys
import unittest
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'src'), str(Path(__file__).resolve().parent)]
import numpy as np
from clickagent.anchored_retention_v29 import fit, choose
from test_coordinate_calibration_v29 import CalibrationChecks


class AnchoredChecks(unittest.TestCase):
    def test_all_discordant_rivals_and_no_free_movement(self):
        r = CalibrationChecks().record();model = fit([r])
        self.assertEqual(model['eligible_queries'], 1);self.assertEqual(model['pairs'], 2)
        j, reason = choose(model, r, np.zeros(3), .999999)
        self.assertEqual(j, 2);self.assertEqual(r['points'][j].tolist(), [31, 33])
        self.assertEqual(reason['reason'], 'keep_exact_free')

    def test_tie_labels_supply_no_preference(self):
        r = CalibrationChecks().record();r['y'][:] = True
        model = fit([r]);self.assertEqual(model['pairs'], 0)


if __name__ == '__main__': unittest.main()
