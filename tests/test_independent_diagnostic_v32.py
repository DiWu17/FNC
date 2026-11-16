import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_independent_diagnostic_v32 import stratified_take,crop_overlap,classify

class DiagnosticTests(unittest.TestCase):
    def test_stratification_is_fixed_and_disjoint(self):
        pool=[dict(sample_id=str(i),application_group='a' if i<15 else 'b',ui_type='text' if i%2 else 'icon') for i in range(30)]
        a=stratified_take(pool,20,'diagnostic');b=stratified_take(list(reversed(pool)),20,'diagnostic')
        self.assertEqual(a,b);self.assertEqual(len({x['sample_id'] for x in a}),20)
        ids={x['sample_id'] for x in a};reserve=stratified_take([x for x in pool if x['sample_id'] not in ids],10,'reserve')
        self.assertFalse(ids&{x['sample_id'] for x in reserve})
    def test_crop_partial_and_boundary(self):
        c=dict(image_size=[1000,500]);label=dict(boxes=[[.4,.4,.6,.6]])
        self.assertEqual(crop_overlap([0,0,400,500],c,label),0)
        self.assertAlmostEqual(crop_overlap([400,200,500,300],c,label),.5)
        self.assertEqual(crop_overlap([0,0,1000,500],c,label),1)
    def test_exclusive_failure_order(self):
        self.assertEqual(classify(True,True,1),'correct')
        self.assertEqual(classify(False,True,0),'selection_loss')
        self.assertEqual(classify(False,False,0),'crop_miss')
        self.assertEqual(classify(False,False,.01),'local_decode_failure')

if __name__=='__main__':unittest.main()
