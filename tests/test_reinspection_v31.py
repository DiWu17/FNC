import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
from PIL import Image
from clickagent.reinspection_v31 import (plan_diverse,iou,judge_canvas,decide,joint_vector,
    consensus,fit,scores,pair_geometry)

class ReinspectionTests(unittest.TestCase):
    def test_diverse_proposal_avoids_first_crop(self):
        points=np.asarray([[100,100],[130,100],[1400,800],[1440,800]])
        tr,m=plan_diverse(points,np.asarray([3,2,1,1]),[2000,1200],[0,0,640,640])
        self.assertLessEqual(iou(tr['box'],[0,0,640,640]),.25)
        self.assertFalse(m['labels_used'])
        self.assertGreater(tr['box'][0],600)

    def test_global_size_and_duplicate_crop(self):
        tr,m=plan_diverse([],[],[320,240],[0,0,320,240])
        self.assertEqual(tr['box'],[0,0,320,240]);self.assertFalse(m['satisfies_iou_limit'])

    def test_exact_point_retention_and_strict_threshold(self):
        a,b=[17,23],[450,640]
        self.assertEqual(decide(a,b,.1)[0],a)
        self.assertEqual(decide(a,b,.10001)[0],b)
        self.assertEqual(decide(None,b,-1)[0],b)

    def test_pair_reversal(self):
        a=np.arange(1,9);b=np.arange(8,0,-1)
        np.testing.assert_allclose(joint_vector(a,b),-joint_vector(b,a))
        np.testing.assert_allclose(joint_vector(a,a),np.zeros(8))

    def test_canvas_geometry_and_no_gt_interface(self):
        with Image.new('RGB',(1000,800),'white') as im:
            canvas,g=judge_canvas(im,[10,20],[990,790],True)
        self.assertEqual(canvas.size,(512,512));self.assertFalse(g['ground_truth_used'])
        for j,p in enumerate([[10,20],[990,790]]):
            box=g['local'][str(j)]['box'];center=g['local'][str(j)]['local_center']
            restored=[(center[0]-j*256)*128/256+box[0],(center[1]-256)*128/256+box[1]]
            np.testing.assert_allclose(restored,p)
        canvas.close()

    def test_ridge_learns_and_preserves_ties(self):
        rr=[]
        for i,delta in enumerate([-1,1,0,-1,1,0]):
            rr.append(dict(sample_id=str(i),source='s',eligible=True,delta=delta,
                geometry=np.array([1.,0]),content=np.array([float(delta),0]),blurred=np.array([0.,0.])))
        model=fit(rr,'content');p=scores(model,rr)
        self.assertGreater(p[1],.5);self.assertLess(p[0],-.5);self.assertEqual(p[2],0)

    def test_consensus_returns_existing_point(self):
        points=[[20,20],[22,20],[900,900]]
        self.assertEqual(consensus(points,[22,20]),[22,20])

if __name__=='__main__':unittest.main()
