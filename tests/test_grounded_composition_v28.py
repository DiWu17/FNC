import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from clickagent.grounded_composition_v28 import parse,visual_features,relation_features,fixed_crop,pixel,fit,infer

class Checks(unittest.TestCase):
    def test_own_indices(self):
        text='<g><t>Save</t><p>10,20</p><a1>Reset</a1><b1>0,30,60,90</b1><r1>above</r1></g>'
        r=parse(text,[text[:j] for j in range(1,len(text)+1)])
        self.assertEqual(r['target'],'Save');self.assertEqual(r['point'],[10,20]);self.assertTrue(r['anchors'][0]['valid'])
        self.assertEqual(''.join(text[i-1] for i in r['roles']['target']),'Save')
        self.assertFalse(parse(text[:-4],[text[:-4]])['complete'])
        spaced=text.replace('</t>','</t>\n ').replace('</p>','</p>\n')
        p=parse(spaced,[spaced[:j] for j in range(1,len(spaced)+1)])
        self.assertTrue(p['complete']);self.assertEqual(p['point'],[10,20])
    def test_common_component_invariance(self):
        rng=np.random.default_rng(12);v=rng.normal(size=(16,7)).astype('f');a=rng.random((3,96,4)).astype('f')
        p=np.array([[1,1],[3,3]])
        x,_=visual_features(a,v,(2,2),p,[4,4]);y,_=visual_features(a,v+30,(2,2),p,[4,4])
        # Float32 addition of the common offset loses ~1e-6 in the inputs.
        np.testing.assert_allclose(x,y,atol=5e-5);self.assertEqual(x.shape,(2,576))
        z,_=visual_features(a,v,(2,2),p,[4,4],mode='uniform');self.assertGreater(float(np.max(np.abs(x-z))),.01)
    def test_actual_anchor_changes_ranking(self):
        p=np.array([[50,20],[50,80]]);a=dict(valid=True,box=[400,400,600,600],relation='above')
        f=relation_features(p,[100,100],[a]);self.assertGreater(f[0,2],f[1,2])
        a['relation']='below';g=relation_features(p,[100,100],[a]);self.assertLess(g[0,3],g[1,3])
        a['box']=[400,0,600,100];h=relation_features(p,[100,100],[a]);self.assertFalse(np.allclose(g,h))
    def test_crop_and_coordinate(self):
        t=fixed_crop([1430,3000],[1440,3040]);self.assertEqual(t['box'],[800,2400,1440,3040])
        self.assertEqual(pixel([1000,1000],t['crop_size'],t),[1439,3039])
    def test_training_and_empty_evidence(self):
        r=dict(x=np.array([[1.,0],[-1,0]]),base=np.zeros(2),y=np.array([True,False]),source='a',points=np.array([[0,0],[1,1]]),size=np.array([2,2]))
        m=fit([r],.3,True);self.assertGreater(infer(m,r)[0],infer(m,r)[1])
        f,valid=visual_features(np.empty((0,96,4)),np.ones((16,3)),[2,2],np.array([[0,0]]),[4,4]);self.assertFalse(valid.any());self.assertFalse(f.any())

if __name__=='__main__':unittest.main()
