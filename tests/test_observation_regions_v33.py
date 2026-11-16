import unittest
import numpy as np
from clickagent.observation_regions_v33 import coordinate_distribution,plan_regions,box_mass,unique_support
from clickagent.grounded_composition_v28 import fixed_crop

class RegionTests(unittest.TestCase):
    def test_mass_uses_complete_support_and_half_open_edges(self):
        actual=box_mass([[0,0,10,10]],[[0,0],[9,9],[10,9]],[.2,.3,.5])
        self.assertAlmostEqual(actual[0],.5)
    def test_duplicate_support_has_no_extra_weight(self):
        a,w=unique_support([[1,2],[1,2],[4,5]],[1,1,0]);b,v=unique_support([[1,2],[4,5]],[1,0])
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(w,v)
    def test_coordinate_tokens_use_real_prefix_indices(self):
        tokens=['<g><t>','Save','</t><p>','100',',','200','</p>']
        prefixes=[''.join(tokens[:i+1]) for i in range(len(tokens))]
        maps=np.zeros((7,2,4));maps[3:6,:,3]=1
        p,m=coordinate_distribution(dict(all_attention=maps),dict(point=[100,200],grid_shape=[2,2],generated_text=prefixes[-1],generated_prefixes=prefixes))
        self.assertTrue(m['available']);self.assertEqual(m['indices'],[4,5,6]);np.testing.assert_allclose(p,[0,0,0,1])
    def test_guided_coordinate_region_and_shared_geometry(self):
        size=[1280,1280];first=fixed_crop([200,200],size);proposal=dict(all_points=[[100,100],[1100,1100]],all_scores=[5,0]);weights=np.zeros(16);weights[-1]=1
        plans=plan_regions(size,[4,4],first,[200,200],proposal,weights,'x')
        self.assertEqual(plans['coordinate_mass']['transform']['box'],[640,640,1280,1280]);self.assertTrue(plans['coordinate_mass']['satisfies_iou_limit'])
        self.assertEqual(plans['coordinate_mass']['all_anchor_boxes'],plans['name_fullpool']['all_anchor_boxes'])
        for plan in plans.values():self.assertFalse(plan['ground_truth_used']);self.assertEqual(plan['transform']['crop_size'],[1280,1280])
    def test_missing_signal_falls_back_without_dropping_case(self):
        size=[1280,1280];first=fixed_crop([200,200],size);plans=plan_regions(size,[4,4],first,None,dict(all_points=[],all_scores=[]),np.zeros(16),'x')
        self.assertEqual(plans['coordinate_mass']['transform'],plans['random_matched']['transform']);self.assertTrue(plans['name_fullpool']['fallback_to_matched_random'])
    def test_small_image_no_disjoint_region_reported(self):
        size=[320,240];first=fixed_crop([150,120],size);plans=plan_regions(size,[2,2],first,[150,120],dict(all_points=[],all_scores=[]),np.ones(4)/4,'x')
        self.assertEqual(plans['coordinate_mass']['first_crop_iou'],1);self.assertFalse(plans['coordinate_mass']['satisfies_iou_limit'])

if __name__=='__main__':unittest.main()
