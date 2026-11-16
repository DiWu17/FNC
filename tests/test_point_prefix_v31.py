import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
from clickagent.point_prefix_v31 import point_prefix

class PrefixTests(unittest.TestCase):
    def test_complete_point_without_auxiliary_tail(self):
        text='<g><t>Save</t><p>123,456</p><a1>'
        r=point_prefix(text,[text]);self.assertEqual(r['point'],[123,456]);self.assertEqual(r['target'],'Save')
        self.assertTrue(r['parser_only_closing_tag']);self.assertFalse(text.endswith('</g>'))
    def test_never_accept_partial_coordinate(self):
        self.assertIsNone(point_prefix('<g><t>Save</t><p>123,45',[]))
    def test_none_and_unicode(self):
        text='<g><t>保存</t><p>1,999</p>';before=[text]
        self.assertEqual(point_prefix(text,before)['target'],'保存');self.assertEqual(before,[text])
        self.assertIsNone(point_prefix('<g><t>NONE</t><p>NONE</p>',[])['point'])
    def test_wait_for_both_fields(self):
        self.assertIsNone(point_prefix('<g><p>1,2</p>',[]))
        self.assertEqual(point_prefix('<g><p>1,2</p><t>X</t>',[])['point'],[1,2])

if __name__=='__main__':unittest.main()
