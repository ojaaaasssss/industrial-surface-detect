import unittest
import torch
from PIL import Image
from src.core import DefectCNN,predict,transform
class CoreTests(unittest.TestCase):
    def test_output_shape(self):
        self.assertEqual(tuple(DefectCNN()(torch.zeros(2,3,128,128)).shape),(2,))
    def test_prediction(self):
        r=predict(DefectCNN(),Image.new('RGB',(200,100)))
        self.assertIn(r['label'],('defective','non_defective'))
        self.assertTrue(0<=r['defect_probability']<=1)
if __name__=='__main__':unittest.main()
