import unittest
import numpy as np
import test_adj
import test_adj_ptr
import test_adj_many
import test_lpmethods
import test_foam
import test_imp
import test_gpus_many
import test_gpus_many_map


class TestLpMethods(unittest.TestCase):

    def test_adj(self):
        [scale,err] = test_adj.test_adj()
        self.assertTrue((np.abs(scale-1)<0.1) & (err<0.01))

    def test_adj_ptr(self):
        err = test_adj_ptr.test_adj_ptr()
        self.assertTrue(err<0.01)

    def test_adj_many(self):
        [err0,err1,err2] = test_adj_many.test_adj_many()
        self.assertTrue((err0<0.003) & (err1<0.006) & (err2<0.0006))
    
    def test_lpmethods(self):
        [norm0,norm1,norm2,norm3,norm4] = test_lpmethods.test_lpmethods()
        dif = 0.001
        self.assertTrue(((norm0-222.4238)<dif) & ((norm1-348.78424)<dif) & (norm2-347.4832<dif) & (norm3-328.2814<dif) & (norm4-324.60355<dif))

    def test_foam(self):
        [norm0,norm1,norm2,norm3,norm4] = test_foam.test_foam()
        dif = 0.001
        self.assertTrue(((norm0-602.86017)<dif) & ((norm1-1200.2976)<dif) & (norm2-1139.5126<dif) & (norm3-444.35742<dif) & (norm4-926.6201<dif))
    
    def test_imp(self):
        [norm0,norm1,norm2,norm3,norm4] = test_imp.test_imp()
        dif = 0.001
        self.assertTrue(((norm0-1993.7756)<dif) & ((norm1-3521.5103)<dif) & (norm2-3454.9438<dif) & (norm3-2933.0613<dif) & (norm4-3355.1555<dif))
 
    def test_gpus_many(self):
        norm = test_gpus_many.test_gpus_many()
        dif = 0.001
        self.assertTrue(norm-381.52753<dif) 

    def test_gpus_many_map(self):
        norm = test_gpus_many_map.test_gpus_many_map()
        dif = 0.001
        self.assertTrue(norm-678.70526<dif) 

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLpMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)