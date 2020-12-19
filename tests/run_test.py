import unittest
import numpy as np
import test_adj_ptr
import test_adj_many
import test_lpmethods
import test_foam
import test_imp
import test_gpus_many
import test_gpus_many_map


class TestLpMethods(unittest.TestCase):
    
    def test_adj_ptr(self):
        err = test_adj_ptr.test_adj_ptr()
        self.assertTrue(err < 0.01)

    def test_adj_many(self):
        [err0, err1, err2] = test_adj_many.test_adj_many()
        self.assertTrue((err0 < 0.005) & (err1 < 0.007) & (err2 < 0.0007))

    def test_lpmethods(self):
        [norm0, norm1, norm2, norm3, norm4] = test_lpmethods.test_lpmethods()
        dif = 10
        self.assertTrue(((norm0-222.4238) < dif) & ((norm1-348.78424) < dif) &
                        (norm2-347.4832 < dif) & (norm3-328.2814 < dif) & (norm4-324.60355 < dif))

    def test_foam(self):
        [norm0, norm1, norm2, norm3, norm4, norm5] = test_foam.test_foam()
        dif = 100
        self.assertTrue(((norm0-602.86015) < dif) & ((norm1-1195.5931) < dif) &
                        (norm2-1139.5114 < dif) & (norm3-444.35743 < dif) & (norm4-926.6204 < dif) & (norm5-1003.76623 < dif))

    def test_imp(self):
        [norm0, norm1, norm2, norm3, norm4] = test_imp.test_imp()
        dif = 100
        self.assertTrue(((norm0-1993.7756) < dif) & ((norm1-3546.1950) < dif) &
                        (norm2-3454.9408 < dif) & (norm3-2933.0618 < dif) & (norm4-3355.1609 < dif))

    def test_gpus_many(self):
        norm = test_gpus_many.test_gpus_many()
        dif = 10
        self.assertTrue(norm-381.53 < dif)

    def test_gpus_many_map(self):
        norm = test_gpus_many_map.test_gpus_many_map()
        dif = 10
        self.assertTrue(norm-678.1923 < dif)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLpMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)
