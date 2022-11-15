import os
import tensorflow as tf

from mgcora.CIT import AvailableDatasets, one, add2, div2, mlt1, isNN1, summation, train_model, maxg, sumg, mul100
from mgcora.datasets.citation_dataset import get_dataset
from libmg.loaders import SingleGraphLoader
from libmg.compiler import GNNCompiler, FixPointConfig, Bottom, CompilationConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"


class CudaTest(tf.test.TestCase):
    def setUp(self):
        super(CudaTest, self).setUp()

    def test_cuda(self):
        self.assertEqual(tf.test.is_built_with_cuda(), True)


def preprocess(g):
        gnn = train_model(AvailableDatasets.CORA)
        loader = SingleGraphLoader(get_dataset(AvailableDatasets.CORA))
        new_x = gnn((loader.load().__iter__().__next__()[0]))
        g.x = new_x
        return g


class InfluenceTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.dataset = get_dataset(AvailableDatasets.CORA)
        self.dataset.apply(preprocess)
        self.compiler = GNNCompiler(psi_functions={'1': one, 'add2': add2, 'div2': div2, 'mlt1': mlt1, 'isNN1': isNN1,
                                                   'max': maxg, 'sum': sumg, 'mul100': mul100},
                                    sigma_functions={'sum': summation},
                                    phi_functions={},
                                    bottoms={'f': FixPointConfig(Bottom(1, 0.), 0.001)},
                                    tops={},
                                    config=CompilationConfig.xa_config(tf.float32, 1433, tf.float32))

    def test_run(self):
        model = self.compiler.compile('mu X,f . ( (( mlt1 ; isNN1 ; <| sum ) || (((X ; <| sum) || (1 ; <| sum ; max)) '
                                      '; div2)) ; add2 ) ; (mul100 || sum) ; div2')
        loader = SingleGraphLoader(self.dataset, epochs=1)
        for inputs, y in loader.load():
            print(model.call([inputs], training=False))

    def test_indegrees(self):
        model = self.compiler.compile('1 ; <| sum')
        loader = SingleGraphLoader(self.dataset, epochs=1)
        for inputs, y in loader.load():
            model.call([inputs], training=False)

    def test_max_indegree(self):
        model = self.compiler.compile('1 ; <| sum ; max')
        loader = SingleGraphLoader(self.dataset, epochs=1)
        for inputs, y in loader.load():
            print(model.call([inputs], training=False))

    def test_normalize(self):
        model = self.compiler.compile('1 ; (mul100 || sum) ; div2')
        loader = SingleGraphLoader(self.dataset, epochs=1)
        for inputs, y in loader.load():
            print(model.call([inputs], training=False))


if __name__ == '__main__':
    tf.test.main()
