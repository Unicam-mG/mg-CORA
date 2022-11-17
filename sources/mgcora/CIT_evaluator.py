import tensorflow as tf
from spektral.data import SingleLoader
from libmg.compiler import GNNCompiler, FixPointConfig, Bottom, CompilationConfig
from libmg.evaluator import PredictPerformance, CallPerformance

from sources.mgcora.CIT import AvailableDatasets, one, add2, div2, mlt1, isNN1, maxg, sumg, mul100, summation
from sources.mgcora.datasets.citation_dataset import get_dataset


compiler = GNNCompiler(psi_functions={'1': one, 'add2': add2, 'div2': div2, 'mlt1': mlt1, 'isNN1': isNN1,
                                      'max': maxg, 'sum': sumg, 'mul100': mul100},
                       sigma_functions={'sum': summation},
                       phi_functions={},
                       bottoms={'f': FixPointConfig(Bottom(1, 0.), 0.001)},
                       tops={},
                       config=CompilationConfig.xa_config(tf.float32, 1433, tf.float32))

if __name__ == '__main__':
    d = get_dataset(AvailableDatasets.CORA)
    PredictPerformance(lambda _: compiler.compile('mu X,f . ( (( mlt1 ; isNN1 ; <| sum ) || (((X ; <| sum) || (1 ; <| '
                                                  'sum ; max)) '
                                                  '; div2)) ; add2 ) ; (mul100 || sum) ; div2'),
                       lambda dataset: SingleLoader(dataset, epochs=1))(d)
    CallPerformance(
        lambda _: compiler.compile('mu X,f . ( (( mlt1 ; isNN1 ; <| sum ) || (((X ; <| sum) || (1 ; <| sum ; max)) '
                                   '; div2)) ; add2 ) ; (mul100 || sum) ; div2'),
        lambda dataset: SingleLoader(dataset, epochs=1))(d)
