import tensorflow as tf
import os.path
from enum import Enum
import numpy as np
from keras.optimizers import Adam
from spektral.data.loaders import SingleLoader
from spektral.models.gcn import GCN
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.losses import CategoricalCrossentropy
from libmg.layers import PsiLocal, Sigma, PsiGlobal

from sources.mgcora.datasets.citation_dataset import get_dataset


class AvailableDatasets(Enum):
    CORA = 'cora'
    CITESEER = 'citeseer'
    PUBMED = 'pubmed'


def train_model(dataset_name):
    if os.path.isdir('trained_models/base_gcn_' + dataset_name.value):
        return load_model('trained_models/base_gcn_' + dataset_name.value)

    learning_rate = 1e-2
    epochs = 200
    patience = 10

    # Load data
    dataset = get_dataset(dataset_name)

    # We convert the binary masks to sample weights so that we can compute the
    # average loss over the nodes (following original implementation by
    # Kipf & Welling)
    def mask_to_weights(mask):
        return mask.astype(np.float32) / np.count_nonzero(mask)

    weights_tr, weights_va, weights_te = (
        mask_to_weights(mask)
        for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
    )

    model = GCN(n_labels=dataset.n_labels)
    model.compile(
        optimizer=Adam(learning_rate),
        loss=CategoricalCrossentropy(reduction="sum"),
        weighted_metrics=["acc"],
    )

    # Train model
    loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
    loader_va = SingleLoader(dataset, sample_weights=weights_va)
    model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
        verbose=0
    )

    loader_te = SingleLoader(dataset, sample_weights=weights_te)
    model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch, verbose=0)

    model.save('trained_models/base_gcn_' + dataset_name.value)
    return model


one = PsiLocal(lambda x: tf.ones(shape=(tf.shape(x)[0], 1), dtype=x.dtype))
add2 = PsiLocal(lambda x: tf.math.add(x[:, :1], x[:, 1:]))
div2 = PsiLocal(lambda x: tf.math.divide(x[:, :1], x[:, 1:]))
mlt1 = PsiLocal(lambda x: tf.expand_dims(tf.math.argmax(x, axis=1), axis=-1))
isNN1 = PsiLocal(lambda x: tf.cast(tf.math.equal(x, 4), dtype=tf.float32))
mul100 = PsiLocal(lambda x: tf.math.multiply(x[:, :1], 100))
maxg = PsiGlobal(single_op=lambda x: tf.math.reduce_max(x, axis=0, keepdims=True))
sumg = PsiGlobal(single_op=lambda x: tf.math.reduce_sum(x, axis=0, keepdims=True))
summation = Sigma(lambda m, i, n, x: tf.math.unsorted_segment_sum(m, i, n))
