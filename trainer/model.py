from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import TFDistilBertForSequenceClassification


def input_fn(features, labels, shuffle, num_epochs, batch_size):
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def create_keras_model(learning_rate):
    
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=2)

    # Compile Keras model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    return model