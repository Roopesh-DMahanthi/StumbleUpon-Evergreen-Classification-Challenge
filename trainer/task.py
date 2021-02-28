from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from . import model
from . import util

import tensorflow as tf
import numpy as np
import pandas as pd


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


class CustomCallback(tf.keras.callbacks.TensorBoard):
    """Callback to write out a custom metric used by CAIP for HP Tuning."""

    def __init__(self,patience=0):
        self.patience=10
        self.best_weights=None
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>=0.99):
            print("/nReached 99% Accuracy....Stopping Further Training")
            self.model.stop_training=True
        current = logs.get('loss')
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)


def train_and_evaluate(args):
    
    train_x, train_y, eval_x, eval_y = util.load_data()

    # dimensions
    num_train_examples= len(dict(train_x)['input_ids'])
    num_eval_examples = len(dict(eval_x)['input_ids'])

    # Create the Keras Model
    keras_model = model.create_keras_model(learning_rate=args.learning_rate)

    # Pass a numpy array by passing DataFrame.values
    training_dataset = model.input_fn(
        features=train_x,
        labels=train_y,
        shuffle=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size)

    # Pass a numpy array by passing DataFrame.values
    validation_dataset = model.input_fn(
        features=eval_x,
        labels=eval_y,
        shuffle=False,
        num_epochs=args.num_epochs,
        batch_size=num_eval_examples)

    # Setup Learning Rate decay.
    lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
        verbose=True)

    # Setup TensorBoard callback.
    #custom_cb = CustomCallback(os.path.join(args.job_dir, 'metric_tb'))

    # Train model
    keras_model.fit(
        training_dataset,
        steps_per_epoch=int(num_train_examples / args.batch_size),
        epochs=args.num_epochs,
        validation_data=validation_dataset,
        validation_steps=1,
        verbose=1,
        callbacks=[lr_decay_cb])

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.models.save_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)