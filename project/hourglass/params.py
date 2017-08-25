from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from project.flags import FLAGS
from project.utils.tfrecords import tfrecords_count

# MODEL PARAMETERS
N_LANDMARKS = 'n_landmarks'
N_FEATURES = 'n_features'
N_HOURGLASS = 'n_hourglass'
N_RESIDUALS = 'n_residuals'

# TRAINING PARAMETERS
LEARNING_RATE = 'initial_learning_rate'
DECAY_RATE = 'learning_rate_decay_rate'
DECAY_STEPS = 'learning_rate_decay_epochs'


def params_from_flags():
    params = {
        N_LANDMARKS: FLAGS.n_landmarks,
        N_FEATURES: FLAGS.n_features,
        N_HOURGLASS: FLAGS.n_hourglass,
        N_RESIDUALS: FLAGS.n_residuals,

    }

    return params


def train_params_from_flags():
    train_epoch_size = tfrecords_count(FLAGS.train_data)
    train_epoch_steps = train_epoch_size / FLAGS.batch_size

    params = params_from_flags()

    train_params = {
        LEARNING_RATE: FLAGS.initial_learning_rate,
        DECAY_RATE: FLAGS.learning_rate_decay_rate,
        DECAY_STEPS: FLAGS.learning_rate_decay_epochs * train_epoch_steps
    }

    params = dict(params, **train_params)

    return params
