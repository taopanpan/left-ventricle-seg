from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import project.hourglass.params as hgparams
from project.flags import FLAGS
from project.hourglass import estimator
from project.input import data
from project.utils.tfrecords import tfrecords_count

tf.logging.set_verbosity(tf.logging.DEBUG)


def _input_fn(filename, n_landmarks):
    with tf.name_scope("input"):
        provider = data.DataProvider(
            filename=filename,
            n_landmarks=n_landmarks,
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocess_threads,
            augmentation=True)

        images, gt_heatmaps, gt_lms, scale, marked = provider.get()
        images /= 255.

        features = {'image': images,
                    'marked_idx': marked,
                    'scale': scale}

        targets = {'heatmap': gt_heatmaps,
                   'coordinates': gt_lms}

        return features, targets


def _configure_validation(eval_data, train_epoch_steps, n_landmarks):
    if eval_data is None or FLAGS.eval_every_n_epochs <= 0:
        return None

    eval_epoch_size = tfrecords_count(FLAGS.eval_data)
    eval_epoch_steps = eval_epoch_size / FLAGS.batch_size

    every_n_steps = FLAGS.eval_every_n_epochs * train_epoch_steps
    early_stopping_rounds = FLAGS.patience * train_epoch_steps

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: _input_fn(eval_data, n_landmarks),
        eval_steps=eval_epoch_steps,
        every_n_steps=every_n_steps,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=early_stopping_rounds)

    return [validation_monitor]


def train(model_dir, train_data, eval_data=None, params=None):
    train_epoch_size = tfrecords_count(train_data)
    train_epoch_steps = train_epoch_size / FLAGS.batch_size

    max_steps = None
    if FLAGS.max_epochs is not None:
        max_steps = FLAGS.max_epochs * train_epoch_steps

    # Instantiate Estimator
    run_config = tf.contrib.learn.RunConfig(
        save_checkpoints_secs=None,
        save_checkpoints_steps=train_epoch_steps,
        keep_checkpoint_max=max(FLAGS.patience, 5)
    )
    nn = estimator.get_estimator(model_dir=model_dir, params=params, run_config=run_config)

    # Fit
    nn.fit(input_fn=lambda: _input_fn(train_data, params[hgparams.N_LANDMARKS]),
           max_steps=max_steps,
           monitors=_configure_validation(eval_data, train_epoch_steps, params[hgparams.N_LANDMARKS]))


def main(unused_argv):
    params = hgparams.params_from_flags()
    train(FLAGS.model_dir, FLAGS.train_data, FLAGS.eval_data, params)


if __name__ == '__main__':
    tf.app.run(main=main)
