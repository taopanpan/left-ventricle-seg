from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import project.hourglass.params as hgparams
import project.input.augmentation as augmentation
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
            augmentation=True,
            flip_fn=augmentation.catface_flip_fn
        )

        images, gt_heatmaps, gt_lms, scale, marked = provider.get()
        images /= 255.

        features = {'image': images,
                    'marked_idx': marked,
                    'scale': scale}

        targets = {'heatmap': gt_heatmaps,
                   'coordinates': gt_lms}

        return features, targets


def evaluate(model_dir, eval_data, params):
    params[hgparams.N_LANDMARKS] = 34
    params['eval_fn'] = estimator.catface_evaluation

    # Instantiate Estimator
    nn = estimator.get_estimator(model_dir=model_dir, params=params)

    eval_epoch_size = tfrecords_count(eval_data)
    eval_epoch_steps = eval_epoch_size / FLAGS.batch_size

    # Evaluate
    metrics = nn.evaluate(input_fn=lambda: _input_fn(eval_data, params[hgparams.N_LANDMARKS]), steps=eval_epoch_steps)
    return metrics


def main(unused_argv):
    params = hgparams.params_from_flags()
    metrics = evaluate(model_dir=FLAGS.model_dir, eval_data=FLAGS.eval_data, params=params)

    for metric, value in metrics.items():
        print('{}: {}'.format(metric, value))


if __name__ == '__main__':
    tf.app.run(main=main)
