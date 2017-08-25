from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import project.hourglass.params as hgparams
import project.metrics.metrics as metrics
from project.hourglass import hourglass
from project.utils import utils


def get_estimator(model_dir, params, run_config=None):
    """ Returns an instance of the Estimator """
    estimator = learn.Estimator(model_fn=_model_fn,
                                model_dir=model_dir,
                                params=params,
                                config=run_config)
    return estimator


def _model_fn(features, labels, mode, params):
    """    
    :param features:
     * 'image': a tensor of shape `[batch_size, 256, 256, 3]`.
            The images to be processed by the network.
     * 'scale': a tensor of shape `[batch_size]`.
            Holds information used to normalise evaluation metrics between sample.
            It could represent the size of a cat's face, for example.
     * 'marked_idx': a tensor of shape `[batch_size, n_landmarks]`.
            Contains values of 1's or 0's which signify whether each landmark annotation exists
            (e.g. in case of severe occlusion). 
    :param labels:
     * 'heatmap': a tensor of shape `[batch_size, 256, 256, n_landmarks]`.
            The ground truth heatmaps for landmark locations.
     * 'coordinates': a tensor of shape `[batch_size, 2]`.
            The ground truth image coordinates for the landmark locations.
    :param mode: supplied by TensorFlow. Either TRAIN, EVAL or INFER
    :param params:
    
      * eval_fn: Model function. Follows the signature:
        * Args:
          * `coordinate_predictions`:
          * `gt_coordinates`:
          * `scales`:
          * `marked`:

        * Returns:
          `eval_metric_ops`:
    
     
    :return: ModelFnOps
    """
    # Input Images
    input_images = features['image']

    # Construct the network and make predictions
    all_heatmap_predictions = _build_network(input_images, params, mode)
    final_heatmap_predictions = all_heatmap_predictions[-1]
    coordinate_predictions = utils.get_coordinates(final_heatmap_predictions)

    final_prediction_loss = None
    train_op = None
    eval_metric_ops = {}
    predictions_dict = {}

    if mode == learn.ModeKeys.INFER:
        predictions_dict = {
            'images': input_images,
            'heatmaps': final_heatmap_predictions,
            'coordinates': coordinate_predictions
        }

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        gt_heatmaps = labels['heatmap']
        marked = features['marked_idx']

        with tf.name_scope("loss"):
            all_predictions_losses = _build_losses(all_heatmap_predictions, gt_heatmaps, marked)
            final_prediction_loss = all_predictions_losses[-1]

        _build_summaries(input_images, all_heatmap_predictions, gt_heatmaps)

        for i, loss in enumerate(all_predictions_losses):
            tf.summary.scalar('loss/hourglass_{}'.format(i + 1), loss)
        tf.summary.scalar('loss/final_hourglass', all_predictions_losses[-1])

    # Calculate Evaluation Metrics:
    if mode == learn.ModeKeys.EVAL:
        scales = features['scale']
        gt_coordinates = labels['coordinates']
        marked = features['marked_idx']

        with tf.name_scope("evaluation"):
            eval_fn = params['eval_fn']
            eval_metric_ops = eval_fn(coordinate_predictions, gt_coordinates, marked, scales)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            params[hgparams.LEARNING_RATE],
            tf.contrib.framework.get_global_step(),
            decay_steps=params[hgparams.DECAY_STEPS],
            decay_rate=params[hgparams.DECAY_RATE],
            staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        train_op = _build_train_op(all_predictions_losses, learning_rate)

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=final_prediction_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def _build_network(input_images, params, mode):
    with tf.variable_scope("network"):
        model = hourglass.StackedHourglassNetwork(
            n_landmarks=params[hgparams.N_LANDMARKS],
            n_hourglass=params[hgparams.N_HOURGLASS],
            n_residuals=params[hgparams.N_RESIDUALS],
            n_features=params[hgparams.N_FEATURES],
            training=mode == learn.ModeKeys.TRAIN)

        all_heatmap_predictions = model.model(input_images)
        return all_heatmap_predictions


def _build_losses(all_heatmap_predictions, gt_heatmaps, marked_idx):
    """ Landmark-regression losses:
        - Mean Squared Error (L2 loss):
        loss on a pixel is weighted depending on whether it has any probability mass or not """
    with tf.name_scope("mse"):
        with tf.name_scope("weights"):
            mask = tf.expand_dims(tf.expand_dims(marked_idx, 1), 1)
            heatmap_weights = utils.heatmap_weights(keypoints=gt_heatmaps,
                                                    mask=mask,
                                                    ng_w=1, ps_w=1)

    # calculate losses for the predictions from each hourglass

    losses = [tf.losses.mean_squared_error(predictions, gt_heatmaps, weights=heatmap_weights)
              for predictions in all_heatmap_predictions]

    return losses


def _build_train_op(all_predictions_losses, learning_rate):
    train_ops = []
    for loss in all_predictions_losses:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            optimizer='Adam',
            summaries=['learning_rate'])
        train_ops.append(train_op)
    train_op = tf.group(*train_ops)
    return train_op


def catface_evaluation(coordinate_predictions, gt_coordinates, marked, scales):
    eval_metric_ops = {}

    # Probability of Correct Keypoints
    pck_metric_ops = metrics.pck(metrics.Indices.CATFACE, coordinate_predictions, gt_coordinates, scales, tolerance=0.1)

    # Normalised Mean Error (NME)
    nme_metric_ops = metrics.normalised_mean_error(metrics.Indices.CATFACE, coordinate_predictions, gt_coordinates,
                                                   scales)

    eval_metric_ops = dict(eval_metric_ops, **pck_metric_ops)
    eval_metric_ops = dict(eval_metric_ops, **nme_metric_ops)

    return eval_metric_ops


def catpose_evaluation(coordinate_predictions, gt_coordinates, marked, scales):
    eval_metric_ops = {}

    # Probability of Correct Keypoints
    pck_metric_ops = metrics.pck(metrics.Indices.CATPOSE, coordinate_predictions, gt_coordinates, scales, tolerance=0.5)

    # Normalised Mean Error (NME)
    nme_metric_ops = metrics.normalised_mean_error(metrics.Indices.CATPOSE, coordinate_predictions, gt_coordinates,
                                                   scales)

    eval_metric_ops = {eval_metric_ops, pck_metric_ops}
    eval_metric_ops = {eval_metric_ops, nme_metric_ops}
    return eval_metric_ops


def _build_summaries(images, all_predictions, gt_heatmaps):
    """ Image summaries """
    # input images
    tf.summary.image('input_image',
                     images,
                     max_outputs=4)

    # prediction heatmaps
    for i, predictions in enumerate(all_predictions):
        tf.summary.image('predictions/hourglass_{}'.format(i + 1),
                         tf.reduce_sum(predictions, -1)[..., None] * 255.0,
                         max_outputs=4)

    tf.summary.image('predictions/final_hourglass',
                     tf.reduce_sum(all_predictions[-1], -1)[..., None] * 255.0,
                     max_outputs=4)

    # ground truth heatmaps
    tf.summary.image('ground_truth',
                     tf.reduce_sum(gt_heatmaps * tf.ones_like(gt_heatmaps), -1)[..., None] * 255.0,
                     max_outputs=4)
