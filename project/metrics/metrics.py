from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Indices(object):
    """Dictionaries which provide a mapping from landmark label to its indices in the annotation schema.
    These may be used for generating the TensorBoard summaries for the PCK metrics
  
    The following keys are defined:
  
    * `CATPOSE`:
    * `CATFACE`:
    """

    
    CATFACE = {
        'Endocardium': list(range(0, 17)),
        'Epicardium': list(range(17, 34))        
    }


def pck(labels_to_indices, predictions, gts, scales, tolerance):
    scaled_distances = _calculate_distances(predictions, gts, scales)

    # Calculate pck for landmark groups
    labels_to_pck = {'pck/{}'.format(label): tf.metrics.mean(_accuracy(scaled_distances, tolerance, indices))
                     for label, indices in labels_to_indices.items()}

    # Calculate overall pck

    all_indices = list(range(len(labels_to_pck)))
    labels_to_pck['pck/all'] = tf.metrics.mean(
        _accuracy(scaled_distances, tolerance, indices=all_indices))

    metrics_to_values, metrics_to_updates = tf.contrib.metrics.aggregate_metric_map(labels_to_pck)

    return metrics_to_updates


def normalised_mean_error(labels_to_indices, predictions, gts, scales):
    scaled_distances = _calculate_distances(predictions, gts, tf.sqrt(scales))

    # Calculate mean scaled distance for landmark groups
    labels_to_pck = {
        'normalised_mean_error/{}'.format(label): tf.metrics.mean(_part_distances(scaled_distances, indices))
        for label, indices in labels_to_indices.items()}

    # Calculate overall mean scaled distance
    all_indices = list(range(len(labels_to_pck)))
    labels_to_pck['normalised_mean_error/all'] = tf.metrics.mean(_part_distances(scaled_distances, all_indices))

    metrics_to_values, metrics_to_updates = tf.contrib.metrics.aggregate_metric_map(labels_to_pck)

    return metrics_to_updates


def _calculate_distances(predictions, gts, scales):
    euclidean_distances = tf.sqrt(tf.reduce_sum(tf.pow(predictions - gts, 2), axis=-1))
    scaled_distances = euclidean_distances / tf.expand_dims(scales, -1)

    return scaled_distances, euclidean_distances


def _accuracy(distances, tolerance, indices):
    part_distances = tf.transpose(
        tf.gather(
            tf.transpose(distances),
            indices))

    return tf.reduce_mean(tf.to_double(part_distances <= tolerance), axis=1)


def _part_distances(distances, indices):
    part_distances = tf.transpose(
        tf.gather(
            tf.transpose(distances),
            indices))

    nonnan_indices = ~tf.is_nan(part_distances)

    part_distances = tf.boolean_mask(part_distances, nonnan_indices)

    return part_distances
