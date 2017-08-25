from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import project.hourglass.params as hgparams
from project.flags import FLAGS
from project.hourglass import estimator
from project.input import inference_provider
from menpo.landmark import labeller, left_ventricle_34,left_ventricle_34_trimesh,left_ventricle_34_trimesh1

from menpo.image import Image
from menpo.landmark import labeller
from menpo.shape import PointCloud

import project.utils.labeller_lv as labels

tf.logging.set_verbosity(tf.logging.DEBUG)


def _input_fn(filename):
    with tf.name_scope("input"):
        provider = inference_provider.InferenceProvider(filename, FLAGS.batch_size, FLAGS.num_preprocess_threads)
        images = provider.get()
        images /= 255.

        features = {'image': images}
        return features


def predict(model_dir, infer_data, params):
    params[hgparams.N_LANDMARKS] = 34

    # Instantiate Estimator
    nn = estimator.get_estimator(model_dir=model_dir, params=params)

    predictions = nn.predict(input_fn=lambda: _input_fn(infer_data),
                             as_iterable=True)

    return predictions


def main(unused_argv):
    params = hgparams.params_from_flags()
    predictions = predict(FLAGS.model_dir, FLAGS.infer_data, params)
    visualise(predictions)


def visualise(predictions):
    if isinstance(predictions, dict):
        for i in range(len(predictions['images'])):
            input_image = predictions['images'][i]
            heatmaps = predictions['heatmaps'][i]
            coordinates = predictions['coordinates'][i]
            _visualise_predictions(input_image, heatmaps, coordinates)
    else:
        for prediction in predictions:
            _visualise_predictions(prediction['images'], prediction['heatmaps'], prediction['coordinates'])


def _visualise_prediction(input_image, heatmaps, coordinates):
    summed = np.mean(heatmaps, axis=-1)
    plt.figure()
    f, (axis1, axis2) = plt.subplots(1, 2)
    axis1.imshow(input_image)
    axis2.imshow(summed)
    plt.show()


def _visualise_predictions(input_image, heatmaps, coordinates):
    group_sizes = [17, 17]
    group_labels = ['Endocardium', 'Epicardium']

    plt.figure()

    # input image
    menpo_image = Image.init_from_channels_at_back(input_image)
    menpo_image.landmarks['predictions'] = PointCloud(coordinates)
    labeller(menpo_image, 'predictions', left_ventricle_34)
    del menpo_image.landmarks['predictions']
    rasterised_image = menpo_image.rasterize_landmarks(group='lv_34')

    ax_input_image = plt.subplot2grid((7, 14), (0, 0), colspan=6, rowspan=6)
    ax_input_image.imshow(rasterised_image.pixels_with_channels_at_back())

    index = 0
    heatmap_plots = []

    # plot individual predictions
    for i in range(len(group_sizes)):
        for j in range(group_sizes[i]):
            axis = plt.subplot2grid((7, 14), (i, 7 + j))
            axis.imshow(heatmaps[..., index])
            index += 1

            heatmap_plots.append(axis)

    add_group_labels(heatmap_plots, group_labels, group_sizes)
    make_ticklabels_invisible(heatmap_plots)

    plt.show()


def add_group_labels(heatmap_plots, group_labels, group_sizes):
    i = 0
    for group_index, label in enumerate(group_labels):
        heatmap_plots[i].set_ylabel(label)
        i += group_sizes[group_index]


def make_ticklabels_invisible(heatmap_plots):
    for i, ax in enumerate(heatmap_plots):
        ax.set_title("{}".format(i))
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)


if __name__ == '__main__':
    tf.app.run(main=main)
