from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.ndimage
import tensorflow as tf
from menpo.transform import Translation, Scale


def crop_image(img, center, scale, res, base=384):
    h = base * scale

    t = Translation(
        [
            res[0] * (-center[0] / h + .5),
            res[1] * (-center[1] / h + .5)
        ]) \
        .compose_after(
        Scale(
            (res[0] / h,
             res[1] / h)
        )).pseudoinverse()

    # Upper left point of original image
    ul = np.floor(t.apply([0, 0]))
    # Bottom right point of original image
    br = np.ceil(t.apply(res).astype(np.int))

    # crop and rescale
    cimg, trans = img.warp_to_shape(
        br - ul,
        Translation(-(br - ul) / 2 + (br + ul) / 2),
        return_transform=True)

    c_scale = np.min(cimg.shape) / np.mean(res)
    new_img = cimg.rescale(1 / c_scale).resize(res)

    return new_img, trans, c_scale


def lms_to_heatmap(lms, h, w, n_landmarks, marked_index):
    xs, ys = tf.meshgrid(tf.range(0., tf.to_float(w)),
                         tf.range(0., tf.to_float(h)))
    sigma = 5.
    normalisation = (1. / (sigma * np.sqrt(2. * np.pi)))

    def heatmap_fn(lms):
        y, x, idx = tf.unstack(lms)
        idx = tf.to_int32(idx)

        def gaussian():
            gaussian_hm = tf.exp(-0.5 * (tf.pow(ys - y, 2) + tf.pow(xs - x, 2)) * tf.pow(1. / sigma, 2.)) \
                          * normalisation * 17.
            return gaussian_hm

        def blank():
            blank_hm = tf.zeros((h, w))
            return blank_hm

        # depending on the marked index, return either a gaussian around the landmark, or a blank heatmap
        is_marked = tf.equal(marked_index[idx], 1)

        return tf.cond(is_marked, gaussian, blank)

    # stack the heatmaps for each landmark
    img_hm = tf.stack(
        tf.map_fn(heatmap_fn,
                  tf.concat(axis=1,
                            values=[lms, tf.to_float(tf.range(0, n_landmarks))[..., None]])))

    return img_hm


def get_coordinates(heatmap_predictions):
    """ Converts the n_landmarks predicted landmark heatmaps
        into n_landmarks predicted (x,y) landmark coordinates """

    def gaussian_blur(heatmaps):
        bsize, h, w, n_ch = heatmaps.shape
        lms_hm_prediction_filter = np.stack(list(map(
            lambda x: scipy.ndimage.filters.gaussian_filter(*x),
            # apply gaussian conv filters to each heatmap (with variance=3?)

            # (batch size / #channels / height / width) , [3,3,3,3,3,3,3...]
            # get bsize * n_ch heatmaps of size [h x w], zip up with 3's
            zip(heatmaps.transpose(0, 3, 1, 2).reshape(-1, h, w), [3] * (bsize * n_ch)))))

        # revert back to [bsize, h, w, n_ch]
        result = lms_hm_prediction_filter.reshape(
            bsize, n_ch, h, w).transpose(0, 2, 3, 1)

        return np.array(result).astype(np.float32)

    # apply Gaussian blur to each heatmap
    with tf.name_scope('blur'):
        blurred_predictions, = tf.py_func(func=gaussian_blur, inp=[heatmap_predictions], Tout=[tf.float32])

    with tf.name_scope('extract_coordinates'):
        # dimensions are now [0:bsize, 1:h, 2:w, 3:n_ch]
        # find the most likely x's and y's (i.e. the most confidently predicted)

        x_predictions = tf.argmax(tf.reduce_max(blurred_predictions, 2), 1)
        y_predictions = tf.argmax(tf.reduce_max(blurred_predictions, 1), 1)
        xy_predictions = tf.transpose(
            tf.to_float(tf.stack([x_predictions, y_predictions])),
            perm=[1, 2, 0])

    return xy_predictions


def heatmap_weights(keypoints, mask=None, ng_w=0.01, ps_w=1.0):
    """ Returns a tensor of weights to be applied for the loss function
    If a pixel has non-zero probability of being a landmark, it has weight ps_w
    Otherwise, if a pixel has zero probability of being a landmark, it has weight ng_w
    """

    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))

    weights = tf.where(
        condition=is_background,
        x=ones * ng_w,
        y=ones * ps_w)

    if mask is not None:
        weights *= tf.to_float(mask)

    return weights


def put_kernels_on_grid(kernel, pad=1):
    """
    from: https://gist.github.com/kukuruza/03731dc494603ceab0c5

    Visualize conv. features as an image (mostly for the 1st layer).
    Place kernels into a grid, with some padding between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    """

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(np.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))

    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
    print('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7
