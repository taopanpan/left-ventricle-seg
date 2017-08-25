from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class StackedHourglassNetwork(object):
    def __init__(self, n_landmarks, n_hourglass=2, n_residuals=3, n_features=256, training=False):

        self.n_hourglass = n_hourglass
        self.n_residuals = n_residuals
        self.n_landmarks = n_landmarks
        self.n_features = n_features

        self.training = training

    def model(self, inputs):
        """ Builds a Stacked Hourglass network, using `n_hourglass` modules """

        # Keep track of all (final and intermediate) predictions
        all_predictions = []

        # Initial processing of the images
        with tf.name_scope('initial_processing'):
            inter = self._initial_processing(inputs)

        # Stack the hourglasses!
        for i in range(self.n_hourglass):
            with tf.name_scope('hourglass'):
                hg = self._hourglass(inter, 4)

            with tf.name_scope('between_hourglass'):
                # Residual layers
                ll = self._residuals(hg)

                # Linear layer to produce first set of predictions
                ll = self._linear_layer(ll)

                with tf.name_scope('make_predictions'):
                    predictions = self._make_predictions(ll)
                    all_predictions.append(predictions)

                # Add predictions back for the next hourglass
                if i < (self.n_hourglass - 1):
                    with tf.name_scope('reintegrate_predictions'):
                        # down-sample
                        predictions_ = tf.layers.conv2d(inputs=predictions,
                                                        filters=self.n_landmarks,
                                                        kernel_size=[4, 4],
                                                        strides=4)

                        # back into feature space
                        predictions_ = tf.layers.conv2d(inputs=predictions_,
                                                        filters=self.n_features,
                                                        kernel_size=[1, 1],
                                                        strides=1)

                    ll_ = tf.layers.conv2d(inputs=ll,
                                           filters=self.n_features,
                                           kernel_size=[1, 1],
                                           strides=1)

                    # add together:
                    #   1: the predictions we just made
                    #   2: features output from last hourglass
                    #   3: features output from the hourglass before that
                    inter = tf.add_n([predictions_, ll_, inter])

        return all_predictions

    def _initial_processing(self, inputs):
        """ Initial processing before heading into the hourglasses """
        cnv1 = tf.layers.conv2d(inputs=inputs,
                                filters=64,
                                kernel_size=[7, 7],
                                strides=2,
                                padding='same')
        cnv1 = tf.layers.batch_normalization(inputs=cnv1, training=self.training)
        cnv1 = tf.nn.relu(cnv1)

        r1 = self._residual(cnv1, channels_out=128)

        pool = tf.layers.max_pooling2d(inputs=r1,
                                       pool_size=[2, 2],
                                       strides=2,
                                       padding='same')

        r4 = self._residual(pool, channels_out=128)
        r5 = self._residual(r4, channels_out=self.n_features)

        return r5

    def _hourglass(self, inputs, level):
        """ Recursively constructs an hourglass module, whose size depends on the initial value of level """
        # Upper branch
        with tf.name_scope('upper_branch'):
            up1 = self._residuals(inputs)

        # Lower branch
        with tf.name_scope('lower_branch'):
            low1 = tf.layers.max_pooling2d(inputs=inputs,
                                           pool_size=[2, 2],
                                           strides=2,
                                           padding='same')
            low1 = self._residuals(low1)

            if level > 1:
                low2 = self._hourglass(low1, level - 1)
            else:
                low2 = self._residuals(low1)

            low3 = self._residuals(low2)
            up2 = tf.layers.conv2d_transpose(inputs=low3,
                                             filters=self.n_features,
                                             kernel_size=[2, 2],
                                             strides=2,
                                             padding='same')
        # Bring the two branches together
        return tf.add(up1, up2)

    def _linear_layer(self, inputs):
        ll = tf.layers.conv2d(inputs=inputs,
                              filters=self.n_features,
                              kernel_size=[1, 1],
                              strides=1)
        ll = tf.layers.batch_normalization(inputs=ll, training=self.training)
        ll = tf.nn.relu(ll)
        return ll

    def _make_predictions(self, inputs):
        """ Generates a set of heatmaps for each landmark """
        # Generate low-resolution heatmaps
        predictions_ = tf.layers.conv2d(inputs=inputs,
                                        filters=self.n_landmarks,
                                        kernel_size=[1, 1],
                                        strides=1)

        # Bring the heatmaps up to the resolution of the original input images
        predictions = tf.layers.conv2d_transpose(
            inputs=predictions_,
            filters=self.n_landmarks,
            kernel_size=4,
            strides=4,
            activation=None,
            padding='same'
        )
        return predictions

    def _residuals(self, inputs):
        """ Back-to-back residual blocks """
        net = inputs
        with tf.name_scope('residuals'):
            for _ in range(self.n_residuals):
                net = self._residual(net, self.n_features)
        return net

    def _residual(self, inputs, channels_out):
        """ A residual bottleneck block, as seen in ResNet-152 """
        channels_in = tf.shape(inputs)[3]

        with tf.name_scope('residual'):
            with tf.name_scope('bottleneck'):
                # reduce dimensionality
                net = tf.layers.batch_normalization(inputs=inputs, training=self.training)
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(inputs=net,
                                       filters=channels_out / 2,
                                       kernel_size=[1, 1],
                                       strides=1)

                net = tf.layers.batch_normalization(inputs=net, training=self.training)
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(inputs=net,
                                       filters=channels_out / 2,
                                       kernel_size=[3, 3],
                                       strides=1,
                                       padding='same')

                # increase dimensionality
                net = tf.layers.batch_normalization(inputs=net, training=self.training)
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(inputs=net,
                                       filters=channels_out,
                                       kernel_size=[1, 1],
                                       strides=1)

            with tf.name_scope('skip'):
                def identity(): return tf.identity(inputs)

                def change_dimensionality(): return tf.layers.conv2d(inputs=inputs,
                                                                     filters=channels_out,
                                                                     kernel_size=[1, 1],
                                                                     strides=1)

                skip = tf.cond(tf.equal(channels_in, channels_out), identity, change_dimensionality)

            # skip connection
            residual = tf.add(net, skip)

        return residual
