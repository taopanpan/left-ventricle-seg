from pathlib import Path

import tensorflow as tf


class InferenceProvider(object):
    def __init__(self, filename, batch_size=4, num_threads=1):
        self.filename = Path(filename)
        self.image_extension = 'jpg'
        self.batch_size = batch_size
        self.num_threads = num_threads

    def get(self):
        filename_queue = tf.train.string_input_producer([str(self.filename)],
                                                        num_epochs=1,
                                                        shuffle=False)
        image = self._get_data_protobuf(filename_queue)

        return tf.train.batch(
            [image],
            self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def _get_data_protobuf(self, filename_queue):
        """ Given the filename of a protobuf, ... """

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = self._get_features(serialized_example)

        # image
        with tf.name_scope("deserialise_image"):
            image, image_height, image_width = self._image_from_feature(features)

        return image

    def _get_features(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                # images
                'image': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
            }
        )
        return features

    def _image_from_feature(self, features):
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])
        #
        image = tf.reshape(image, (image_height, image_width, 3))
        image = tf.to_float(image)

        image.set_shape([256, 256, 3])

        return image, image_height, image_width
