from pathlib import Path

import tensorflow as tf

import project.input.augmentation
from project.utils.utils import lms_to_heatmap


class DataProvider(object):
    def __init__(self, filename, n_landmarks, batch_size=1, batch_shuffling_pool=40, num_threads=1, augmentation=False,
                 flip_fn=None):
        self.filename = Path(filename)
        self.image_extension = 'jpg'
        self.n_landmarks = n_landmarks
        self.batch_size = batch_size
        self.batch_shuffling_pool = batch_shuffling_pool
        self.num_threads = num_threads
        self.augmentation = augmentation

        self.flip_fn = flip_fn

    def get(self):
        image, gt_heatmap, gt_lms, scale, marked = self._get_data_protobuf(self.filename)

        # Recommendation:
        # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
        capacity = self.batch_shuffling_pool + (self.num_threads + 2) * self.batch_size

        return tf.train.shuffle_batch(
            [image, gt_heatmap, gt_lms, scale, marked],
            self.batch_size,
            capacity=capacity,
            min_after_dequeue=self.batch_shuffling_pool,
            num_threads=self.num_threads)

    def _get_data_protobuf(self, filename):
        """ Given the filename of a protobuf, ... """
        filename_queue = tf.train.string_input_producer([str(filename)],
                                                        num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = self._get_features(serialized_example)

        # image
        with tf.name_scope("deserialise_image"):
            image, image_height, image_width = self._image_from_features(features)

        # ground truth landmarks
        with tf.name_scope("deserialise_landmarks"):
            gt_heatmaps, gt_lms, n_landmarks, visible, marked = self._heatmaps_from_features(features)

        # information
        with tf.name_scope("deserialise_info"):
            scale = self._info_from_features(features)

        # augmentation
        with tf.name_scope("image_augmentation"):
            if self.augmentation:
                gt_heatmaps, gt_lms, image, image_height, image_width = project.input.augmentation.augmentation(
                    gt_heatmaps, gt_lms, image, image_height, image_width,
                    max_scale=1.25, min_scale=0.75,
                    max_rotate=30., min_rotate=-30.,
                    flip_probability=0.5, flip_fn=self.flip_fn)

        with tf.name_scope("crop"):
            # crop to 256 * 256
            gt_heatmaps, gt_lms, image = self._crop(gt_heatmaps, gt_lms, image, image_height, image_width)

        self._set_shape(image, gt_heatmaps, gt_lms)

        return image, gt_heatmaps, gt_lms, scale, marked

    def _get_features(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                # image
                'image': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                # landmarks
                'n_landmarks': tf.FixedLenFeature([], tf.int64),
                'gt': tf.FixedLenFeature([], tf.string),
                'scale': tf.FixedLenFeature([], tf.float32),
                'visible': tf.FixedLenFeature([], tf.string),
                'marked': tf.FixedLenFeature([], tf.string),
                # original information
                'original_scale': tf.FixedLenFeature([], tf.float32),
                'original_centre': tf.FixedLenFeature([], tf.string),
                'original_lms': tf.FixedLenFeature([], tf.string),
                # inverse transformation to original landmarks
                'restore_translation': tf.FixedLenFeature([], tf.string),
                'restore_scale': tf.FixedLenFeature([], tf.float32)
            }
        )
        return features

    def _image_from_features(self, features):
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])
        #
        image = tf.reshape(image, (image_height, image_width, 3))
        image = tf.to_float(image)
        return image, image_height, image_width

    def _heatmaps_from_features(self, features):
        n_landmarks = tf.to_int32(features['n_landmarks'])
        gt_lms = tf.decode_raw(features['gt'], tf.float32)

        visible = tf.to_int32(tf.decode_raw(features['visible'], tf.int64))
        marked = tf.to_int32(tf.decode_raw(features['marked'], tf.int64))
        visible.set_shape([self.n_landmarks])
        marked.set_shape([self.n_landmarks])

        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])

        gt_lms = tf.reshape(tensor=gt_lms, shape=(n_landmarks, 2))
        gt_heatmap = lms_to_heatmap(
            gt_lms, image_height, image_width, n_landmarks, marked)
        gt_heatmap = tf.transpose(gt_heatmap, perm=[1, 2, 0])

        return gt_heatmap, gt_lms, n_landmarks, visible, marked

    def _info_from_features(self, features):
        scale = features['scale']
        return scale

    def _crop(self, gt_heatmaps, gt_lms, image, image_height, image_width):
        target_h = tf.to_int32(256)
        target_w = tf.to_int32(256)
        offset_h = tf.to_int32((image_height - target_h) / 2)
        offset_w = tf.to_int32((image_width - target_w) / 2)
        image = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, target_h, target_w)
        gt_heatmaps = tf.image.crop_to_bounding_box(
            gt_heatmaps, offset_h, offset_w, target_h, target_w)
        gt_lms -= tf.to_float(tf.stack([offset_h, offset_w]))
        return gt_heatmaps, gt_lms, image

    def _set_shape(self, image, gt_heatmap, gt_lms):
        # height x width x [RGB]
        image.set_shape([None, None, 3])

        # 2d heatmap for each landmark
        gt_heatmap.set_shape([None, None, self.n_landmarks])

        # 2d coordinates for each landmark
        gt_lms.set_shape([self.n_landmarks, 2])
