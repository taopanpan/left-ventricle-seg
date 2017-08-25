import numpy as np
import tensorflow as tf


def augmentation(gt_heatmaps, gt_lms, image, image_height, image_width, max_scale=1, min_scale=1, max_rotate=0,
                 min_rotate=0, flip_probability=0, flip_fn=None):
    if flip_fn is None:
        flip_fn = _default_flip_fn

    max_scale = float(max_scale)
    min_scale = float(min_scale)
    max_rotate = float(max_rotate)
    min_rotate = float(min_rotate)
    flip_probability = float(flip_probability)

    # rescale
    if min_scale != 1 and max_scale != 1:
        scale_factor = min_scale + tf.random_uniform([1]) * (max_scale - min_scale)
        if scale_factor != 1:
            gt_heatmaps, gt_lms, image, image_height, image_width = _scale(scale_factor[0], gt_heatmaps, gt_lms, image,
                                                                           image_height, image_width)
    # rotate
    if min_rotate != 1 and max_rotate != 1:
        rotation_angle = min_rotate + (tf.random_uniform([1]) * (max_rotate - min_rotate))
        rotation_angle = rotation_angle * (np.pi / 180)

        if rotation_angle != 0:
            gt_heatmaps, gt_lms, image = _rotate(rotation_angle, gt_heatmaps, gt_lms, image)

    # flip
    if flip_probability > 0:
        do_flip = np.random.uniform([1])

        if do_flip > flip_probability:
            gt_heatmaps, gt_lms, image = _flip(flip_fn, gt_heatmaps, gt_lms, image)

    return gt_heatmaps, gt_lms, image, image_height, image_width


def _scale(scale_factor, gt_heatmaps, gt_lms, image, image_height, image_width):
    image_height = tf.to_int32(tf.to_float(image_height) * scale_factor)
    image_width = tf.to_int32(tf.to_float(image_width) * scale_factor)
    image = tf.image.resize_images(image, tf.stack([image_height, image_width]))
    gt_heatmaps = tf.image.resize_images(gt_heatmaps, tf.stack([image_height, image_width]))
    gt_lms *= scale_factor
    return gt_heatmaps, gt_lms, image, image_height, image_width


def _rotate(angle, gt_heatmaps, gt_lms, image):
    image = rotate_image_tensor(image, angle)
    gt_heatmaps = rotate_image_tensor(gt_heatmaps, angle)
    gt_lms = rotate_points_tensor(gt_lms, image, angle)
    return gt_heatmaps, gt_lms, image


def _flip(flip_fn, gt_heatmaps, gt_lms, image):
    gt_heatmaps, gt_lms, image = flip_fn(image, gt_heatmaps, gt_lms)
    return gt_heatmaps, gt_lms, image


def rotate_points_tensor(points, image, angle):
    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # center coordinates since rotation center is supposed to be in the image center
    points_centered = points - image_center

    rot_matrix = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(angle), -tf.sin(angle), tf.sin(angle), tf.cos(angle)])
    rot_matrix = tf.reshape(rot_matrix, shape=[2, 2])

    points_centered_rot = tf.matmul(rot_matrix, tf.transpose(points_centered))

    return tf.transpose(points_centered_rot) + image_center


def rotate_image_tensor(image, angle):
    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # Coordinates of new image
    xs, ys = tf.meshgrid(tf.range(0., tf.to_float(s[1])), tf.range(0., tf.to_float(s[0])))
    coords_new = tf.reshape(tf.stack([ys, xs], 2), [-1, 2])

    # center coordinates since rotation center is supposed to be in the image center
    coords_new_centered = tf.to_float(coords_new) - image_center

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.stack(
        [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(
        rot_mat_inv, tf.transpose(coords_new_centered))
    coord_old = tf.to_int32(tf.round(
        tf.transpose(coord_old_centered) + image_center))

    # Find nearest neighbor in old image
    coord_old_y, coord_old_x = tf.unstack(coord_old, axis=1)

    # Clip values to stay inside image coordinates
    outside_y = tf.logical_or(tf.greater(
        coord_old_y, s[0] - 1), tf.less(coord_old_y, 0))
    outside_x = tf.logical_or(tf.greater(
        coord_old_x, s[1] - 1), tf.less(coord_old_x, 0))
    outside_ind = tf.logical_or(outside_y, outside_x)

    inside_mask = tf.logical_not(outside_ind)
    inside_mask = tf.tile(tf.reshape(inside_mask, s[:2])[..., None], tf.stack([1, 1, s[2]]))

    coord_old_y = tf.clip_by_value(coord_old_y, 0, s[0] - 1)
    coord_old_x = tf.clip_by_value(coord_old_x, 0, s[1] - 1)
    coord_flat = coord_old_y * s[1] + coord_old_x

    im_flat = tf.reshape(image, tf.stack([-1, s[2]]))
    rot_image = tf.gather(im_flat, coord_flat)
    rot_image = tf.reshape(rot_image, s)

    return tf.where(inside_mask, rot_image, tf.zeros_like(rot_image))


def catface_flip_fn(image, gt_heatmaps, gt_lms):
    image = tf.image.flip_left_right(image)
    gt_heatmaps = tf.image.flip_left_right(gt_heatmaps)

    flip_hm_list = []
    flip_lms_list = []
    for idx in [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,\
                33, 32,31,30,29,28,27,26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        flip_hm_list.append(gt_heatmaps[:, :, idx])
        flip_lms_list.append(gt_lms[idx, :])

    gt_heatmaps = tf.stack(flip_hm_list, axis=2)
    gt_lms = tf.stack(flip_lms_list)

    return gt_heatmaps, gt_lms, image


def catpose_flip_fn(image, gt_heatmaps, gt_lms):
    image = tf.image.flip_left_right(image)
    gt_heatmaps = tf.image.flip_left_right(gt_heatmaps)

    flip_hm_list = []
    flip_lms_list = []
    for idx in [
        # HEAD
        2, 0, 1, 3, 4,
        # TORSO
        5, 6, 7,
        # FRONT LEFT LEG
        11, 12, 13,
        # FRONT RIGHT LEG
        8, 9, 10,
        # HIND LEFT LEG
        17, 18, 19,
        # HIND RIGHT LEG
        14, 15, 16,
        # TAIL
        20, 21, 22, 23]:
        flip_hm_list.append(gt_heatmaps[:, :, idx])
        flip_lms_list.append(gt_lms[idx, :])

    gt_heatmaps = tf.stack(flip_hm_list, axis=2)
    gt_lms = tf.stack(flip_lms_list)

    return gt_heatmaps, gt_lms, image


def _default_flip_fn(image, gt_heatmaps, gt_lms):
    image = tf.image.flip_left_right(image)
    gt_heatmaps = tf.image.flip_left_right(gt_heatmaps)

    return gt_heatmaps, gt_lms, image
