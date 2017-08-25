from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
from pathlib import Path

import menpo.io as mio
import tensorflow as tf


def tfrecords_count(tfrecords_file):
    """    
    :param tfrecords_file: The path of a .tfrecords file
    :return: The number of serialised examples contained in `tfrecords`
    """
    count = 0
    file = str(Path(tfrecords_file))
    for record in tf.python_io.tf_record_iterator(file):
        count += 1
    return count


def int_feature(value):
    """    
    :param value: 
    :return: 
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """    
    :param value: 
    :return: 
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """    
    :param value: 
    :return: 
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def jpg_feature(image):
    """    
    :param image: 
    :return: 
    """
    fp = BytesIO()
    mio.export_image(image, fp, extension='jpg')
    fp.seek(0)
    jpg_bytes = fp.read()
    return bytes_feature(jpg_bytes)
