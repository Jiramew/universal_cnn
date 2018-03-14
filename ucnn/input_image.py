import os.path

import tensorflow as tf

from ucnn import static

RECORD_DIR = static.RECORD_DIR
TRAIN_FILE = static.TRAIN_FILE
VALID_FILE = static.VALID_FILE


def read_and_decode(filename_queue, image_height, image_width, chars_num, classes_num):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(features['image_raw'], tf.int16)
    image.set_shape([image_height * image_width])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    reshape_image = tf.reshape(image, [image_height, image_width, 1])
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label.set_shape([chars_num * classes_num])
    reshape_label = tf.reshape(label, [chars_num, classes_num])
    return tf.cast(reshape_image, tf.float32), tf.cast(reshape_label, tf.float32)


def inputs(train, batch_size, image_height, image_width, chars_num, classes_num):
    filename = os.path.join(RECORD_DIR, TRAIN_FILE if train else VALID_FILE)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename])
        image, label = read_and_decode(filename_queue, image_height, image_width, chars_num, classes_num)
        if train:
            images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                           batch_size=batch_size,
                                                           num_threads=6,
                                                           capacity=2000 + 3 * batch_size,
                                                           min_after_dequeue=2000)
        else:
            images, sparse_labels = tf.train.batch([image, label],
                                                   batch_size=batch_size,
                                                   num_threads=6,
                                                   capacity=2000 + 3 * batch_size)

        return images, sparse_labels
