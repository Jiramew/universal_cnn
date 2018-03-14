import os.path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.platform import gfile

from ucnn import static

RECORD_DIR = static.RECORD_DIR
TRAIN_FILE = static.TRAIN_FILE
VALID_FILE = static.VALID_FILE


class Record(object):
    def __init__(self, configuration):
        self.IMAGE_HEIGHT = configuration['image_height']
        self.IMAGE_WIDTH = configuration['image_width']
        self.CHARS_NUM = configuration['chars_length']
        self.CLASSES_NUM = configuration['charsets_length']
        self.CHAR_SETS = configuration['charsets']

    def main(self, train_dir, valid_dir):
        training_data = self.create_data_list(train_dir)
        self.conver_to_tfrecords(training_data, TRAIN_FILE)

        validation_data = self.create_data_list(valid_dir)
        self.conver_to_tfrecords(validation_data, VALID_FILE)

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def label_to_one_hot(self, label):
        one_hot_label = np.zeros([self.CHARS_NUM, self.CLASSES_NUM])
        offset = []
        index = []
        for i, c in enumerate(label):
            offset.append(i)
            # print(c)
            index.append(self.CHAR_SETS.index(c))
        one_hot_index = [offset, index]
        one_hot_label[one_hot_index] = 1.0
        return one_hot_label.astype(np.uint8)

    def conver_to_tfrecords(self, data_set, name):
        """转换成 tfrecords."""
        if not os.path.exists(RECORD_DIR):
            os.makedirs(RECORD_DIR)
        filename = os.path.join(RECORD_DIR, name)
        print('>> Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        data_set = list(data_set)
        num_examples = len(data_set)
        for index in range(num_examples):
            image = data_set[index][0]
            height = image.shape[0]
            width = image.shape[1]
            image_raw = image.tostring()
            label = data_set[index][1]
            label_raw = self.label_to_one_hot(label).tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': self._int64_feature(height),
                'width': self._int64_feature(width),
                'label_raw': self._bytes_feature(label_raw),
                'image_raw': self._bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
        print('>> Writing Done!')

    def create_data_list(self, image_dir):
        if not gfile.Exists(image_dir):
            print("Image director '" + image_dir + "' not found.")
            return None
        extensions = ['jpg', 'jpeg', 'png']
        print("Looking for images in '" + image_dir + "'")
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(image_dir, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print("No files found in '" + image_dir + "'")
            return None
        images = []
        labels = []
        for file_name in file_list:
            image = Image.open(file_name)
            image_gray = image.convert('L')
            image_resize = image_gray.resize(size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
            input_img = np.array(image_resize, dtype='int16')
            image.close()
            label_name = os.path.basename(file_name).split('.')[0].split('_')[1]
            images.append(input_img)
            labels.append(label_name)
        return zip(images, labels)
