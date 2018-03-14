import os.path
import numpy as np
import tensorflow as tf

from PIL import Image
from datetime import datetime
from tensorflow.python.platform import gfile

from ucnn import cnn_model
from ucnn.cnn_model import CNNModel


class Predict(object):
    def __init__(self, configuration):
        self.configuration = configuration
        self.IMAGE_HEIGHT = configuration['image_height']
        self.IMAGE_WIDTH = configuration['image_width']
        self.CHARS_NUM = configuration['chars_length']
        self.CLASSES_NUM = configuration['charsets_length']
        self.CHAR_SETS = configuration['charsets']

    def one_hot_to_texts(self, recog_result, logits_result):
        texts = []
        for i in range(recog_result.shape[0]):
            index = recog_result[i]
            score = [max(l) for l in logits_result[i]]
            texts.append(
                ''.join([self.CHAR_SETS[i] for i in index]) + ' ' + str(max(score)) + " " + str(
                    sum(score) / len(score)))
        return texts

    def input_data(self, image_dir):
        if not gfile.Exists(image_dir):
            print(">> Image director '" + image_dir + "' not found.")
            return None
        extensions = ['jpg', 'jpeg', 'png']
        print(">> Looking for images in '" + image_dir + "'")
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(image_dir, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print(">> No files found in '" + image_dir + "'")
            return None
        batch_size = len(file_list)
        images = np.zeros([batch_size, self.IMAGE_HEIGHT * self.IMAGE_WIDTH], dtype='float32')
        files = []
        i = 0
        for file_name in file_list:
            image = Image.open(file_name)
            image_gray = image.convert('L')
            image_resize = image_gray.resize(size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
            image.close()
            input_img = np.array(image_resize, dtype='float32')
            input_img = np.multiply(input_img.flatten(), 1. / 255) - 0.5
            images[i, :] = input_img
            base_name = os.path.basename(file_name)
            files.append(base_name)
            i += 1
        return images, files

    def predict(self, test_dir, checkpoint):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            cm = CNNModel(self.configuration)
            input_images, input_filenames = self.input_data(test_dir)
            images = tf.constant(input_images)
            logits = cm.model(images, keep_prob=1)
            result = cnn_model.output(logits)
            saver = tf.train.Saver()
            sess = tf.Session()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
            print(tf.train.latest_checkpoint(checkpoint))
            logits_result = sess.run(logits)
            recog_result = sess.run(result)
            sess.close()
            text = self.one_hot_to_texts(recog_result, logits_result)
            total_count = len(input_filenames)
            true_count = 0.
            for i in range(total_count):
                print(input_filenames[i] + " " + text[i])
                if text[i].strip().split(" ")[0] in input_filenames[i]:
                    true_count += 1
            precision = true_count / total_count
            print('%s true/total: %d/%d recognize @ 1 = %.3f'
                  % (datetime.now(), true_count, total_count, precision))
            return text
