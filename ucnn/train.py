import time
import tensorflow as tf
from datetime import datetime

from ucnn import cnn_model
from ucnn.cnn_model import CNNModel


def train(batch_size_, checkpoint, configuration):
    with tf.Graph().as_default():
        cm = CNNModel(configuration)
        images, labels = cm.inputs(train=True, batch_size=batch_size_)
        logits = cm.model(images, keep_prob=0.5)
        loss = cnn_model.loss(logits, labels)
        train_op = cnn_model.training(loss)

        if tf.train.latest_checkpoint('training_checkpoint') is not None:
            saver = tf.train.Saver(tf.global_variables())
            sess = tf.Session()
            saver.restore(sess, tf.train.latest_checkpoint('training_checkpoint'))
        else:
            saver = tf.train.Saver(tf.global_variables())
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess = tf.Session()
            sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 1
        try:
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                if step % 10 == 0:
                    print('>> Step %d run_train: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                if step % 100 == 0:
                    print('>> %s Saving in %s' % (datetime.now(), checkpoint))
                    saver.save(sess, checkpoint, global_step=step)
                if step % 100000 == 0:
                    print("continue or break?")
                    corb = input("please input c or b: ")
                    if corb == "c":
                        print('>> %s Saving in %s' % (datetime.now(), checkpoint))
                        saver.save(sess, checkpoint, global_step=step)
                    else:
                        break
                step += 1
        except Exception as e:
            print('>> %s Saving in %s' % (datetime.now(), checkpoint))
            saver.save(sess, checkpoint, global_step=step)
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
