import sys
import argparse


def handle_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='train or predict.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./train_data',
        help='Directory where to get training images.'
    )
    parser.add_argument(
        '--valid_dir',
        type=str,
        default='./valid_data',
        help='Directory where to get validating images.'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='./test_data',
        help='Directory where to get testing images.'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./training_checkpoint/checkpoint',
        help='Directory where to write checkpoint.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size.'
    )
    parser.add_argument(
        '--height',
        type=int,
        help='Image height.'
    )
    parser.add_argument(
        '--width',
        type=int,
        help='Image width.'
    )
    parser.add_argument(
        '--charsets',
        type=str,
        default='abcdefghijkmnpqrstuvwxyz123456789ABCDEFGHIJKLMNPQRSTUVWXYZ',
        help='Possible characters of label.'
    )
    parser.add_argument(
        '--chars_length',
        type=int,
        help='Fixed length label length.'
    )
    return parser.parse_args(argv)


def main():
    argv = sys.argv[1:]
    args = handle_arguments(argv)

    mode = args.mode

    train_dir = args.train_dir
    valid_dir = args.valid_dir
    test_dir = args.test_dir

    checkpoint_dir = args.checkpoint_dir
    batch_size = args.batch_size

    image_height = args.height
    image_width = args.width
    charsets = args.charsets
    chars_length = args.chars_length
    charsets_length = len(charsets)

    configuration = {"image_height": image_height,
                     "image_width": image_width,
                     "charsets": charsets,
                     "charsets_length": charsets_length,
                     "chars_length": chars_length}

    if mode == "train":
        print("Using Argument for Training. mode {0}, "
              "train_dir {1}, valid_dir {2}, checkpoint_dir {3}, batch_size {4}".format(mode,
                                                                                        train_dir,
                                                                                        valid_dir,
                                                                                        checkpoint_dir,
                                                                                        batch_size))

        from ucnn.train import train
        from ucnn.record import Record
        rd = Record(configuration)
        rd.main(train_dir, valid_dir)
        print("Load Image files as TFRecord.")
        train(batch_size, checkpoint_dir, configuration)

    elif mode == "predict":
        print("Using Argument for Predicting. mode {0}, test_dir {1}, checkpoint_dir {2}".format(mode,
                                                                                                 train_dir,
                                                                                                 valid_dir,
                                                                                                 test_dir,
                                                                                                 checkpoint_dir,
                                                                                                 batch_size))
        from ucnn.predict import Predict
        pred = Predict(configuration)

        print("Predicting...")
        pred.predict(test_dir, checkpoint_dir)


if __name__ == '__main__':
    main()
