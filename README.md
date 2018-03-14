# Universal Cnn
Universal CNN Training & Predicting For Fixed-Length Label of Images(captcha).

## Installation
    python setup.py install

## Usage
    Image Preparing:
    The name of image files should follow the format of `image_{label}.{png|jpg|jpeg}`          
    
    Training:
    ucnn --mode train --train_dir ./train_data --valid_dir ./valid_data --height 40 --width 150 --chars_length 5
    
    Predicting:
    ucnn --mode predict --test_dir ./test_data --checkpoint_dir ./training_checkpoint --height 40 --width 150 --chars_length 5

## Options
    -h, --help                        Show this help message and exit
    --mode MODE                       'train' or 'predict'
    --train_dir TRAIN_DIR             Directory where to get training images.
    --valid_dir VALID_DIR             Directory where to get validating images.
    --test_dir TEST_DIR               Directory where to get testing images.
    --checkpoint_dir CHECKPOINT_DIR   Directory where to write checkpoint.
    --batch_size BATCH_SIZE           Batch size.
    --height HEIGHT                   Image height.
    --width WIDTH                     Image width.
    --charsets CHARSETS               Possible characters of label.
    --chars_length CHARS_LENGTH       Fixed length label length.
