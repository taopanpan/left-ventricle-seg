import json

import tensorflow as tf

from pathlib import Path

FLAGS = tf.app.flags.FLAGS

# ---------- MODEL options --------------------------------------
tf.app.flags.DEFINE_string('model_dir', 'models/lv/lv_1hg_lr1e-3_decay10',
                           '''Directory to save and load model parameters, graph and etc.
                           If a saved model exists, its latest checkpoint is used, unless otherwise specified by the
                           `checkpoint_path` option.''')

tf.app.flags.DEFINE_string('checkpoint_path', 'models/lv/lv_2hg_lr1e-3_decay10',
                           '''Path of a specific model checkpoint to use.''')

tf.app.flags.DEFINE_integer('n_landmarks', 34,
                            '''Number of landmarks needed to annotate your data.''')

tf.app.flags.DEFINE_integer('n_features', 128,
                            '''Number of features in the hourglass.''')

tf.app.flags.DEFINE_integer('n_hourglass', 1,
                            '''Number of hourglasses to stack.''')

tf.app.flags.DEFINE_integer('n_residuals', 3,
                            '''Number of residual modules at each location in the hourglass.''')

# ---------- TRAINING options --------------------------------------
tf.app.flags.DEFINE_string('train_data', 'data/tfrecords/lv/train.tfrecords',
                           '''The .tfrecords file containing the training data.''')

tf.app.flags.DEFINE_integer('batch_size', 40, '''How many examples should be processed in each step?''')

tf.app.flags.DEFINE_integer('max_epochs', None,
                            '''The maximum number of epochs to process for training.''')

tf.app.flags.DEFINE_boolean('augmentation', True,
                            '''Whether to perform data augmentation on training images.
                            This involves rescaling, rotating and horizontally flipping the data.''')

# ---------- VALIDATION options --------------------------------------
tf.app.flags.DEFINE_integer('eval_every_n_epochs', 10,
                            '''After how many training epochs should validation be performed?''')

tf.app.flags.DEFINE_integer('patience', 50,
                            '''If the evaluation metric does not improve in this many epochs, then cease training.
                            NOTE: This also affects the number of checkpoints to save!''')

tf.app.flags.DEFINE_string('eval_data', 'data/tfrecords/lv/eval.tfrecords',
                           '''The .tfrecords file containing the validation data.''')

# ---------- HYPER-PARAMETER options --------------------------------------
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          '''The initial learning rate.''')

tf.app.flags.DEFINE_float('learning_rate_decay_rate', 0.99,
                          '''The learning rate decay factor - usually a value between 0 and 1.
                          The next learning rate is obtained by multiplication with this value.''')

tf.app.flags.DEFINE_float('learning_rate_decay_epochs', 5,
                          '''The learning rate should decay after training over this many epochs.''')

# ---------- PREDICTION options --------------------------------------
tf.app.flags.DEFINE_string('infer_data', 'data/tfrecords/lv/test.tfrecords',
                           '''The .tfrecords file containing the data to make inferences from.''')

# ---------- MISC. options --------------------------------------
tf.app.flags.DEFINE_string('device', '/gpu:0',
                           '''Device (GPU/CPU) to run on.''')

tf.app.flags.DEFINE_integer('batch_shuffling_pool', 20,
                            '''How big a buffer we will randomly sample from for batching --
                            bigger means better shuffling but slower start up and more memory used.''')

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            '''How many threads to use for preprocessing the image data.''')

 
def dump_flags():
    path = Path(FLAGS.model_dir) / 'PARAMS.json'
    path.parent.mkdir(parents=True)
    if not path.exists():
        with open(str(path), 'w') as handle:
            json.dump(tf.flags.FLAGS.__flags, handle, sort_keys=True, indent=4)
