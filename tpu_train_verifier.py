import numpy as np
import pandas as pd
import tensorflow as tf

from scripts import tf2_0_baseline_w_bert_translated_to_tf2_0 as tf2baseline # Oliviera's script
from scripts.tf2_0_baseline_w_bert_translated_to_tf2_0 import AnswerType
from scripts import bert_modeling as modeling
from scripts import bert_optimization
from scripts import albert_optimization
from scripts import albert

import tqdm
import json
import absl
import sys
import os

### Define Flags ###

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(absl.flags.FLAGS)

flags = absl.flags

flags.DEFINE_string(
    "model", "albert",
    "The name of model to use. Choose from ['bert', 'albert'].")

flags.DEFINE_string(
    "config_file", "models/albert_xxl/config.json",
    "The config json file corresponding to the pre-trained BERT/ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "models/albert_xxl/vocab/modified-30k-clean.model",
                    "The vocabulary file that the ALBERT/BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "output/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("train_precomputed_file", "data/verifier_train.tf_record",
                    "Precomputed tf records for training.")

flags.DEFINE_integer("train_num_precomputed", -1,
                     "Number of precomputed tf records for training.")

flags.DEFINE_string(
    "output_checkpoint_file", "albert_verifier.h5",
    "Where to save finetuned checkpoints to.")

flags.DEFINE_string(
    "output_predictions_file", "predictions.json",
    "Where to print predictions in NQ prediction format, to be passed to"
    "natural_questions.nq_eval.")

flags.DEFINE_string(
    "log_dir", "logs/",
    "Where logs, specifically Tensorboard logs, will be saved to.")

flags.DEFINE_integer(
    "log_freq", 128,
    "How many samples between each training log update.")

flags.DEFINE_string(
    "init_checkpoint", "models/bert_joint_baseline/tf2_bert_joint.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

# This should be changed to 512 at some point,
# as training was done with that value, it may
# not make a big difference though
flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 1, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 3,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 10000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "verbosity", 1, "How verbose our error messages should be")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("tpu_name", None, "Name of the TPU to use.")

flags.DEFINE_string("tpu_zone", None, "Which zone the TPU is in.")

flags.DEFINE_bool("use_one_hot_embeddings", False, "Whether to use use_one_hot_embeddings")

absl.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal NQ evaluation.")

# TODO(Edan): Look at nested contents too at some point
# Around 5% of long answers are nested, and around 50% of questions have
# long answers
# This means that this setting alone restricts us from a correct answer
# around 2.5% of the time
flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")

flags.DEFINE_integer("task_id", 0,
                     "Train and dev shard to read from and write to.")

flags.DEFINE_integer("max_contexts", 48,
                     "Maximum number of contexts to output for an example.")

flags.DEFINE_integer(
    "max_position", 50,
    "Maximum context position for which to generate special tokens.")

## Custom flags

flags.DEFINE_integer(
    "n_examples", -1,
    "Number of examples to read from files. Only applicable during testing")

flags.DEFINE_string(
    "train_file", "data/simplified-nq-train.jsonl",
    "NQ json for training. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")

## Special flags - do not change

flags.DEFINE_string(
    "predict_file", "data/simplified-nq-test.jsonl",
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")
flags.DEFINE_boolean("logtostderr", True, "Logs to stderr")
flags.DEFINE_boolean("undefok", True, "it's okay to be undefined")
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('HistoryManager.hist_file', '', 'k
￼
')

FLAGS = flags.FLAGS
FLAGS(sys.argv) # Parse the flags

VOCAB_SIZE = 30209

# https://stackoverflow.com/questions/9629179/python-counting-lines-in-a-huge-10gb-file-as-fast-as-possible
def blocks(f, size=65536):
    while True:
        b = f.read(size)
        if not b:
            break
        yield b

n_records = 0
for record in tf.compat.v1.python_io.tf_record_iterator(FLAGS.train_precomputed_file):
    n_records += 1

# with open(FLAGS.train_file, 'r') as f:
#     n_train_examples = sum([bl.count('\n') for bl in blocks(f)])

# print('# Training Examples:', n_train_examples)
print('# Training Records:', n_records)

if FLAGS.do_train and FLAGS.train_num_precomputed != n_records:
    print('Changing the number of precomuted records listed to use all avaliable data.')
    FLAGS.train_num_precomputed = n_records

### Define Functions to Build the Model ###

class TDense(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self,input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
          raise TypeError("Unable to build `TDense` layer with "
                          "non-floating point (and non-complex) "
                          "dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
          raise ValueError("The last dimension of the inputs to "
                           "`TDense` should be defined. "
                           "Found `None`.")
        last_dim = tf.compat.dimension_value(input_shape[-1])
        ### tf 2.1 rc min_ndim=3 -> min_ndim=2
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.output_size,last_dim],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            "bias",
            shape=[self.output_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        super(TDense, self).build(input_shape)

    def call(self,x):
        return tf.matmul(x,self.kernel,transpose_b=True)+self.bias

# def get_bert_model(config_file, max_seq_length):
#     """Builds and returns a BERT"""
#     config = modeling.BertConfig.from_json_file(config_file)
#     input_ids = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='input_ids')
#     input_mask = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='input_mask')
#     segment_ids = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='segment_ids')
    
#     bert_layer = modeling.BertModel(config=config, name='bert')
#     pooled_output, sequence_output = bert_layer(input_word_ids=input_ids,
#                                                 input_mask=input_mask,
#                                                 input_type_ids=segment_ids)
    
#     # Maybe try sharing the start and end logits variables
#     seq_layer = TDense(2, name='td_seq')
#     # seq_layer = tf.keras.layers.TimeDistributed(seq_layer, name='td_seq')
    
#     seq_logits = seq_layer(sequence_output)
#     start_logits, end_logits = tf.split(seq_logits, axis=-1, num_or_size_splits=2, name='split')
#     start_logits = tf.squeeze(start_logits, axis=-1, name='start_logits')
#     end_logits = tf.squeeze(end_logits, axis=-1, name='end_logits')
    
#     ans_type_layer = TDense(len(tf2baseline.AnswerType), name='ans_type_logits')
#     ans_type_logits = ans_type_layer(pooled_output)
    
#     return tf.keras.Model([input_ids, input_mask, segment_ids],
#                           [start_logits, end_logits, ans_type_logits],
#                           name='bert_baseline')

# this is the helper function to create the albert model
# config_file is used to create the model
# pretrain_ckpt is used to load the pretrain weights except for the embedding layer
def get_albert_model(config_file, max_seq_length, vocab_size, pretrain_ckpt=None):
    """ create albert model from pretrained configuration file with vocab_size changed to VOCAB_SIZE
        and optionally loads the pretrained weights
    """
    
    config = albert.AlbertConfig.from_json_file(config_file)
    config.vocab_size = vocab_size
    albert_layer = albert.AlbertModel(config=config)
    
    input_ids = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='input_ids')
    input_mask = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='input_mask')
    segment_ids = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='segment_ids')

    pooled_output, sequence_output = albert_layer(input_word_ids=input_ids,
                                                    input_mask=input_mask,
                                                    input_type_ids=segment_ids)
    
    # # Maybe try sharing the start and end logits variables
    # seq_layer = TDense(2, name='td_seq')
    # # seq_layer = tf.keras.layers.TimeDistributed(seq_layer, name='td_seq')
    
    # seq_logits = seq_layer(sequence_output)
    # start_logits, end_logits = tf.split(seq_logits, axis=-1, num_or_size_splits=2, name='split')
    # start_logits = tf.squeeze(start_logits, axis=-1, name='start_logits')
    # end_logits = tf.squeeze(end_logits, axis=-1, name='end_logits')
    
    valid_layer = TDense(2, name='valid_logits')
    valid_logits = valid_layer(pooled_output)
    
    albert_model = tf.keras.Model([input_ids, input_mask, segment_ids],
                          [valid_logits],
                          name='albert')
    
    if pretrain_ckpt:
        if 'pretrain' in pretrain_ckpt:
            load_pretrain_weights(albert_model, config_file, pretrain_ckpt, max_seq_length)
        else:
            albert_model.load_weights(pretrain_ckpt)

    return albert_model

def load_pretrain_weights(model, config_file, ckpt_file, max_seq_length):
    """loads the pretrained model's weights, except for the embedding layer,
    into the new model, which has [0:29999] loaded
    
    Args:
        model: the same model architecture as the pre-trained model except for embedding
        config_file: path to the config file to re-create the pre-trained model
        ckpt_file: path to the checkpoint of the pre-trained model
    """
    
    # re-create the pre-trained model
    config = albert.AlbertConfig.from_json_file(config_file)
    albert_layer_pretrain = albert.AlbertModel(config=config, name='albert_pretrain')

    input_ids_pretrain = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='input_ids_pretrain')
    input_mask_pretrain = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='input_mask_pretrain')
    segment_ids_pretrain = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='segment_ids_pretrain')

    pooled_output_pretrain, sequence_output_pretrain = albert_layer_pretrain(input_word_ids=input_ids_pretrain,
                                                    input_mask=input_mask_pretrain,
                                                    input_type_ids=segment_ids_pretrain)

    albert_model_pretrain = tf.keras.Model(inputs=[input_ids_pretrain,input_mask_pretrain,segment_ids_pretrain], 
           outputs=[pooled_output_pretrain, sequence_output_pretrain])
    
    # load the weights into the pre-trained model
    albert_model_pretrain.load_weights(ckpt_file)
    
    # set the pre-train weights on the new model
    albert_layer = model.get_layer('albert_model')
    albert_layer.embedding_postprocessor.set_weights(albert_layer_pretrain.embedding_postprocessor.get_weights())
    albert_layer.encoder.set_weights(albert_layer_pretrain.encoder.get_weights())
    albert_layer.pooler_transform.set_weights(albert_layer_pretrain.pooler_transform.get_weights())
    # load the embedding
    embedding_weights = albert_layer.embedding_lookup.get_weights()
    embedding_weights_pretrain = albert_layer_pretrain.embedding_lookup.get_weights()
    # the embedding weights are stored in a list of size 1, so we need to do [0] to get the actual weights
    new_embedding_weights = tf.concat([embedding_weights_pretrain[0], embedding_weights[0][30000:]],axis=0)
    # then we unsqueeze the first dimension after concat
    new_embedding_weights = tf.expand_dims(new_embedding_weights, axis=0)
    albert_layer.embedding_lookup.set_weights(new_embedding_weights)

# function that builds bert/albert from config, optionally loads the pretrain weights for albert
def build_model(model_name, config_file, max_seq_length, init_ckpt, vocab_size=VOCAB_SIZE):
    """ build model according to model_name
    
    Args:
        model_name: ['bert', 'albert']
        config_file: path to config file
        max_seq_length: the maximum length for each scan
        pretrain_ckpt: path to pretrain checkpoint (albert only)
        vocab_size: size of the new vocab, (albert only)
    Returns:
        the specified model
    """

    if model_name == 'albert':
        model = get_albert_model(config_file=config_file, 
                                 max_seq_length=max_seq_length, 
                                 pretrain_ckpt=init_ckpt,
                                 vocab_size=vocab_size)
    # elif model_name == 'bert':
    #     model = get_bert_model(config_file, max_seq_length)
    #     model.load_weights(init_ckpt)
    else:
        raise ValueError('{} is not supported'.format(model_name))
    return model

# # freeze all pretrain weights of albert
# def freeze_pretrain_weights(model):
#     """Freeze pretrain weights of the albert model.
#     """
#     albert_layer = model.get_layer('albert_model')
#     albert_layer.embedding_postprocessor.trainable = False
#     albert_layer.encoder.trainable = False
#     albert_layer.pooler_transform.trainable = False
    
def compile_model(model, model_type, learning_rate,
                  num_train_steps, num_warmup_steps):
    
    if model_type.lower() not in ('bert', 'albert'):
        raise ValueError('`model_type` must be one of the following values: ["bert", "albert"]!')
    
    # # TODO(Edan): Add a way to have no loss on this for when there is no answer
    # # Computes the loss for positions.
    # def compute_loss(positions, logits):
    #     one_hot_positions = tf.one_hot(
    #         tf.cast(positions, tf.int32), depth=FLAGS.max_seq_length, dtype=tf.float32)
    #     log_probs = tf.nn.log_softmax(logits, axis=-1)
    #     loss = -tf.reduce_mean(
    #         input_tensor=tf.reduce_sum(input_tensor=one_hot_positions * log_probs, axis=-1))
    #     return loss

    # Computes the loss for labels.
    def compute_label_loss(labels, logits):
        # one_hot_labels = tf.one_hot(
        #     tf.cast(labels, tf.int32), depth=2, dtype=tf.float32)
        labels = tf.cast(labels, tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        log_probs = tf.clip(log_probs, -100.0, 0.0)
        loss = -tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=labels * log_probs, axis=-1))
        return loss
    
    losses = {
        'valid_logits': compute_label_loss
    }
    loss_weights = {
        'valid_logits': 1.0
    }

    if model_type.lower() == 'bert':
        optimization = bert_optimization
    elif model_type.lower() == 'albert':
        optimization = albert_optimization

    optimizer = optimization.create_optimizer(learning_rate,
                                            num_train_steps,
                                            num_warmup_steps)
    
    model.compile(optimizer=optimizer,
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics=[tf.keras.metrics.categorical_accuracy])

### Build the Model ###

num_train_features = FLAGS.train_num_precomputed
num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                      FLAGS.num_train_epochs)
num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


if FLAGS.use_tpu:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu_name,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    with strategy.scope():
        model = build_model(model_name=FLAGS.model,
                            config_file=FLAGS.config_file,
                            max_seq_length=FLAGS.max_seq_length,
                            init_ckpt=FLAGS.init_checkpoint)
        compile_model(model=model,
                      model_type=FLAGS.model,
                      learning_rate=FLAGS.learning_rate,
                      num_train_steps=num_train_steps,
                      num_warmup_steps=num_warmup_steps)
else:
    model = build_model(model_name=FLAGS.model,
                        config_file=FLAGS.config_file,
                        max_seq_length=FLAGS.max_seq_length,
                        init_ckpt=FLAGS.init_checkpoint)
    compile_model(model=model,
                    model_type=FLAGS.model,
                    learning_rate=FLAGS.learning_rate,
                    num_train_steps=num_train_steps,
                    num_warmup_steps=num_warmup_steps)

print('Model generated.')
model.summary()

### Create Generator for Training Data ###

train_filenames = tf.io.gfile.glob(FLAGS.train_precomputed_file)

name_to_features = {
    "input_ids": tf.io.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
    "input_mask": tf.io.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
    "segment_ids": tf.io.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
}
if FLAGS.do_train:
    name_to_features["valid"] = tf.io.FixedLenFeature([2], tf.int64)

def decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = t

    output = ({
        'input_ids': example['input_ids'],
        'input_mask': example['input_mask'],
        'segment_ids': example['segment_ids']
    },
    {
        'valid_logits': example['valid']
    })

    return output

def data_generator(batch_size=32, seed=42, valid_frac=0.05):
    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    dataset = tf.data.TFRecordDataset(train_filenames)
    dataset = dataset.map(lambda r: decode_record(r, name_to_features))

    if valid_frac <= 0:
        dataset = dataset.shuffle(buffer_size=20000, seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
        dataset = dataset.repeat()
        return dataset, None

    train_size = int(num_train_features * (1.0 - valid_frac))

    train_dataset = dataset.take(train_size)
    valid_dataset = dataset.skip(train_size)

    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=False)
    valid_dataset = valid_dataset.batch(batch_size=batch_size, drop_remainder=False)

    train_dataset = train_dataset.shuffle(buffer_size=20000, seed=seed, reshuffle_each_iteration=True)
    valid_dataset = valid_dataset.shuffle(buffer_size=5000, seed=seed, reshuffle_each_iteration=True)

    train_dataset = train_dataset.repeat()
    valid_dataset = valid_dataset.repeat()

    return train_dataset, valid_dataset

# https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, update_freq, **kwargs):
        self._n_curr_samples = 0
        super().__init__(log_dir=log_dir, update_freq=update_freq, **kwargs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        n_steps = self._total_batches_seen + 1
        self._n_curr_samples += logs.get('size', 1)
        
        if self.update_freq != 'epoch' and self._n_curr_samples >= self.update_freq:
            logs.update({'lr': model.optimizer.lr(n_steps).numpy()})
            self._n_curr_samples = 0
        super().on_batch_end(batch, logs)

# Create training callbacks
tensorboard_callback = CustomTensorBoard(
    log_dir=FLAGS.log_dir, update_freq=FLAGS.log_freq)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(FLAGS.output_dir, FLAGS.output_checkpoint_file), monitor='val_acc', verbose=0, save_best_only=False,
    save_weights_only=True, mode='max', save_freq=FLAGS.save_checkpoints_steps)

# if not os.path.exists(FLAGS.log_dir):
#     os.makedirs(FLAGS.log_dir)

if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

### Train the Model ###

valid_frac = 0.02
train_dataset, valid_dataset = data_generator(batch_size=FLAGS.train_batch_size, valid_frac=valid_frac)
n_valid = np.ceil(FLAGS.train_num_precomputed * valid_frac)

H = model.fit(x=train_dataset,
              epochs=FLAGS.num_train_epochs,
              steps_per_epoch=int(np.ceil((FLAGS.train_num_precomputed - n_valid) / FLAGS.train_batch_size)),
              validation_data=valid_dataset,
              validation_steps=int(np.ceil(n_valid / FLAGS.train_batch_size)),
              callbacks=[ckpt_callback, tensorboard_callback])

model.save_weights(os.path.join(FLAGS.output_dir, FLAGS.output_checkpoint_file + '.final'))
