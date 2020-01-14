import tensorflow as tf

from scripts import albert
from scripts import bert_modeling as modeling

VOCAB_SIZE = 30209

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

def get_bert_model(config_file, max_seq_length):
    """Builds and returns a BERT"""
    config = modeling.BertConfig.from_json_file(config_file)
    input_ids = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='input_ids')
    input_mask = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='input_mask')
    segment_ids = tf.keras.Input(shape=(max_seq_length,),dtype=tf.int32,name='segment_ids')
    
    bert_layer = modeling.BertModel(config=config, name='bert')
    pooled_output, sequence_output = bert_layer(input_word_ids=input_ids,
                                                input_mask=input_mask,
                                                input_type_ids=segment_ids)
    
    # Maybe try sharing the start and end logits variables
    seq_layer = TDense(2, name='td_seq')
    # seq_layer = tf.keras.layers.TimeDistributed(seq_layer, name='td_seq')
    
    seq_logits = seq_layer(sequence_output)
    start_logits, end_logits = tf.split(seq_logits, axis=-1, num_or_size_splits=2, name='split')
    start_logits = tf.squeeze(start_logits, axis=-1, name='start_logits')
    end_logits = tf.squeeze(end_logits, axis=-1, name='end_logits')
    
    ans_type_layer = TDense(len(tf2baseline.AnswerType), name='ans_type_logits')
    ans_type_logits = ans_type_layer(pooled_output)
    
    return tf.keras.Model([input_ids, input_mask, segment_ids],
                          [start_logits, end_logits, ans_type_logits],
                          name='bert_baseline')

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
    
    # Maybe try sharing the start and end logits variables
    seq_layer = TDense(2, name='td_seq')
    # seq_layer = tf.keras.layers.TimeDistributed(seq_layer, name='td_seq')
    
    seq_logits = seq_layer(sequence_output)
    start_logits, end_logits = tf.split(seq_logits, axis=-1, num_or_size_splits=2, name='split')
    start_logits = tf.squeeze(start_logits, axis=-1, name='start_logits')
    end_logits = tf.squeeze(end_logits, axis=-1, name='end_logits')
    
    ans_type_layer = TDense(len(tf2baseline.AnswerType), name='ans_type_logits')
    ans_type_logits = ans_type_layer(pooled_output)
    
    albert_model = tf.keras.Model([input_ids, input_mask, segment_ids],
                          [start_logits, end_logits, ans_type_logits],
                          name='albert')
    
    if pretrain_ckpt:
        albert_model.load_weights(pretrain_ckpt)

    return albert_model

def get_albert_verifier(config_file, max_seq_length, vocab_size, pretrain_ckpt=None):
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
    
    valid_layer = TDense(2, name='valid_logits')
    valid_logits = valid_layer(pooled_output)
    
    albert_model = tf.keras.Model([input_ids, input_mask, segment_ids],
                          [valid_logits],
                          name='albert_verifier')
    
    if pretrain_ckpt:
        albert_model.load_weights(pretrain_ckpt)

    return albert_model

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
    elif model_name == 'bert':
        model = get_bert_model(config_file, max_seq_length)
        model.load_weights(init_ckpt)
    elif model_name == 'verifier':
        model = get_albert_verifier(config_file=config_file, 
                                    max_seq_length=max_seq_length, 
                                    pretrain_ckpt=init_ckpt,
                                    vocab_size=vocab_size)
    else:
        raise ValueError('{} is not supported'.format(model_name))
    return model