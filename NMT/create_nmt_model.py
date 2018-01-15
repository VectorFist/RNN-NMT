import tensorflow as tf
from collections import namedtuple
from tensorflow.python.ops import lookup_ops

from nmt_model import NMTModel
from iterator import get_iterator
from iterator import get_infer_iterator

SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
UNK_ID = 0
TrainModel = namedtuple('TrainModel', ['graph', 'model', 'iterator'])
EvalModel = namedtuple('EvalModel', ['graph', 'model', 'src_file_placeholder',
                                     'tgt_file_placeholder', 'iterator'])
InferModel = namedtuple('InferModel', ['graph', 'model', 'src_placeholder',
                                       'batch_size_placeholder', 'iterator'])


def create_train_model(hparams):
    src_file = hparams.src_train_file
    tgt_file = hparams.tgt_train_file
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file
    graph = tf.Graph()
    with graph.as_default(), tf.container('train'):
        src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)
        iterator = get_iterator(src_dataset, tgt_dataset, src_vocab_table,
                                tgt_vocab_table, hparams.batch_size, SOS, EOS,
                                src_max_len=hparams.src_max_len,
                                tgt_max_len=hparams.tgt_max_len)
        model = NMTModel(hparams, 'train', iterator, src_vocab_table, tgt_vocab_table)
        return TrainModel(graph=graph, model=model, iterator=iterator)


def create_eval_model(hparams):
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file
    graph = tf.Graph()
    with graph.as_default(), tf.container('eval'):
        src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
        src_file_placeholder = tf.placeholder(shape=[], dtype=tf.string)
        tgt_file_placeholder = tf.placeholder(shape=[], dtype=tf.string)
        src_dataset = tf.data.TextLineDataset(src_file_placeholder)
        tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
        iterator = get_iterator(src_dataset, tgt_dataset, src_vocab_table,
                                tgt_vocab_table, hparams.batch_size, SOS, EOS,
                                src_max_len=hparams.src_max_len,
                                tgt_max_len=hparams.tgt_max_len)
        model = NMTModel(hparams, 'eval', iterator, src_vocab_table, tgt_vocab_table)
        return EvalModel(graph=graph, model=model, src_file_placeholder=src_file_placeholder,
                         tgt_file_placeholder=tgt_file_placeholder, iterator=iterator)


def create_infer_model(hparams):
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file
    graph = tf.Graph()
    with graph.as_default(), tf.container('infer'):
        src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(tgt_vocab_file,
                                                                             default_value=UNK)
        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)
        iterator = get_infer_iterator(src_dataset, src_vocab_table, batch_size_placeholder,
                                      EOS, src_max_len=hparams.src_max_len_infer)
        model = NMTModel(hparams, 'infer', iterator, src_vocab_table, tgt_vocab_table,
                         reverse_tgt_vocab_table)
        return InferModel(graph=graph, model=model, src_placeholder=src_placeholder,
                          batch_size_placeholder=batch_size_placeholder, iterator=iterator)
