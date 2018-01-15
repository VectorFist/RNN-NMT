import model_train
import model_infer
import codecs
import tensorflow as tf


class HParams(object):
    def __init__(self):
        self.src_train_file = 'nmt_data_en_ch/train_english.txt'
        self.tgt_train_file = 'nmt_data_en_ch/train_chinese.txt'
        self.src_test_file = 'nmt_data_en_ch/dev_english.txt'
        self.tgt_test_file = 'nmt_data_en_ch/dev_chinese.txt'
        self.src_vocab_file = 'nmt_data_en_ch/english_vocab.txt'
        self.tgt_vocab_file = 'nmt_data_en_ch/chinese_vocab.txt'
        self.src_vocab_size = self.compute_vocab_size(self.src_vocab_file)
        self.tgt_vocab_size = self.compute_vocab_size(self.tgt_vocab_file)
        self.model_dir = 'nmt_model_en_ch'

        self.sos = '<s>'
        self.eos = '</s>'
        self.unk = '</unk>'
        self.src_max_len = 50
        self.tgt_max_len = 50
        self.src_max_len_infer = 50
        self.tgt_max_len_infer = 50

        self.num_units = 128
        self.num_layers = 6
        self.encoder_type = 'bi'   # {bi, uni}

        self.optimizer = 'sgd'     # {sgd, adam}
        self.learning_rate = 1.0
        self.num_train_steps = 20000
        self.init_weight = 0.1

        self.unit_type = 'lstm'    # {lstm ,gru}
        self.forget_bias = 1.0
        self.dropout = 0.2
        self.max_gradient_norm = 5.0
        self.batch_size = 128
        self.infer_batch_size = 40
        self.beam_width = 5
        self.length_penalty_weight = 0.0

        self.steps_per_stats = 100
        self.num_gpus = 1

    def compute_vocab_size(self, vocab_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            vocab_size = 0
            for word in f:
                vocab_size += 1
        return vocab_size


def main():
    hparams = HParams()
    model_train.train(hparams)

    string = '待翻译文本'
    translation = model_infer.translate(hparams, string)
    print(translation)


if __name__ == '__main__':
    main()
