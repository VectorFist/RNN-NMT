import tensorflow as tf
import time
from tensorflow.python.layers import core as layers_core


class NMTModel(object):
    def __init__(self, hparams, mode, iterator, source_vocab_table,
                 target_vocab_table, reverse_target_vocab_table=None):
        self.iterator = iterator
        self.mode = mode
        self.src_vocab_table = source_vocab_table
        self.tgt_vocab_table = target_vocab_table

        self.src_vocab_size = hparams.src_vocab_size
        self.tgt_vocab_size = hparams.tgt_vocab_size

        self.num_layers = hparams.num_layers

        initializer = tf.random_uniform_initializer(-hparams.init_weight, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        self.embedding_encoder, self.embedding_decoder = self.init_embedding(hparams.num_units)
        self.batch_size = tf.size(self.iterator.source_sequence_length)

        with tf.variable_scope('decoder/output_projection'):
            self.output_layer = layers_core.Dense(hparams.tgt_vocab_size, use_bias=False,
                                                  name='output_projection')

        graph = self.build_graph(hparams)
        if self.mode == 'train':
            self.train_loss = graph[1]
            self.word_count = tf.reduce_sum(self.iterator.source_sequence_length) + \
                              tf.reduce_sum(self.iterator.target_sequence_length)
        elif self.mode == 'eval':
            self.eval_loss = graph[1]
        elif self.mode == 'infer':
            self.infer_logits, _, self.final_context_state, self.sample_id = graph
            self.sample_words = reverse_target_vocab_table.lookup(tf.to_int64(self.sample_id))

        if self.mode != 'infer':
            self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)

        self.global_step = tf.Variable(0, trainable=False)
        train_variables = tf.trainable_variables()

        if self.mode == 'train':
            self.learning_rate = tf.constant(hparams.learning_rate)
            self.learning_rate = self._learning_rate_decay(hparams.num_train_steps)
            if hparams.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif hparams.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.train_loss, train_variables, colocate_gradients_with_ops=True)
            clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)
            self.grad_norm = gradient_norm

            self.train_opt = optimizer.apply_gradients(zip(clipped_gradients, train_variables),
                                                       global_step=self.global_step)
            self.train_summary = tf.summary.merge([
                tf.summary.scalar('learning_rate', self.learning_rate),
                tf.summary.scalar('train_loss', self.train_loss)])
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        count_train_variables = self._count_train_variables(train_variables)
        print('='*5, 'build %s nmt model done' % self.mode, '='*5)
        print('='*5, 'total train variables count: {:d}'.format(count_train_variables), '='*5)

    def reinitializer_or_load_model(self, sess, model_dir, load=True):
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        if latest_ckpt and load:
            start_time = time.time()
            self.saver.restore(sess, latest_ckpt)
            sess.run(tf.tables_initializer())
            print('load model from {:s}, time {:.3f}s'.format(latest_ckpt, time.time()-start_time))
        else:
            start_time = time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            print('reinirializer model, time {:.3f}s'.format(time.time() - start_time))
        global_step = self.global_step.eval(session=sess)
        return global_step

    def init_embedding(self, embed_size):
        with tf.variable_scope('embeddings', dtype=tf.float32):
            with tf.variable_scope('encoder'):
                embedding_encoder = tf.get_variable('embedding_encoder',
                                                    [self.src_vocab_size, embed_size])

            with tf.variable_scope('decoder'):
                embedding_decoder = tf.get_variable('embedding_decoder',
                                                    [self.tgt_vocab_size, embed_size])
        return embedding_encoder, embedding_decoder

    def train(self, sess):
        assert self.mode == 'train'
        return sess.run([self.train_opt, self.train_loss, self.predict_count,
                         self.train_summary, self.global_step, self.word_count,
                         self.batch_size, self.grad_norm, self.learning_rate])

    def eval(self, sess):
        assert self.mode == 'eval'
        return sess.run([self.eval_loss, self.predict_count, self.batch_size])

    def infer(self, sess):
        assert self.mode == 'infer'
        return sess.run([self.infer_logits, self.sample_id, self.sample_words])

    def decode(self, sess):
        _, _, sample_words = self.infer(sess)

        sample_words = sample_words.transpose()
        return sample_words

    def build_graph(self, hparams):
        with tf.variable_scope('dynamic_seq2seq', dtype=tf.float32):
            encoder_outputs, encoder_state = self._build_encoder(hparams)
            logits, sample_id, final_context_state = self._build_decoder(
                encoder_outputs, encoder_state, hparams)
            if self.mode != 'infer':
                loss = self._compute_loss(logits)
            else:
                loss = None
        return logits, loss, final_context_state, sample_id

    def _build_encoder(self, hparams):
        num_layers = hparams.num_layers
        source = tf.transpose(self.iterator.source)

        with tf.variable_scope('encoder') as scope:
            encoder_emb_in = tf.nn.embedding_lookup(self.embedding_encoder, source)
            if hparams.encoder_type == 'uni':
                cell = self._build_encoder_cell(hparams, num_layers)
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell, encoder_emb_in, dtype=scope.dtype,
                    sequence_length=self.iterator.source_sequence_length,
                    time_major=True, swap_memory=True)
            elif hparams.encoder_type == 'bi':
                num_bi_layers = num_layers // 2
                fw_cell = self._build_encoder_cell(hparams, num_bi_layers)
                bw_cell = self._build_encoder_cell(hparams, num_bi_layers)
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell, bw_cell, encoder_emb_in, dtype=scope.dtype,
                    sequence_length=self.iterator.source_sequence_length,
                    time_major=True, swap_memory=True)
                encoder_outputs = tf.concat(bi_outputs, -1)
                bi_encoder_state = bi_state

                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])
                        encoder_state.append(bi_encoder_state[1][layer_id])
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError('Unknown encoder_type %s' % hparams.encode_type)
        return encoder_outputs, encoder_state

    def _build_encoder_cell(self, hparams, num_layers):
        return self._build_rnn_cell(hparams, num_layers)

    def _build_decoder(self, encoder_outputs, encoder_state, hparams):
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)), tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)), tf.int32)
        iterator = self.iterator

        max_encoder_length = tf.reduce_max(iterator.source_sequence_length)
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length)*2.0))

        with tf.variable_scope('decoder') as scope:
            cell, decoder_initial_state = self._build_decoder_cell(
                hparams, encoder_outputs, encoder_state, iterator.source_sequence_length)
            if self.mode != 'infer':
                target_input = tf.transpose(iterator.target_input)
                decoder_emb_in = tf.nn.embedding_lookup(self.embedding_decoder, target_input)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_in,
                                                           iterator.target_sequence_length,
                                                           time_major=True)
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder, output_time_major=True, swap_memory=True, scope=scope)
                sample_id = outputs.sample_id
                logits = self.output_layer(outputs.rnn_output)
            else:
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                if beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell, embedding=self.embedding_decoder,
                        start_tokens=start_tokens, end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width, output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight)
                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                                      start_tokens, end_token)
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state,
                                                                 output_layer=self.output_layer)
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder, maximum_iterations=maximum_iterations,
                    output_time_major=True, swap_memory=True, scope=scope)
                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id
        return logits, sample_id, final_context_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state, source_sequence_length):
        num_units = hparams.num_units
        num_layers = self.num_layers
        beam_width = hparams.beam_width
        dtype = tf.float32

        memory = tf.transpose(encoder_outputs, [1, 0, 2])

        if self.mode == 'infer' and beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length,
                                                                   multiplier=beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
            batch_size = self.batch_size * beam_width
        else:
            batch_size = self.batch_size

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, memory,
                                                                memory_sequence_length=source_sequence_length)
        cell = self._build_rnn_cell(hparams, num_layers)

        alignment_history = (self.mode == 'infer' and beam_width == 0)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                   attention_layer_size=num_units,
                                                   alignment_history=alignment_history,
                                                   output_attention=True,
                                                   name='attention')
        decoder_initial_state = cell.zero_state(batch_size, dtype).clone(cell_state=encoder_state)
        return cell, decoder_initial_state

    def _build_rnn_cell(self, hparams, num_layers):
        def _single_cell(unit_type, num_units, forget_bias, dropout, mode, device_id):
            dropout = dropout if mode == 'train' else 0.0
            if unit_type == 'lstm':
                single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
            elif unit_type == 'gru':
                single_cell = tf.contrib.rnn.GRUCell(num_units)
            else:
                raise ValueError('Unknown unit type %s!' % unit_type)
            if dropout > 0:
                single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell,
                                                            input_keep_prob=1.0-dropout)
            single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, '/gpu:%d' % (device_id % hparams.num_gpus))
            return single_cell
        cell_list = []
        for i in range(num_layers):
            single_cell_ = _single_cell(hparams.unit_type, hparams.num_units,
                                        hparams.forget_bias, hparams.dropout, self.mode,
                                        device_id=i)
            cell_list.append(single_cell_)
        if num_layers == 1:
            return cell_list[0]
        else:
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _compute_loss(self, logits):
        target_output = tf.transpose(self.iterator.target_output)
        max_time = target_output.shape[0].value
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output,
                                                                  logits=logits)
        target_weights = tf.sequence_mask(self.iterator.target_sequence_length, max_time,
                                          dtype=logits.dtype)
        target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)
        return loss

    def _learning_rate_decay(self, num_train_steps):
        decay_rate = 0.5
        start_decay_step = int(num_train_steps*2/3)
        decay_steps = (num_train_steps-start_decay_step) // 4
        learning_rate = tf.cond(self.global_step < start_decay_step,
                                lambda: self.learning_rate,
                                lambda: tf.train.exponential_decay(
                                    self.learning_rate, self.global_step-start_decay_step,
                                    decay_steps, decay_rate, staircase=True),
                                name='learning_rate_decay')
        return learning_rate

    def _count_train_variables(self, train_variables):
        count = 0
        for variable in train_variables:
            temp = 1
            for dim in variable.shape:
                temp *= dim.value
            count += temp
        return count



