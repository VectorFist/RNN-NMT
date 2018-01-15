import tensorflow as tf
import create_nmt_model
import os
import time
import model_infer
import random
import utils

SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'


def train(hparams):
    model_dir = hparams.model_dir
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats
    steps_per_eval = 10 * steps_per_stats

    test_src_file = hparams.src_test_file
    test_tgt_file = hparams.tgt_test_file
    sample_src_data = model_infer.load_data(test_src_file)
    sample_tgt_data = model_infer.load_data(test_tgt_file)

    train_model = create_nmt_model.create_train_model(hparams)
    eval_model = create_nmt_model.create_eval_model(hparams)
    infer_model = create_nmt_model.create_infer_model(hparams)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    train_sess = tf.Session(graph=train_model.graph, config=config)
    eval_sess = tf.Session(graph=eval_model.graph, config=config)
    infer_sess = tf.Session(graph=infer_model.graph, config=config)

    with train_model.graph.as_default():
        global_step = train_model.model.reinitializer_or_load_model(train_sess, model_dir)
        train_sess.run(train_model.iterator.initializer)

    summary_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_log'), train_model.graph)

    dev_bleu_score, _ = run_external_eval(infer_model, infer_sess, model_dir, hparams)
    while global_step < num_train_steps:
        start_time = time.time()
        try:
            step_result = train_model.model.train(train_sess)
        except tf.errors.OutOfRangeError:
            print('=========== Finished an epoch ==============')
            run_sample_decode(infer_model, infer_sess, hparams, sample_src_data, sample_tgt_data)
            train_sess.run(train_model.iterator.initializer)
            continue
        train_loss = step_result[1]
        train_summary = step_result[3]
        global_step = step_result[4]
        learning_rate = step_result[8]
        if global_step % steps_per_stats == 0:
            print('train step {:d}, train loss {:.4f}, test_bleu_score: {:.3f}, lr: {:.3f}, time per step: {:.3f}s'.
                  format(global_step, train_loss, dev_bleu_score, learning_rate, time.time()-start_time))
        if global_step % steps_per_eval == 0:
            summary_writer.add_summary(train_summary, global_step=global_step)
            train_model.model.saver.save(train_sess, os.path.join(model_dir, 'translate.ckpt'),
                                         global_step=global_step)
            run_sample_decode(infer_model, infer_sess, hparams, sample_src_data, sample_tgt_data)
            dev_bleu_score, _ = run_external_eval(infer_model, infer_sess, model_dir, hparams)

    summary_writer.close()
    print('train done')


def run_external_eval(infer_model, infer_sess, model_dir, hparams):
    with infer_model.graph.as_default():
        global_step = infer_model.model.reinitializer_or_load_model(infer_sess, model_dir)
        src_test_file = hparams.src_test_file
        tgt_test_file = hparams.tgt_test_file
        dev_infer_iterator_feed_dict = {
            infer_model.src_placeholder: model_infer.load_data(src_test_file),
            infer_model.batch_size_placeholder: hparams.infer_batch_size}
        model_dir = hparams.model_dir
        infer_sess.run(infer_model.iterator.initializer, feed_dict=dev_infer_iterator_feed_dict)

        trans_file = os.path.join(model_dir, "output_%s" % 'dev')
        bleu_score = utils.decode_and_evaluate_bleu(infer_model.model, infer_sess, trans_file, tgt_test_file,
                                                    hparams.beam_width, tgt_eos=hparams.eos)
        print('{:s} bleu score: {:.3f}'.format('test', bleu_score))
        return bleu_score, global_step


def run_sample_decode(infer_model, infer_sess, hparams, src_data, tgt_data):
    sample_id = random.randint(0, len(src_data) - 1)
    with infer_model.graph.as_default():
        infer_model.model.reinitializer_or_load_model(infer_sess, hparams.model_dir)

        iterator_feed_dict = {infer_model.src_placeholder: [src_data[sample_id]],
                              infer_model.batch_size_placeholder: 1}
        infer_sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
        nmt_outputs = infer_model.model.decode(infer_sess)
    if hparams.beam_width > 0:
        nmt_outputs = nmt_outputs[0]
    translation = utils.get_translation(nmt_outputs, 0, EOS)
    print('src: %s' % src_data[sample_id])
    print('ref: %s' % tgt_data[sample_id])
    print('nmt: ' + translation)
