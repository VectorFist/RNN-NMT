import codecs
import tensorflow as tf
import create_nmt_model
import utils


def load_data(inference_input_file):
    with codecs.getreader("utf-8")(tf.gfile.GFile(inference_input_file, mode="rb")) as f:
        inference_data = f.read().splitlines()
    return inference_data


def translate(hparams, string):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    infer_model = create_nmt_model.create_infer_model(hparams)
    infer_sess = tf.Session(graph=infer_model.graph, config=config)
    with infer_model.graph.as_default():
        infer_model.model.reinitializer_or_load_model(infer_sess, hparams.model_dir)

        iterator_feed_dict = {infer_model.src_placeholder: [string],
                              infer_model.batch_size_placeholder: 1}
        infer_sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
        nmt_outputs = infer_model.model.decode(infer_sess)
    if hparams.beam_width > 0:
        nmt_outputs = nmt_outputs[0]
    translation = utils.get_translation(nmt_outputs, 0, hparams.eos)
    return translation


def translate_file(hparams, src_infer_file, trans_file):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    infer_model = create_nmt_model.create_infer_model(hparams)
    infer_sess = tf.Session(graph=infer_model.graph, config=config)
    with infer_model.graph.as_default():
        dev_infer_iterator_feed_dict = {
            infer_model.src_placeholder: load_data(src_infer_file),
            infer_model.batch_size_placeholder: hparams.infer_batch_size}
        infer_sess.run(infer_model.iterator.initializer, feed_dict=dev_infer_iterator_feed_dict)

        utils.decode_and_evaluate_bleu(infer_model, infer_sess, trans_file, '', hparams.beam_width,
                                       hparams.eos)
    print('translate file %s done, save translation to file %s' % (src_infer_file, trans_file))
