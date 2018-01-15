import time
import tensorflow as tf
import codecs
import numpy as np
import bleu


def get_translation(nmt_outputs, sent_id, tgt_eos):
    if tgt_eos:
        tgt_eos = tgt_eos.encode('utf-8')
    output = nmt_outputs[sent_id, :].tolist()
    if tgt_eos and tgt_eos in output:
        output = output[: output.index(tgt_eos)]

    translation = b' '.join(output)
    return str(translation, encoding='utf-8')


def _bleu(ref_file, trans_file):
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            translations.append(line.split(" "))

    bleu_score, _, _, _, _, _ = bleu.compute_bleu(per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


def decode_and_evaluate_bleu(model, sess, trans_file, ref_file, beam_width, tgt_eos, num_translations_per_input=1,
                             decode=True):
    if decode:
        print("decoding to output %s." % trans_file)
        num_sentences = 0
        with codecs.getwriter("utf-8")(
            tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
            trans_f.write("")  # Write empty string to ensure file is created.

        num_translations_per_input = max(
            min(num_translations_per_input, beam_width), 1)
        while True:
            try:
                nmt_outputs = model.decode(sess)
                if beam_width == 0:
                    nmt_outputs = np.expand_dims(nmt_outputs, 0)

                batch_size = nmt_outputs.shape[1]
                num_sentences += batch_size

                for sent_id in range(batch_size):
                    for beam_id in range(num_translations_per_input):
                        translation = get_translation(nmt_outputs[beam_id], sent_id, tgt_eos=tgt_eos)
                        #print(translation)
                        #trans_f.write((translation + b"\n").decode("utf-8"))
                        trans_f.write(translation+'\n')
            except tf.errors.OutOfRangeError:
                print('decode done...')
                break

    if ref_file and tf.gfile.Exists(trans_file):
        bleu_score = _bleu(ref_file, trans_file)
        return bleu_score





