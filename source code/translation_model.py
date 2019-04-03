import tensorflow as tf
import os
import pickle
import re
import sys
import math
import shutil
import itertools
import utils, evaluation
from seq2seq_model import Seq2SeqModel
from subprocess import Popen, PIPE
import time


class TranslationModel:
    def __init__(self, encoders, decoders, checkpoint_dir, learning_rate, learning_rate_decay_factor,
                 batch_size, keep_best=1, dev_prefix=None, score_function='corpus_scores', name=None, ref_ext=None,
                 pred_edits=False, dual_output=False, binary=None, truncate_lines=True, ensemble=False,
                 checkpoints=None, beam_size=1, len_normalization=1, early_stopping=True, **kwargs):

        self.batch_size = batch_size
        self.character_level = {}
        self.binary = []

        for encoder_or_decoder in encoders + decoders:
            encoder_or_decoder.ext = encoder_or_decoder.ext or encoder_or_decoder.name
            self.character_level[encoder_or_decoder.ext] = encoder_or_decoder.character_level
            self.binary.append(encoder_or_decoder.get('binary', False))

        self.char_output = decoders[0].character_level

        self.src_ext = [encoder.ext for encoder in encoders]
        self.trg_ext = [decoder.ext for decoder in decoders]

        self.extensions = self.src_ext + self.trg_ext

        self.ref_ext = ref_ext
        if self.ref_ext is not None:
            self.binary.append(False)

        self.pred_edits = pred_edits
        self.dual_output = dual_output

        self.dev_prefix = dev_prefix
        self.name = name

        self.max_input_len = [encoder.max_len for encoder in encoders]
        self.max_output_len = [decoder.max_len for decoder in decoders]

        if truncate_lines:
            self.max_len = None  # we let seq2seq.get_batch handle long lines (by truncating them)
        else:  # the line reader will drop lines that are too long
            self.max_len = dict(zip(self.extensions, self.max_input_len + self.max_output_len))

        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate', dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        with tf.device('/cpu:0'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.baseline_step = tf.Variable(0, trainable=False, name='baseline_step')

        self.filenames = utils.get_filenames(extensions=self.extensions, dev_prefix=dev_prefix, name=name,
                                             ref_ext=ref_ext, binary=self.binary, **kwargs)
        utils.debug('reading vocabularies')
        self.vocabs = None
        self.src_vocab, self.trg_vocab = None, None
        self.read_vocab()

        for encoder_or_decoder, vocab in zip(encoders + decoders, self.vocabs):
            if vocab:
                encoder_or_decoder.vocab_size = len(vocab.reverse)

        utils.debug('creating model')

        self.models = []
        if ensemble and checkpoints is not None:
            for i, _ in enumerate(checkpoints, 1):
                with tf.variable_scope('model_{}'.format(i)):
                    model = Seq2SeqModel(encoders, decoders, self.learning_rate, self.global_step, name=name,
                                         pred_edits=pred_edits, dual_output=dual_output,
                                         baseline_step=self.baseline_step, **kwargs)
                    self.models.append(model)
            self.seq2seq_model = self.models[0]
        else:
            self.seq2seq_model = Seq2SeqModel(encoders, decoders, self.learning_rate, self.global_step, name=name,
                                              pred_edits=pred_edits, dual_output=dual_output,
                                              baseline_step=self.baseline_step, **kwargs)
            self.models.append(self.seq2seq_model)

        self.seq2seq_model.create_beam_op(self.models, beam_size, len_normalization, early_stopping)

        self.batch_iterator = None
        self.dev_batches = None
        self.train_size = None
        self.saver = None
        self.keep_best = keep_best
        self.checkpoint_dir = checkpoint_dir
        self.epoch = None

        self.training = utils.AttrDict()  # used to keep track of training

        try:
            self.reversed_scores = getattr(evaluation, score_function).reversed  # the lower the better
        except AttributeError:
            self.reversed_scores = False  # the higher the better

    def read_data(self, max_train_size, max_dev_size, read_ahead=10, batch_mode='standard', shuffle=True,
                  crash_test=False, use_unknown=True, **kwargs):
        utils.debug('reading training data')
        self.batch_iterator, self.train_size = utils.get_batch_iterator(
            self.filenames.train, self.extensions, self.vocabs, self.batch_size,
            max_size=max_train_size, character_level=self.character_level, max_seq_len=self.max_len,
            read_ahead=read_ahead, mode=batch_mode, shuffle=shuffle, binary=self.binary, crash_test=crash_test, use_unknown=use_unknown
        )

        utils.debug('reading development data')

        dev_sets = [
            utils.read_dataset(dev, self.extensions, self.vocabs, max_size=max_dev_size,
                               character_level=self.character_level, binary=self.binary, use_unknown=use_unknown)[0]
            for dev in self.filenames.dev
            ]
        # subset of the dev set whose loss is periodically evaluated
        self.dev_batches = [utils.get_batches(dev_set, batch_size=self.batch_size) for dev_set in dev_sets]

    def read_vocab(self):
        # don't try reading vocabulary for encoders that take pre-computed features
        self.vocabs = [
            None if binary else utils.initialize_vocabulary(vocab_path)
            for vocab_path, binary in zip(self.filenames.vocab, self.binary)
            ]
        self.src_vocab, self.trg_vocab = self.vocabs[:len(self.src_ext)], self.vocabs[len(self.src_ext):]

    def eval_step(self):
        # compute loss on dev set
        for prefix, dev_batches in zip(self.dev_prefix, self.dev_batches):
            eval_loss = sum(
                self.seq2seq_model.step(batch, update_model=False).loss * len(batch)
                for batch in dev_batches
            )
            eval_loss /= sum(map(len, dev_batches))

            utils.log("  {} eval: loss {:.2f}".format(prefix, eval_loss))

    def decode_sentence(self, sentence_tuple, remove_unk=False):
        return next(self.decode_batch([sentence_tuple], remove_unk))

    def decode_batch(self, sentence_tuples, batch_size, remove_unk=False, fix_edits=True):
        if batch_size == 1:
            batches = ([sentence_tuple] for sentence_tuple in sentence_tuples)  # lazy
        else:
            batch_count = int(math.ceil(len(sentence_tuples) / batch_size))
            batches = [sentence_tuples[i * batch_size:(i + 1) * batch_size] for i in range(batch_count)]

        def map_to_ids(sentence_tuple):
            token_ids = [
                sentence if vocab is None else
                utils.sentence_to_token_ids(sentence, vocab.vocab, ext, character_level=self.character_level.get(ext))
                for ext, vocab, sentence in zip(self.extensions, self.vocabs, sentence_tuple)
                ]
            return token_ids

        for batch_id, batch in enumerate(batches):
            token_ids = list(map(map_to_ids, batch))
            batch_token_ids, attn_weights = self.seq2seq_model.greedy_decoding(token_ids)
            batch_token_ids = zip(*batch_token_ids)

            for src_tokens, trg_token_ids, attn_weight in zip(batch, batch_token_ids, attn_weights):
                trg_tokens = []

                for trg_token_ids_, vocab in zip(trg_token_ids, self.trg_vocab):
                    trg_token_ids_ = list(trg_token_ids_)  # from np array to list
                    if utils.EOS_ID in trg_token_ids_:
                        trg_token_ids_ = trg_token_ids_[:trg_token_ids_.index(utils.EOS_ID)]

                    trg_tokens_ = [vocab.reverse[i] if i < len(vocab.reverse) else utils._UNK
                                   for i in trg_token_ids_]
                    trg_tokens.append(trg_tokens_)

                if self.pred_edits:
                    # first output is ops, second output is words
                    raw_hypothesis = ' '.join('_'.join(tokens) for tokens in zip(*trg_tokens))
                    trg_tokens = utils.reverse_edits(src_tokens[0].split('\t')[1].split(), trg_tokens, fix=fix_edits)
                    trg_tokens = [token for token in trg_tokens if token not in utils._START_VOCAB]
                    # FIXME: char-level
                else:
                    trg_tokens = trg_tokens[0]
                    raw_hypothesis = ''.join(trg_tokens) if self.char_output else ' '.join(trg_tokens)

                if remove_unk:
                    trg_tokens = [token for token in trg_tokens if token != utils._UNK]

                if self.char_output:
                    hypothesis = ''.join(trg_tokens)
                else:
                    hypothesis = ' '.join(trg_tokens).replace('@@ ', '')  # merge subwords units

                yield hypothesis, raw_hypothesis, attn_weight

    def align(self, output=None, align_encoder_id=0, **kwargs):
        # if self.binary and any(self.binary):
        #     raise NotImplementedError

        if len(self.filenames.test) != len(self.extensions):
            raise Exception('wrong number of input files')

        binary = self.binary and any(self.binary)

        paths = self.filenames.test or [None]
        lines = utils.read_lines(paths, binary=self.binary)

        for line_id, lines in enumerate(lines):
            token_ids = [
                sentence if vocab is None else
                utils.sentence_to_token_ids(sentence, vocab.vocab, character_level=self.character_level.get(ext))
                for ext, vocab, sentence in zip(self.extensions, self.vocabs, lines)
                ]

            _, weights = self.seq2seq_model.step(data=[token_ids], align=True, update_model=False)

            trg_vocab = self.trg_vocab[0]
            trg_token_ids = token_ids[len(self.src_ext)]
            trg_tokens = [trg_vocab.reverse[i] if i < len(trg_vocab.reverse) else utils._UNK for i in trg_token_ids]

            weights = weights.squeeze()
            max_len = weights.shape[1]

            if binary:
                src_tokens = None
            else:
                src_tokens = lines[align_encoder_id].split()[:max_len - 1] + [utils._EOS]
            trg_tokens = trg_tokens[:weights.shape[0] - 1] + [utils._EOS]

            output_file = '{}.{}.svg'.format(output, line_id + 1) if output is not None else None

            utils.heatmap(src_tokens, trg_tokens, weights, output_file=output_file)

    def decode(self, output=None, remove_unk=False, raw_output=False, max_test_size=None, **kwargs):
        utils.log('starting decoding')

        # empty `test` means that we read from standard input, which is not possible with multiple encoders
        # assert len(self.src_ext) == 1 or self.filenames.test
        # check that there is the right number of files for decoding
        # assert not self.filenames.test or len(self.filenames.test) == len(self.src_ext)

        output_file = None
        try:
            output_file = sys.stdout if output is None else open(output, 'w')
            paths = self.filenames.test or [None]
            lines = utils.read_lines(paths, binary=self.binary)

            if max_test_size:
                lines = itertools.islice(lines, max_test_size)

            if not self.filenames.test:  # interactive mode
                batch_size = 1
            else:
                batch_size = self.batch_size
                lines = list(lines)

            hypothesis_iter = self.decode_batch(lines, batch_size, remove_unk=remove_unk)

            for hypothesis, raw, attn in hypothesis_iter:
                if raw_output:
                    hypothesis = raw

                output_file.write(hypothesis + '\n')
                output_file.flush()
        finally:
            if output_file is not None:
                output_file.close()

    def evaluate(self, score_function, on_dev=True, output=None, remove_unk=False, max_dev_size=None,
                 raw_output=False, fix_edits=True, max_test_size=None, post_process_script=None, **kwargs):
        """
        Decode a dev or test set, and perform evaluation with respect to gold standard, using the provided
        scoring function. If `output` is defined, also save the decoding output to this file.
        When evaluating development data (`on_dev` to True), several dev sets can be specified (`dev_prefix` parameter
        in configuration files), and a score is computed for each of them.

        :param score_function: name of the scoring function used to score and rank models (typically 'bleu_score')
        :param on_dev: if True, evaluate the dev corpus, otherwise evaluate the test corpus
        :param output: save the hypotheses to this file
        :param remove_unk: remove the UNK symbols from the output
        :param max_dev_size: maximum number of lines to read from dev files
        :param max_test_size: maximum number of lines to read from test files
        :param raw_output: save raw decoder output (don't do post-processing like UNK deletion or subword
            concatenation). The evaluation is still done with the post-processed output.
        :param fix_edits: when predicting edit operations, pad shorter hypotheses with KEEP symbols.
        :return: scores of each corpus to evaluate
        """
        utils.log('starting decoding')

        if on_dev:
            filenames = self.filenames.dev
        else:
            filenames = [self.filenames.test]

        # convert `output` into a list, for zip
        if isinstance(output, str):
            output = [output]
        elif output is None:
            output = [None] * len(filenames)

        scores = []
        new_lines = []
        for filenames_, output_, prefix in zip(filenames, output, self.dev_prefix):  # evaluation on multiple corpora
            extensions = list(self.extensions)
            if self.ref_ext is not None:
                extensions.append(self.ref_ext)

            lines = list(utils.read_lines(filenames_, binary=self.binary))
            if on_dev and max_dev_size:
                lines = lines[:max_dev_size]
            elif not on_dev and max_test_size:
                lines = lines[:max_test_size]

            hypotheses = []
            references = []

            output_file = None

            try:
                if output_ is not None:
                    output_file = open(output_, 'w')

                lines_ = list(zip(*lines))

                src_sentences = list(zip(*lines_[:len(self.src_ext)]))
                trg_sentences = list(zip(*lines_[len(self.src_ext):]))

                hypothesis_iter = self.decode_batch(lines, self.batch_size, remove_unk=remove_unk,
                                                    fix_edits=fix_edits)
                #ref_file_path = '../data/test/ref.out'
                #ref_file = open(ref_file_path, 'w')
                #gen_file_path = "../data/test/hyp.out"
                #gen_file = open(gen_file_path, 'w')
                for i, (sources, hypothesis, reference) in enumerate(zip(src_sentences, hypothesis_iter,
                                                                         trg_sentences)):
                    if self.ref_ext is not None and on_dev:
                        reference = reference[-1]
                    else:
                        reference = reference[0]  # single output for now

                    hypothesis, raw, attn = hypothesis

                    hypotheses.append(hypothesis)
                    reference = reference.strip()
                    references.append(reference)
                    if output_file is not None:
                        if raw_output:
                            hypothesis = raw

                        output_file.write(hypothesis  +'\n')
                        output_file.flush()
                    #ref_file.write(reference + '\n')
                    #gen_file.write(hypothesis + '\n' )
                    #gen_file.flush() 

            finally:
                if output_file is not None:
                    output_file.close()

            if post_process_script is not None:
                data = '\n'.join(hypotheses).encode()
                data = Popen([post_process_script], stdout=PIPE, stdin=PIPE).communicate(input=data)[0].decode()
                hypotheses = data.splitlines()
            print("hypotheses:%d, references: %d" % (len(hypotheses), len(references)))
            # default scoring function is utils.bleu_score
            score, avg_score = getattr(evaluation, score_function)(hypotheses, references)

            # print scoring information
            score_info = [prefix, 'score={:.4f} avg_score={:.4f}'.format(score, avg_score)]

            # if score_summary:
            #    score_info.append(score_summary)

            if self.name is not None:
                score_info.insert(0, self.name)

            utils.log(' '.join(map(str, score_info)))
            scores.append(score)

        return scores

    def train(self, baseline_steps=0, loss_function='xent', **kwargs):
        self.init_training(**kwargs)

        utils.log('starting training')
        while True:
            try:
                self.train_step(loss_function=loss_function, **kwargs)
            except (utils.FinishedTrainingException, KeyboardInterrupt):
                utils.log('exiting...')
                self.save()
                return
            except utils.EvalException:
                self.save()
                step, score = self.training.scores[-1]
                self.manage_best_checkpoints(step, score)
            except utils.CheckpointException:
                self.save()

    def init_training(self, sgd_after_n_epoch=None, **kwargs):
        self.read_data(**kwargs)
        self.epoch = self.batch_size * self.global_step // self.train_size

        global_step = self.global_step.eval()
        epoch = self.epoch.eval()
        if sgd_after_n_epoch is not None and epoch >= sgd_after_n_epoch:  # already switched to SGD
            self.training.use_sgd = True
        else:
            self.training.use_sgd = False

        if kwargs.get('batch_mode') != 'random' and not kwargs.get('shuffle'):
            # read all the data up to this step (only if the batch iteration method is deterministic)
            for _ in range(global_step):
                next(self.batch_iterator)

        # those parameters are used to track the progress of training
        self.training.time = 0
        self.training.steps = 0
        self.training.loss = 0
        self.training.baseline_loss = 0
        self.training.losses = []
        self.training.last_decay = global_step
        self.training.scores = []

    def train_step(self, steps_per_checkpoint, model_dir, steps_per_eval=None, max_steps=0,
                   max_epochs=0, eval_burn_in=0, decay_if_no_progress=None, decay_after_n_epoch=None,
                   decay_every_n_epoch=None, sgd_after_n_epoch=None, sgd_learning_rate=None, min_learning_rate=None,
                   loss_function='xent', **kwargs):
        if min_learning_rate is not None and self.learning_rate.eval() < min_learning_rate:
            utils.debug('learning rate is too small: stopping')
            raise utils.FinishedTrainingException
        if 0 < max_steps <= self.global_step.eval() or 0 < max_epochs <= self.epoch.eval():
            raise utils.FinishedTrainingException

        start_time = time.time()

        step_function = self.seq2seq_model.step

        res = step_function(next(self.batch_iterator), update_model=True, use_sgd=self.training.use_sgd,
                            update_baseline=True)

        self.training.loss += res.loss
        self.training.baseline_loss += getattr(res, 'baseline_loss', 0)

        self.training.time += time.time() - start_time
        self.training.steps += 1

        global_step = self.global_step.eval()
        epoch = self.epoch.eval()

        if decay_after_n_epoch is not None and self.batch_size * global_step >= decay_after_n_epoch * self.train_size:
            if decay_every_n_epoch is not None and (self.batch_size * (global_step - self.training.last_decay)
                                                        >= decay_every_n_epoch * self.train_size):
                self.learning_rate_decay_op.eval()
                utils.debug('  decaying learning rate to: {:.3g}'.format(self.learning_rate.eval()))
                self.training.last_decay = global_step

        if sgd_after_n_epoch is not None and epoch >= sgd_after_n_epoch:
            if not self.training.use_sgd:
                utils.debug('epoch {}, starting to use SGD'.format(epoch + 1))
                self.training.use_sgd = True
                if sgd_learning_rate is not None:
                    self.learning_rate.assign(sgd_learning_rate).eval()
                self.training.last_decay = global_step  # reset learning rate decay

        if steps_per_checkpoint and global_step % steps_per_checkpoint == 0:
            loss = self.training.loss / self.training.steps
            baseline_loss = self.training.baseline_loss / self.training.steps
            step_time = self.training.time / self.training.steps

            summary = 'step {} epoch {} learning rate {:.3g} step-time {:.3f} loss {:.3f}'.format(
                global_step, epoch + 1, self.learning_rate.eval(), step_time, loss)

            if self.name is not None:
                summary = '{} {}'.format(self.name, summary)

            utils.log(summary)

            if decay_if_no_progress and len(self.training.losses) >= decay_if_no_progress:
                if loss >= max(self.training.losses[:decay_if_no_progress]):
                    self.learning_rate_decay_op.eval()

            self.training.losses.append(loss)
            self.training.loss, self.training.time, self.training.steps, self.training.baseline_loss = 0, 0, 0, 0
            self.eval_step()

        if steps_per_eval and global_step % steps_per_eval == 0 and 0 <= eval_burn_in <= global_step:
            eval_dir = 'eval' if self.name is None else 'eval_{}'.format(self.name)
            eval_output = os.path.join(model_dir, eval_dir)

            os.makedirs(eval_output, exist_ok=True)

            # if there are several dev files, we define several output files
            output = [
                os.path.join(eval_output, '{}.{}.out'.format(prefix, global_step))
                for prefix in self.dev_prefix
                ]

            kwargs_ = dict(kwargs)
            kwargs_['output'] = output
            score, *_ = self.evaluate(on_dev=True, **kwargs_)
            self.training.scores.append((global_step, score))

        if steps_per_eval and global_step % steps_per_eval == 0:
            raise utils.EvalException
        elif steps_per_checkpoint and global_step % steps_per_checkpoint == 0:
            raise utils.CheckpointException

    def manage_best_checkpoints(self, step, score):
        score_filename = os.path.join(self.checkpoint_dir, 'scores.txt')
        # try loading previous scores
        try:
            with open(score_filename) as f:
                # list of pairs (score, step)
                scores = [(float(line.split()[0]), int(line.split()[1])) for line in f]
        except IOError:
            scores = []

        if any(step_ >= step for _, step_ in scores):
            utils.warn('inconsistent scores.txt file')

        best_scores = sorted(scores, reverse=not self.reversed_scores)[:self.keep_best]

        def full_path(filename):
            return os.path.join(self.checkpoint_dir, filename)

        lower = (lambda x, y: y < x) if self.reversed_scores else (lambda x, y: x < y)

        if any(lower(score_, score) for score_, _ in best_scores) or not best_scores:
            # if this checkpoint is in the top, save it under a special name

            prefix = 'translate-{}.'.format(step)
            dest_prefix = 'best-{}.'.format(step)

            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith(prefix):
                    dest_filename = filename.replace(prefix, dest_prefix)
                    shutil.copy(full_path(filename), full_path(dest_filename))

                    # also copy to `best` if this checkpoint is the absolute best
                    if all(lower(score_, score) for score_, _ in best_scores):
                        dest_filename = filename.replace(prefix, 'best.')
                        shutil.copy(full_path(filename), full_path(dest_filename))

            best_scores = sorted(best_scores + [(score, step)], reverse=not self.reversed_scores)

            for _, step_ in best_scores[self.keep_best:]:
                # remove checkpoints that are not in the top anymore
                prefix = 'best-{}'.format(step_)
                for filename in os.listdir(self.checkpoint_dir):
                    if filename.startswith(prefix):
                        os.remove(full_path(filename))

        # save scores
        scores.append((score, step))

        with open(score_filename, 'w') as f:
            for score_, step_ in scores:
                f.write('{:.2f} {}\n'.format(score_, step_))

    def initialize(self, checkpoints=None, reset=False, reset_learning_rate=False, max_to_keep=1,
                   keep_every_n_hours=0, sess=None, use_transfer=False, api_params=None, **kwargs):
        """
        :param checkpoints: list of checkpoints to load (instead of latest checkpoint)
        :param reset: don't load latest checkpoint, reset learning rate and global step
        :param reset_learning_rate: reset the learning rate to its initial value
        :param max_to_keep: keep this many latest checkpoints at all times
        :param keep_every_n_hours: and keep checkpoints every n hours
        """
        sess = sess or tf.get_default_session()

        if keep_every_n_hours <= 0 or keep_every_n_hours is None:
            keep_every_n_hours = float('inf')

        self.saver = tf.train.Saver(max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=keep_every_n_hours,
                                    sharded=False)

        sess.run(tf.global_variables_initializer())
        blacklist = ['dropout_keep_prob']

        if reset_learning_rate or reset:
            blacklist.append('learning_rate')
        if reset:
            blacklist.append('global_step')

        params = {k: kwargs.get(k) for k in ('variable_mapping', 'reverse_mapping')}

        if checkpoints and len(self.models) > 1:
            assert len(self.models) == len(checkpoints)
            for i, checkpoint in enumerate(checkpoints, 1):
                load_checkpoint(sess, None, checkpoint, blacklist=blacklist, prefix='model_{}'.format(i), **params)
        elif checkpoints:  # load partial checkpoints
            for checkpoint in checkpoints:  # checkpoint files to load
                load_checkpoint(sess, None, checkpoint, blacklist=blacklist, **params)
        elif not reset:
            load_checkpoint(sess, self.checkpoint_dir, blacklist=blacklist, **params)
        print(use_transfer)
        if api_params and use_transfer:
            param_variables = tf.global_variables()
            for v in param_variables:
                if 'api' in v.name and v.name in api_params.keys():
                    # print("Assign param %s with api model" % v.name)
                    sess.run(v.assign(api_params[v.name]))
                    utils.debug('Assign param: {} with api model'.format(v.name))

        utils.debug('global step: {}'.format(self.global_step.eval()))
        utils.debug('baseline step: {}'.format(self.baseline_step.eval()))

    def save(self):
        save_checkpoint(tf.get_default_session(), self.saver, self.checkpoint_dir, self.global_step)


# hard-coded variables which can also be defined in config file (variable_mapping and reverse_mapping)
global_variable_mapping = []  # map old names to new names
global_reverse_mapping = [  # map new names to old names
    (r'decoder_(.*?)/.*/initial_state_projection/', r'decoder_\1/initial_state_projection/'),
]


def load_checkpoint(sess, checkpoint_dir, filename=None, blacklist=(), prefix=None, variable_mapping=None,
                    reverse_mapping=None):
    """
    if `filename` is None, we load last checkpoint, otherwise
      we ignore `checkpoint_dir` and load the given checkpoint file.
    """
    variable_mapping = variable_mapping or []
    reverse_mapping = reverse_mapping or []

    variable_mapping = list(variable_mapping) + global_variable_mapping
    reverse_mapping = list(reverse_mapping) + global_reverse_mapping

    if filename is None:
        # load last checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt is not None:
            filename = ckpt.model_checkpoint_path
    else:
        checkpoint_dir = os.path.dirname(filename)

    vars_ = []
    var_names = []
    for var in tf.global_variables():
        if prefix is None or var.name.startswith(prefix):
            name = var.name if prefix is None else var.name[len(prefix) + 1:]
            vars_.append(var)
            var_names.append(name)

    var_file = os.path.join(checkpoint_dir, 'vars.pkl')
    if os.path.exists(var_file):
        with open(var_file, 'rb') as f:
            old_names = pickle.load(f)
    else:
        old_names = list(var_names)

    name_mapping = {}
    for name in old_names:
        name_ = name
        for key, value in variable_mapping:
            name_ = re.sub(key, value, name_)
        name_mapping[name] = name_

    var_names_ = []
    for name in var_names:
        name_ = name
        for key, value in reverse_mapping:
            name_ = re.sub(key, value, name_)
        if name_ in list(name_mapping.values()):
            name = name_
        var_names_.append(name)
    vars_ = dict(zip(var_names_, vars_))

    variables = {old_name[:-2]: vars_[new_name] for old_name, new_name in name_mapping.items()
                 if new_name in vars_ and not any(prefix in new_name for prefix in blacklist)}

    if filename is not None:
        utils.log('reading model parameters from {}'.format(filename))
        tf.train.Saver(variables).restore(sess, filename)

        utils.debug('retrieved parameters ({})'.format(len(variables)))
        for var in sorted(variables.values(), key=lambda var: var.name):
            utils.debug('  {} {}'.format(var.name, var.get_shape()))


def save_checkpoint(sess, saver, checkpoint_dir, step=None, name=None):
    var_file = os.path.join(checkpoint_dir, 'vars.pkl')
    name = name or 'translate'
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(var_file, 'wb') as f:
        var_names = [var.name for var in tf.global_variables()]
        pickle.dump(var_names, f)

    utils.log('saving model to {}'.format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, name)
    saver.save(sess, checkpoint_path, step, write_meta_graph=False)

    utils.log('finished saving model')
