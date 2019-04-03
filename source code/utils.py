import os
import sys
import numpy as np
import logging
import struct
import random
import math
import shutil
import functools
import operator
import heapq

from collections import namedtuple
from contextlib import contextmanager

# special vocabulary symbols

_BOS = '<S>'
_EOS = '</S>'
_UNK = '<UNK>'
_KEEP = '<KEEP>'
_DEL = '<DEL>'
_INS = '<INS>'
_SUB = '<SUB>'
_NONE = '<NONE>'

_START_VOCAB = [_BOS, _EOS, _UNK, _KEEP, _DEL, _INS, _SUB, _NONE]

BOS_ID = 0
EOS_ID = 1
UNK_ID = 2
KEEP_ID = 3
DEL_ID = 4
INS_ID = 5
SUB_ID = 6
NONE_ID = 7


class FinishedTrainingException(Exception):
    def __init__(self):
        debug('finished training')

class CheckpointException(Exception):
    pass
class EvalException(Exception):
    pass


@contextmanager
def open_files(names, mode='r'):
    """ Safely open a list of files in a context manager.
    Example:
    >>> with open_files(['foo.txt', 'bar.csv']) as (f1, f2):
    ...   pass
    """

    files = []
    try:
        for name_ in names:
            if name_ is None:
                file_ = sys.stdin if 'r' in mode else sys.stdout
            else:
                file_ = open(name_, mode=mode)
            files.append(file_)
        yield files
    finally:
        for file_ in files:
            file_.close()


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    Example:
    >>> d = AttrDict(x=1, y=2)
    >>> d.x
    1
    >>> d.y = 3
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self  # dark magic

    def __getattr__(self, item):
        return self.__dict__.get(item)


def reverse_edits(source, edits, fix=True, strict=False):
    if len(edits) == 1:    # transform list of edits as a list of (op, word) tuples
        edits = edits[0]
        for i, edit in enumerate(edits):
            if edit in (_KEEP, _DEL, _INS, _SUB):
                edit = (edit, edit)
            elif edit.startswith(_INS + '_'):
                edit = (_INS, edit[len(_INS + '_'):])
            elif edit.startswith(_SUB + '_'):
                edit = (_SUB, edit[len(_SUB + '_'):])
            else:
                edit = (_INS, edit)

            edits[i] = edit
    else:
        edits = zip(*edits)

    src_words = source
    target = []
    consistent = True
    i = 0

    for op, word in edits:
        if strict and not consistent:
            break
        if op in (_DEL, _KEEP, _SUB):
            if i >= len(src_words):
                consistent = False
                continue

            if op == _KEEP:
                target.append(src_words[i])
            elif op == _SUB:
                target.append(word)

            i += 1
        else:   # op is INS
            target.append(word)

    if fix:
        target += src_words[i:]

    return target


def initialize_vocabulary(vocabulary_path):
    """
    Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {'dog': 0, 'cat': 1}, and a reversed vocabulary ['dog', 'cat'].

    :param vocabulary_path: path to the file containing the vocabulary.
    :return:
      the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path) as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.rstrip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return namedtuple('vocab', 'vocab reverse')(vocab, rev_vocab)
    else:
        raise ValueError("vocabulary file %s not found", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, ext, character_level=False, use_unknown=True):
    """
    Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    :param sentence: a string, the sentence to convert to token-ids
    :param vocabulary: a dictionary mapping tokens to integers
    :param character_level: treat sentence as a string of characters, and
        not as a string of words
    :return: a list of integers, the token-ids for the sentence.
    """
    sentence = sentence.strip()
    sentence = sentence.rstrip('\n') if character_level else sentence.split(' ')
    if ext =='nl':
        use_unknown=True
    if use_unknown:
        return [vocabulary.get(w, UNK_ID) for w in sentence]
    else:
        tks = []
        for w in sentence:
            if w not in vocabulary:
                w = w.split('_')[0]
            tks.append(vocabulary[w])
        return tks


def get_filenames(data_dir, model_dir, extensions, train_prefix, dev_prefix, vocab_prefix, name=None,
                  ref_ext=None, binary=None, decode=None, eval=None, align=None, **kwargs):
    """
    Get a bunch of file prefixes and extensions, and output the list of filenames to be used
    by the model.

    :param data_dir: directory where all the the data is stored
    :param extensions: list of file extensions, in the right order (last extension is always the target)
    :param train_prefix: name of the training corpus (usually 'train')
    :param dev_prefix: name of the dev corpus (usually 'dev')
    :param vocab_prefix: prefix of the vocab files (usually 'vocab')
    :param kwargs: optional contains an additional 'decode', 'eval' or 'align' parameter
    :return: namedtuple containing the filenames
    """
    train_path = os.path.join(data_dir, train_prefix)
    dev_path = [os.path.join(data_dir, prefix) for prefix in dev_prefix]

    train = ['{}/{}.token.{}'.format(train_path, train_prefix, ext) for ext in extensions]

    dev_extensions = list(extensions)
    if ref_ext is not None and ref_ext != extensions[-1]:
        dev_extensions.append(ref_ext)

    dev = [['{}/test.token.{}'.format(path, ext) for ext in dev_extensions] for path in dev_path]

    vocab_path = os.path.join(data_dir, vocab_prefix)
    vocab_src = ['{}.{}'.format(vocab_path, ext) for ext in extensions]

    data = 'data' if name is None else 'data_{}'.format(name)
    vocab_path = os.path.join(model_dir, data, 'vocab')
    vocab = ['{}.{}'.format(vocab_path, ext) for ext in extensions]
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    binary = binary or [False] * len(vocab)
    for src, dest, binary_ in zip(vocab_src, vocab, binary):
        if not binary_ and not os.path.exists(dest):
            debug('copying vocab to {}'.format(dest))
            shutil.copy(src, dest)

    exts = list(extensions)
    if decode is not None:  # empty list means we decode from standard input
        test = decode
        exts.pop(-1)
    elif eval is not None:
        if ref_ext is not None:
            exts[-1] = ref_ext
        test = eval or dev_prefix[:1]
    else:
        test = align or dev_prefix[:1]

    if len(test) == 1 and not (decode and os.path.exists(test[0])):
        corpus_path = os.path.join(data_dir, test[0])
        test = ['{}/test.token.{}'.format(corpus_path, ext) for ext in exts]

    filenames = namedtuple('filenames', ['train', 'dev', 'test', 'vocab'])
    return filenames(train, dev, test, vocab)


def read_dataset(paths, extensions, vocabs, max_size=None, character_level=None, sort_by_length=False,
                 max_seq_len=None, from_position=None, binary=None, use_unknown=True):
    data_set = []

    if from_position is not None:
        debug('reading from position: {}'.format(from_position))

    line_reader = read_lines_from_position(paths, from_position=from_position, binary=binary)
    character_level = character_level or {}

    positions = None

    for inputs, positions in line_reader:
        if len(data_set) > 0 and len(data_set) % 100000 == 0:
            debug("  lines read: {}".format(len(data_set)))
        lines = [
            input_ if binary_ else
            sentence_to_token_ids(input_, vocab.vocab,ext, character_level=character_level.get(ext), use_unknown=use_unknown)
            for input_, vocab, binary_, ext in zip(inputs, vocabs, binary, extensions)
        ]

        if not all(lines):  # skip empty inputs
            continue
        # skip lines that are too long
        if max_seq_len and any(len(line) > max_seq_len[ext] for line, ext in zip(lines, extensions)):
            continue

        data_set.append(lines)

        if max_size and len(data_set) >= max_size:
            break

    debug('files: {}'.format(' '.join(paths)))
    debug('lines reads: {}'.format(len(data_set)))

    if sort_by_length:
        data_set.sort(key=lambda lines: list(map(len, lines)))

    return data_set, positions


def random_batch_iterator(data, batch_size):
    """
    The most basic form of batch iterator.

    :param data: the dataset to segment into batches
    :param batch_size: the size of a batch
    :return: an iterator which yields random batches (indefinitely)
    """
    while True:
        yield random.sample(data, batch_size)


def basic_batch_iterator(data, batch_size, shuffle=False, allow_smaller=True):
    if shuffle:
        random.shuffle(data)

    batch_count = len(data) // batch_size

    if allow_smaller and batch_count * batch_size < len(data):
        batch_count += 1

    for i in range(batch_count):
        yield data[i * batch_size:(i + 1) * batch_size]


def cycling_batch_iterator(data, batch_size, shuffle=True, allow_smaller=True):
    """
    Indefinitely cycle through a dataset and yield batches (the dataset is shuffled
    at each new epoch)

    :param data: the dataset to segment into batches
    :param batch_size: the size of a batch
    :return: an iterator which yields batches (indefinitely)
    """
    while True:
        iterator = basic_batch_iterator(data, batch_size, shuffle=shuffle, allow_smaller=allow_smaller)
        for batch in iterator:
            yield batch


def read_ahead_batch_iterator(data, batch_size, read_ahead=10, shuffle=True, allow_smaller=True,
                              mode='standard', cycle=True, crash_test=False, **kwargs):
    """
    Same iterator as `cycling_batch_iterator`, except that it reads a number of batches
    at once, and sorts their content according to their size.

    This is useful for training, where all the sequences in one batch need to be padded
     to the same length as the longest sequence in the batch.

    :param data: the dataset to segment into batches
    :param batch_size: the size of a batch
    :param read_ahead: number of batches to read ahead of time and sort (larger numbers
      mean faster training, but less random behavior)
    :return: an iterator which yields batches (indefinitely)
    """
    if not cycle:
        iterator = basic_batch_iterator(data, batch_size, shuffle=shuffle, allow_smaller=allow_smaller)
    elif mode == 'random':
        iterator = random_batch_iterator(data, batch_size)
    else:
        iterator = cycling_batch_iterator(data, batch_size, shuffle=shuffle, allow_smaller=allow_smaller)

    if crash_test:
        n = batch_size // 2
        dummy_batch = heapq.nlargest(n, data, key=lambda p: len(p[0]))
        dummy_batch += heapq.nlargest(batch_size - n, data, key=lambda p: len(p[1]))

        while True:
            yield dummy_batch

    if read_ahead is None or read_ahead <= 1:
        yield from iterator

    while True:
        batches = []
        for batch in iterator:
            batches.append(batch)
            if len(batches) >= read_ahead:
                break

        data_ = sorted(sum(batches, []), key=lambda lines: len(lines[-1]))
        batches = [data_[i * batch_size:(i + 1) * batch_size] for i in range(read_ahead)]
        batches = [batch for batch in batches if batch]  # filter empty batches

        if not any(batches):
            break

        if shuffle:  # TODO: enable shuffling here without epoch shuffling
            random.shuffle(batches)
        for batch in batches:
            yield batch


def get_batch_iterator(paths, extensions, vocabs, batch_size, max_size=None, character_level=None,
                       sort_by_length=False, max_seq_len=None, read_ahead=10, shuffle=True,
                       binary=None, mode='standard', crash_test=False, use_unknown=True):
    read_shard = functools.partial(read_dataset,
        paths=paths, extensions=extensions, vocabs=vocabs, max_size=max_size, max_seq_len=max_seq_len,
        character_level=character_level, sort_by_length=sort_by_length, binary=binary, use_unknown=use_unknown)
    batch_iterator = functools.partial(read_ahead_batch_iterator, batch_size=batch_size, read_ahead=read_ahead,
                                       shuffle=shuffle, mode=mode, crash_test=crash_test)

    # FIXME: crash test only for first shard

    with open(paths[-1]) as f:   # count lines
        line_count = sum(1 for _ in f)
        debug('total line count: {}'.format(line_count))

    shard, position = read_shard()
    if not max_size or line_count <= max_size:
        # training set is small enough to fit entirely into memory (single shard)
        return batch_iterator(shard), line_count
    else:
        batch_iterator = functools.partial(batch_iterator, cycle=False)

        def generator(position, shard):
            while True:
                if len(shard) < max_size:
                    # last shard, start again from the beginning of the dataset
                    position = None

                size = 0
                for batch in batch_iterator(shard):
                    size += len(batch)
                    yield batch

                    if size >= len(shard):  # cycle through this shard only once, then read next shard
                        shard, position = read_shard(from_position=position)
                        break

        return generator(position, shard), line_count


def get_batches(data, batch_size, batches=0, allow_smaller=True):
    """
    Segment `data` into a given number of fixed-size batches. The dataset is automatically shuffled.

    This function is for smaller datasets, when you need access to the entire dataset at once (e.g. dev set).
    For larger (training) datasets, where you may want to lazily iterate over batches
    and cycle several times through the entire dataset, prefer batch iterators
    (such as `cycling_batch_iterator`).

    :param data: the dataset to segment into batches (a list of data points)
    :param batch_size: the size of a batch
    :param batches: number of batches to return (0 for the largest possible number)
    :param allow_smaller: allow the last batch to be smaller
    :return: a list of batches (which are lists of `batch_size` data points)
    """
    if not allow_smaller:
        max_batches = len(data) // batch_size
    else:
        max_batches = int(math.ceil(len(data) / batch_size))

    if batches < 1 or batches > max_batches:
        batches = max_batches

    random.shuffle(data)
    batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(batches)]
    return batches


def read_binary_features(filename, from_position=None):
    """
    Reads a binary file containing vector features. First two (int32) numbers correspond to
    number of entries (lines), and dimension of the vectors.
    Each entry starts with a 32 bits integer indicating the number of frames, followed by
    (frames * dimension) 32 bits floats.

    Use `scripts/extract-audio-features.py` to create such a file for audio (MFCCs).

    :param filename: path to the binary file containing the features
    :return: list of arrays of shape (frames, dimension)
    """
    all_feats = []

    with open(filename, 'rb') as f:
        lines, dim = struct.unpack('ii', f.read(8))
        if from_position is not None:
            f.seek(from_position)

        while True:
            x = f.read(4)
            if len(x) < 4:
                break
            frames, = struct.unpack('i', x)
            n = frames * dim
            x = f.read(4 * n)
            if len(x) < 4 * n:
                break
            feats = struct.unpack('f' * n, x)
            yield list(np.array(feats).reshape(frames, dim)), f.tell()


def read_lines(paths, binary=None):
    binary = binary or [False] * len(paths)
    return zip(*[sys.stdin if path is None else
                 map(operator.itemgetter(0), read_binary_features(path)) if binary_
                 else open(path)
                 for path, binary_ in zip(paths, binary)])


def read_text_from_position(filename, from_position=None):
    with open(filename) as f:
        if from_position is not None:
            f.seek(from_position)
        while True:
            line = f.readline()
            if not line:
                break
            yield line, f.tell()


def read_lines_from_position(paths, from_position=None, binary=None):
    binary = binary or [False] * len(paths)
    from_position = from_position or [None] * len(paths)

    iterators = [
        read_binary_features(path, from_position_) if binary_ else
        read_text_from_position(path, from_position_)
        for path, binary_, from_position_ in zip(paths, binary, from_position)
    ]

    for data in zip(*iterators):
        yield tuple(zip(*data))


def create_logger(log_file=None):
    """
    Initialize global logger and return it.

    :param log_file: log to this file, or to standard output if None
    :return: created logger
    """
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S')
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    return logger


def log(msg, level=logging.INFO):
    logging.getLogger(__name__).log(level, msg)


def debug(msg): log(msg, level=logging.DEBUG)


def warn(msg): log(msg, level=logging.WARN)


def alignment_to_text(xlabels=None, ylabels=None, weights=None, output_file=None):
    """
    :param xlabels: input words
    :param ylabels: output words
    :param weights: numpy array of shape (len(xlabels), len(ylabels))
    :param output_file: write the matrix in this file
    """
    with open(output_file.replace('svg', 'txt').replace('jpg', 'txt'), 'w') as output_file:
        output_file.write(' \t' + '\t'.join(xlabels) + '\n')
        for i in range(len(ylabels)):
            output_file.write(ylabels[i])
            for j in range(len(xlabels)):
                output_file.write('\t' + str(weights[i][j]))
            output_file.write('\n')
