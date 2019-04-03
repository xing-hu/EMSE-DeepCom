import tensorflow as tf
import utils


"""
Code from: https://github.com/vahidk/EffectiveTensorflow
"""

def get_weights(sequence, eos_id, include_first_eos=True):
    cumsum = tf.cumsum(tf.to_float(tf.not_equal(sequence, eos_id)), axis=1)
    range_ = tf.range(start=1, limit=tf.shape(sequence)[1] + 1)
    range_ = tf.tile(tf.expand_dims(range_, axis=0), [tf.shape(sequence)[0], 1])
    weights = tf.to_float(tf.equal(cumsum, tf.to_float(range_)))

    if include_first_eos:
        weights = weights[:,:-1]
        shape = [tf.shape(weights)[0], 1]
        weights = tf.concat([tf.ones(tf.stack(shape)), weights], axis=1)

    return tf.stop_gradient(weights)


def resize_like(src, dst):
    batch_size = tf.shape(src)[0]
    beam_size = tf.shape(dst)[0] // batch_size
    shape = get_shape(src)[1:]
    src = tf.tile(tf.expand_dims(src, axis=1), [1, beam_size] + [1] * len(shape))
    src = tf.reshape(src, tf.stack([batch_size * beam_size] + shape))
    return src


def get_shape(tensor):
    """Returns static shape if available and dynamic shape otherwise."""
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def batch_gather(tensor, indices):
    """Gather in batch from a tensor of arbitrary size.

    In pseduocode this module will produce the following:
    output[i] = tf.gather(tensor[i], indices[i])

    Args:
      tensor: Tensor of arbitrary size.
      indices: Vector of indices.
    Returns:
      output: A tensor of gathered values.
    """
    shape = get_shape(tensor)
    flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    indices = tf.convert_to_tensor(indices)
    offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
    offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
    output = tf.gather(flat_first, indices + offset)
    return output


def log_softmax(x, axis, temperature=None):
    T = temperature or 1.0
    my_max = tf.reduce_max(x/T, axis=axis, keep_dims=True)
    return x - (tf.log(tf.reduce_sum(tf.exp(x/T - my_max), axis, keep_dims=True)) + my_max)


def rnn_beam_search(update_funs, initial_states, sequence_length, beam_width, len_normalization=None,
                    early_stopping=True, temperature=None):
    """
    :param update_funs: function to compute the next state and logits given the current state and previous ids
    :param initial_states: recurrent model states
    :param sequence_length: maximum output length
    :param beam_width: beam size
    :param len_normalization: length normalization coefficient (0 or None for no length normalization)
    :param early_stopping: same behavior as the old beam-search implementation, beam size is reduced by one
        each time a new finished hypothesis is encountered. This doesn't improve decoding speed, but
        might produce better results.
        FIXME: this takes a lot of memory. This causes crashes with very long sequences
    :return: tensor of size (batch_size, beam_size, seq_len) containing the beam-search hypotheses sorted by
        best score (axis 1), and tensor of size (batch_size, beam_size) containing the said scores.
    """
    batch_size = tf.shape(initial_states[0])[0]

    states = []
    for initial_state in initial_states:
        state = tf.tile(tf.expand_dims(initial_state, axis=1), [1, beam_width, 1])
        states.append(state)

    sel_sum_logprobs = tf.log([[1.] + [0.] * (beam_width - 1)])
    sel_sum_logprobs = tf.tile(sel_sum_logprobs, [batch_size, 1])

    ids = tf.tile([[utils.BOS_ID]], [batch_size, beam_width])
    sel_ids = tf.expand_dims(ids, axis=2)

    mask = tf.ones([batch_size, beam_width], dtype=tf.float32)

    finished_hypotheses = tf.zeros(shape=[batch_size, 0, sequence_length + 1], dtype=tf.int32)
    finished_scores = tf.zeros(shape=[batch_size, 0], dtype=tf.float32)

    beam_sizes = tf.ones(shape=[batch_size], dtype=tf.int32) * beam_width

    for i in range(sequence_length):
        ids = tf.reshape(ids, [batch_size * beam_width])
        logits = None
        new_states = []
        for k, (update_fun, state) in enumerate(zip(update_funs, states), 1):
            state = tf.reshape(state, [batch_size * beam_width, tf.shape(state)[2]])

            scope = tf.get_variable_scope() if len(states) == 1 else 'model_{}'.format(k)
            with tf.variable_scope(scope, reuse=True):
                state, logits_ = update_fun(state, ids, i)

            state = tf.reshape(state, [batch_size, beam_width, tf.shape(state)[1]])
            new_states.append(state)

            num_classes = logits_.shape.as_list()[-1]
            logits_ = tf.reshape(logits_, [batch_size, beam_width, num_classes])
            logits_ = log_softmax(logits_, axis=2, temperature=temperature)

            if logits is None:
                logits = logits_
            else:
                logits += logits_

        states = new_states

        if early_stopping:
            pad = tf.ones(shape=[batch_size, beam_width, sequence_length - tf.shape(sel_ids)[2] + 1], dtype=tf.int32)
            pad *= utils.EOS_ID
            finished_hypotheses_ = tf.concat([sel_ids, pad], axis=2)
            finished_scores_ = sel_sum_logprobs + logits[:,:,utils.EOS_ID]
            logits = logits[:,:,utils.EOS_ID + 1:]  # FIXME, only works if EOS is before any other symbol that matters
        else:
            finished_scores_ = None
            finished_hypotheses_ = None

        sum_logprobs = (tf.expand_dims(sel_sum_logprobs, axis=2) + (logits * tf.expand_dims(mask, axis=2)))

        num_classes = logits.shape.as_list()[-1]
        sel_sum_logprobs, indices = tf.nn.top_k(
            tf.reshape(sum_logprobs, [batch_size, num_classes * beam_width]),
            k=beam_width)

        if early_stopping:
            scores_ = tf.concat([sel_sum_logprobs, finished_scores_], axis=1)
            _, indices_ = tf.nn.top_k(scores_, k=beam_width)
            dummy_indices = tf.ones(shape=[batch_size, beam_width], dtype=tf.int32) * 2 * beam_width

            indices_ = tf.stack([indices_, dummy_indices], axis=1)

            beam_mask = (1 - tf.sequence_mask(beam_sizes, maxlen=beam_width, dtype=tf.int32))
            batch_range = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, beam_width])
            beam_range = tf.tile(tf.expand_dims(tf.range(beam_width), axis=0), [batch_size, 1])
            filter_ = tf.stack([batch_range, beam_mask, beam_range], axis=2)
            indices_ = tf.gather_nd(indices_, filter_)

            mask_ = tf.reduce_sum(tf.one_hot(indices_, depth=2 * beam_width, dtype=tf.int32), axis=1)
            hyp_count = tf.reduce_sum(mask_[:,beam_width:], axis=1)
            beam_sizes = tf.maximum(beam_sizes - hyp_count, 0)

            penalty = (1.0 - tf.to_float(mask_)) * 1e30
            sel_sum_logprobs -= penalty[:,:beam_width]
            finished_scores_ -= penalty[:,beam_width:]

            finished_hypotheses = tf.concat([finished_hypotheses, finished_hypotheses_], axis=1)
            finished_scores = tf.concat([finished_scores, finished_scores_], axis=1)
            ids = indices % num_classes + utils.EOS_ID + 1
        else:
            ids = indices % num_classes

        beam_ids = indices // num_classes
        states = [batch_gather(state, beam_ids) for state in states]
        sel_ids = tf.concat([batch_gather(sel_ids, beam_ids), tf.expand_dims(ids, axis=2)], axis=2)
        mask = (batch_gather(mask, beam_ids) * tf.to_float(tf.not_equal(ids, utils.EOS_ID)))

    if early_stopping:
        sel_sum_logprobs = tf.concat([sel_sum_logprobs, finished_scores], axis=1)
        sel_ids = tf.concat([sel_ids, finished_hypotheses], axis=1)

    sel_ids = sel_ids[:, :, 1:]  # remove BOS symbol

    if len_normalization:
        n = tf.shape(sel_ids)[1]
        sel_ids_ = tf.reshape(sel_ids, shape=[batch_size * n, sequence_length])
        mask = get_weights(sel_ids_, utils.EOS_ID, include_first_eos=True)
        length = tf.reduce_sum(mask, axis=1)
        length = tf.reshape(length, shape=[batch_size, n])
        sel_sum_logprobs /= (length ** len_normalization)

    if early_stopping or len_normalization:
        sel_sum_logprobs, indices = tf.nn.top_k(sel_sum_logprobs, k=beam_width, sorted=True)
        indices = tf.stack([tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, beam_width]), indices], axis=2)
        sel_ids = tf.gather_nd(sel_ids, indices)

    return sel_ids, sel_sum_logprobs
