import tensorflow as tf
import math
from tensorflow.contrib.rnn import BasicLSTMCell, RNNCell, DropoutWrapper, MultiRNNCell
from rnn import stack_bidirectional_dynamic_rnn, CellInitializer, GRUCell, DropoutGRUCell
import utils, beam_search

def auto_reuse(fun):
    """
    Wrapper that automatically handles the `reuse' parameter.
    This is rather risky, as it can lead to reusing variables
    by mistake.
    """
    def fun_(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except ValueError as e:
            if 'reuse' in str(e):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    return fun(*args, **kwargs)
            else:
                raise e
    return fun_


get_variable = auto_reuse(tf.get_variable)
dense = auto_reuse(tf.layers.dense)


class CellWrapper(RNNCell):
    """
    Wrapper around LayerNormBasicLSTMCell, BasicLSTMCell and MultiRNNCell, to keep
    the state_is_tuple=False behavior (soon to be deprecated).
    """
    def __init__(self, cell):
        super(CellWrapper, self).__init__()
        self.cell = cell
        self.num_splits = len(cell.state_size) if isinstance(cell.state_size, tuple) else 1

    @property
    def state_size(self):
        return sum(self.cell.state_size)

    @property
    def output_size(self):
        return self.cell.output_size

    def __call__(self, inputs, state, scope=None):
        state = tf.split(value=state, num_or_size_splits=self.num_splits, axis=1)
        new_h, new_state = self.cell(inputs, state, scope=scope)
        return new_h, tf.concat(new_state, 1)


def multi_encoder(encoder_inputs, encoders, encoder_input_length, other_inputs=None, **kwargs):
    """
    Build multiple encoders according to the configuration in `encoders`, reading from `encoder_inputs`.
    The result is a list of the outputs produced by those encoders (for each time-step), and their final state.

    :param encoder_inputs: list of tensors of shape (batch_size, input_length), one tensor for each encoder.
    :param encoders: list of encoder configurations
    :param encoder_input_length: list of tensors of shape (batch_size,) (one tensor for each encoder)
    :return:
      encoder outputs: a list of tensors of shape (batch_size, input_length, encoder_cell_size), hidden states of the
        encoders.
      encoder state: concatenation of the final states of all encoders, tensor of shape (batch_size, sum_of_state_sizes)
      new_encoder_input_length: list of tensors of shape (batch_size,) with the true length of the encoder outputs.
        May be different than `encoder_input_length` because of maxout strides, and time pooling.
    """
    encoder_states = []
    encoder_outputs = []

    # create embeddings in the global scope (allows sharing between encoder and decoder)
    embedding_variables = []
    for encoder in encoders:
        if encoder.binary:
            embedding_variables.append(None)
            continue
        # inputs are token ids, which need to be mapped to vectors (embeddings)
        embedding_shape = [encoder.vocab_size, encoder.embedding_size]

        if encoder.embedding_initializer == 'sqrt3':
            initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
        else:
            initializer = None

        device = '/cpu:0' if encoder.embeddings_on_cpu else None
        with tf.device(device):  # embeddings can take a very large amount of memory, so
            # storing them in GPU memory can be impractical
            embedding = get_variable('embedding_{}'.format(encoder.name), shape=embedding_shape,
                                     initializer=initializer)
        embedding_variables.append(embedding)

    new_encoder_input_length = []

    for i, encoder in enumerate(encoders):
        if encoder.use_lstm is False:
            encoder.cell_type = 'GRU'

        with tf.variable_scope('encoder_{}'.format(encoder.name)):
            encoder_inputs_ = encoder_inputs[i]
            encoder_input_length_ = encoder_input_length[i]

            def get_cell(input_size=None, reuse=False):
                if encoder.cell_type.lower() == 'lstm':
                    cell = CellWrapper(BasicLSTMCell(encoder.cell_size, reuse=reuse))
                elif encoder.cell_type.lower() == 'dropoutgru':
                    cell = DropoutGRUCell(encoder.cell_size, reuse=reuse, layer_norm=encoder.layer_norm,
                                          input_size=input_size, input_keep_prob=encoder.rnn_input_keep_prob,
                                          state_keep_prob=encoder.rnn_state_keep_prob)
                else:
                    cell = GRUCell(encoder.cell_size, reuse=reuse, layer_norm=encoder.layer_norm)

                if encoder.use_dropout and encoder.cell_type.lower() != 'dropoutgru':
                    cell = DropoutWrapper(cell, input_keep_prob=encoder.rnn_input_keep_prob,
                                          output_keep_prob=encoder.rnn_output_keep_prob,
                                          state_keep_prob=encoder.rnn_state_keep_prob,
                                          variational_recurrent=encoder.pervasive_dropout,
                                          dtype=tf.float32, input_size=input_size)
                return cell

            embedding = embedding_variables[i]

            batch_size = tf.shape(encoder_inputs_)[0]
            time_steps = tf.shape(encoder_inputs_)[1]

            if embedding is not None:
                flat_inputs = tf.reshape(encoder_inputs_, [tf.multiply(batch_size, time_steps)])
                flat_inputs = tf.nn.embedding_lookup(embedding, flat_inputs)
                encoder_inputs_ = tf.reshape(flat_inputs,
                                             tf.stack([batch_size, time_steps, flat_inputs.get_shape()[1].value]))

            if other_inputs is not None:
                encoder_inputs_ = tf.concat([encoder_inputs_, other_inputs], axis=2)

            if encoder.use_dropout:
                noise_shape = [1, time_steps, 1] if encoder.pervasive_dropout else [batch_size, time_steps, 1]
                encoder_inputs_ = tf.nn.dropout(encoder_inputs_, keep_prob=encoder.word_keep_prob,
                                                noise_shape=noise_shape)

                size = tf.shape(encoder_inputs_)[2]
                noise_shape = [1, 1, size] if encoder.pervasive_dropout else [batch_size, time_steps, size]
                encoder_inputs_ = tf.nn.dropout(encoder_inputs_, keep_prob=encoder.embedding_keep_prob,
                                                noise_shape=noise_shape)

            if encoder.input_layers:
                for j, layer_size in enumerate(encoder.input_layers):
                    if encoder.input_layer_activation is not None and encoder.input_layer_activation.lower() == 'relu':
                        activation = tf.nn.relu
                    else:
                        activation = tf.tanh

                    encoder_inputs_ = dense(encoder_inputs_, layer_size, activation=activation, use_bias=True,
                                            name='layer_{}'.format(j))
                    if encoder.use_dropout:
                        encoder_inputs_ = tf.nn.dropout(encoder_inputs_, keep_prob=encoder.input_layer_keep_prob)

            # Contrary to Theano's RNN implementation, states after the sequence length are zero
            # (while Theano repeats last state)
            inter_layer_keep_prob = None if not encoder.use_dropout else encoder.inter_layer_keep_prob

            parameters = dict(
                inputs=encoder_inputs_, sequence_length=encoder_input_length_,
                dtype=tf.float32, parallel_iterations=encoder.parallel_iterations
            )

            input_size = encoder_inputs_.get_shape()[2].value
            state_size = (encoder.cell_size * 2 if encoder.cell_type.lower() == 'lstm' else encoder.cell_size)

            def get_initial_state(name='initial_state'):
                if encoder.train_initial_states:
                    initial_state = get_variable(name, initializer=tf.zeros(state_size))
                    return tf.tile(tf.expand_dims(initial_state, axis=0), [batch_size, 1])
                else:
                    return None

            if encoder.bidir:
                rnn = lambda reuse: stack_bidirectional_dynamic_rnn(
                    cells_fw=[get_cell(input_size if j == 0 else 2 * encoder.cell_size, reuse=reuse)
                              for j in range(encoder.layers)],
                    cells_bw=[get_cell(input_size if j == 0 else 2 * encoder.cell_size, reuse=reuse)
                              for j in range(encoder.layers)],
                    initial_states_fw=[get_initial_state('initial_state_fw')] * encoder.layers,
                    initial_states_bw=[get_initial_state('initial_state_bw')] * encoder.layers,
                    time_pooling=encoder.time_pooling, pooling_avg=encoder.pooling_avg,
                    **parameters)

                initializer = CellInitializer(encoder.cell_size) if encoder.orthogonal_init else None
                with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
                    try:
                        encoder_outputs_, _, encoder_states_ = rnn(reuse=False)
                    except ValueError:   # Multi-task scenario where we're reusing the same RNN parameters
                        encoder_outputs_, _, encoder_states_ = rnn(reuse=True)
            else:
                if encoder.time_pooling or encoder.final_state == 'concat_last':
                    raise NotImplementedError

                if encoder.layers > 1:
                    cell = MultiRNNCell([get_cell(input_size if j == 0 else encoder.cell_size)
                                         for j in range(encoder.layers)])
                    initial_state = (get_initial_state(),) * encoder.layers
                else:
                    cell = get_cell(input_size)
                    initial_state = get_initial_state()

                encoder_outputs_, encoder_states_ = auto_reuse(tf.nn.dynamic_rnn)(cell=cell,
                                                                                  initial_state=initial_state,
                                                                                  **parameters)

            last_backward = encoder_outputs_[:, 0, encoder.cell_size:]
            indices = tf.stack([tf.range(batch_size), encoder_input_length_ - 1], axis=1)
            last_forward = tf.gather_nd(encoder_outputs_[:, :, :encoder.cell_size], indices)
            last_forward.set_shape([None, encoder.cell_size])

            if encoder.final_state == 'concat_last': # concats last states of all backward layers (full LSTM states)
                encoder_state_ = tf.concat(encoder_states_, axis=1)
            elif encoder.final_state == 'average':
                mask = tf.sequence_mask(encoder_input_length_, maxlen=tf.shape(encoder_outputs_)[1], dtype=tf.float32)
                mask = tf.expand_dims(mask, axis=2)
                encoder_state_ = tf.reduce_sum(mask * encoder_outputs_, axis=1) / tf.reduce_sum(mask, axis=1)
            elif encoder.final_state == 'average_inputs':
                mask = tf.sequence_mask(encoder_input_length_, maxlen=tf.shape(encoder_inputs_)[1], dtype=tf.float32)
                mask = tf.expand_dims(mask, axis=2)
                encoder_state_ = tf.reduce_sum(mask * encoder_inputs_, axis=1) / tf.reduce_sum(mask, axis=1)
            elif encoder.bidir and encoder.final_state == 'last_both':
                encoder_state_ = tf.concat([last_forward, last_backward], axis=1)
            elif encoder.bidir and not encoder.final_state == 'last_forward':   # last backward hidden state
                encoder_state_ = last_backward
            else:  # last forward hidden state
                encoder_state_ = last_forward

            if encoder.bidir and encoder.bidir_projection:
                encoder_outputs_ = dense(encoder_outputs_, encoder.cell_size, use_bias=False, name='bidir_projection')

            encoder_outputs.append(encoder_outputs_)
            encoder_states.append(encoder_state_)
            new_encoder_input_length.append(encoder_input_length_)

    encoder_state = tf.concat(encoder_states, 1)
    return encoder_outputs, encoder_state, new_encoder_input_length


def compute_energy(hidden, state, attn_size, attn_keep_prob=None, pervasive_dropout=False, layer_norm=False,
                   mult_attn=False, **kwargs):
    if attn_keep_prob is not None:
        state_noise_shape = [1, tf.shape(state)[1]] if pervasive_dropout else None
        state = tf.nn.dropout(state, keep_prob=attn_keep_prob, noise_shape=state_noise_shape)
        hidden_noise_shape = [1, 1, tf.shape(hidden)[2]] if pervasive_dropout else None
        hidden = tf.nn.dropout(hidden, keep_prob=attn_keep_prob, noise_shape=hidden_noise_shape)

    if mult_attn:
        state = dense(state, attn_size, use_bias=False, name='state')
        hidden = dense(hidden, attn_size, use_bias=False, name='hidden')
        return tf.einsum('ijk,ik->ij', hidden, state)
    else:
        y = dense(state, attn_size, use_bias=not layer_norm, name='W_a')
        y = tf.expand_dims(y, axis=1)

        if layer_norm:
            y = tf.contrib.layers.layer_norm(y, scope='layer_norm_state')
            hidden = tf.contrib.layers.layer_norm(hidden, center=False, scope='layer_norm_hidden')

        f = dense(hidden, attn_size, use_bias=False, name='U_a')

        v = get_variable('v_a', [attn_size])
        s = f + y
        return tf.reduce_sum(v * tf.tanh(s), axis=2)


def compute_energy_with_filter(hidden, state, prev_weights, attn_filters, attn_filter_length,
                               **kwargs):
    hidden = tf.expand_dims(hidden, 2)

    batch_size = tf.shape(hidden)[0]
    time_steps = tf.shape(hidden)[1]
    attn_size = hidden.get_shape()[3].value

    filter_shape = [attn_filter_length * 2 + 1, 1, 1, attn_filters]
    filter_ = get_variable('filter', filter_shape)
    u = get_variable('U', [attn_filters, attn_size])
    prev_weights = tf.reshape(prev_weights, tf.stack([batch_size, time_steps, 1, 1]))
    conv = tf.nn.conv2d(prev_weights, filter_, [1, 1, 1, 1], 'SAME')
    shape = tf.stack([tf.multiply(batch_size, time_steps), attn_filters])
    conv = tf.reshape(conv, shape)
    z = tf.matmul(conv, u)
    z = tf.reshape(z, tf.stack([batch_size, time_steps, 1, attn_size]))

    y = dense(state, attn_size, use_bias=True, name='y')
    y = tf.reshape(y, [-1, 1, 1, attn_size])

    k = get_variable('W', [attn_size, attn_size])
    # dot product between tensors requires reshaping
    hidden = tf.reshape(hidden, tf.stack([tf.multiply(batch_size, time_steps), attn_size]))
    f = tf.matmul(hidden, k)
    f = tf.reshape(f, tf.stack([batch_size, time_steps, 1, attn_size]))

    v = get_variable('V', [attn_size])
    s = f + y + z
    return tf.reduce_sum(v * tf.tanh(s), [2, 3])


def global_attention(state, hidden_states, encoder, encoder_input_length, scope=None, context=None, **kwargs):
    with tf.variable_scope(scope or 'attention_{}'.format(encoder.name)):
        if context is not None and encoder.use_context:
            state = tf.concat([state, context], axis=1)

        if encoder.attn_filters:
            e = compute_energy_with_filter(hidden_states, state, attn_size=encoder.attn_size,
                                           attn_filters=encoder.attn_filters,
                                           attn_filter_length=encoder.attn_filter_length, **kwargs)
        else:
            e = compute_energy(hidden_states, state, attn_size=encoder.attn_size,
                               attn_keep_prob=encoder.attn_keep_prob, pervasive_dropout=encoder.pervasive_dropout,
                               layer_norm=encoder.layer_norm, mult_attn=encoder.mult_attn, **kwargs)

        e -= tf.reduce_max(e, axis=1, keep_dims=True)
        mask = tf.sequence_mask(encoder_input_length, maxlen=tf.shape(hidden_states)[1], dtype=tf.float32)

        T = encoder.attn_temperature or 1.0
        exp = tf.exp(e / T) * mask
        weights = exp / tf.reduce_sum(exp, axis=-1, keep_dims=True)
        weighted_average = tf.reduce_sum(tf.expand_dims(weights, 2) * hidden_states, axis=1)

        return weighted_average, weights


def no_attention(state, hidden_states, *args, **kwargs):
    batch_size = tf.shape(state)[0]
    weighted_average = tf.zeros(shape=tf.stack([batch_size, 0]))
    weights = tf.zeros(shape=[batch_size, tf.shape(hidden_states)[1]])
    return weighted_average, weights


def average_attention(hidden_states, encoder_input_length, *args, **kwargs):
    # attention with fixed weights (average of all hidden states)
    lengths = tf.to_float(tf.expand_dims(encoder_input_length, axis=1))
    mask = tf.sequence_mask(encoder_input_length, maxlen=tf.shape(hidden_states)[1])
    weights = tf.to_float(mask) / lengths
    weighted_average = tf.reduce_sum(hidden_states * tf.expand_dims(weights, axis=2), axis=1)
    return weighted_average, weights


def last_state_attention(hidden_states, encoder_input_length, *args, **kwargs):
    weights = tf.one_hot(encoder_input_length - 1, tf.shape(hidden_states)[1])
    weights = tf.to_float(weights)

    weighted_average = tf.reduce_sum(hidden_states * tf.expand_dims(weights, axis=2), axis=1)
    return weighted_average, weights


def local_attention(state, hidden_states, encoder, encoder_input_length, pos=None, scope=None,
                    context=None, **kwargs):
    batch_size = tf.shape(state)[0]
    attn_length = tf.shape(hidden_states)[1]

    if context is not None and encoder.use_context:
        state = tf.concat([state, context], axis=1)

    state_size = state.get_shape()[1].value

    with tf.variable_scope(scope or 'attention_{}'.format(encoder.name)):
        encoder_input_length = tf.to_float(tf.expand_dims(encoder_input_length, axis=1))

        if pos is not None:
            pos = tf.reshape(pos, [-1, 1])
            pos = tf.minimum(pos, encoder_input_length - 1)

        if pos is not None and encoder.attn_window_size > 0:
            # `pred_edits` scenario, where we know the aligned pos
            # when the windows size is non-zero, we concatenate consecutive encoder states
            # and map it to the right attention vector size.
            weights = tf.to_float(tf.one_hot(tf.to_int32(tf.squeeze(pos, axis=1)), depth=attn_length))

            weighted_average = []
            for offset in range(-encoder.attn_window_size, encoder.attn_window_size + 1):
                pos_ = pos + offset
                pos_ = tf.minimum(pos_, encoder_input_length - 1)
                pos_ = tf.maximum(pos_, 0)  # TODO: when pos is < 0, use <S> or </S>
                weights_ = tf.to_float(tf.one_hot(tf.to_int32(tf.squeeze(pos_, axis=1)), depth=attn_length))
                weighted_average_ = tf.reduce_sum(tf.expand_dims(weights_, axis=2) * hidden_states, axis=1)
                weighted_average.append(weighted_average_)

            weighted_average = tf.concat(weighted_average, axis=1)
            weighted_average = dense(weighted_average, encoder.attn_size)
        elif pos is not None:
            weights = tf.to_float(tf.one_hot(tf.to_int32(tf.squeeze(pos, axis=1)), depth=attn_length))
            weighted_average = tf.reduce_sum(tf.expand_dims(weights, axis=2) * hidden_states, axis=1)
        else:
            # Local attention of Luong et al. (http://arxiv.org/abs/1508.04025)
            wp = get_variable('Wp', [state_size, state_size])
            vp = get_variable('vp', [state_size, 1])

            pos = tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(state, wp)), vp))
            pos = tf.floor(encoder_input_length * pos)
            pos = tf.reshape(pos, [-1, 1])
            pos = tf.minimum(pos, encoder_input_length - 1)

            idx = tf.tile(tf.to_float(tf.range(attn_length)), tf.stack([batch_size]))
            idx = tf.reshape(idx, [-1, attn_length])

            low = pos - encoder.attn_window_size
            high = pos + encoder.attn_window_size

            mlow = tf.to_float(idx < low)
            mhigh = tf.to_float(idx > high)
            m = mlow + mhigh
            m += tf.to_float(idx >= encoder_input_length)

            mask = tf.to_float(tf.equal(m, 0.0))

            e = compute_energy(hidden_states, state, attn_size=encoder.attn_size, **kwargs)

            weights = softmax(e, mask=mask)

            sigma = encoder.attn_window_size / 2
            numerator = -tf.pow((idx - pos), tf.convert_to_tensor(2, dtype=tf.float32))
            div = tf.truediv(numerator, 2 * sigma ** 2)
            weights *= tf.exp(div)  # result of the truncated normal distribution
            # normalize to keep a probability distribution
            # weights /= (tf.reduce_sum(weights, axis=1, keep_dims=True) + 10e-12)

            weighted_average = tf.reduce_sum(tf.expand_dims(weights, axis=2) * hidden_states, axis=1)

        return weighted_average, weights


def attention(encoder, **kwargs):
    attention_functions = {
        'global': global_attention,
        'local': local_attention,
        'none': no_attention,
        'average': average_attention,
        'last_state': last_state_attention
    }

    attention_function = attention_functions.get(encoder.attention_type, global_attention)

    return attention_function(encoder=encoder, **kwargs)


def multi_attention(state, hidden_states, encoders, encoder_input_length, pos=None, aggregation_method='sum',
                    prev_weights=None, **kwargs):
    attns = []
    weights = []

    context_vector = None
    for i, (hidden, encoder, input_length) in enumerate(zip(hidden_states, encoders, encoder_input_length)):
        pos_ = pos[i] if pos is not None else None
        prev_weights_ = prev_weights[i] if prev_weights is not None else None

        hidden = beam_search.resize_like(hidden, state)
        input_length = beam_search.resize_like(input_length, state)

        context_vector, weights_ = attention(state=state, hidden_states=hidden, encoder=encoder,
                                             encoder_input_length=input_length, pos=pos_, context=context_vector,
                                             prev_weights=prev_weights_, **kwargs)
        attns.append(context_vector)
        weights.append(weights_)

    if aggregation_method == 'sum':
        context_vector = tf.reduce_sum(tf.stack(attns, axis=2), axis=2)
    else:
        context_vector = tf.concat(attns, axis=1)

    return context_vector, weights


def attention_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder, encoder_input_length,
                      feed_previous=0.0, align_encoder_id=0, feed_argmax=True, **kwargs):
    """
    :param decoder_inputs: int32 tensor of shape (batch_size, output_length)
    :param initial_state: initial state of the decoder (usually the final state of the encoder),
      as a float32 tensor of shape (batch_size, initial_state_size). This state is mapped to the
      correct state size for the decoder.
    :param attention_states: list of tensors of shape (batch_size, input_length, encoder_cell_size),
      the hidden states of the encoder(s) (one tensor for each encoder).
    :param encoders: configuration of the encoders
    :param decoder: configuration of the decoder
    :param encoder_input_length: list of int32 tensors of shape (batch_size,), tells for each encoder,
     the true length of each sequence in the batch (sequences in the same batch are padded to all have the same
     length).
    :param feed_previous: scalar tensor corresponding to the probability to use previous decoder output
      instead of the ground truth as input for the decoder (1 when decoding, between 0 and 1 when training)
    :param feed_argmax: boolean tensor, when True the greedy decoder outputs the word with the highest
    probability (argmax). When False, it samples a word from the probability distribution (softmax).
    :param align_encoder_id: outputs attention weights for this encoder. Also used when predicting edit operations
    (pred_edits), to specifify which encoder reads the sequence to post-edit (MT).

    :return:
      outputs of the decoder as a tensor of shape (batch_size, output_length, decoder_cell_size)
      attention weights as a tensor of shape (output_length, encoders, batch_size, input_length)
    """
    assert not decoder.pred_maxout_layer or decoder.cell_size % 2 == 0, 'cell size must be a multiple of 2'

    if decoder.use_lstm is False:
        decoder.cell_type = 'GRU'

    embedding_shape = [decoder.vocab_size, decoder.embedding_size]
    if decoder.embedding_initializer == 'sqrt3':
        initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    else:
        initializer = None

    device = '/cpu:0' if decoder.embeddings_on_cpu else None
    with tf.device(device):
        embedding = get_variable('embedding_{}'.format(decoder.name), shape=embedding_shape, initializer=initializer)

    input_shape = tf.shape(decoder_inputs)
    batch_size = input_shape[0]
    time_steps = input_shape[1]

    scope_name = 'decoder_{}'.format(decoder.name)
    scope_name += '/' + '_'.join(encoder.name for encoder in encoders)

    def embed(input_):
        embedded_input = tf.nn.embedding_lookup(embedding, input_)

        if decoder.use_dropout and decoder.word_keep_prob is not None:
            noise_shape = [1, 1] if decoder.pervasive_dropout else [batch_size, 1]
            embedded_input = tf.nn.dropout(embedded_input, keep_prob=decoder.word_keep_prob, noise_shape=noise_shape)
        if decoder.use_dropout and decoder.embedding_keep_prob is not None:
            size = tf.shape(embedded_input)[1]
            noise_shape = [1, size] if decoder.pervasive_dropout else [batch_size, size]
            embedded_input = tf.nn.dropout(embedded_input, keep_prob=decoder.embedding_keep_prob,
                                           noise_shape=noise_shape)

        return embedded_input

    def get_cell(input_size=None, reuse=False):
        cells = []

        for j in range(decoder.layers):
            input_size_ = input_size if j == 0 else decoder.cell_size

            if decoder.cell_type.lower() == 'lstm':
                cell = CellWrapper(BasicLSTMCell(decoder.cell_size, reuse=reuse))
            elif decoder.cell_type.lower() == 'dropoutgru':
                cell = DropoutGRUCell(decoder.cell_size, reuse=reuse, layer_norm=decoder.layer_norm,
                                      input_size=input_size_, input_keep_prob=decoder.rnn_input_keep_prob,
                                      state_keep_prob=decoder.rnn_state_keep_prob)
            else:
                cell = GRUCell(decoder.cell_size, reuse=reuse, layer_norm=decoder.layer_norm)

            if decoder.use_dropout and decoder.cell_type.lower() != 'dropoutgru':
                cell = DropoutWrapper(cell, input_keep_prob=decoder.rnn_input_keep_prob,
                                      output_keep_prob=decoder.rnn_output_keep_prob,
                                      state_keep_prob=decoder.rnn_state_keep_prob,
                                      variational_recurrent=decoder.pervasive_dropout,
                                      dtype=tf.float32, input_size=input_size_)
            cells.append(cell)

        if len(cells) == 1:
            return cells[0]
        else:
            return CellWrapper(MultiRNNCell(cells))

    def look(state, input_, prev_weights=None, pos=None):
        prev_weights_ = [prev_weights if i == align_encoder_id else None for i in range(len(encoders))]
        pos_ = None
        if decoder.pred_edits:
            pos_ = [pos if i == align_encoder_id else None for i in range(len(encoders))]
        if decoder.attn_prev_word:
            state = tf.concat([state, input_], axis=1)

        parameters = dict(hidden_states=attention_states, encoder_input_length=encoder_input_length,
                          encoders=encoders, aggregation_method=decoder.aggregation_method)
        context, new_weights = multi_attention(state, pos=pos_, prev_weights=prev_weights_, **parameters)
        if decoder.context_mapping:
            with tf.variable_scope(scope_name):
                activation = tf.nn.tanh if decoder.context_mapping_activation == 'tanh' else None
                use_bias = not decoder.context_mapping_no_bias
                context = dense(context, decoder.context_mapping, use_bias=use_bias, activation=activation,
                                name='context_mapping')

        return context, new_weights[align_encoder_id]

    def update(state, input_, context=None, symbol=None):
        if context is not None and decoder.rnn_feed_attn:
            input_ = tf.concat([input_, context], axis=1)
        input_size = input_.get_shape()[1].value

        initializer = CellInitializer(decoder.cell_size) if decoder.orthogonal_init else None
        with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
            try:
                output, new_state = get_cell(input_size)(input_, state)
            except ValueError:  # auto_reuse doesn't work with LSTM cells
                output, new_state = get_cell(input_size, reuse=True)(input_, state)

        if decoder.skip_update and decoder.pred_edits and symbol is not None:
            is_del = tf.equal(symbol, utils.DEL_ID)
            new_state = tf.where(is_del, state, new_state)

        if decoder.cell_type.lower() == 'lstm' and decoder.use_lstm_full_state:
            output = new_state

        return output, new_state

    def update_pos(pos, symbol, max_pos=None):
        if not decoder.pred_edits:
            return pos

        is_keep = tf.equal(symbol, utils.KEEP_ID)
        is_del = tf.equal(symbol, utils.DEL_ID)
        is_not_ins = tf.logical_or(is_keep, is_del)

        pos = beam_search.resize_like(pos, symbol)
        max_pos = beam_search.resize_like(max_pos, symbol)

        pos += tf.to_float(is_not_ins)
        if max_pos is not None:
            pos = tf.minimum(pos, tf.to_float(max_pos))
        return pos

    def generate(state, input_, context):
        if decoder.pred_use_lstm_state is False:  # for back-compatibility
            state = state[:,-decoder.cell_size:]

        projection_input = [state, context]
        if decoder.use_previous_word:
            projection_input.insert(1, input_)  # for back-compatibility

        output_ = tf.concat(projection_input, axis=1)

        if decoder.pred_deep_layer:
            deep_layer_size = decoder.pred_deep_layer_size or decoder.embedding_size
            if decoder.layer_norm:
                output_ = dense(output_, deep_layer_size, use_bias=False, name='deep_output')
                output_ = tf.contrib.layers.layer_norm(output_, activation_fn=tf.nn.tanh, scope='output_layer_norm')
            else:
                output_ = dense(output_, deep_layer_size, activation=tf.tanh, use_bias=True, name='deep_output')

            if decoder.use_dropout:
                size = tf.shape(output_)[1]
                noise_shape = [1, size] if decoder.pervasive_dropout else None
                output_ = tf.nn.dropout(output_, keep_prob=decoder.deep_layer_keep_prob, noise_shape=noise_shape)
        else:
            if decoder.pred_maxout_layer:
                maxout_size = decoder.maxout_size or decoder.cell_size
                output_ = dense(output_, maxout_size, use_bias=True, name='maxout')
                if decoder.old_maxout:  # for back-compatibility with old models
                    output_ = tf.nn.pool(tf.expand_dims(output_, axis=2), window_shape=[2], pooling_type='MAX',
                                         padding='SAME', strides=[2])
                    output_ = tf.squeeze(output_, axis=2)
                else:
                    output_ = tf.maximum(*tf.split(output_, num_or_size_splits=2, axis=1))

            if decoder.pred_embed_proj:
                # intermediate projection to embedding size (before projecting to vocabulary size)
                # this is useful to reduce the number of parameters, and
                # to use the output embeddings for output projection (tie_embeddings parameter)
                output_ = dense(output_, decoder.embedding_size, use_bias=False, name='softmax0')

        if decoder.tie_embeddings and (decoder.pred_embed_proj or decoder.pred_deep_layer):
            bias = get_variable('softmax1/bias', shape=[decoder.vocab_size])
            output_ = tf.matmul(output_, tf.transpose(embedding)) + bias
        else:
            output_ = dense(output_, decoder.vocab_size, use_bias=True, name='softmax1')
        return output_

    state_size = (decoder.cell_size * 2 if decoder.cell_type.lower() == 'lstm' else decoder.cell_size) * decoder.layers

    if decoder.use_dropout:
        initial_state = tf.nn.dropout(initial_state, keep_prob=decoder.initial_state_keep_prob)

    with tf.variable_scope(scope_name):
        if decoder.layer_norm:
            initial_state = dense(initial_state, state_size, use_bias=False, name='initial_state_projection')
            initial_state = tf.contrib.layers.layer_norm(initial_state, activation_fn=tf.nn.tanh,
                                                         scope='initial_state_layer_norm')
        else:
            initial_state = dense(initial_state, state_size, use_bias=True, name='initial_state_projection',
                                  activation=tf.nn.tanh)

    if decoder.cell_type.lower() == 'lstm' and decoder.use_lstm_full_state:
        initial_output = initial_state
    else:
        initial_output = initial_state[:, -decoder.cell_size:]

    time = tf.constant(0, dtype=tf.int32, name='time')
    outputs = tf.TensorArray(dtype=tf.float32, size=time_steps)
    samples = tf.TensorArray(dtype=tf.int64, size=time_steps)
    inputs = tf.TensorArray(dtype=tf.int64, size=time_steps).unstack(tf.to_int64(tf.transpose(decoder_inputs)))

    states = tf.TensorArray(dtype=tf.float32, size=time_steps)
    weights = tf.TensorArray(dtype=tf.float32, size=time_steps)
    attns = tf.TensorArray(dtype=tf.float32, size=time_steps)

    initial_symbol = inputs.read(0)  # first symbol is BOS
    initial_input = embed(initial_symbol)
    initial_pos = tf.zeros([batch_size], tf.float32)
    initial_weights = tf.zeros(tf.shape(attention_states[align_encoder_id])[:2])
    initial_context, _ = look(initial_output, initial_input, pos=initial_pos, prev_weights=initial_weights)
    initial_data = tf.concat([initial_state, initial_context, tf.expand_dims(initial_pos, axis=1), initial_weights],
                             axis=1)
    context_size = initial_context.shape[1].value

    def get_logits(state, ids, time):  # for beam-search decoding
        with tf.variable_scope('decoder_{}'.format(decoder.name)):
            state, context, pos, prev_weights = tf.split(state, [state_size, context_size, 1, -1], axis=1)
            input_ = embed(ids)

            pos = tf.squeeze(pos, axis=1)
            pos = tf.cond(tf.equal(time, 0),
                          lambda: pos,
                          lambda: update_pos(pos, ids, encoder_input_length[align_encoder_id]))

            if decoder.cell_type.lower() == 'lstm' and decoder.use_lstm_full_state:
                output = state
            else:
                # output is always the right-most part of state. However, this only works at test time,
                # because different dropout operations can be used on state and output.
                output = state[:, -decoder.cell_size:]

            if decoder.conditional_rnn:
                with tf.variable_scope('conditional_1'):
                    output, state = update(state, input_)
            elif decoder.update_first:
                output, state = update(state, input_, None, ids)
            elif decoder.generate_first:
                output, state = tf.cond(tf.equal(time, 0),
                                        lambda: (output, state),
                                        lambda: update(state, input_, context, ids))

            context, new_weights = look(output, input_, pos=pos, prev_weights=prev_weights)

            if decoder.conditional_rnn:
                with tf.variable_scope('conditional_2'):
                    output, state = update(state, context)
            elif not decoder.generate_first:
                output, state = update(state, input_, context, ids)

            logits = generate(output, input_, context)

            pos = tf.expand_dims(pos, axis=1)
            state = tf.concat([state, context, pos, new_weights], axis=1)
            return state, logits

    def _time_step(time, input_, input_symbol, pos, state, output, outputs, states, weights, attns, prev_weights,
                   samples):
        if decoder.conditional_rnn:
            with tf.variable_scope('conditional_1'):
                output, state = update(state, input_)
        elif decoder.update_first:
            output, state = update(state, input_, None, input_symbol)

        context, new_weights = look(output, input_, pos=pos, prev_weights=prev_weights)

        if decoder.conditional_rnn:
            with tf.variable_scope('conditional_2'):
                output, state = update(state, context)
        elif not decoder.generate_first:
            output, state = update(state, input_, context, input_symbol)

        output_ = generate(output, input_, context)

        argmax = lambda: tf.argmax(output_, 1)
        target = lambda: inputs.read(time + 1)
        softmax = lambda: tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(output_)), num_samples=1),
                                     axis=1)

        use_target = tf.logical_and(time < time_steps - 1, tf.random_uniform([]) >= feed_previous)
        predicted_symbol = tf.case([
            (use_target, target),
            (tf.logical_not(feed_argmax), softmax)],
            default=argmax)   # default case is useful for beam-search

        predicted_symbol.set_shape([None])
        predicted_symbol = tf.stop_gradient(predicted_symbol)
        samples = samples.write(time, predicted_symbol)

        input_ = embed(predicted_symbol)
        pos = update_pos(pos, predicted_symbol, encoder_input_length[align_encoder_id])

        attns = attns.write(time, context)
        weights = weights.write(time, new_weights)
        states = states.write(time, state)
        outputs = outputs.write(time, output_)

        if not decoder.conditional_rnn and not decoder.update_first and decoder.generate_first:
            output, state = update(state, input_, context, predicted_symbol)

        return (time + 1, input_, predicted_symbol, pos, state, output, outputs, states, weights, attns, new_weights,
                samples)

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        _, _, _, new_pos, new_state, _, outputs, states, weights, attns, new_weights, samples = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, initial_input, initial_symbol, initial_pos, initial_state, initial_output, outputs,
                       weights, states, attns, initial_weights, samples),
            parallel_iterations=decoder.parallel_iterations,
            swap_memory=decoder.swap_memory)

    outputs = outputs.stack()
    weights = weights.stack()  # batch_size, encoders, output time, input time
    states = states.stack()
    attns = attns.stack()
    samples = samples.stack()

    # put batch_size as first dimension
    outputs = tf.transpose(outputs, perm=(1, 0, 2))
    weights = tf.transpose(weights, perm=(1, 0, 2))
    states = tf.transpose(states, perm=(1, 0, 2))
    attns = tf.transpose(attns, perm=(1, 0, 2))
    samples = tf.transpose(samples)

    return outputs, weights, states, attns, samples, get_logits, initial_data


def encoder_decoder(encoders, decoders, encoder_inputs, targets, feed_previous, align_encoder_id=0,
                    encoder_input_length=None, feed_argmax=True, **kwargs):
    decoder = decoders[0]
    targets = targets[0]  # single decoder

    if encoder_input_length is None:
        encoder_input_length = []
        for encoder_inputs_ in encoder_inputs:
            weights = get_weights(encoder_inputs_, utils.EOS_ID, include_first_eos=True)
            encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

    parameters = dict(encoders=encoders, decoder=decoder, encoder_inputs=encoder_inputs,
                      feed_argmax=feed_argmax)

    target_weights = get_weights(targets[:, 1:], utils.EOS_ID, include_first_eos=True)

    attention_states, encoder_state, encoder_input_length = multi_encoder(
        encoder_input_length=encoder_input_length, **parameters)

    outputs, attention_weights, _, _, samples, beam_fun, initial_data = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state, feed_previous=feed_previous,
        decoder_inputs=targets[:, :-1], align_encoder_id=align_encoder_id, encoder_input_length=encoder_input_length,
        **parameters
    )

    xent_loss = sequence_loss(logits=outputs, targets=targets[:, 1:], weights=target_weights)
    losses = xent_loss

    return losses, [outputs], encoder_state, attention_states, attention_weights, samples, beam_fun, initial_data


def chained_encoder_decoder(encoders, decoders, encoder_inputs, targets, feed_previous,
                            chaining_strategy=None, align_encoder_id=0, chaining_non_linearity=False,
                            chaining_loss_ratio=1.0, chaining_stop_gradient=False, **kwargs):
    decoder = decoders[0]
    targets = targets[0]  # single decoder

    assert len(encoders) == 2

    encoder_input_length = []
    input_weights = []
    for encoder_inputs_ in encoder_inputs:
        weights = get_weights(encoder_inputs_, utils.EOS_ID, include_first_eos=True)
        input_weights.append(weights)
        encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

    target_weights = get_weights(targets[:, 1:], utils.EOS_ID, include_first_eos=True)

    parameters = dict(encoders=encoders[1:], decoder=encoders[0])

    attention_states, encoder_state, encoder_input_length[1:] = multi_encoder(
        encoder_inputs[1:], encoder_input_length=encoder_input_length[1:], **parameters)

    decoder_inputs = encoder_inputs[0][:, :-1]
    batch_size = tf.shape(decoder_inputs)[0]

    pad = tf.ones(shape=tf.stack([batch_size, 1]), dtype=tf.int32) * utils.BOS_ID
    decoder_inputs = tf.concat([pad, decoder_inputs], axis=1)

    outputs, _, states, attns, _, _, _ = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state, decoder_inputs=decoder_inputs,
        encoder_input_length=encoder_input_length[1:], **parameters
    )

    chaining_loss = sequence_loss(logits=outputs, targets=encoder_inputs[0], weights=input_weights[0])

    if decoder.cell_type.lower() == 'lstm':
        size = states.get_shape()[2].value
        decoder_outputs = states[:, :, size // 2:]
    else:
        decoder_outputs = states

    if chaining_strategy == 'share_states':
        other_inputs = states
    elif chaining_strategy == 'share_outputs':
        other_inputs = decoder_outputs
    else:
        other_inputs = None

    if other_inputs is not None and chaining_stop_gradient:
        other_inputs = tf.stop_gradient(other_inputs)

    parameters = dict(encoders=encoders[:1], decoder=decoder, encoder_inputs=encoder_inputs[:1],
                      other_inputs=other_inputs)

    attention_states, encoder_state, encoder_input_length[:1] = multi_encoder(
        encoder_input_length=encoder_input_length[:1], **parameters)

    if chaining_stop_gradient:
        attns = tf.stop_gradient(attns)
        states = tf.stop_gradient(states)
        decoder_outputs = tf.stop_gradient(decoder_outputs)

    if chaining_strategy == 'concat_attns':
        attention_states[0] = tf.concat([attention_states[0], attns], axis=2)
    elif chaining_strategy == 'concat_states':
        attention_states[0] = tf.concat([attention_states[0], states], axis=2)
    elif chaining_strategy == 'sum_attns':
        attention_states[0] += attns
    elif chaining_strategy in ('map_attns', 'map_states', 'map_outputs'):
        if chaining_strategy == 'map_attns':
            x = attns
        elif chaining_strategy == 'map_outputs':
            x = decoder_outputs
        else:
            x = states

        shape = [x.get_shape()[-1], attention_states[0].get_shape()[-1]]

        w = tf.get_variable("map_attns/matrix", shape=shape)
        b = tf.get_variable("map_attns/bias", shape=shape[-1:])

        x = tf.einsum('ijk,kl->ijl', x, w) + b
        if chaining_non_linearity:
            x = tf.nn.tanh(x)

        attention_states[0] += x

    outputs, attention_weights, _, _, samples, beam_fun, initial_data = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state,
        feed_previous=feed_previous, decoder_inputs=targets[:,:-1],
        align_encoder_id=align_encoder_id, encoder_input_length=encoder_input_length[:1],
        **parameters
    )

    xent_loss = sequence_loss(logits=outputs, targets=targets[:, 1:],
                              weights=target_weights)

    if chaining_loss is not None and chaining_loss_ratio:
        xent_loss += chaining_loss_ratio * chaining_loss

    losses = [xent_loss, None, None]

    return losses, [outputs], encoder_state, attention_states, attention_weights, samples, beam_fun, initial_data


def softmax(logits, dim=-1, mask=None):
    e = tf.exp(logits)
    if mask is not None:
        e *= mask

    return e / tf.clip_by_value(tf.reduce_sum(e, axis=dim, keep_dims=True), 10e-37, 10e+37)


def sequence_loss(logits, targets, weights, average_across_timesteps=False, average_across_batch=True):
    batch_size = tf.shape(targets)[0]
    time_steps = tf.shape(targets)[1]

    logits_ = tf.reshape(logits, tf.stack([time_steps * batch_size, logits.get_shape()[2].value]))
    targets_ = tf.reshape(targets, tf.stack([time_steps * batch_size]))

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=targets_)
    crossent = tf.reshape(crossent, tf.stack([batch_size, time_steps]))


    log_perp = tf.reduce_sum(crossent * weights, axis=1)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, axis=1)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        log_perp /= total_size

    cost = tf.reduce_sum(log_perp)

    if average_across_batch:
        return cost / tf.to_float(batch_size)
    else:
        return cost


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
