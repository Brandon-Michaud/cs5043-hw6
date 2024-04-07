import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, AveragePooling1D, Dense, Embedding, Conv1D


def create_simple_gru(input_size,
                      n_classes,
                      n_tokens,
                      n_embedding,
                      pp_filters,
                      pp_kernel_size,
                      pp_strides,
                      pp_padding,
                      pp_activation,
                      gru_layers,
                      dense_layers,
                      activation_gru=None,
                      activation_dense=None,
                      unroll=True,
                      bidirectional=False,
                      pool_size=2,
                      padding='valid',
                      lambda_regularization=None,
                      grad_clip=None,
                      lrate=0.0001,
                      loss=None,
                      metrics=None):
    '''
    Creates a CNN for sequence modeling
    :param input_size: Length of sequence
    :param n_classes: Number of output classes
    :param n_tokens: Number of distinct tokens
    :param n_embedding: Desired size of embedding
    :param rnn_layers: Number of nodes for convolutional layers; array
    :param dense_layers: Number of nodes for dense layers: array
    :param pool_size: Size for max pooling between layers
    :param padding: Type of padding to use; valid or same
    :param activation_rnn: Activation function for recurrent layers
    :param activation_dense: Activation function for dense layers
    :param unroll: Unroll RNN
    :param bidirectional: Make simple RNN layers bidirectional
    :param lambda_regularization: L2 regularization hyperparameter
    :param grad_clip: Gradient clipping cutoff
    :param lrate: Learning rate
    :param loss: Loss function
    :param metrics: Additional metrics to record
    '''
    # Regularization
    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    model = Sequential()

    # Create embeddings
    model.add(Embedding(input_dim=n_tokens, output_dim=n_embedding, input_length=input_size))

    # Preprocessing
    model.add(Conv1D(filters=pp_filters, kernel_size=pp_kernel_size, strides=pp_strides, padding=pp_padding,
                     activation=pp_activation))

    # RNN layers
    for i, n_neurons in enumerate(gru_layers):
        if bidirectional:
            model.add(Bidirectional(GRU(n_neurons,
                                        activation=activation_gru,
                                        use_bias=True,
                                        return_sequences=(i != len(gru_layers) - 1),
                                        kernel_initializer='random_uniform',
                                        bias_initializer='zeros',
                                        kernel_regularizer=lambda_regularization,
                                        unroll=unroll)))
        else:
            model.add(GRU(n_neurons,
                          activation=activation_gru,
                          use_bias=True,
                          return_sequences=(i != len(gru_layers) - 1),
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          unroll=unroll))
        if i != len(gru_layers) - 1:
            model.add(AveragePooling1D(pool_size=pool_size, strides=pool_size, padding=padding))

    # Dense layers
    for n_neurons in dense_layers:
        model.add(Dense(units=n_neurons,
                        activation=activation_dense,
                        use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=lambda_regularization))

    # Output layer
    model.add(Dense(units=n_classes,
                    activation='softmax',
                    kernel_initializer='random_uniform',
                    kernel_regularizer=lambda_regularization))

    # The optimizer determines how the gradient descent is to be done
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False, clipnorm=grad_clip)

    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model
