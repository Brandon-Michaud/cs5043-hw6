import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, GlobalMaxPooling1D, Dense, Embedding, Input, Conv1D
from positional_encoder import *


def create_simple_mha(input_size,
                      n_classes,
                      n_tokens,
                      n_embedding,
                      pp_filters,
                      pp_kernel_size,
                      pp_strides,
                      pp_padding,
                      pp_activation,
                      num_heads,
                      key_dim,
                      dense_layers,
                      activation_dense=None,
                      lambda_regularization=None,
                      grad_clip=None,
                      lrate=0.0001,
                      loss=None,
                      metrics=None):
    '''
    Creates a MHA for sequence modeling
    :param input_size: Length of sequence
    :param n_classes: Number of output classes
    :param n_tokens: Number of distinct tokens
    :param n_embedding: Desired size of embedding
    :param pp_filters: Number of filters for preprocessing convolution
    :param pp_activation: Activation for preprocessing convolution
    :param pp_padding: Padding for preprocessing convolution
    :param pp_strides: Strides for preprocessing convolution
    :param pp_kernel_size: Kernel size for preprocessing convolution
    :param num_heads: Number of attention heads per MHA layer; array
    :param key_dim: Number of embedded dimensions per attention head per MHA layer; array
    :param dense_layers: Number of nodes for dense layers: array
    :param activation_dense: Activation function for dense layers
    :param lambda_regularization: L2 regularization hyperparameter
    :param grad_clip: Gradient clipping cutoff
    :param lrate: Learning rate
    :param loss: Loss function
    :param metrics: Additional metrics to record
    '''
    # Regularization
    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    # Input
    tensor = Input(shape=(input_size,))
    input_tensor = tensor

    # Embeddings
    tensor = Embedding(input_dim=n_tokens, output_dim=n_embedding, input_length=input_size)(tensor)

    # Preprocessing
    tensor = Conv1D(filters=pp_filters, kernel_size=pp_kernel_size, strides=pp_strides, padding=pp_padding,
                    activation=pp_activation)(tensor)

    # Positional encoding
    tensor = PositionalEncoding(max_steps=input_size, max_dims=pp_filters)(tensor)

    # Multi-head attention
    for nh, kd in zip(num_heads, key_dim):
        tensor = MultiHeadAttention(num_heads=nh, key_dim=kd)(tensor, tensor)

    # Reduce MHA output to single hyper token
    tensor = GlobalMaxPooling1D()(tensor)

    # Dense layers
    for n_neurons in dense_layers:
        tensor = Dense(units=n_neurons,
                       activation=activation_dense,
                       use_bias=True,
                       kernel_initializer='random_uniform',
                       bias_initializer='zeros',
                       kernel_regularizer=lambda_regularization)(tensor)

    # Output layer
    tensor = Dense(units=n_classes,
                   activation='softmax',
                   kernel_initializer='random_uniform',
                   kernel_regularizer=lambda_regularization)(tensor)

    output_tensor = tensor

    # Create model from data flow
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # The optimizer determines how the gradient descent is to be done
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False, clipnorm=grad_clip)

    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model
