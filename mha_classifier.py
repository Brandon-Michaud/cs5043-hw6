import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, GlobalMaxPooling1D, Dense, Embedding
from positional_encoder import *


def create_simple_mha(input_size,
                      n_classes,
                      n_tokens,
                      n_embedding,
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
    # Regularization
    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    # Create embeddings
    tensor = Embedding(input_dim=n_tokens, output_dim=n_embedding, input_length=input_size)

    # MHA
    tensor = MultiHeadAttention

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
