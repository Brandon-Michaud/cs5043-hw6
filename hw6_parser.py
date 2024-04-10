'''
Advanced Machine Learning, 2024

Argument parser needed by multiple programs.

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
'''

import argparse

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='HW6', fromfile_prefix_chars='@')

    # High-level info for WandB
    parser.add_argument('--project', type=str, default='hw6', help='WandB project name')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', help='Perform the experiment even if the it was completed previously')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files")
    parser.add_argument('--dataset', type=str, default='', help='Data set directory')

    parser.add_argument('--rotation', type=int, default=0, help='Rotation to use for splits')
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')

    # Specific experiment configuration
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--sequence_length', type=int, default=10, help='Maximum length of sequence')
    parser.add_argument('--n_embedding', type=int, default=16, help='Size of embeddings')

    # Preprocessing parameters
    parser.add_argument('--pp_filters', type=int, default=64, help='Number of filters in preprocessing')
    parser.add_argument('--pp_kernel_size', type=int, default=4, help='Kernel size in preprocessing')
    parser.add_argument('--pp_strides', type=int, default=4, help='Strides in preprocessing')
    parser.add_argument('--pp_padding', type=str, default='valid', help='Padding in preprocessing')
    parser.add_argument('--pp_activation', type=str, default='elu', help='Activation in preprocessing')

    # GRU parameters
    parser.add_argument('--gru_layers', nargs='+', type=int, default=[10, 5], help='Number of units per rnn layer (sequence of ints)')
    parser.add_argument('--gru_activation', type=str, default='elu', help='Activation for rnn units')
    parser.add_argument('--unroll', action='store_true', help='Unroll rnn')
    parser.add_argument('--bidirectional', action='store_true', help='Make RNN layers bidirectional')

    # MHA parameters
    parser.add_argument('--num_heads', nargs='+', type=int, default=[4], help='Number of attention heads per MHA')
    parser.add_argument('--key_dim', nargs='+', type=int, default=[4], help='Number of embedded values examined each head per MHA')

    # GRU and MHA parameters
    parser.add_argument('--pool', type=int, default=2, help='Max pooling size')
    parser.add_argument('--padding', type=str, default='valid', help='Padding type for convolutional layers')
    parser.add_argument('--dense_layers', nargs='+', type=int, default=[10, 5], help='Number of units per dense layer (sequence of ints)')
    parser.add_argument('--dense_activation', type=str, default='elu', help='Activation for dense units')

    parser.add_argument('--batch_normalization', action='store_true', help='Turn on batch normalization')

    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate for dense layers')
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization parameter")
    parser.add_argument('--grad_clip', type=float, default=None, help='Threshold for gradient clipping')

    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=10, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=3, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4, help="Number of threads to use during batch construction")
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")
    
    parser.add_argument('--generator_seed', type=int, default=42, help="Seed used for generator configuration")
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Number of training batches per epoch (must use --repeat if you are using this)")
    parser.add_argument('--no_use_py_func', action='store_true', help="False = use py_function in creating the dataset")

    # Post
    parser.add_argument('--render', action='store_true', default=False , help='Write model image')
    parser.add_argument('--save_model', action='store_true', default=False , help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')
    
    return parser

