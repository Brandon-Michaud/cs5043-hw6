import numpy as np
import matplotlib.pyplot as plt
import pickle


def scatter_accuracies():
    # Load accuracies from results files
    rnn_test_accuracy = np.empty(5)
    cnn_test_accuracy = np.empty(5)
    for r in range(5):
        with open(f'results/rnn_rnns_32_16_8_ract_tanh_dense_128_64_32_dact_elu_gc_0.001000_lrate_0.000100_rot_{r}_'
                  f'results.pkl', "rb") as fp:
            results = pickle.load(fp)
            rnn_test_accuracy[r] = results['predict_testing_eval'][1]
        with open(f'results/cnn_filters_32_64_128_256_ksizes_128_64_32_16_pool_2_pad_valid_cact_elu_dense_128_64_32_'
                  f'dact_elu_gc_0.010000_lrate_0.000100_rot_{r}_results.pkl', "rb") as fp:
            results = pickle.load(fp)
            cnn_test_accuracy[r] = results['predict_testing_eval'][1]

    # Make scatter plot of accuracies
    fig = plt.figure()
    for i in range(5):
        plt.scatter(rnn_test_accuracy[i], cnn_test_accuracy[i], label=f'Rotation {i}')
    plt.legend()
    plt.xlabel('RNN Accuracy')
    plt.ylabel('CNN Accuracy')
    plt.title('Model Accuracy')
    plt.axis('equal')
    fig.savefig('figures/model_accuracies.png')


if __name__ == '__main__':
    scatter_accuracies()
