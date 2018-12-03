import numpy as np
import matplotlib.pyplot as plt
from calc_metrics import point_metrics


def plot_pz_results(train_true, train_photo_z, test_true, test_photo_z,
                    outname, z_high=3.5, n_bins=15):

    # Plot scatter plots
    fig = plt.figure(figsize=(12, 12))

    fig.add_subplot(2, 2, 1)

    train_len = len(train_true)

    plt.hexbin(train_true, 
               train_photo_z, bins='log', cmap='viridis')
    plt.plot(np.arange(0, z_high, 0.01), np.arange(0, z_high, 0.01),
             ls='--', c='r')
    plt.xlabel('True Z')
    plt.ylabel('Photo Z')
    plt.title('Training Results: %i objects' % train_len)
    plt.colorbar()

    fig.add_subplot(2, 2, 2)

    test_len = len(test_true)

    plt.hexbin(test_true, test_photo_z, bins='log',
               cmap='viridis')
    plt.plot(np.arange(0, z_high, 0.01), np.arange(0, z_high, 0.01),
             ls='--', c='r')
    plt.xlabel('True Z')
    plt.ylabel('Photo Z')
    plt.title('Test Results: %i objects' % test_len)
    plt.colorbar()

    fig.add_subplot(2, 2, 3)

    pm = point_metrics()

    bias = pm.photo_z_robust_bias(test_photo_z,
                                  test_true, z_high, n_bins)
    plt.plot(np.linspace(0, 3.5, 15), bias)
    plt.xlabel('True Z')
    plt.ylabel('Robust Bias')

    fig.add_subplot(2, 2, 4)

    stdev_iqr = pm.photo_z_robust_stdev(test_photo_z,
                                        test_true, z_high, n_bins)
    plt.plot(np.linspace(0, 3.5, 15), stdev_iqr)
    plt.xlabel('True Z')
    plt.ylabel('Robust Standard Deviation')

    plt.tight_layout()
    plt.savefig('%s.pdf' % outname)
