import numpy as np
import matplotlib.pyplot as plt
from calc_metrics import point_metrics


class plot_pz_nn():

    def plot_single_results(self, train_df, test_df,
                            outname, z_high=3.5, n_bins=15):

        # Plot scatter plots
        fig = plt.figure(figsize=(12, 12))

        fig.add_subplot(2, 2, 1)

        train_len = len(train_df)

        plt.hexbin(train_df['true_z'],
                   train_df['photo_z'], bins='log', cmap='viridis')
        plt.plot(np.arange(0, z_high, 0.01), np.arange(0, z_high, 0.01),
                 ls='--', c='r')
        plt.xlabel('True Z')
        plt.ylabel('Photo Z')
        plt.title('Training Results: %i objects' % train_len)
        plt.colorbar()

        fig.add_subplot(2, 2, 2)

        test_len = len(test_df)

        plt.hexbin(test_df['true_z'], test_df['photo_z'], bins='log',
                   cmap='viridis')
        plt.plot(np.arange(0, z_high, 0.01), np.arange(0, z_high, 0.01),
                 ls='--', c='r')
        plt.xlabel('True Z')
        plt.ylabel('Photo Z')
        plt.title('Test Results: %i objects' % test_len)
        plt.colorbar()

        fig.add_subplot(2, 2, 3)

        pm = point_metrics()

        bias = pm.photo_z_robust_bias(test_df['photo_z'],
                                      test_df['true_z'], z_high, n_bins)
        plt.plot(np.linspace(0, 3.5, 15), bias)
        plt.xlabel('True Z')
        plt.ylabel('Robust Bias')

        fig.add_subplot(2, 2, 4)

        stdev_iqr = pm.photo_z_robust_stdev(test_df['photo_z'],
                                            test_df['true_z'], z_high, n_bins)
        plt.plot(np.linspace(0, 3.5, 15), stdev_iqr)
        plt.xlabel('True Z')
        plt.ylabel('Robust Standard Deviation')

        plt.tight_layout()
        plt.savefig(outname)

    def plot_multiple_results(self, train_results_list, test_results_list,
                              suffixes, outname, z_high=3.5, n_bins=15):

        fig = plt.figure(figsize=(12, 12))

        fig.add_subplot(2, 2, 1)

        pm = point_metrics()

        for idx in range(len(suffixes)):
            bias = pm.photo_z_robust_bias(train_results_list[idx]['photo_z'],
                                          train_results_list[idx]['true_z'],
                                          z_high, n_bins)
            plt.plot(np.linspace(0, z_high, n_bins), bias, label=suffixes[idx])
        plt.xlabel('True Z')
        plt.ylabel('Robust Bias')
        plt.title('Training Set Bias')
        plt.legend()

        fig.add_subplot(2, 2, 2)

        for idx in range(len(suffixes)):
            stdev_iqr = pm.photo_z_robust_stdev(
                            train_results_list[idx]['photo_z'],
                            train_results_list[idx]['true_z'],
                            z_high, n_bins)
            plt.plot(np.linspace(0, z_high, n_bins), stdev_iqr,
                     label=suffixes[idx])
        plt.xlabel('True Z')
        plt.ylabel('Robust Standard Deviation')
        plt.title('Training Set Standard Deviation')

        fig.add_subplot(2, 2, 3)

        pm = point_metrics()

        for idx in range(len(suffixes)):
            bias = pm.photo_z_robust_bias(test_results_list[idx]['photo_z'],
                                          test_results_list[idx]['true_z'],
                                          z_high, n_bins)
            plt.plot(np.linspace(0, z_high, n_bins), bias, label=suffixes[idx])
        plt.xlabel('True Z')
        plt.ylabel('Robust Bias')
        plt.title('Test Set Bias')
        plt.legend()

        fig.add_subplot(2, 2, 4)

        for idx in range(len(suffixes)):
            stdev_iqr = pm.photo_z_robust_stdev(
                            test_results_list[idx]['photo_z'],
                            test_results_list[idx]['true_z'],
                            z_high, n_bins)
            plt.plot(np.linspace(0, z_high, n_bins), stdev_iqr,
                     label=suffixes[idx])
        plt.xlabel('True Z')
        plt.ylabel('Robust Standard Deviation')
        plt.title('Test Set Standard Deviaiton')

        plt.tight_layout()
        plt.savefig(outname)

    def plot_gap_results(self, train_results_list, test_results_list,
                         suffixes, outname, z_high=3.5, n_bins=15):

        fig = plt.figure(figsize=(12, 12))

        fig.add_subplot(2, 2, 1)

        pm = point_metrics()

        for idx in range(len(suffixes)):
            plt.hist(test_results_list[idx]['true_z'], histtype='step',
                     label=suffixes[idx], bins=np.linspace(0, z_high, n_bins),
                     normed=True)

        fig.add_subplot(2, 2, 3)

        pm = point_metrics()

        for idx in range(len(suffixes)):
            bias = pm.photo_z_robust_bias(test_results_list[idx]['photo_z'],
                                          test_results_list[idx]['true_z'],
                                          z_high, n_bins)
            plt.plot(np.linspace(0, z_high, n_bins), bias, label=suffixes[idx])
        plt.xlabel('True Z')
        plt.ylabel('Robust Bias')
        plt.title('Test Set Bias')
        plt.legend()

        fig.add_subplot(2, 2, 4)

        for idx in range(len(suffixes)):
            stdev_iqr = pm.photo_z_robust_stdev(
                            test_results_list[idx]['photo_z'],
                            test_results_list[idx]['true_z'],
                            z_high, n_bins)
            plt.plot(np.linspace(0, z_high, n_bins), stdev_iqr,
                     label=suffixes[idx])
        plt.xlabel('True Z')
        plt.ylabel('Robust Standard Deviation')
        plt.title('Test Set Standard Deviaiton')

        plt.tight_layout()
        plt.savefig(outname)
