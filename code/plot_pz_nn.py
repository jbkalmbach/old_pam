import numpy as np
import matplotlib.pyplot as plt
from calc_metrics import point_metrics


class plot_pz_nn():

    def plot_single_results(self, train_df, test_df,
                            outname, z_low=0.0, z_high=3.5, n_bins=15,
                            density_plot=True):

        # Plot scatter plots
        fig = plt.figure(figsize=(12, 12))

        fig.add_subplot(2, 2, 1)

        train_len = len(train_df)

        if density_plot is True:
            plt.hexbin(train_df['true_z'],
                       train_df['photo_z'], bins='log', cmap='viridis')
            plt.colorbar()
        else:
            plt.scatter(train_df['true_z'], train_df['photo_z'], alpha=0.2,
                        s=2)
        plt.plot(np.arange(z_low, z_high, 0.01),
                 np.arange(z_low, z_high, 0.01),
                 ls='--', c='r')
        plt.xlabel('True Z')
        plt.ylabel('Photo Z')
        plt.xlim((z_low, z_high))
        plt.title('Training Results: %i objects' % train_len)

        fig.add_subplot(2, 2, 2)

        test_len = len(test_df)

        if density_plot is True:
            plt.hexbin(test_df['true_z'], test_df['photo_z'], bins='log',
                       cmap='viridis')
            plt.colorbar()
        else:
            plt.scatter(test_df['true_z'], test_df['photo_z'], alpha=0.2,
                        s=2)
        plt.plot(np.arange(z_low, z_high, 0.01),
                 np.arange(z_low, z_high, 0.01),
                 ls='--', c='r')
        plt.xlabel('True Z')
        plt.ylabel('Photo Z')
        plt.title('Test Results: %i objects' % test_len)
        plt.xlim((z_low, z_high))

        fig.add_subplot(2, 2, 3)

        pm = point_metrics()

        bias = pm.photo_z_robust_bias(test_df['photo_z'],
                                      test_df['true_z'], z_high, n_bins)
        plt.plot(np.linspace(z_low, z_high, n_bins), bias)
        plt.xlabel('True Z')
        plt.ylabel('Robust Bias')

        fig.add_subplot(2, 2, 4)

        stdev_iqr = pm.photo_z_robust_stdev(test_df['photo_z'],
                                            test_df['true_z'], z_high, n_bins)
        plt.plot(np.linspace(z_low, z_high, n_bins), stdev_iqr)
        plt.xlabel('True Z')
        plt.ylabel('Robust Standard Deviation')

        plt.tight_layout()
        plt.savefig(outname)

    def plot_multiple_results(self, train_results_list, test_results_list,
                              suffixes, outname, scatter_index=1,
                              z_low=0.0, z_high=3.5, n_bins=15):

        fig = plt.figure(figsize=(12, 12))
        
        fig.add_subplot(2, 2, 1)

        pm = point_metrics()

        for idx in range(len(suffixes)):
            bias = pm.photo_z_robust_bias(train_results_list[idx]['photo_z'],
                                          train_results_list[idx]['true_z'],
                                          z_high, n_bins)
            plt.plot(np.linspace(z_low, z_high, n_bins), bias, label=suffixes[idx])
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
            plt.plot(np.linspace(z_low, z_high, n_bins), stdev_iqr,
                     label=suffixes[idx])
        plt.xlabel('True Z')
        plt.ylabel('Robust Standard Deviation')
        plt.title('Training Set Standard Deviaiton')

        fig.add_subplot(2, 2, 3)

        pm = point_metrics()

        for idx in range(len(suffixes)):
            #bias = pm.photo_z_robust_bias(test_results_list[idx]['photo_z'],
            bias = pm.photo_z_bias(test_results_list[idx]['photo_z'],
                                          test_results_list[idx]['true_z'],
                                          z_high, n_bins)
            plt.plot(np.linspace(z_low, z_high, n_bins), bias, label=suffixes[idx])
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
            plt.plot(np.linspace(z_low, z_high, n_bins), stdev_iqr,
                     label=suffixes[idx])
        plt.xlabel('True Z')
        plt.ylabel('Robust Standard Deviation')
        plt.title('Test Set Standard Deviaiton')

        plt.tight_layout()
        plt.savefig(outname)

    def plot_cut_results(self, test_results_list,
                         suffixes, outname, scatter_index=1,
                         z_low=0.0, z_high=3.5, n_bins=15):

        fig = plt.figure(figsize=(12, 12))

        fig.add_subplot(2, 2, 1)

        pm = point_metrics()

        for idx in range(len(suffixes)):
            plt.hist(test_results_list[idx]['true_z'], histtype='step',
                     label=suffixes[idx], bins=np.linspace(z_low, z_high, n_bins),
                     density=True, lw=4)
            plt.legend()
        plt.xlabel('Redshift')
        plt.ylabel('Density')

        fig.add_subplot(2, 2, 2)

        plt.scatter(test_results_list[scatter_index]['true_z'],
                    test_results_list[scatter_index]['photo_z'], alpha=0.2, s=2)
        plt.plot(np.arange(0, z_high, 0.01), np.arange(z_low, z_high, 0.01),
                 ls='--', c='r')
        plt.xlabel('True Z')
        plt.ylabel('Photo Z')

        fig.add_subplot(2, 2, 3)

        pm = point_metrics()

        for idx in range(len(suffixes)):
            bias = pm.photo_z_robust_bias(test_results_list[idx]['photo_z'],
                                          test_results_list[idx]['true_z'],
                                          z_high, n_bins)
            plt.plot(np.linspace(z_low, z_high, n_bins), bias, label=suffixes[idx])
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
            plt.plot(np.linspace(z_low, z_high, n_bins), stdev_iqr,
                     label=suffixes[idx])
        plt.xlabel('True Z')
        plt.ylabel('Robust Standard Deviation')
        plt.title('Test Set Standard Deviaiton')

        plt.tight_layout()
        plt.savefig(outname)