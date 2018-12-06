import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class create_cats():

    def __init__(self, cat_file, names):

        self.cat_file = cat_file
        self.cat_names = names

    def get_catalog(self):

        cat_array = np.genfromtxt(self.cat_file,
                                  names=self.cat_names)

        return cat_array

    def plot_color_color(self, mag_cat, filename):

        color_cat = mag_cat[:, :-1] - mag_cat[:, 1:]

        fig = plt.figure(figsize=(12, 12))
        color_names = ['u-g', 'g-r', 'r-i', 'i-z', 'z-y']

        for color_idx in range(4):

            fig.add_subplot(2, 2, color_idx+1)

            plt.hexbin(color_cat[:, color_idx],
                       color_cat[:, color_idx+1], bins='log')

            plt.xlabel(color_names[color_idx])
            plt.ylabel(color_names[color_idx+1])

        plt.tight_layout()
        plt.savefig(filename)

    def create_base_cats(self, out_suffix, train_len,
                         out_dir='.', plot_color=True,
                         cc_plot_name='train_color_color'):

        cat_df = pd.DataFrame(self.get_catalog())

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(cat_df))
        test_len = len(cat_df) - train_len

        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[:train_len]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[train_len:]

        if plot_color is True:
            self.plot_color_color(train_input[['u', 'g', 'r',
                                               'i', 'z', 'y']].values,
                                  os.path.join(out_dir,
                                               '%s.pdf' % cc_plot_name))

        print('Training set size: %i. Test set size: %i.' % (train_len,
                                                             test_len))

        train_input.to_csv(os.path.join(out_dir,
                                        'train_cat_%s.dat' % out_suffix),
                           index=False)
        test_input.to_csv(os.path.join(out_dir,
                                       'test_cat_%s.dat' % out_suffix),
                          index=False)

    def create_sparse_cats(self, out_suffix, train_len, sparsity=2,
                           out_dir='.', plot_color=True,
                           cc_plot_name='sparse_color_color'):

        cat_df = pd.DataFrame(self.get_catalog())

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(pz_cat))
        test_len = len(cat_df) - train_len

        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[:train_len:sparsity]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[train_len:]

        if plot_color is True:
            self.plot_color_color(train_input[['u', 'g', 'r',
                                               'i', 'z', 'y']].values,
                                  os.path.join(out_dir,
                                               '%s.pdf' % cc_plot_name))

        train_len = len(train_input)
        print('Training set size: %i. Test set size: %i.' % (train_len,
                                                             test_len))

        train_input.to_csv(os.path.join(out_dir,
                                        'train_cat_%s.dat' % out_suffix),
                           index=False)
        test_input.to_csv(os.path.join(out_dir,
                                       'test_cat_%s.dat' % out_suffix),
                          index=False)

    def create_color_cut_cats(self, out_suffix, train_len, color_groups,
                              choose_out=None, out_dir='.', plot_color=True,
                              random_state=None,
                              cc_plot_name='color_cut_color_color'):

        cat_df = pd.DataFrame(self.get_catalog())

        if random_state is None:
            random_state = np.random.RandomState()
        elif type(random_state) is int:
            random_state = np.random.RandomState(random_state)

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(cat_df))
        test_len = len(cat_df) - train_len

        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[:train_len]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[train_len:]

        cat_colors = cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values
        train_colors = cat_colors[:train_len, :-1] - cat_colors[:train_len, 1:]
        test_colors = cat_colors[train_len:, :-1] - cat_colors[train_len, 1:]

        kmeans = KMeans(n_clusters=color_groups,
                        random_state=random_state).fit(train_colors)

        train_labels = kmeans.labels_
        test_labels = kmeans.predict(test_colors)

        print('Color group histogram: ')
        print(np.histogram(train_labels, bins=color_groups))

        if choose_out is None:
            choose_out = random_state.randint(0, high=color_groups)
        print('Removing Color Group %i' % choose_out)
        keep_train = np.where(train_labels != choose_out)
        train_input = train_input.iloc[keep_train]

        if plot_color is True:
            self.plot_color_color(train_input[['u', 'g', 'r',
                                               'i', 'z', 'y']].values,
                                  os.path.join(out_dir,
                                               '%s.pdf' % cc_plot_name))
            fig = plt.figure(figsize=(12,12))
            color_names = ['u-g', 'g-r', 'r-i', 'i-z', 'z-y']

            for color_idx in range(4):

                fig.add_subplot(2, 2, color_idx+1)

                plt.scatter(train_colors[::10, color_idx],
                            train_colors[::10, color_idx+1], alpha=0.2,
                            c=train_labels[::10],
                            cmap=plt.get_cmap('tab10'))

                plt.xlabel(color_names[color_idx])
                plt.ylabel(color_names[color_idx+1])
                plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'color_gap_groups.pdf'))

        train_len = len(train_input)
        print('Training set size: %i. Test set size: %i.' % (train_len,
                                                             test_len))

        train_input.to_csv(os.path.join(out_dir,
                                        'train_cat_%s.dat' % out_suffix),
                           index=False)
        test_input.to_csv(os.path.join(out_dir,
                                       'test_cat_%s.dat' % out_suffix),
                          index=False)
        np.savetxt(os.path.join(out_dir, 'train_labels_%s.dat' % out_suffix),
                   train_labels)
        np.savetxt(os.path.join(out_dir, 'test_labels_%s.dat' % out_suffix),
                   test_labels)

    def create_redshift_cut_cats(self, out_suffix, train_len, z_cut_low,
                                 z_cut_high, sparsity=None, out_dir='.',
                                 plot_color=True,
                                 cc_plot_name='redshift_cut_color_color'):

        cat_df = pd.DataFrame(self.get_catalog())

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(pz_cat))
        test_len = len(cat_df) - train_len

        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[:train_len]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[train_len:]

        if sparsity is None:
            # Cut out all points in redshift space
            train_input = train_input.query('redshift < %f or redshift > %f' %
                                            (z_cut_low, z_cut_high))
        else:
            keep_idx = np.where((train_input['redshift'].values >= z_cut_low) &
                                (train_input['redshift'].values <= z_cut_high))[0]
            # Thin redshift space out by factor of `sparsity`.
            final_idx = []
            for idx in range(train_len):
                if idx not in keep_idx:
                    final_idx.append(idx)
            
            keep_idx = keep_idx[::sparsity]
            for idx in keep_idx:
                final_idx.append(idx)

            train_input = train_input.iloc[final_idx]

        train_len = len(train_input)

        if plot_color is True:
            self.plot_color_color(train_input[['u', 'g', 'r',
                                               'i', 'z', 'y']].values,
                                  os.path.join(out_dir,
                                               '%s.pdf' % cc_plot_name))

        print('Training set size: %i. Test set size: %i.' % (train_len,
                                                             test_len))

        train_input.to_csv(os.path.join(out_dir,
                                        'train_cat_%s.dat' % out_suffix),
                           index=False)
        test_input.to_csv(os.path.join(out_dir,
                                       'test_cat_%s.dat' % out_suffix),
                          index=False)