import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
                         cc_plot_name='train_color_color',
                         random_state=None):

        if ((type(random_state) is int) | (random_state is None)):
            random_state = np.random.RandomState(random_state)

        cat_df = pd.DataFrame(self.get_catalog())

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(cat_df))
        test_len = len(cat_df) - train_len

        shuffled_idx = random_state.choice(np.arange(len(cat_df)),
                                           size=len(cat_df),
                                           replace=False)

        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[shuffled_idx[:train_len]]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[shuffled_idx[train_len:]]

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
                           out_dir='.', plot_color=True, save_test=False,
                           cc_plot_name='sparse_color_color',
                           random_state=None):

        if ((type(random_state) is int) | (random_state is None)):
            random_state = np.random.RandomState(random_state)

        cat_df = pd.DataFrame(self.get_catalog())

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(cat_df))
        test_len = len(cat_df) - train_len

        shuffled_idx = random_state.choice(np.arange(len(cat_df)),
                                           size=len(cat_df),
                                           replace=False)

        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[shuffled_idx[:train_len:sparsity]]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[shuffled_idx[train_len:]]

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
        if save_test is True:
            test_input.to_csv(os.path.join(out_dir,
                                           'test_cat_%s.dat' % out_suffix),
                              index=False)

    def create_color_group_cats(self, out_suffix, train_len, color_groups,
                                sparsity=None,
                                choose_out=None, out_dir='.', plot_color=True,
                                random_state=None, save_test=False,
                                cc_plot_name='group_cut_color_color'):

        cat_df = pd.DataFrame(self.get_catalog())

        if random_state is None:
            random_state = np.random.RandomState()
        elif type(random_state) is int:
            random_state = np.random.RandomState(random_state)

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(cat_df))
        test_len = len(cat_df) - train_len

        shuffled_idx = random_state.choice(np.arange(len(cat_df)),
                                           size=len(cat_df),
                                           replace=False)

        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[shuffled_idx[:train_len]]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[shuffled_idx[train_len:]]

        cat_colors = cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values
        train_colors = cat_colors[shuffled_idx[:train_len], :-1] -\
            cat_colors[shuffled_idx[:train_len], 1:]
        test_colors = cat_colors[shuffled_idx[train_len:], :-1] -\
            cat_colors[shuffled_idx[train_len:], 1:]

        kmeans = KMeans(n_clusters=color_groups,
                        random_state=random_state).fit(train_colors)

        train_labels = kmeans.labels_
        test_labels = kmeans.predict(test_colors)

        print('Color group histogram: ')
        print(np.histogram(train_labels, bins=color_groups))

        if choose_out is None:
            choose_out = random_state.randint(0, high=color_groups)
        print('Removing Color Group %i' % choose_out)
        keep_train = np.where(train_labels != choose_out)[0]
        if sparsity is None:
            train_input = train_input.iloc[keep_train]
        else:
            out_train = np.where(train_labels == choose_out)[0]
            out_train = out_train[::sparsity]
            keep_train = np.concatenate((keep_train, out_train))
            train_input = train_input.iloc[keep_train]
            train_labels = train_labels[keep_train]

        if plot_color is True:
            self.plot_color_color(train_input[['u', 'g', 'r',
                                               'i', 'z', 'y']].values,
                                  os.path.join(out_dir,
                                               '%s.pdf' % cc_plot_name))
            fig = plt.figure(figsize=(14, 14))
            color_names = ['u-g', 'g-r', 'r-i', 'i-z', 'z-y']

            cmap = plt.get_cmap('tab10')
            color_vals = cmap(np.arange(color_groups))
            cm = ListedColormap(color_vals)

            for color_idx in range(4):

                fig.add_subplot(2, 2, color_idx+1)

                plt.scatter(train_colors[::10, color_idx],
                            train_colors[::10, color_idx+1], alpha=0.2,
                            c=train_labels[::10],
                            cmap=cm, vmin=0, vmax=color_groups)

                plt.xlabel(color_names[color_idx], size=14)
                plt.ylabel(color_names[color_idx+1], labelpad=1., size=14)
                plt.xticks(size=14)
                plt.yticks(size=14)
                if color_idx == 0:
                    plt.yticks([-1., 0., 1., 2., 3.], ['-1.0', '0.0', '1.0', '2.0', '3.0'])
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.8])
            cbar = plt.colorbar(cax=cbar_ax)
            cbar.solids.set(alpha=1.)
            cbar.ax.invert_yaxis()
            cbar.set_ticks(np.arange(color_groups)+0.5)
            cbar.set_ticklabels(['Group %i' % g_num for g_num
                                 in range(color_groups)])
            cbar.ax.tick_params(labelsize=14)

            #plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'color_gap_groups.pdf'))

        train_len = len(train_input)
        print('Training set size: %i. Test set size: %i.' % (train_len,
                                                             test_len))

        train_input.to_csv(os.path.join(out_dir,
                                        'train_cat_%s.dat' % out_suffix),
                           index=False)
        if save_test is True:
            test_input.to_csv(os.path.join(out_dir,
                                           'test_cat_%s.dat' % out_suffix),
                              index=False)
        np.savetxt(os.path.join(out_dir, 'train_labels_%s.dat' % out_suffix),
                   train_labels)
        np.savetxt(os.path.join(out_dir, 'test_labels_%s.dat' % out_suffix),
                   test_labels)

    def create_color_cut_cats(self, out_suffix, train_len,
                              cut_index, cut_low, cut_high, sparsity=None,
                              out_dir='.', plot_color=True, save_test=False,
                              cc_plot_name='color_cut_color_color',
                              random_state=None):

        if ((type(random_state) is int) | (random_state is None)):
            random_state = np.random.RandomState(random_state)
        
        cat_df = pd.DataFrame(self.get_catalog())

        if ((len(cut_index) != len(cut_low)) |
           (len(cut_low) != len(cut_high))):
            raise ValueError('Cut Index, Cut Low and Cut High lists ' +
                             'must be equal length')

        if sparsity is not None:
            if len(sparsity) != len(cut_index):
                raise ValueError('If sparsity is not None must be list with' +
                                 ' same length as Cut Index')
        else:
            sparsity = [None]*len(cut_index)

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(cat_df))
        test_len = len(cat_df) - train_len

        shuffled_idx = random_state.choice(np.arange(len(cat_df)),
                                           size=len(cat_df),
                                           replace=False)
                                        
        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[shuffled_idx[:train_len]]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[shuffled_idx[train_len:]]

        cat_colors = cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values
        train_colors = cat_colors[shuffled_idx[:train_len], :-1] -\
            cat_colors[shuffled_idx[:train_len], 1:]
        test_colors = cat_colors[shuffled_idx[train_len:], :-1] -\
            cat_colors[shuffled_idx[train_len:], 1:]
        train_labels = np.zeros(train_len)
        test_labels = np.zeros(test_len)

        # Cut out all points in color space
        for color_val, color_low, color_high, sparse_factor in zip(
                cut_index, cut_low, cut_high, sparsity):
            print(color_val, color_low, color_high)

            train_cut_idx = np.where((train_colors[:, color_val]
                                      >= color_low) &
                                     (train_colors[:, color_val]
                                      < color_high))[0]
            test_cut_idx = np.where((test_colors[:, color_val] >= color_low) &
                                    (test_colors[:, color_val]
                                     < color_high))[0]

            if sparse_factor is not None:
                train_cut_idx = np.delete(train_cut_idx,
                                          slice(None, None, sparse_factor))

            train_labels[train_cut_idx] = 1
            test_labels[test_cut_idx] = 1

        if plot_color is True:
            self.plot_color_color(train_input[['u', 'g', 'r',
                                               'i', 'z', 'y']].values[
                                                np.where(train_labels == 0)],
                                  os.path.join(out_dir,
                                               '%s.pdf' % cc_plot_name))

        train_input = train_input.iloc[np.where(train_labels == 0)]
        train_len = len(train_input)
        print('Training set size: %i. Test set size: %i.' % (train_len,
                                                             test_len))

        train_input.to_csv(os.path.join(out_dir,
                                        'train_cat_%s.dat' % out_suffix),
                           index=False)
        if save_test is True:
            test_input.to_csv(os.path.join(out_dir,
                                           'test_cat_%s.dat' % out_suffix),
                              index=False)
        np.savetxt(os.path.join(out_dir, 'train_labels_%s.dat' % out_suffix),
                   train_labels)
        np.savetxt(os.path.join(out_dir, 'test_labels_%s.dat' % out_suffix),
                   test_labels)

    def create_mag_cut_cats(self, out_suffix, train_len,
                            cut_band, cut_low, cut_high, sparsity=None,
                            out_dir='.', plot_color=True, save_test=False,
                            cc_plot_name='mag_cut_color_color',
                            random_state=None):

        if ((type(random_state) is int) | (random_state is None)):
            random_state = np.random.RandomState(random_state)
        
        cat_df = pd.DataFrame(self.get_catalog())

        if ((len(cut_band) != len(cut_low)) |
           (len(cut_low) != len(cut_high))):
            raise ValueError('Cut Index, Cut Low and Cut High lists ' +
                             'must be equal length')

        if sparsity is not None:
            if len(sparsity) != len(cut_band):
                raise ValueError('If sparsity is not None must be list with' +
                                 ' same length as Cut Index')
        else:
            sparsity = [None]*len(cut_band)

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(cat_df))
        test_len = len(cat_df) - train_len

        shuffled_idx = random_state.choice(np.arange(len(cat_df)),
                                           size=len(cat_df),
                                           replace=False)
                                        
        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[shuffled_idx[:train_len]]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[shuffled_idx[train_len:]]
        mag_index_dict = {'u':1, 'g':2, 'r':3, 'i':4, 'z':5, 'y':6}
        train_labels = np.zeros(train_len)
        test_labels = np.zeros(test_len)

        # Cut out all points in color space
        for mag_on, mag_low, mag_high, sparse_factor in zip(
                cut_band, cut_low, cut_high, sparsity):

            mag_val = mag_index_dict[mag_on]
            print(mag_val, mag_low, mag_high, sparse_factor)

            train_cut_idx = np.where((train_input.iloc[:, mag_val]
                                      >= mag_low) &
                                     (train_input.iloc[:, mag_val]
                                      < mag_high))[0]
            test_cut_idx = np.where((test_input.iloc[:, mag_val] >= mag_low) &
                                    (test_input.iloc[:, mag_val]
                                     < mag_high))[0]

            if sparse_factor is not None:
                train_cut_idx = np.delete(train_cut_idx,
                                          slice(None, None, sparse_factor))

            train_labels[train_cut_idx] = 1
            test_labels[test_cut_idx] = 1

        if plot_color is True:
            self.plot_color_color(train_input[['u', 'g', 'r',
                                               'i', 'z', 'y']].values[
                                                np.where(train_labels == 0)],
                                  os.path.join(out_dir,
                                               '%s.pdf' % cc_plot_name))

        train_input = train_input.iloc[np.where(train_labels == 0)]
        train_len = len(train_input)
        print('Training set size: %i. Test set size: %i.' % (train_len,
                                                             test_len))

        train_input.to_csv(os.path.join(out_dir,
                                        'train_cat_%s.dat' % out_suffix),
                           index=False)
        if save_test is True:
            test_input.to_csv(os.path.join(out_dir,
                                           'test_cat_%s.dat' % out_suffix),
                              index=False)
        np.savetxt(os.path.join(out_dir, 'train_labels_%s.dat' % out_suffix),
                   train_labels)
        np.savetxt(os.path.join(out_dir, 'test_labels_%s.dat' % out_suffix),
                   test_labels)
        
    def create_redshift_cut_cats(self, out_suffix, train_len, z_cut_low,
                                 z_cut_high, sparsity=None, out_dir='.',
                                 plot_color=True, save_test=False,
                                 cc_plot_name='redshift_cut_color_color',
                                 random_state=None):

        if ((type(random_state) is int) | (random_state is None)):
            random_state = np.random.RandomState(random_state)
        
        cat_df = pd.DataFrame(self.get_catalog())

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = int(train_len*len(cat_df))
        test_len = len(cat_df) - train_len

        shuffled_idx = random_state.choice(np.arange(len(cat_df)),
                                           size=len(cat_df),
                                           replace=False)
                                        
        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[shuffled_idx[:train_len]]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[shuffled_idx[train_len:]]

        if sparsity is None:
            # Cut out all points in redshift space
            train_input = train_input.query('redshift < %f or redshift > %f' %
                                            (z_cut_low, z_cut_high))
        else:
            keep_idx = np.where((train_input['redshift'].values >= z_cut_low) &
                                (train_input['redshift'].values
                                 <= z_cut_high))[0]
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
        if save_test is True:
            test_input.to_csv(os.path.join(out_dir,
                                           'test_cat_%s.dat' % out_suffix),
                              index=False)
