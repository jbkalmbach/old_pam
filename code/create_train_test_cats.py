import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    def create_base_cats(self, out_suffix, train_len=500000,
                         out_dir='.', plot_color=True,
                         cc_plot_name='cat_color_color'):

        cat_df = pd.DataFrame(self.get_catalog())

        # If train_len < 1 then it is a fraction of catalog
        if train_len < 1.0:
            train_len = train_len*len(pz_cat)
        test_len = len(cat_df) - train_len

        train_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                              'z', 'y']].iloc[:train_len]
        test_input = cat_df[['redshift', 'u', 'g', 'r', 'i',
                             'z', 'y']].iloc[train_len:]

        if plot_color is True:
            self.plot_color_color(cat_df[['u', 'g', 'r',
                                          'i', 'z', 'y']].values,
                                  '%s.pdf' % cc_plot_name)

        print('Training set size: %i. Test set size: %i.' % (train_len,
                                                             test_len))

        train_input.to_csv('train_cat_%s.dat' % out_suffix, index=False)
        test_input.to_csv('test_cat_%s.dat' % out_suffix, index=False)


if __name__ == "__main__":

    names = ['index', 'redshift', 'u', 'g', 'r', 'i',
             'z', 'y', 'g_abs', 'r_abs']

    cat_name = 'Euclid_trim_25p2_3p5.dat'
    filename = os.path.join('/Users/Bryce/Desktop/NN_PhotoZ/',
                            'scripts_bryce/cats',
                            cat_name)

    cc = create_cats(filename, names)
    cc.create_base_cats('full')