import sys
import numpy as np
import pandas as pd
from photoz_nn import photoz_nn
from pz_errors import pz_errors as pze

if __name__ == "__main__":

    cat_suffix = sys.argv[1]

    pz_nn = photoz_nn()
    train_filename = '../data/train_cat_%s.dat' % cat_suffix
    train_colors, train_specz = pz_nn.load_catalog(train_filename)
    test_filename = '../data/test_cat_%s.dat' % cat_suffix
    test_colors, test_specz = pz_nn.load_catalog(test_filename)

    train_results_dict = {}
    test_results_dict = {}

    #for cat_suffix in ['full', 'group_color_cut_3', 'group_color_cut_1', 'r_23_mag_cut', 'r_23_mag_cut_keep_10',
    #'sparse', 'ug_color_cut', 'z_2_cut', 'z_2_cut_all']:
    train_results_dict[cat_suffix] = pd.read_csv('../data/train_results_%s.csv' % cat_suffix)
    test_results_dict[cat_suffix] = pd.read_csv('../data/test_results_%s.csv' % cat_suffix)

    pz_err = pze()
    full_error = pz_err.nn_error(train_colors, test_colors[:500000],
                                 train_results_dict[cat_suffix], test_results_dict[cat_suffix].iloc[:500000],
                                 n_neighbors=100)

    np.savetxt('../data/full_error.csv', full_error)
