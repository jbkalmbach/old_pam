import pandas as pd
import numpy as np
import sys
from photoz_nn import photoz_nn

if __name__ == "__main__":

    train_suffix = sys.argv[1]
    test_suffix = sys.argv[2]
    n_epochs = int(sys.argv[3])

    pz_nn = photoz_nn(seed=1446)

    train_filename = '../data/train_cat_%s.dat' % train_suffix
    train_colors, train_specz = pz_nn.load_catalog(train_filename)
    test_filename = '../data/test_cat_%s.dat' % test_suffix
    test_colors, test_specz = pz_nn.load_catalog(test_filename)

    train_len = len(train_colors)
    test_len = len(test_colors)

    net, error = pz_nn.train_model(train_colors, train_specz,
                                   n_epochs, return_error=True)

    pz_nn.save_model(net, '../data/pz_network_%s.pt' % train_suffix)

    train_colors, train_specz = pz_nn.load_catalog(train_filename)
    train_photoz = pz_nn.run_model(net, train_colors)
    test_photoz = pz_nn.run_model(net, test_colors)

    train_results = {'true_z': train_specz.reshape(train_len),
                     'photo_z': train_photoz.reshape(train_len)}
    train_results_df = pd.DataFrame.from_dict(data=train_results)
    test_results = {'true_z': test_specz.reshape(test_len),
                    'photo_z': test_photoz.reshape(test_len)}
    test_results_df = pd.DataFrame.from_dict(data=test_results)
    train_results_df.to_csv('../data/train_results_%s.csv' % train_suffix,
                            index=False)
    test_results_df.to_csv('../data/test_results_%s.csv' % train_suffix,
                           index=False)
    np.savetxt('../data/train_error_%s.csv' % train_suffix, error)
