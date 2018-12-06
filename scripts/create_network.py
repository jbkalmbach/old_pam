import pandas as pd
import sys
from photoz_nn import photoz_nn

if __name__ == "__main__":

    cat_suffix = sys.argv[1]

    pz_nn = photoz_nn(seed=1446)

    train_filename = '../data/train_cat_%s.dat' % cat_suffix
    train_colors, train_specz = pz_nn.load_catalog(train_filename)
    test_filename = '../data/test_cat_%s.dat' % cat_suffix
    test_colors, test_specz = pz_nn.load_catalog(test_filename)

    train_len = len(train_colors)
    test_len = len(test_colors)

    net = pz_nn.train_model(train_colors, train_specz)

    full_train_pz = pz_nn.run_model(net, train_colors)
    full_test_pz = pz_nn.run_model(net, test_colors)

    pz_nn.save_model(net, '../data/pz_network_%s.pt' % cat_suffix)

    train_results = {'true_z':train_specz.reshape(train_len),
                     'photo_z': full_train_pz.reshape(train_len)}
    train_results_df = pd.DataFrame.from_dict(data=train_results)
    test_results = {'true_z':test_specz.reshape(test_len),
                    'photo_z': test_photoz.reshape(test_len)}
    test_results_df = pd.DataFrame.from_dict(data=test_results)
    train_results_df.to_csv('../data/train_results_%s.csv' % cat_suffix, index=False)
    test_results_df.to_csv('../data/test_results_%s.csv' % cat_suffix, index=False)
