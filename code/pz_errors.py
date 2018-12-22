import numpy as np
from sklearn.neighbors import NearestNeighbors


class pz_errors():

    def __init__(self):

        return

    def nn_error(self, train_input, test_input,
                 train_z, test_z,
                 n_neighbors=100, calc_colors=False):

        if calc_colors is True:
            train_input = train_input[:, :-1] - train_input[:, 1:]
            test_input = test_input[:, :-1] - test_input[:, 1:]  

        nn_obj = NearestNeighbors(n_neighbors=n_neighbors)
        print('Fitting Training Set')
        nn_obj.fit(train_input)
        print('Calculating Test Set Nearest Neighbors')
        dist, nn_idx = nn_obj.kneighbors(test_input)

        test_err = []
        print('Calculating Errors')
        for test_idx in range(len(test_z)):
            if test_idx % 10000 == 0:
                print(test_idx)
            true_z_vals = train_z['true_z'].iloc[nn_idx[test_idx]]
            photo_z_vals = train_z['photo_z'].iloc[nn_idx[test_idx]]
            z_errs = photo_z_vals - true_z_vals
            l, r = np.percentile(z_errs, [16, 84])
            test_err.append(r-l)

        return test_err
