import numpy as np
from sklearn.neighbors import NearestNeighbors


class pz_errors():

    def __init__():

        return

    def nn_error(self, train_input, test_input,
                 train_z, test_z, use_colors=True):

        if use_colors is True:
            train_input = train_input[:, :-1] - train_input[:, 1:]
            test_input = test_input[:, :-1] - test_input[:, 1:]  

        nn_obj = NearestNeighbors(n_neighbors=100)
        nn_obj.fit(train_input)
        dist, nn_idx = nn_obj.kneighbors(test_input)

        test_err = []

        for test_idx in range(len(test_z)):
            true_z_vals = train_z['true_z'].iloc[nn_idx]
            photo_z_vals = train_z['test_z'].iloc[nn_idx]
            z_errs = photo_z_vals - true_z_vals
            l, r = np.percentile(z_errs, [16, 84])
            test_err.append(r-l)

        return test_err
