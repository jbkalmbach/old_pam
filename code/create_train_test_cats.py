import numpy as np


class create_train_test_cats():

    def __init__(self, cat_file, names):

        self.cat_file = cat_file
        self.cat_names = names

    def get_catalog(self):

        cat_array = np.genfromtxt(self.cat_file, names=['index', 'redshift',
                                                   'u', 'g', 'r', 'i',
                                                   'z', 'y', 'g_abs',
                                                   'r_abs'])

        return cat_df

    def create_base_cats(self, train_length=500000):

        cat_fraction = False

        if train_length < 1.0:
            cat_fraction = True

        train_cat = data_reader()
        filename = os.path.join(os.environ['PZ_CAT_FOLDER'],
                                cat_name)
        cat_array = train_cat.get_catalog(filename)