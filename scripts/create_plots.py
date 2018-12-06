from plot_pz_nn import plot_pz_nn

if __name__ == "__main__":

    # cat_suffixes = ['full', 'sparse']
    # cat_suffixes = ['full', 'color_gap_2', 'color_gap_4', 'color_gap_7']
    cat_suffixes = ['full', 'high_z_cut', 'z_2_cut']
    plot_pz = plot_pz_nn()

    color_gap = False
    train_df_list = []
    test_df_list = []

    for suffix in cat_suffixes:
        train_df_list.append(pd.read_csv('../data/train_results_%s.csv' % suffix))
        test_df_list.append(pd.read_csv('../data/test_results_%s.csv' % suffix))

        plot_pz.plot_single_results(train_df_list[-1], test_df_list[-1], suffix)

    out_str = 'compare'
    for suffix in cat_suffixes:
        out_str += '_%s' % suffix

    plot_pz.plot_multiple_results(train_df_list, test_df_list, cat_suffixes, out_str)

    if color_gap is True:

        train_df_list = []
        test_df_list = []

        for suffix in cat_suffixes:

            train_df = pd.read_csv('../data/train_results_%s.csv' % suffix)
            test_df = pd.read_csv('../data/test_results_%s.csv' % suffix)

            if suffix == 'full':
                train_df_list.append(train_df)
                test_df_list.append(test_df)
            else:
                label_list = np.genfromtxt('../data/test_labels_%s.dat' % suffix)
                train_df_list.append(train_df)
                keep_idx = np.where(label_list == int(suffix[-1]))
                test_df = test_df.iloc[keep_idx].reset_index(drop=True)
                print(len(test_df))
                test_df_list.append(test_df)

        out_str = 'compare_gap'
        for suffix in cat_suffixes:
            out_str += '_%s' % suffix

        plot_pz.plot_gap_results(train_df_list, test_df_list, cat_suffixes, out_str)