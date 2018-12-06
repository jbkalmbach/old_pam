from create_cats import create_cats

if __name__ == "__main__":

    names = ['index', 'redshift', 'u', 'g', 'r', 'i',
             'z', 'y', 'g_abs', 'r_abs']

    cat_name = 'Euclid_trim_25p2_3p5.dat'
    filename = os.path.join(os.environ['PZ_CAT_FOLDER'],
                            cat_name)

    cc = create_cats(filename, names)
    # cc.create_base_cats('full', 500000, out_dir='/home/brycek/sd_card/pam/data')
    # cc.create_sparse_cats('sparse', 500000, sparsity=5,
    #                       out_dir='/home/brycek/sd_card/pam/data')
    # cc.create_color_cut_cats('color_gap_2', 500000, 8, choose_out=2, plot_color=False,
    #                          out_dir='/home/brycek/sd_card/pam/data', random_state=17)
    cc.create_redshift_cut_cats('z_2_cut', 500000, 2.0, 5.0, sparsity=4,
                                out_dir='/home/brycek/sd_card/pam/data',
                                plot_color=False,
                                cc_plot_name='redshift_cut_color_color')