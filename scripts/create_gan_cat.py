import sys
import matplotlib as mpl
mpl.use('pdf')
import pandas as pd
from gan_nn import gan_nn

if __name__ == "__main__":

    train_suffix = sys.argv[1]
    use_colors = bool(int(sys.argv[2]))
    cat_length = int(sys.argv[3]) # Total catalog length including training size

    pz_gan = gan_nn(seed=1222, use_colors=use_colors)

    train_filename = '../data/train_cat_%s.dat' % train_suffix
    # Load in training set with a k-means group taken out
    train_cat = pz_gan.load_catalog(train_filename)
    train_cat_df = pd.read_csv(train_filename)

    pz_gan.load_model('../data/gan_model_%s.pt' % train_suffix)
    train_len = len(train_cat)
    if cat_length <= train_len:
        raise ValueError('New catalog must be greater than training catalog size.')
    add_samples = cat_length - train_len
    gan_cat_df = pz_gan.create_gan_cat(add_samples)
    gan_cat_complete_df = pd.concat([train_cat_df, gan_cat_df])
    gan_cat_complete_df.to_csv('../data/train_cat_gan_%s_%ik.dat' %
                               (train_suffix, cat_length/1000),
                               index=False)
