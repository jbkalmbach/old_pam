import sys
import matplotlib as mpl
mpl.use('pdf')
from gan_nn import gan_nn

if __name__ == "__main__":

    train_suffix = sys.argv[1]
    use_colors = bool(int(sys.argv[2]))
    n_epochs = int(sys.argv[3])

    pz_gan = gan_nn(seed=1222, use_colors=use_colors)

    train_filename = '../data/train_cat_%s.dat' % train_suffix
    # Load in training set with a k-means group taken out
    train_cat = pz_gan.load_catalog(train_filename)

    pz_gan.train_gan(train_cat, n_epochs, plot_suffix=train_suffix)
    gan_cat_df = pz_gan.create_gan_cat(10000)
    gan_cat_df.to_csv('../data/train_cat_gan_%s.dat' % train_suffix, index=False)
