import sys
import matplotlib as mpl
mpl.use('pdf')
import pandas as pd
from gan_nn import gan_nn

if __name__ == "__main__":

    train_suffix = sys.argv[1]
    use_colors = bool(int(sys.argv[2]))
    n_epochs = int(sys.argv[3])

    pz_gan = gan_nn(seed=1222, use_colors=use_colors)

    train_filename = '../data/train_cat_%s.dat' % train_suffix
    # Load in training set with a k-means group taken out
    train_cat = pz_gan.load_catalog(train_filename)
    train_cat_df = pd.read_csv(train_filename)

    pz_gan.train_gan(train_cat, n_epochs, plot_suffix=train_suffix)
    pz_gan.save_model('../data/gan_model_%s.pt' % train_suffix)
