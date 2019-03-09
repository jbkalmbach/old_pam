import sys
from gan_nn import gan_nn

if __name__ == "__main__":

    train_suffix = sys.argv[1]
    use_colors = bool(int(sys.argv[2]))

    pz_gan = gan_nn(seed=1222, use_colors=use_colors)

    train_filename = '../data/train_cat_%s.dat' % train_suffix
    # Load in training set with a k-means group taken out
    train_cat = pz_gan.load_catalog(train_filename)

    pz_gan.train_gan(train_cat, 20)
